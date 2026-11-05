# =============================================================================
# From formula + data to design matrices, following R's `model.matrix`.
#
# brms builds its population-level design with `model.matrix`, so matching
# brms means matching R's coding rules exactly — which columns a factor
# expands into, in which order, and under which names. Getting a rule wrong
# here does not fail: it silently fits a differently parameterised model, and
# comparisons against brms are what catch it.
#
# The two rules that matter (R's `TermCode`, src/library/stats/src/model.c):
#
#   1. A factor in a term is coded by CONTRASTS (treatment: drop the first
#      level) if the term with that factor removed is empty or is contained
#      in some EARLIER term. Otherwise it gets FULL dummy coding.
#   2. With no intercept, the first factor in the first term that contains a
#      factor gets full coding instead (so `y ~ 0 + g` gives one column per
#      level).
#
# Within an interaction, the FIRST factor's columns vary fastest, and column
# names join the atom names with `:`. A factor's column name is the variable
# name with the level appended (`gb`), and a logical is coded as a factor with
# levels FALSE/TRUE (`xTRUE`).
# =============================================================================

"""
    AtomInfo

What the training data said about one atom: `levels === nothing` for a
continuous variable, else the factor levels in coding order. Stored so that
new data is coded identically (same levels, same columns).
"""
struct AtomInfo
    levels::Union{Nothing,Vector{Any}}
    basis::Any           # nothing, or a PolyBasis / SplineBasis (poly, bs, ns atoms)
end
AtomInfo(levels) = AtomInfo(levels, nothing)

iscategorical(info::AtomInfo) = info.levels !== nothing

# --- poly(x, degree) ---------------------------------------------------------
#
# R's orthogonal polynomials. stats::poly orthogonalises the centred monomials
# by QR and rescales by diag(R), which makes column k the MONIC polynomial of
# degree k orthogonal (over the training x) to all lower ones — exactly what
# the three-term (Stieltjes) recurrence produces, and that recurrence with the
# stored `alpha`, `norm2` is how predict.poly codes new data:
#     Z_0 = 1, Z_1 = x - alpha_1,
#     Z_k = (x - alpha_k) Z_{k-1} - (norm2_{k+1} / norm2_k) Z_{k-2}
# each column divided by sqrt(norm2_{k+2}) (R's 1-based norm2 = c(1, n, ...)).
# `raw = TRUE` gives plain powers x, x^2, ...

struct PolyBasis
    degree::Int
    raw::Bool
    alpha::Vector{Float64}
    norm2::Vector{Float64}     # R's norm2, length degree + 2
end

function _poly_eval(alpha, norm2, degree, x)
    n = length(x)
    Z = ones(Float64, n, degree + 1)
    degree >= 1 && (Z[:, 2] .= x .- alpha[1])
    for k in 2:degree
        @views Z[:, k + 1] .= (x .- alpha[k]) .* Z[:, k] .- (norm2[k + 1] / norm2[k]) .* Z[:, k - 1]
    end
    return Z
end

function PolyBasis(x::AbstractVector{<:Real}, degree::Int, raw::Bool)
    degree >= 1 || throw(ArgumentError("poly() needs degree >= 1"))
    raw && return PolyBasis(degree, true, Float64[], Float64[])
    length(unique(x)) > degree ||
        throw(ArgumentError("poly(): degree $degree needs more than $degree unique values"))
    xf = Float64.(x)
    alpha = zeros(degree); norm2 = ones(degree + 2)
    # build the columns one at a time, each alpha from the columns so far
    Z = ones(Float64, length(xf), degree + 1)
    norm2[2] = length(xf)
    for k in 1:degree
        alpha[k] = sum(xf .* Z[:, k] .^ 2) / norm2[k + 1]
        if k == 1
            Z[:, 2] .= xf .- alpha[1]
        else
            @views Z[:, k + 1] .= (xf .- alpha[k]) .* Z[:, k] .- (norm2[k + 1] / norm2[k]) .* Z[:, k - 1]
        end
        norm2[k + 2] = sum(abs2, @view Z[:, k + 1])
    end
    return PolyBasis(degree, false, alpha, norm2)
end

function _poly_block(pb::PolyBasis, x)
    xf = Float64.(x)
    pb.raw && return reduce(hcat, [xf .^ k for k in 1:pb.degree])
    Z = _poly_eval(pb.alpha, pb.norm2, pb.degree, xf)
    return Z[:, 2:end] ./ sqrt.(pb.norm2[3:end])'
end

# `poly(x, k)` / `poly(x, k, raw = TRUE)` as an atom: (inner expression, degree, raw)
function _poly_args(ex)
    (ex isa Expr && ex.head == :call && ex.args[1] === :poly) || return nothing
    length(ex.args) >= 3 && ex.args[3] isa Integer ||
        throw(ArgumentError("poly() needs a literal integer degree: poly(x, 2)"))
    raw = false
    for a in ex.args[4:end]
        a isa Expr && a.head == :kw && a.args[1] === :raw ||
            throw(ArgumentError("poly(): only `raw = TRUE` is supported as an extra argument"))
        raw = a.args[2] in (:TRUE, :T, true)
    end
    return ex.args[2], Int(ex.args[3]), raw
end

_basis_ncol(pb::PolyBasis) = pb.degree
_basis_block(pb::PolyBasis, x) = _poly_block(pb, x)

# the variable inside a basis-expansion atom (poly / bs / ns), or nothing
function _basis_inner(ex)
    pa = _poly_args(ex)
    pa === nothing || return pa[1]
    sa = _spline_args(ex)
    return sa === nothing ? nothing : sa[1]
end

function _atom_info(a::Atom, v)
    pa = _poly_args(a.ex)
    sa = pa === nothing ? _spline_args(a.ex) : nothing
    pa === nothing && sa === nothing && return _atom_info(v)
    nonmissingtype(eltype(v)) <: Real ||
        throw(ArgumentError("$(a.ex.args[1])() needs a numeric variable"))
    pa === nothing || return AtomInfo(nothing, PolyBasis(v, pa[2], pa[3]))
    return AtomInfo(nothing, SplineBasis(v, sa[2], sa[3], sa[4]))
end

# Classify a column. Anything not a (non-Bool) Real is a factor, matching R,
# where character, factor and logical columns all expand into dummies.
function _atom_info(v)
    T = nonmissingtype(eltype(v))
    T <: AbstractArray && return AtomInfo(nothing)      # a matrix response, cbind(a, b, c)
    if T <: Bool
        return AtomInfo(Any[l for l in (false, true) if any(isequal(l), v)])
    elseif T <: Real
        return AtomInfo(nothing)
    end
    # DataAPI.levels respects a CategoricalArray's own level order and is a
    # sorted unique for anything else — R's `factor()` default.
    return AtomInfo(collect(Any, DataAPI.levels(v)))
end

_level_name(l::Bool) = l ? "TRUE" : "FALSE"
_level_name(l) = string(l)

# --- evaluating atoms against data -----------------------------------------
#
# A tiny interpreter instead of `eval`: an atom is a column name or a call
# tree over column names, numbers and functions. Functions are looked up in
# the formula's module. Doing it this way keeps formulas plain data (no
# closures, no world-age issues) and means a string formula behaves exactly
# like a macro one.
function _eval_atom(a::Atom, cols, mod)
    # a poly/bs/ns atom's values are its variable's; the basis is in its AtomInfo
    inner = _basis_inner(a.ex)
    v = _interp(inner === nothing ? a.ex : inner, cols, mod)
    v isa AbstractVector || (v = fill(v, _nrows(cols)))
    return v
end

_nrows(cols) = length(first(values(cols)))

function _interp(s::Symbol, cols, mod)
    haskey(cols, s) && return getproperty(cols, s)
    # brms's `0 + Intercept + x`: the intercept as an ordinary (uncentred)
    # coefficient b_Intercept, class b -- a column of ones unless the data has one
    s === :Intercept && return 1.0
    isdefined(mod, s) && return getfield(mod, s)
    throw(ArgumentError("variable `$s` is neither a data column nor defined in $(mod)"))
end
_interp(x::Number, cols, mod) = x
_interp(x::QuoteNode, cols, mod) = x.value
_interp(x::AbstractString, cols, mod) = x
function _interp(ex::Expr, cols, mod)
    if ex.head == :call
        f = ex.args[1]
        # R's I() only protects arithmetic from formula interpretation;
        # here it is the identity.
        f === :I && return _interp(ex.args[2], cols, mod)
        # cbind(a, b, c): a matrix response (multinomial, dirichlet, ...).
        # Returned as one row vector per observation so it passes through the
        # complete-case filter like any other atom; a row with any missing
        # entry is itself `missing`, so the whole row is dropped.
        if f === :cbind
            cs = map(a -> _interp(a, cols, mod), ex.args[2:end])
            return [any(c -> ismissing(c[i]), cs) ? missing : [c[i] for c in cs] for i in 1:_nrows(cols)]
        end
        fn = f isa Symbol ? _interp(f, NamedTuple(), mod) : _interp(f, cols, mod)
        args = map(a -> _interp(a, cols, mod), ex.args[2:end])
        return broadcast(fn, args...)
    elseif ex.head == :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        # qualified name, e.g. Base.log
        return getfield(_interp(ex.args[1], NamedTuple(), mod), ex.args[2].value)
    end
    throw(ArgumentError("unsupported expression `$ex` in a formula"))
end

# --- population-level design -------------------------------------------------

"""
    DesignSpec

Everything needed to rebuild a design matrix for new data: the terms, how
each factor in each term is coded, and the resulting column names. The
intercept is NOT a column here; brms handles it separately (centering).
"""
struct DesignSpec
    intercept::Bool
    terms::Vector{Vector{Atom}}
    full::Vector{Vector{Bool}}   # full[t][a]: atom a of term t gets full dummy coding
    colnames::Vector{String}
end

function design_spec(intercept::Bool, terms, infos::Dict{Atom,AtomInfo})
    full = [falses(length(t)) for t in terms]
    for (ti, t) in enumerate(terms), (ai, a) in enumerate(t)
        iscategorical(infos[a]) || continue
        rest = Set(x for x in t if x != a)
        isempty(rest) && continue                       # rule 1, empty margin
        any(pj -> rest ⊆ Set(terms[pj]), 1:ti-1) && continue   # rule 1, earlier margin
        full[ti][ai] = true
    end
    if !intercept                                        # rule 2
        found = false
        for (ti, t) in enumerate(terms)
            for (ai, a) in enumerate(t)
                if iscategorical(infos[a]) && length(infos[a].levels) > 1
                    full[ti][ai] = true
                    found = true
                    break
                end
            end
            found && break
        end
    end
    names = String[]
    for (ti, t) in enumerate(terms)
        append!(names, _term_names(t, full[ti], infos))
    end
    return DesignSpec(intercept, terms, full, names)
end

function _atom_colnames(a::Atom, full::Bool, info::AtomInfo)
    info.basis === nothing || return [a.label * string(k) for k in 1:_basis_ncol(info.basis)]
    iscategorical(info) || return [a.label]
    lv = full ? info.levels : info.levels[2:end]
    return [a.label * _level_name(l) for l in lv]
end

# first atom varies fastest, as in R
function _term_names(t, full, infos)
    cur = _atom_colnames(t[1], full[1], infos[t[1]])
    for j in 2:length(t)
        nxt = _atom_colnames(t[j], full[j], infos[t[j]])
        cur = [c * ":" * n for n in nxt for c in cur]
    end
    return cur
end

function _atom_block(v, full::Bool, info::AtomInfo)
    info.basis === nothing || return _basis_block(info.basis, v)
    if !iscategorical(info)
        return reshape(Float64.(v), :, 1)
    end
    lv = full ? info.levels : info.levels[2:end]
    B = zeros(Float64, length(v), length(lv))
    for (k, l) in enumerate(lv), i in eachindex(v)
        isequal(v[i], l) && (B[i, k] = 1.0)
    end
    # a value outside the training levels would silently give an all-zero
    # row, i.e. be treated as the reference level
    for i in eachindex(v)
        any(l -> isequal(v[i], l), info.levels) ||
            throw(ArgumentError("level `$(v[i])` was not present in the data the model was built from"))
    end
    return B
end

"""
    model_matrix(spec, vals, N) -> Matrix{Float64}

The design matrix (no intercept column) for atom values `vals`.
"""
function model_matrix(spec::DesignSpec, vals::Dict{Atom,<:AbstractVector}, infos, N::Int)
    blocks = Matrix{Float64}[]
    for (ti, t) in enumerate(spec.terms)
        cur = _atom_block(vals[t[1]], spec.full[ti][1], infos[t[1]])
        for j in 2:length(t)
            nxt = _atom_block(vals[t[j]], spec.full[ti][j], infos[t[j]])
            cur = hcat((cur[:, c] .* nxt[:, n] for n in axes(nxt, 2) for c in axes(cur, 2))...)
        end
        push!(blocks, cur)
    end
    isempty(blocks) && return zeros(Float64, N, 0)
    return reduce(hcat, blocks)
end

# --- grouping factors --------------------------------------------------------

"""
    GroupInfo

Levels of one grouping factor (`g`, or the combination `g1:g2`) and the
level index of every observation. Levels of a combination are the joined
labels `a_b`, as brms makes them.
"""
struct GroupInfo
    levels::Vector{String}
    index::Dict{String,Int}
end

function _group_values(gt::GroupTerm, vals)
    if length(gt.group) == 1
        return [_level_name(x) for x in vals[gt.group[1]]]
    end
    cols = [vals[a] for a in gt.group]
    return [join((_level_name(c[i]) for c in cols), "_") for i in eachindex(cols[1])]
end

function _group_info(gt::GroupTerm, vals, infos)
    if !isempty(gt.mm)
        # multi-membership: the union of every membership factor's levels
        lv = sort(unique(reduce(vcat, [_group_values(GroupTerm(true, Vector{Atom}[], m, false), vals) for m in gt.mm])))
        return GroupInfo(lv, Dict(l => i for (i, l) in enumerate(lv)))
    end
    if length(gt.group) == 1 && iscategorical(infos[gt.group[1]])
        # a single factor keeps its own level order (numeric groups sort
        # numerically, as R's factor() does)
        lv = [_level_name(l) for l in infos[gt.group[1]].levels]
    elseif length(gt.group) == 1
        lv = [_level_name(l) for l in sort(unique(vals[gt.group[1]]))]
    else
        lv = sort(unique(_group_values(gt, vals)))
    end
    return GroupInfo(lv, Dict(l => i for (i, l) in enumerate(lv)))
end

function _group_index(gi::GroupInfo, gt::GroupTerm, vals)
    labels = _group_values(gt, vals)
    return map(labels) do l
        haskey(gi.index, l) ||
            throw(ArgumentError("group level `$l` of `$(group_label(gt))` was not in the data the model was built from (new levels are not supported yet)"))
        gi.index[l]
    end
end

# --- naming ------------------------------------------------------------------

# brms's `make_stan_names`-style renaming of coefficient names: drop spaces
# and brackets, spell out operators. `log(z)` -> `logz`, `I(x^2)` -> `IxE2`.
function brms_rename(s::AbstractString)
    for (a, b) in ((" ", ""), ("(", ""), (")", ""), ("[", ""), ("]", ""),
                   (",", ""), ("\"", ""), ("'", ""), ("?", ""),
                   ("+", "P"), ("-", "M"), ("*", "MU"), ("/", "D"),
                   ("^", "E"), ("=", "EQ"))
        s = replace(s, a => b)
    end
    return s
end
