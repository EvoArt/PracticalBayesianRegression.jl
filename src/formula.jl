# =============================================================================
# brms formula syntax, parsed from Julia expressions.
#
# Julia's parser already accepts brms formulas unchanged:
#
#     y | trials(n) ~ x * z + log(w) + (1 + x | g) + (1 || g:h)
#
# parses to `Expr(:call, :~, <lhs>, <rhs>)`, with `|` as an ordinary call and
# `||` as its own `Expr(:||)` head. So there is no tokenizer here, only a walk
# over the parsed tree. The same walk serves the `@bf` macro and the string
# form `bf("y ~ x")`, so one formula string can be used in both R and Julia.
#
# Nothing here touches data. A formula is a description; `design.jl` turns it
# into matrices once the data is known.
# =============================================================================

"""
    Atom

One variable-like factor of a model term: either a bare data column (`x`) or
an expression evaluated against the data (`log(z)`, `I(x^2)`). Two atoms are
the same atom iff they print the same, which is also R's rule for term
identity.
"""
struct Atom
    label::String   # as written; becomes part of the coefficient name
    ex::Any         # a Symbol for a column, an Expr for anything computed
end
Atom(s::Symbol) = Atom(String(s), s)
Atom(ex::Expr) = Atom(_strip_spaces(string(ex)), ex)
Base.:(==)(a::Atom, b::Atom) = a.label == b.label
Base.hash(a::Atom, h::UInt) = hash(a.label, h)
Base.show(io::IO, a::Atom) = print(io, a.label)

# Julia prints `x^2` as `x ^ 2`; R prints it without spaces. Coefficient names
# are built from labels, so normalise here once.
_strip_spaces(s) = replace(s, " " => "")

"""
    GroupTerm

A group-level ("random") effect `(terms | group)` or `(terms || group)`.
`cor` is `true` for `|` (correlated effects, an LKJ prior on their
correlation) and `false` for `||` (independent effects).
"""
struct GroupTerm
    intercept::Bool
    terms::Vector{Vector{Atom}}
    group::Vector{Atom}     # [g], or [g1, g2] for the interaction g1:g2
    cor::Bool
    id::String              # brms's `(1 | p | g)` ID: terms sharing it are correlated, "" = none
    by::Any                 # gr(g, by = h): an Atom (sd/cor per level of h), or nothing
    mm::Vector{Vector{Atom}}    # mm(g1, g2): the membership factors (empty = ordinary)
    mmw::Vector{Atom}           # mm(..., weights = cbind(w1, w2)) (empty = equal weights)
    mmscale::Bool               # mm(..., scale = TRUE): weights sum to one per row
    student::Bool               # gr(g, dist = "student"): student-t group effects
    cov::Any                    # gr(g, cov = A): the name of a known covariance among levels (or nothing)
end
GroupTerm(intercept, terms, group, cor) = GroupTerm(intercept, terms, group, cor, "")
GroupTerm(intercept, terms, group, cor, id) = GroupTerm(intercept, terms, group, cor, id, nothing, Vector{Atom}[], Atom[], true)
GroupTerm(intercept, terms, group, cor, id, by, mm, mmw, mmscale) =
    GroupTerm(intercept, terms, group, cor, id, by, mm, mmw, mmscale, false, nothing)
GroupTerm(intercept, terms, group, cor, id, by, mm, mmw, mmscale, student) =
    GroupTerm(intercept, terms, group, cor, id, by, mm, mmw, mmscale, student, nothing)

group_label(gt::GroupTerm) = isempty(gt.mm) ? join((a.label for a in gt.group), ":") :
                              "mm" * join(brms_rename(join((a.label for a in m), ":")) for m in gt.mm)

"""
    RHS

The right-hand side of one formula: an intercept flag, the population-level
terms (each a product of atoms, so `x:z` is `[x, z]`), the group-level terms,
and any `offset(...)` atoms.
"""
struct RHS
    intercept::Bool
    terms::Vector{Vector{Atom}}
    groups::Vector{GroupTerm}
    offsets::Vector{Atom}
    mo::Vector{Atom}                 # monotonic effects mo(x)
    ar::Any                          # nothing, or (time, gr, p, q, kind, M): ar/ma/arma (kind :arma,
                                     # or :arma_cov with cov = TRUE), :cosy, :fcor, :lagsar, :errorsar
    sm::Vector{Any}                  # smooth terms: (kind, covariates, k, by, text) for s() / t2()
    cs::Vector{Vector{Atom}}         # category-specific terms cs(x) (ordinal models)
    me::Vector{Tuple{Atom,Atom,String}}   # measurement-error terms me(x, sdx): (x, sdx, brms label)
    gp::Vector{Any}                  # Gaussian processes gp(x, ...): (covariates, k, c, scale)
    mi::Vector{Atom}                 # mi(x): a predictor imputed by its own model (multivariate)
    car::Vector{Any}                 # car(W, gr = loc, type = ): (W expression, gr atom, type)
    moint::Vector{Vector{Atom}}      # interaction terms with mo(): mo(x):z, mo(x1):mo(x2) (atoms as written)
end
RHS(intercept, terms, groups, offsets, mo, ar, sm, cs, me, gp, mi, car) =
    RHS(intercept, terms, groups, offsets, mo, ar, sm, cs, me, gp, mi, car, Vector{Atom}[])
RHS(intercept, terms, groups, offsets) =
    RHS(intercept, terms, groups, offsets, Atom[], nothing, Any[], Vector{Atom}[], Tuple{Atom,Atom,String}[], Any[], Atom[], Any[])
RHS(intercept, terms, groups, offsets, mo, ar, sm) =
    RHS(intercept, terms, groups, offsets, mo, ar, sm, Vector{Atom}[], Tuple{Atom,Atom,String}[], Any[], Atom[], Any[])
_with_intercept(r::RHS, b::Bool) = RHS(b, r.terms, r.groups, r.offsets, r.mo, r.ar, r.sm, r.cs, r.me, r.gp, r.mi, r.car, r.moint)
_with_offsets(r::RHS, offs) = RHS(r.intercept, r.terms, r.groups, offs, r.mo, r.ar, r.sm, r.cs, r.me, r.gp, r.mi, r.car, r.moint)
# an atom written as mo(x)
_is_mo(a::Atom) = a.ex isa Expr && a.ex.head == :call && a.ex.args[1] === :mo

"""
    DparFormula

A formula for one distributional parameter. `dpar === :mu` is the main
formula; `sigma ~ x` etc. give distributional regression.
"""
struct DparFormula
    dpar::Symbol
    rhs::RHS
end

"""
    BrmsFormula

A parsed model formula, made by [`@bf`](@ref) or [`bf`](@ref). Add
[`lf`](@ref) formulas or [`set_mecor`](@ref) to it with `+`; adding two
formulas gives a multivariate [`MVFormula`](@ref).
"""
struct BrmsFormula
    response::Atom
    trials::Union{Nothing,Atom,Int}
    weights::Union{Nothing,Atom}
    cens::Union{Nothing,Atom}        # y | cens(c)
    cens_y2::Union{Nothing,Atom}     # y | cens(c, y2), interval upper bound
    dpars::Vector{DparFormula}   # dpars[1] is always :mu
    mod::Module
    text::String
    # non-linear models (nl = true): the mean is nlfun(nlpars..., nlvars...)
    nl::Bool
    nlfun::Any
    nlpars::Vector{Symbol}
    nlvars::Vector{Symbol}
    # further addition terms: :se => (atom, sigma::Bool), :trunc => (lb, ub)
    # (each an Atom or a number); rate() becomes an offset of mu
    extra::Dict{Symbol,Any}
end

function Base.show(io::IO, f::BrmsFormula)
    print(io, f.text)
end

"""
    bf(formula, dpar_formulas...; nl = false, center = true, mod = Main) -> BrmsFormula

Parse a model formula written in brms syntax and given as a string (or a
quoted expression). Strings let you copy formulas to and from R unchanged; in
Julia code, [`@bf`](@ref) does the same without quotes.

The first formula is `response ~ terms`; each further formula `dpar ~ terms`
models one distributional parameter of the family (`sigma ~ z`) or, with
`nl = true`, one non-linear parameter. See [`@bf`](@ref) for the terms.

# Keywords
- `nl`: the main formula's right-hand side is a non-linear Julia expression
  in parameters, each given its own formula (`bf("y ~ a * exp(-b * x)",
  "a ~ 1", "b ~ 1"; nl = true)`). The expression is compiled at run time, so
  inside a function call `brm` through `Base.invokelatest`.
- `center`: centre the population-level predictors, as brms does by default;
  with `center = false` the `Intercept` is the intercept at zero predictors.
- `mod`: the module in which functions used in the formula (`log(x)`,
  `myfun(x)`) and named matrices are looked up. Inside a package, pass
  `mod = @__MODULE__`.

# Examples
```julia
bf("y ~ x + (1 | g)")
bf("y ~ x + (1 | g)", "sigma ~ x")              # distributional model
bf("count | trials(n) ~ treatment + (1 | site)")
```
"""
function bf(exs...; mod::Module=Main, nl::Bool=false, nlfun=nothing, center::Bool=true)
    isempty(exs) && throw(ArgumentError("bf() needs at least a main formula"))
    raw = map(e -> e isa AbstractString ? Meta.parse(e) : e, exs)
    main = raw[1]
    _is_tilde(main) || throw(ArgumentError("expected a formula `y ~ ...`, got `$main`"))
    nlpars, nlvars = Symbol[], Symbol[]
    if nl
        # the main right-hand side is an ordinary Julia expression, NOT formula
        # syntax: `a * exp(-b * x)` must not be read as interaction terms
        nlpars = [e.args[2] for e in raw[2:end] if _is_tilde(e) && e.args[2] isa Symbol &&
                  _mentions(main.args[3], e.args[2])]
        isempty(nlpars) && throw(ArgumentError("a non-linear formula needs `par ~ ...` formulas for its parameters"))
        nlvars = setdiff(_free_symbols(main.args[3]), nlpars)
        if nlfun === nothing
            # string/expression form: compile the function here (world age: call
            # `brm` through `Base.invokelatest` when this runs inside a function).
            # `mod` defaults to Main; inside a package pass `mod = @__MODULE__`
            # (evaluating into another package's closed module breaks its precompilation)
            nlfun = Core.eval(mod, Expr(:->, Expr(:tuple, nlpars..., nlvars...), main.args[3]))
        end
    end
    parsed = map(e -> _fix_precedence(e), raw)
    resp, trials, weights, cens, y2, extra = _parse_lhs(parsed[1].args[2])
    center || (extra[:center] = false)     # brms bf(center = FALSE): the intercept is not centred
    dpars = DparFormula[DparFormula(:mu, nl ? RHS(false, Vector{Atom}[], GroupTerm[], Atom[]) : _parse_rhs(parsed[1].args[3]))]
    for e in parsed[2:end]
        _is_tilde(e) || throw(ArgumentError("expected `dpar ~ ...`, got `$e`"))
        d = e.args[2]
        d isa Symbol || throw(ArgumentError("left-hand side of a distributional formula must be a parameter name, got `$d`"))
        any(x -> x.dpar === d, dpars) && throw(ArgumentError("`$d` has two formulas"))
        push!(dpars, DparFormula(d, _parse_rhs(e.args[3])))
    end
    if haskey(extra, :rate)
        # rate(r): exposure, i.e. log(r) as an offset of mu
        r = dpars[1].rhs
        off = Atom(Expr(:call, :log, extra[:rate].ex))
        dpars[1] = DparFormula(:mu, _with_offsets(r, vcat(r.offsets, off)))
    end
    text = join((_strip_spaces(string(p)) for p in (nl ? raw : parsed)), ", ")
    return BrmsFormula(resp, trials, weights, cens, y2, dpars, mod, text, nl, nlfun, nlpars, nlvars, extra)
end
bf(f::BrmsFormula) = f

# does `ex` mention symbol `s` (outside call heads)?
_mentions(ex, s) = s in _free_symbols(ex)
# the symbols an expression reads, excluding called function names
function _free_symbols(ex)
    out = Symbol[]
    _walk_syms!(out, ex)
    return unique(out)
end
_walk_syms!(out, s::Symbol) = push!(out, s)
function _walk_syms!(out, ex::Expr)
    if ex.head == :call
        foreach(a -> _walk_syms!(out, a), ex.args[2:end])
    elseif ex.head == :.               # qualified name: a function, not data
        return
    else
        foreach(a -> _walk_syms!(out, a), ex.args)
    end
end
_walk_syms!(out, x) = nothing

struct LinearFormulas
    exs::Vector{Any}
end

"""
    lf(dpar_formulas...)

Formulas for distributional parameters, given as strings, to add to a model
formula with `+`, as brms's `lf()`. `bf("y ~ x") + lf("sigma ~ z")` is the same
model as `bf("y ~ x", "sigma ~ z")`. Non-linear models take all their formulas
in [`bf`](@ref) itself.

See also [`@lf`](@ref).
"""
lf(exs...) = LinearFormulas(Any[e isa AbstractString ? Meta.parse(e) : e for e in exs])

"""
    @lf(dpar_formulas...)

The macro form of [`lf`](@ref): `@bf(y ~ x) + @lf(sigma ~ z)` is
`@bf(y ~ x, sigma ~ z)`.
"""
macro lf(exs...)
    return :($(@__MODULE__).lf($(map(QuoteNode, exs)...)))
end
function Base.:+(f::BrmsFormula, l::LinearFormulas)
    f.nl && throw(ArgumentError("lf(): give a non-linear model's formulas to bf() itself"))
    dps = copy(f.dpars)
    for e in l.exs
        e = _fix_precedence(e)
        _is_tilde(e) && e.args[2] isa Symbol || throw(ArgumentError("lf(): expected `dpar ~ ...`, got `$e`"))
        d = e.args[2]
        any(x -> x.dpar === d, dps) && throw(ArgumentError("`$d` has two formulas"))
        push!(dps, DparFormula(d, _parse_rhs(e.args[3])))
    end
    text = join(vcat(f.text, [_strip_spaces(string(_fix_precedence(e))) for e in l.exs]), ", ")
    return BrmsFormula(f.response, f.trials, f.weights, f.cens, f.cens_y2, dps, f.mod, text, f.nl, f.nlfun,
                       f.nlpars, f.nlvars, f.extra)
end

struct _SetMecor
    flag::Bool
end

"""
    set_mecor(flag = true)

Whether the latent values of two or more `me()` terms are correlated, as
brms's `set_mecor`. Correlated latents are the default; add
`set_mecor(false)` to the formula with `+` for independent ones:

```julia
@bf(y ~ me(x1, sdx1) + me(x2, sdx2)) + set_mecor(false)
```
"""
set_mecor(flag::Bool=true) = _SetMecor(flag)
function Base.:+(f::BrmsFormula, m::_SetMecor)
    extra = copy(f.extra); extra[:mecor] = m.flag
    return BrmsFormula(f.response, f.trials, f.weights, f.cens, f.cens_y2, f.dpars, f.mod, f.text, f.nl, f.nlfun,
                       f.nlpars, f.nlvars, extra)
end

"""
    @bf(formula, dpar_formulas...) -> BrmsFormula
    @bf(formula, dpar_formulas..., nl = true)
    @bf(formula, dpar_formulas..., center = false)

Write a model formula in brms syntax with bare names. The first formula is
`response ~ terms`; each further formula `dpar ~ terms` models one
distributional parameter of the family. Functions in the formula are looked
up in the module where the macro is used. `nl` and `center` are as in
[`bf`](@ref).

Population-level terms follow R's rules: `x + z`, `x * z` (main effects and
interaction), `x:z` or `x & z` (interaction only), `0 + x` or `x - 1` (no
intercept), `log(x)`, `I(x^2)`, `poly(x, 2)`, `ns(x, 3)`, `bs(x, 4)`,
`offset(log(t))`. Julia parses `:` more loosely than `+`, so write
`(a + b):c` as `a:c + b:c`.

Group-level terms: `(1 + x | g)` (correlated), `(1 + x || g)` (uncorrelated),
`(1 | g1:g2)`, `(1 | g1/g2)`, `(1 | p | g)` (correlated across formulas),
`(1 | gr(g, by = h, cor = false, dist = "student", cov = A))`,
`(1 | mm(g1, g2, weights = cbind(w1, w2)))`.

Special terms: `mo(x)`, `cs(x)`, `me(x, sdx)`, `mi(x)`, `s(x)`, `t2(x, z)`,
`gp(x)`, `ar()`, `ma()`, `arma()`, `cosy()`, `unstr()`, `fcor(M)`, `sar(W)`,
`car(W, gr = g)`.

Addition terms on the response: `y | trials(n)`, `weights(w)`, `se(s)`,
`cens(c)`, `trunc(lb = 0)`, `rate(t)`, `mi()`, `thres(3)`, `dec(d)`,
`subset(s)`, `vreal(v)`, `vint(n)`, combined with `+`.

# Examples
```julia
@bf(y ~ x + (1 + x | g))
@bf(y ~ x, sigma ~ x)                            # distributional model
@bf(y ~ a * exp(-b * x), a ~ 1, b ~ 1 + (1 | g), nl = true)
@bf(y1 ~ x) + @bf(y2 ~ x) + set_rescor(false)    # multivariate model
```

See [`bf`](@ref) for the keywords and for formulas given as strings.
"""
macro bf(exs...)
    nl = any(e -> e isa Expr && e.head == :(=) && e.args[1] === :nl && e.args[2] === true, exs)
    center = !any(e -> e isa Expr && e.head == :(=) && e.args[1] === :center && e.args[2] in (false, :FALSE), exs)
    fs = [e for e in exs if !(e isa Expr && e.head == :(=))]
    qs = map(QuoteNode, fs)
    nl || return :($(@__MODULE__).bf($(qs...); mod=$(__module__), center=$center))
    # non-linear: compile the mean function as a closure in the caller's scope,
    # so functions in it resolve there and no `eval` is needed
    main = fs[1]
    pars = [e.args[2] for e in fs[2:end] if _is_tilde(e) && e.args[2] isa Symbol && _mentions(main.args[3], e.args[2])]
    vars = setdiff(_free_symbols(main.args[3]), pars)
    fn = Expr(:->, Expr(:tuple, pars..., vars...), main.args[3])
    return :($(@__MODULE__).bf($(qs...); mod=$(__module__), nl=true, nlfun=$(esc(fn))))
end

_is_tilde(ex) = ex isa Expr && ex.head == :call && length(ex.args) == 3 && ex.args[1] === :~

# --- operator precedence -------------------------------------------------------
#
# Julia's `:` binds MORE LOOSELY than `+`, `-`, `*` and `|`; R's binds more
# tightly. So Julia parses the R formula `y ~ x + a:b` as `(x + a):b`, and
# `y | trials(n) + weights(w)` as `(y | trials(n)) + weights(w)`. Both are
# rotated back to R's reading here, before anything else looks at the tree:
#     (L op M):R  ->  L op (M:R)          a:(B op C)  ->  (a:B) op C
# The one casualty is an explicitly parenthesised `(a + b):c`, which R reads
# as `a:c + b:c` — write that out, or use `&` (StatsModels' interaction
# operator, which Julia parses with the right precedence).
const _LOOSE_OPS = (:+, :-, :*, :|)

_is_call(ex, op) = ex isa Expr && ex.head == :call && ex.args[1] === op
_is_loose(ex) = ex isa Expr && ex.head == :call && ex.args[1] in _LOOSE_OPS && length(ex.args) >= 3

function _fix_precedence(ex)
    ex isa Expr || return ex
    args = map(_fix_precedence, ex.args)
    ex = Expr(ex.head, args...)
    _is_call(ex, :&) && (ex = Expr(:call, :(:), ex.args[2:end]...))
    _is_call(ex, :(:)) || return ex
    ops = ex.args[2:end]
    first_, last_ = ops[1], ops[end]
    if _is_loose(first_)
        # (L op M) : rest  ->  L op (M : rest)
        inner = Expr(:call, :(:), first_.args[end], ops[2:end]...)
        return _fix_precedence(Expr(:call, first_.args[1], first_.args[2:end-1]..., inner))
    elseif length(ops) >= 2 && _is_loose(last_)
        # rest : (B op C)  ->  (rest : B) op C
        inner = Expr(:call, :(:), ops[1:end-1]..., last_.args[2])
        return _fix_precedence(Expr(:call, last_.args[1], inner, last_.args[3:end]...))
    end
    return ex
end

# --- left-hand side ---------------------------------------------------------
#
# `y`, or `y | trials(n)`, `y | weights(w)`, `y | cens(c)`, `y | cens(c, y2)`,
# and sums of these (`y | weights(w) + trials(n)`).
# brms has more addition terms (se, cens, trunc, mi, ...); each is rejected
# by name rather than silently ignored, since ignoring one changes the model.
function _parse_lhs(lhs)
    # `y | a + b` parses as `(y | a) + b` (`|` and `+` share a precedence
    # level); regroup it as `y | (a + b)`
    if _is_call(lhs, :+) && _is_call(lhs.args[2], :|)
        bar = lhs.args[2]
        lhs = Expr(:call, :|, bar.args[2], Expr(:call, :+, bar.args[3], lhs.args[3:end]...))
    end
    if lhs isa Expr && lhs.head == :call && lhs.args[1] === :| && length(lhs.args) == 3
        resp = _atom(lhs.args[2])
        trials = nothing; weights = nothing; cens = nothing; y2 = nothing
        extra = Dict{Symbol,Any}()
        for add in _split_plus(lhs.args[3])
            add isa Expr && add.head == :call ||
                throw(ArgumentError("unrecognised addition term `$add`"))
            name = add.args[1]
            if name === :trials
                a = add.args[2]
                trials = a isa Integer ? Int(a) : _atom(a)
            elseif name === :weights
                weights = _atom(add.args[2])
            elseif name === :cens
                cens = _atom(add.args[2])
                length(add.args) >= 3 && (y2 = _atom(add.args[3]))
            elseif name === :se
                sig = false
                for a in add.args[3:end]
                    a isa Expr && a.head == :kw && a.args[1] === :sigma ||
                        throw(ArgumentError("se() takes `sigma = TRUE/FALSE` only"))
                    sig = a.args[2] in (true, :TRUE, :T)
                end
                extra[:se] = (_atom(add.args[2]), sig)
            elseif name === :rate
                extra[:rate] = _atom(add.args[2])
            elseif name === :mi
                # y | mi(): missing values of y are parameters (brms); needs a gaussian/student y
                extra[:mi] = true
            elseif name === :thres
                # y | thres(3), thres(gr = g), thres(3, gr = g): the number of
                # thresholds and/or one set of thresholds per level of g (brms)
                n = nothing; tg = nothing
                for a in add.args[2:end]
                    if a isa Expr && a.head == :kw && a.args[1] === :gr
                        tg = _atom(a.args[2])
                    elseif a isa Integer && a >= 1
                        n = Int(a)
                    else
                        throw(ArgumentError("thres() takes a number of thresholds and/or gr = "))
                    end
                end
                extra[:thres] = (n, tg)
            elseif name === :dec
                # rt | dec(d): the boundary each response time ended at (wiener)
                extra[:dec] = _atom(add.args[2])
            elseif name in (:vreal, :vint)
                # y | vreal(a, b) / vint(n): extra data for a custom family's log density (brms)
                append!(get!(extra, :vars, Atom[]), [_atom(a) for a in add.args[2:end]])
            elseif name === :subset
                # y | subset(s): this response uses only the rows where s is true (multivariate models)
                extra[:subset] = _atom(add.args[2])
            elseif name === :trunc
                lb, ub = -Inf, Inf
                for a in add.args[2:end]
                    a isa Expr && a.head == :kw && a.args[1] in (:lb, :ub) ||
                        throw(ArgumentError("trunc() takes `lb = ` and/or `ub = `"))
                    v = a.args[2] isa Number ? Float64(a.args[2]) : _atom(a.args[2])
                    a.args[1] === :lb ? (lb = v) : (ub = v)
                end
                extra[:trunc] = (lb, ub)
            else
                throw(ArgumentError("addition term `$name(...)` is not supported yet"))
            end
        end
        return resp, trials, weights, cens, y2, extra
    end
    return _atom(lhs), nothing, nothing, nothing, nothing, Dict{Symbol,Any}()
end

_split_plus(ex) = ex isa Expr && ex.head == :call && ex.args[1] === :+ ?
    reduce(vcat, map(_split_plus, ex.args[2:end])) : Any[ex]

_atom(s::Symbol) = Atom(s)
_atom(ex::Expr) = Atom(ex)
_atom(x) = throw(ArgumentError("`$x` cannot be used as a variable"))

# --- right-hand side --------------------------------------------------------

mutable struct _RHSState
    intercept::Bool
    terms::Vector{Vector{Atom}}
    groups::Vector{GroupTerm}
    offsets::Vector{Atom}
    mo::Vector{Atom}
    ar::Any
    sm::Vector{Any}
    cs::Vector{Vector{Atom}}
    me::Vector{Tuple{Atom,Atom,String}}
    gp::Vector{Any}
    mi::Vector{Atom}
    car::Vector{Any}
end

function _parse_rhs(ex)
    st = _RHSState(true, Vector{Atom}[], GroupTerm[], Atom[], Atom[], nothing, Any[], Vector{Atom}[],
                   Tuple{Atom,Atom,String}[], Any[], Atom[], Any[])
    _walk!(st, ex, true)
    # products containing mo() (mo(x):z, mo(x1):mo(x2), and the mo(x) of mo(x) * z)
    # are monotonic terms, not design columns (brms)
    terms = _normalise_terms(st.terms)
    moint = [t for t in terms if any(_is_mo, t)]
    terms = [t for t in terms if !any(_is_mo, t)]
    # a lone mo(x) from a product expansion is the plain monotonic term
    for t in moint
        length(t) == 1 && (a = _atom(t[1].ex.args[2]); a in st.mo || push!(st.mo, a))
    end
    filter!(t -> length(t) > 1, moint)
    return RHS(st.intercept, terms, st.groups, st.offsets, st.mo, st.ar, st.sm,
               _normalise_terms(st.cs), st.me, st.gp, st.mi, st.car, moint)
end

# `positive` is false inside the right operand of a binary minus, where the
# only thing we accept is the intercept (`y ~ x - 1`).
function _walk!(st, ex, positive)
    if ex isa Integer
        ex == 0 && (st.intercept = false; return)
        ex == 1 && (st.intercept = positive; return)
        throw(ArgumentError("`$ex` is not a valid formula term"))
    end
    if ex isa Expr && ex.head == :call
        f = ex.args[1]
        if f === :+
            foreach(a -> _walk!(st, a, positive), ex.args[2:end])
            return
        elseif f === :- && length(ex.args) == 3
            _walk!(st, ex.args[2], positive)
            ex.args[3] == 1 || throw(ArgumentError("only `- 1` (removing the intercept) is supported, got `$ex`"))
            st.intercept = false
            return
        elseif f === :- && length(ex.args) == 2 && ex.args[2] == 1
            st.intercept = false
            return
        elseif f === :|
            positive || throw(ArgumentError("cannot subtract a group term"))
            lhs = ex.args[2]
            if lhs isa Expr && lhs.head == :call && lhs.args[1] === :|
                # (terms | p | g): parsed as ((terms | p) | g)
                id = lhs.args[3]
                id isa Symbol || throw(ArgumentError("a group-term ID must be a name, as in (1 | p | g)"))
                append!(st.groups, _group_terms(lhs.args[2], ex.args[3], true; id=string(id)))
            else
                append!(st.groups, _group_terms(lhs, ex.args[3], true))
            end
            return
        elseif f === :offset
            push!(st.offsets, _atom(ex.args[2]))
            return
        elseif f === :mo
            push!(st.mo, _atom(ex.args[2]))
            return
        elseif f in (:ar, :ma, :arma)
            # ar(time, gr, p) / ma(time, gr, q) / arma(time, gr, p, q); several
            # of them on the same series combine into one ARMA term
            t = _parse_ar(ex)
            if st.ar === nothing
                st.ar = t
            else
                st.ar[5] in (:arma, :arma_cov) && t[5] === st.ar[5] ||
                    throw(ArgumentError("ar/ma terms cannot be combined with other autocorrelation terms, " *
                                        "and must agree on `cov`"))
                st.ar[1] == t[1] && st.ar[2] == t[2] ||
                    throw(ArgumentError("ar/ma terms must share `time` and `gr`"))
                st.ar = (t[1], t[2], max(st.ar[3], t[3]), max(st.ar[4], t[4]), t[5], nothing)
            end
            return
        elseif f in (:cosy, :fcor, :sar, :unstr)
            # residual covariance structures (brms; timecov.jl): one per model
            st.ar === nothing || throw(ArgumentError("only one autocorrelation structure per model is supported"))
            st.ar = _parse_cov_ac(ex)
            return
        elseif f === :car
            # car(W, gr = loc, type = "escar"): W from data2 (or the calling module)
            W = ex.args[2]; gr = nothing; type = :escar
            for a in ex.args[3:end]
                (a isa Expr && a.head == :kw) || throw(ArgumentError("car(W, gr = , type = )"))
                a.args[1] === :gr ? (gr = _atom(a.args[2])) :
                a.args[1] === :type ? (type = Symbol(strip(string(a.args[2]), '"'))) :
                throw(ArgumentError("car(): unsupported argument `$(a.args[1])`"))
            end
            gr === nothing && throw(ArgumentError("car() needs gr = (the location of each observation)"))
            type in (:escar, :esicar, :icar, :bym2) ||
                throw(ArgumentError("car(): type must be escar, esicar, icar or bym2"))
            push!(st.car, (W, gr, type))
            return
        elseif f === :mi
            length(ex.args) == 2 || throw(ArgumentError("mi() takes one variable: mi(x)"))
            push!(st.mi, _atom(ex.args[2]))
            return
        elseif f === :gp
            # gp(x, z, ...; k = , c = , scale = ): squared-exponential GP (brms defaults)
            covs = Atom[]; k = nothing; c = nothing; scale = true; cov = :exp_quad
            for a in ex.args[2:end]
                if a isa Expr && a.head == :kw
                    kw, v = a.args
                    if kw === :k
                        k = Int(v)
                    elseif kw === :c
                        c = _numconst(v)                                    # e.g. 5/4
                    elseif kw === :scale
                        scale = v in (true, :TRUE)
                    elseif kw === :cov
                        cov = Symbol(strip(string(v), '"'))
                        cov in (:exp_quad, :matern52, :matern32, :exponential) ||
                            throw(ArgumentError("gp(): cov must be exp_quad, matern52, matern32 or exponential"))
                    else
                        throw(ArgumentError("gp(): unsupported argument `$kw` (supported: k, c, scale, cov)"))
                    end
                else
                    push!(covs, _atom(a))
                end
            end
            isempty(covs) && throw(ArgumentError("gp() needs at least one covariate"))
            push!(st.gp, (covs, k, c, scale, cov))
            return
        elseif f === :me
            # me(x, sdx): x observed with standard error sdx (brms)
            length(ex.args) == 3 || throw(ArgumentError("me() takes the variable and its standard error: me(x, sdx)"))
            push!(st.me, (_atom(ex.args[2]), _atom(ex.args[3]), brms_rename(_strip_spaces(string(ex)))))
            return
        elseif f === :cs
            # cs(x): an effect per threshold (sratio / cratio / acat), brms
            inner = _parse_rhs(ex.args[2])
            isempty(inner.groups) || throw(ArgumentError("cs() takes population-level terms only"))
            append!(st.cs, inner.terms)
            return
        elseif f in (:s, :t2)
            # s(x, z, k = , by = ) thin-plate splines, t2(x, z, k = ) tensor products (splines.jl)
            covs = Atom[]; k = nothing; by = nothing; bs = :tp
            for a in ex.args[2:end]
                if a isa Expr && a.head == :kw
                    kw, v = a.args
                    if kw === :k
                        k = v isa Expr ? Int.(v.args[2:end]) : Int(v)       # t2(x, z, k = c(5, 6))
                    elseif kw === :by && f === :s
                        by = _atom(v)
                    elseif kw === :bs && f === :s
                        bs = Symbol(strip(string(v), '"'))
                        bs in (:tp, :cr, :ps, :cc, :ts, :cs, :re) || throw(ArgumentError("s(): bs = \"$bs\" is not supported yet (tp, ts, cr, cs, ps, cc, re; t2() for tensor products)"))
                    else
                        throw(ArgumentError("$f(): unsupported argument `$kw`"))
                    end
                else
                    push!(covs, _atom(a))
                end
            end
            isempty(covs) && throw(ArgumentError("$f() needs at least one covariate"))
            # brms's sds coef is the term as written: s(x, bs = "cr", k = 6)
            text = string(ex)
            push!(st.sm, (f, covs, k, by, text, bs))
            return
        end
    elseif ex isa Expr && ex.head == :||
        append!(st.groups, _group_terms(ex.args[1], ex.args[2], false))
        return
    end
    positive || throw(ArgumentError("cannot subtract term `$ex`"))
    append!(st.terms, _expand(ex))
    return
end

# a numeric constant written as arithmetic (gp(x, c = 5/4)), folded without
# `eval`: evaluating into this module breaks precompilation of any package
# that builds formulas at top level
_numconst(v::Number) = Float64(v)
function _numconst(e::Expr)
    e.head === :call && e.args[1] in (:+, :-, :*, :/, :^) ||
        throw(ArgumentError("expected a number, got `$e`"))
    return Float64(getfield(Base, e.args[1])(map(_numconst, e.args[2:end])...))
end
_numconst(v) = throw(ArgumentError("expected a number, got `$v`"))

# ar(time, gr, p = 1), ma(time, gr, q = 1), arma(time, gr, p = 1, q = 1),
# positional or keyword, as brms. Returns (time, gr, p, q).
function _parse_ar(ex)
    f = ex.args[1]
    time = nothing; gr = nothing; cov = false
    p = f === :ma ? 0 : 1
    q = f === :ar ? 0 : 1
    pos = 0
    for a in ex.args[2:end]
        if a isa Expr && a.head == :kw
            k, v = a.args
            if k === :time
                time = _atom(v)
            elseif k === :gr
                gr = _atom(v)
            elseif k === :p && f !== :ma
                p = Int(v)
            elseif k === :q && f !== :ar
                q = Int(v)
            elseif k === :cov
                cov = v in (true, :TRUE, :T)
                cov || v in (false, :FALSE, :F) || throw(ArgumentError("$f(): cov must be TRUE or FALSE"))
            else
                throw(ArgumentError("unknown $f() argument `$k`"))
            end
        else
            pos += 1
            if pos == 1
                time = _atom(a)
            elseif pos == 2
                gr = _atom(a)
            elseif pos == 3
                f === :ma ? (q = Int(a)) : (p = Int(a))
            elseif pos == 4 && f === :arma
                q = Int(a)
            else
                throw(ArgumentError("too many positional arguments to $f()"))
            end
        end
    end
    (f === :ma || p >= 1) && (f === :ar || q >= 1) && p >= 0 && q >= 0 ||
        throw(ArgumentError("$f(): orders must be at least 1"))
    # brms: the covariance formulation exists for order 1 only
    cov && (p > 1 || q > 1) && throw(ArgumentError("$f(cov = TRUE) supports order 1 only (as brms)"))
    return (time, gr, p, q, cov ? :arma_cov : :arma, nothing)
end

# cosy(time, gr), fcor(M), sar(M, type = "lag" | "error"), as brms. Returns
# the same 6-tuple as _parse_ar: (time, gr, p, q, kind, M expression).
function _parse_cov_ac(ex)
    f = ex.args[1]
    args = ex.args[2:end]
    if f in (:cosy, :unstr)
        time = nothing; gr = nothing; pos = 0
        for a in args
            if a isa Expr && a.head == :kw
                a.args[1] === :time ? (time = _atom(a.args[2])) :
                a.args[1] === :gr ? (gr = _atom(a.args[2])) :
                throw(ArgumentError("unknown $f() argument `$(a.args[1])`"))
            else
                pos += 1
                pos == 1 ? (time = _atom(a)) : pos == 2 ? (gr = _atom(a)) :
                throw(ArgumentError("too many positional arguments to $f()"))
            end
        end
        f === :unstr && (time === nothing || gr === nothing) &&
            throw(ArgumentError("unstr() needs both time and gr (brms)"))
        return (time, gr, 0, 0, f, nothing)
    end
    M = nothing; type = :lag
    for a in args
        if a isa Expr && a.head == :kw
            if f === :sar && a.args[1] === :type
                type = Symbol(strip(string(a.args[2]), '"'))
                type in (:lag, :error) || throw(ArgumentError("sar(): type must be \"lag\" or \"error\""))
            elseif a.args[1] === :M
                M = a.args[2]
            else
                throw(ArgumentError("unknown $f() argument `$(a.args[1])`"))
            end
        else
            M === nothing || throw(ArgumentError("too many positional arguments to $f()"))
            M = a
        end
    end
    M isa Symbol || throw(ArgumentError("$f(M): M must name a matrix (in data2 or the calling module)"))
    return (nothing, nothing, 0, 0, f === :fcor ? :fcor : Symbol(type, :sar), M)
end

# `(1 + x | g1/g2)` is brms/lme4 nesting: `(1 + x | g1) + (1 + x | g1:g2)`.
function _group_terms(effects, group, cor; id::String="")
    inner = _parse_rhs(effects)
    isempty(inner.groups) || throw(ArgumentError("nested `|` inside a group term"))
    if group isa Expr && group.head == :call && group.args[1] === :/
        id == "" || throw(ArgumentError("group-term IDs with nested grouping (g1/g2) are not supported"))
        outer = _group_atoms(group.args[2])
        inner_g = vcat(outer, _group_atoms(group.args[3]))
        return [GroupTerm(inner.intercept, inner.terms, outer, cor),
                GroupTerm(inner.intercept, inner.terms, inner_g, cor)]
    end
    if group isa Expr && group.head == :call && group.args[1] in (:gr, :mm)
        return [_special_group(inner, group, cor, id)]
    end
    return [GroupTerm(inner.intercept, inner.terms, _group_atoms(group), cor, id)]
end

# gr(g, by = h, cor = TRUE, id = ) and mm(g1, g2, weights = cbind(w1, w2), scale = TRUE)
function _special_group(inner, ex, cor, id)
    f = ex.args[1]
    pos = [a for a in ex.args[2:end] if !(a isa Expr && a.head == :kw)]
    kws = Dict(Symbol(a.args[1]) => a.args[2] for a in ex.args[2:end] if a isa Expr && a.head == :kw)
    if f === :gr
        length(pos) == 1 || throw(ArgumentError("gr() takes one grouping factor"))
        by = haskey(kws, :by) ? _atom(kws[:by]) : nothing
        haskey(kws, :cor) && (cor = kws[:cor] in (true, :TRUE))
        haskey(kws, :id) && (id = string(kws[:id]))
        dist = haskey(kws, :dist) ? strip(string(kws[:dist]), '"') : "gaussian"
        dist in ("gaussian", "student") || throw(ArgumentError("gr(): dist must be \"gaussian\" or \"student\""))
        covm = haskey(kws, :cov) ? kws[:cov] : nothing
        covm === nothing || covm isa Symbol || throw(ArgumentError("gr(): cov must name a matrix (in data2 or the calling module)"))
        for k in keys(kws)
            k in (:by, :cor, :id, :dist, :cov) || throw(ArgumentError("gr(): unsupported argument `$k` (supported: by, cor, id, dist, cov)"))
        end
        covm === nothing || (by === nothing && id == "" && dist != "student") ||
            throw(ArgumentError("gr(cov = ) together with by, an ID or dist = \"student\" is not supported yet"))
        by === nothing || id == "" || throw(ArgumentError("gr(): `by` together with an ID is not supported yet"))
        dist == "student" && (by === nothing && id == "" ||
            throw(ArgumentError("gr(dist = \"student\") together with `by` or an ID is not supported yet")))
        return GroupTerm(inner.intercept, inner.terms, _group_atoms(pos[1]), cor, id, by, Vector{Atom}[], Atom[], true,
                         dist == "student", covm)
    end
    length(pos) >= 2 || throw(ArgumentError("mm() needs at least two grouping factors"))
    id == "" || throw(ArgumentError("mm() with a group-term ID is not supported yet"))
    mm = [_group_atoms(p) for p in pos]
    w = Atom[]
    if haskey(kws, :weights)
        we = kws[:weights]
        (we isa Expr && we.head == :call && we.args[1] === :cbind && length(we.args) == length(pos) + 1) ||
            throw(ArgumentError("mm(): weights must be cbind(w1, w2, ...) with one column per factor"))
        w = [_atom(a) for a in we.args[2:end]]
    end
    scale = haskey(kws, :scale) ? kws[:scale] in (true, :TRUE) : true
    for k in keys(kws)
        k in (:weights, :scale) || throw(ArgumentError("mm(): unsupported argument `$k` (supported: weights, scale)"))
    end
    return GroupTerm(inner.intercept, inner.terms, reduce(vcat, mm), cor, id, nothing, mm, w, scale)
end

_group_atoms(g::Symbol) = [Atom(g)]
function _group_atoms(g::Expr)
    g.head == :call && g.args[1] === :(:) && return reduce(vcat, map(_group_atoms, g.args[2:end]))
    throw(ArgumentError("unsupported grouping factor `$g`"))
end

# Expand `*`, `:` and parenthesised sums into a list of product terms.
#   a * b   -> a, b, a:b
#   a : b   -> a:b
#   (a + b):c -> a:c, b:c
function _expand(ex)::Vector{Vector{Atom}}
    if ex isa Expr && ex.head == :call
        f = ex.args[1]
        if f === :*
            return foldl(_star, map(_expand, ex.args[2:end]))
        elseif f === :(:)
            return foldl(_colon, map(_expand, ex.args[2:end]))
        elseif f === :+
            return reduce(vcat, map(_expand, ex.args[2:end]))
        end
    end
    ex isa Symbol || ex isa Expr ||
        throw(ArgumentError("`$ex` is not a valid formula term"))
    return [[_atom(ex)]]
end

_colon(a, b) = [_union(x, y) for y in b for x in a]
_star(a, b) = vcat(a, b, _colon(a, b))

# atoms of a product, de-duplicated (`x:x` is `x`), order of first appearance
_union(x, y) = unique(vcat(x, y))

# R's term order: de-duplicate (as sets), then a stable sort by order (number
# of atoms), so all main effects come before all two-way interactions, etc.
# Coding decisions in design.jl depend on this order.
function _normalise_terms(terms)
    out = Vector{Atom}[]
    seen = Set{Set{Atom}}()
    for t in terms
        s = Set(t)
        s in seen && continue
        push!(seen, s); push!(out, t)
    end
    return sort!(out; by=length, alg=Base.Sort.DEFAULT_STABLE)
end
