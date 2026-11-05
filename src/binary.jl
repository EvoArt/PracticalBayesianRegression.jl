# =============================================================================
# Model binaries: `build_binary` compiles a model into a standalone
# executable that refits it on any CSV with the right columns, as a CmdStan
# executable does for its data block.
#
#     build_binary(@bf(y ~ x + (1 | g)), data; family=poisson(), output="mymodel")
#     $ mymodel/bin/model --data new.csv --output draws.csv --chains 4 --seed 1
#
# Three pieces, all checked at build time:
#   * the model's recipe (recipe.jl), serialised into the generated project and
#     loaded as a `const`, so its concrete type is known to the compiler;
#   * generated Julia source that evaluates every atom of the formula on a
#     DataTable (straight-line broadcasts: `log(abs(z) + 1)` becomes
#     broadcast(log, broadcast(+, broadcast(abs, c_z), 1))), checked against
#     the front end's own interpreter through the CSV path;
#   * `binary_main`, the trimmable driver: read, assemble, sample (the same
#     seeding and sampler calls as `brm`, so `--seed 1` reproduces
#     `brm(...; seed=1)` draw for draw), diagnostics, CSV out.
# The executable is built by juliac with --trim=safe.
# =============================================================================

# --------------------------------------------------------- run time: options

struct BinaryOptions
    data::String
    output::String
    chains::Int
    iter::Int
    warmup::Int
    seed::Int
    delta::Float64
    max_depth::Int
    metric::Symbol
    init::Float64
    threads::Bool
    summary::Bool
    help::Bool
end

const _BINARY_USAGE = """
usage: MODEL --data FILE.csv [options]

Fits the model this executable was built for to the data in FILE.csv (one
row per observation, a header row naming the columns).

  --data FILE         input CSV (required)
  --output FILE       draws CSV to write (default: draws.csv)
  --chains N          number of chains (default 4)
  --iter N            iterations per chain, warm-up included (default 2000)
  --warmup N          warm-up iterations (default iter / 2)
  --seed N            random seed (default: random)
  --delta X           target acceptance rate (default 0.8)
  --max-depth N       maximum tree depth (default 10)
  --metric diag|dense mass matrix (default diag)
  --init X            initial values uniform(-X, X), unconstrained (default 2)
  --no-threads        run chains one after another
  --no-summary        do not print the summary table
  --columns           print the columns this model needs, and exit
  --help              this message

Chains run in parallel when the program is started with more than one
thread, e.g.  JULIA_NUM_THREADS=4 MODEL --data FILE.csv  (default: one).
"""

function _parse_opt_int(k::String, v::String)
    x = tryparse(Int, v)
    x === nothing && throw(ArgumentError("$k needs an integer, got `$v`"))
    return x
end
function _parse_opt_float(k::String, v::String)
    x = tryparse(Float64, v)
    x === nothing && throw(ArgumentError("$k needs a number, got `$v`"))
    return x
end

# A plain index loop: closures capturing the reassigned `i` / `v` were boxed
# (Core.Box, i.e. Any), which juliac --trim cannot compile.
function _parse_binary_args(args::Vector{String})
    data = ""; output = "draws.csv"; chains = 4; iter = 2000; warmup = -1; seed = -1
    delta = 0.8; max_depth = 10; metric = :diag; init = 2.0; threads = true; summ = true; help = false
    n = length(args)
    i = 1
    while i <= n
        a = args[i]
        k = a
        v = ""
        hasv = false
        eq = findfirst('=', a)
        if startswith(a, "--") && eq !== nothing
            k = String(a[1:prevind(a, eq)])
            v = String(a[nextind(a, eq):end])
            hasv = true
        end
        if k == "--columns"
            return BinaryOptions("", "", 0, 0, 0, 0, 0.0, 0, :diag, 0.0, false, false, false), true
        elseif k == "--help" || k == "-h"
            help = true
        elseif k == "--no-threads"
            threads = false
        elseif k == "--no-summary"
            summ = false
        else
            if !hasv
                i < n || throw(ArgumentError("option $k needs a value"))
                i += 1
                v = args[i]
            end
            if k == "--data"
                data = v
            elseif k == "--output"
                output = v
            elseif k == "--chains"
                chains = _parse_opt_int(k, v)
            elseif k == "--iter"
                iter = _parse_opt_int(k, v)
            elseif k == "--warmup"
                warmup = _parse_opt_int(k, v)
            elseif k == "--seed"
                seed = _parse_opt_int(k, v)
            elseif k == "--delta"
                delta = _parse_opt_float(k, v)
            elseif k == "--max-depth"
                max_depth = _parse_opt_int(k, v)
            elseif k == "--metric"
                (v == "diag" || v == "dense") || throw(ArgumentError("--metric must be diag or dense"))
                metric = v == "dense" ? :dense : :diag
            elseif k == "--init"
                init = _parse_opt_float(k, v)
            else
                throw(ArgumentError("unknown option $k (see --help)"))
            end
        end
        i += 1
    end
    if !help
        data == "" && throw(ArgumentError("--data FILE is required (see --help)"))
        chains >= 1 || throw(ArgumentError("--chains must be at least 1"))
        warmup < 0 && (warmup = iter ÷ 2)
        0 <= warmup < iter || throw(ArgumentError("--warmup must be in 0:(iter - 1)"))
        seed < 0 && (seed = Int(rand(Random.RandomDevice(), UInt32)))
    end
    return BinaryOptions(data, output, chains, iter, warmup, seed, delta, max_depth, metric, init, threads,
                         summ, help), false
end

# ------------------------------------------------------- run time: sampling

# _sample_core's :nuts path, without the AdvancedHMC branch the binary must
# not reach: the same rng per chain, initial values and nuts_chain call, so
# the draws equal brm(...; seed) exactly
function _binary_sample(ld::GLMDensity, o::BinaryOptions)
    nkeep = o.iter - o.warmup
    draws = Array{Float64,3}(undef, ld.dim, nkeep, o.chains)
    ndiv = zeros(Int, o.chains)
    times = zeros(o.chains)
    run = c -> begin
        t0 = time()
        rng = Random.Xoshiro(hash((o.seed, c)))
        θ0 = _binary_init(ld, rng, o.init)
        ws = fresh_workspace(ld)
        gradf! = (g, x) -> logdensity_and_gradient!(g, ws, x)
        dr, st = nuts_chain(gradf!, θ0, nkeep, o.warmup; delta=o.delta, max_depth=o.max_depth, thin=1,
                            metric=o.metric, rng=rng)
        draws[:, :, c] .= dr
        ndiv[c] = round(Int, sum(st["divergent__"]))
        times[c] = time() - t0
        nothing
    end
    if o.threads && Threads.nthreads() > 1 && o.chains > 1
        nb = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        try
            tasks = Task[_spawn_chain(_ChainRun(run, c)) for c in 1:o.chains]
            foreach(wait, tasks)
        finally
            BLAS.set_num_threads(nb)
        end
    else
        # the SAME callable as the threaded path: calling it here is what
        # makes the trimmer compile `(::_ChainRun)()`, which the task
        # scheduler later invokes dynamically from C
        for c in 1:o.chains
            _ChainRun(run, c)()
        end
    end
    return draws, ndiv, times
end

# A chain as a task body. Julia 1.12's juliac --trim does not follow task
# bodies through Task construction (the scheduler calls them from C), so a
# Threads.@spawn closure's code is dropped and the binary dies with a
# MethodError on the first spawned chain -- after a clean verify. The known
# 1.12 fix (Reseau.jl #151, FastGeoProjections.jl #32): a named @noinline
# callable that is ALSO called directly from reachable, unfoldable code (the
# sequential branch above), which compiles a standalone specialization for the
# scheduler to dispatch to. Base.Experimental.entrypoint does not help from a
# package (top level runs in the precompile process). Julia 1.13 roots task
# closures itself.
struct _ChainRun{F}
    run::F
    c::Int
end
# @noinline: inlined into the sequential branch, no standalone compiled
# method existed for the scheduler's dynamic call to find (still MethodError)
@noinline (r::_ChainRun)() = (r.run(r.c); nothing)

# Threads.@spawn without its closure
function _spawn_chain(r::_ChainRun)
    t = Task(r)
    t.sticky = false
    isdefined(Threads, :_spawn_set_thrpool) && Threads._spawn_set_thrpool(t, :default)
    schedule(t)
    return t
end

# _initial_values (fit.jl) specialised to Float64: the same draws from rng,
# so the chains start where brm's do (`T::Type` there is not specialised,
# which leaves `T.(...)` a dynamic call under --trim)
function _binary_init(ld::GLMDensity, rng, init::Float64)
    ws = fresh_workspace(ld)
    g = zeros(ld.dim)
    for _ in 1:100
        θ0 = Float64.(init .* (2 .* rand(rng, ld.dim) .- 1))
        lp = _safe_grad!((gg, x) -> logdensity_and_gradient!(gg, ws, x), g, θ0)
        isfinite(lp) && all(isfinite, g) && return θ0
    end
    throw(ArgumentError("no finite log density in 100 random initialisations; try a smaller --init"))
end

# variables() / constrain() for the predictors a recipe can produce (LinPred
# with ordinary group blocks, ScalarDpar), in the same order. The general
# versions (density.jl) cover every feature through abstract containers that
# --trim cannot compile; model_recipe checks these against them exactly.
# `map` over the predictor tuple keeps each predictor's concrete type.
_bpx(d::Symbol) = d === :mu ? "" : string(d, "_")

function _binary_names(ld::GLMDensity)
    out = String[]
    map(ld.preds) do p
        p isa LinPred || return nothing
        x = _bpx(p.dpar)
        p.intercept > 0 && push!(out, "b_$(x)Intercept")
        for c in p.coefnames
            push!(out, "b_$(x)$(c)")
        end
        nothing
    end
    map(ld.preds) do p
        p isa LinPred || return nothing
        for gb in p.groups, c in gb.coefs
            push!(out, "sd_$(gb.label)__$(_bpx(p.dpar))$(c)")
        end
        nothing
    end
    map(ld.preds) do p
        p isa LinPred || return nothing
        x = _bpx(p.dpar)
        for gb in p.groups
            _has_corr(gb) || continue
            q = length(gb.coefs)
            for j in 2:q, i in 1:(j - 1)
                push!(out, "cor_$(gb.label)__$(x)$(gb.coefs[i])__$(x)$(gb.coefs[j])")
            end
        end
        nothing
    end
    map(ld.preds) do p
        p isa ScalarDpar && push!(out, string(p.dpar))
        nothing
    end
    map(ld.preds) do p
        p isa LinPred && p.intercept > 0 && push!(out, p.dpar === :mu ? "Intercept" : "Intercept_$(p.dpar)")
        nothing
    end
    map(ld.preds) do p
        p isa LinPred || return nothing
        for gb in p.groups
            r = p.dpar === :mu ? "r_$(gb.label)" : "r_$(gb.label)__$(p.dpar)"
            for c in gb.coefs, l in gb.levels
                push!(out, "$(r)[$(l),$(c)]")
            end
        end
        nothing
    end
    return out
end

function _binary_b(p::LinPred, θ::Vector{Float64})
    p.bounded || return θ[p.b]
    return Float64[first(_btrans(p.b_lb[j], p.b_ub[j], θ[k])) for (j, k) in enumerate(p.b)]
end

# r = diag(sd) * L * z (q x J); L = I without correlations
function _binary_r(gb::GroupBlock, θ::Vector{Float64})
    q = length(gb.coefs)
    J = length(gb.levels)
    z = reshape(θ[gb.z], q, J)
    M = _has_corr(gb) ? first(_cholcorr(θ[gb.corr], q)) * z : z
    sd = exp.(θ[gb.sd])
    return sd .* M
end

function _binary_values!(out::Vector{Float64}, ld::GLMDensity, θ::Vector{Float64})
    empty!(out)
    map(ld.preds) do p
        p isa LinPred || return nothing
        b = _binary_b(p, θ)
        p.intercept > 0 && push!(out, θ[p.intercept] - dot(p.means, b))
        append!(out, b)
        nothing
    end
    map(ld.preds) do p
        p isa LinPred || return nothing
        for gb in p.groups
            append!(out, exp.(θ[gb.sd]))
        end
        nothing
    end
    map(ld.preds) do p
        p isa LinPred || return nothing
        for gb in p.groups
            _has_corr(gb) || continue
            q = length(gb.coefs)
            L = first(_cholcorr(θ[gb.corr], q))
            Ω = L * L'
            for j in 2:q, i in 1:(j - 1)
                push!(out, Ω[i, j])
            end
        end
        nothing
    end
    map(ld.preds) do p
        p isa ScalarDpar && push!(out, _natural(p, θ[p.idx]))
        nothing
    end
    map(ld.preds) do p
        p isa LinPred && p.intercept > 0 && push!(out, θ[p.intercept])
        nothing
    end
    map(ld.preds) do p
        p isa LinPred || return nothing
        for gb in p.groups
            R = _binary_r(gb, θ)
            for k in axes(R, 1), j in axes(R, 2)
                push!(out, R[k, j])
            end
        end
        nothing
    end
    return out
end

# (iterations x chains x variables), constrained, in variables(ld) order
function _binary_constrain(ld::GLMDensity, draws::Array{Float64,3})
    nkeep, nch = size(draws, 2), size(draws, 3)
    v = Float64[]
    _binary_values!(v, ld, draws[:, 1, 1])
    D = Array{Float64,3}(undef, nkeep, nch, length(v))
    for c in 1:nch, i in 1:nkeep
        _binary_values!(v, ld, draws[:, i, c])
        for k in eachindex(v)
            D[i, c, k] = v[k]
        end
    end
    return D
end

# --------------------------------------------- run time: Rhat and ESS
#
# Rank-normalised split-Rhat and bulk / tail ESS, as the posterior R package
# computes them (Vehtari et al. 2021), written out here so the binary needs no
# FFT: autocovariances are summed lag by lag and the sum stops at Geyer's
# truncation, long before n lags.

# x: draws x chains -> split chains (the middle draw dropped for odd n)
function _split_chains(x::Matrix{Float64})
    n, m = size(x)
    h = n ÷ 2
    out = Matrix{Float64}(undef, h, 2m)
    for c in 1:m, i in 1:h
        out[i, 2c - 1] = x[i, c]
        out[i, 2c] = x[n - h + i, c]
    end
    return out
end

function _ranks(v::Vector{Float64})
    p = sortperm(v)
    r = zeros(length(v))
    i = 1
    while i <= length(v)
        j = i
        while j < length(v) && v[p[j + 1]] == v[p[i]]
            j += 1
        end
        for k in i:j
            r[p[k]] = (i + j) / 2               # average rank for ties
        end
        i = j + 1
    end
    return r
end

# z_scale: normal scores of the pooled ranks
function _z_scale(x::Matrix{Float64})
    S = length(x)
    r = _ranks(vec(x))
    return reshape([sqrt(2.0) * SpecialFunctions.erfinv(2 * ((ri - 0.375) / (S + 0.25)) - 1) for ri in r], size(x))
end

function _rhat_basic(x::Matrix{Float64})
    n = size(x, 1)
    cm = vec(mean(x; dims=1))
    cv = vec(var(x; dims=1))
    vw = mean(cv)
    vb = n * var(cm)
    vw > 0 || return NaN
    return sqrt((vb / vw + n - 1) / n)
end

function _ess_basic(x::Matrix{Float64})
    n, m = size(x)
    n < 4 && return NaN
    cm = vec(mean(x; dims=1))
    acov(t) = mean(sum((x[i, c] - cm[c]) * (x[i + t, c] - cm[c]) for i in 1:(n - t)) / n for c in 1:m)
    mean_var = acov(0) * n / (n - 1)
    var_plus = mean_var * (n - 1) / n
    m > 1 && (var_plus += var(cm))
    var_plus > 0 || return NaN
    rho = zeros(n)
    t = 0
    re = 1.0
    rho[1] = re
    ro = 1 - (mean_var - acov(1)) / var_plus
    rho[2] = ro
    while t < n - 5 && !isnan(re + ro) && re + ro > 0
        t += 2
        re = 1 - (mean_var - acov(t)) / var_plus
        ro = 1 - (mean_var - acov(t + 1)) / var_plus
        if re + ro >= 0
            rho[t + 1] = re
            rho[t + 2] = ro
        end
    end
    max_t = t
    re > 0 && (rho[max_t + 1] = re)
    t = 0
    while t <= max_t - 4                         # Geyer's initial monotone sequence
        t += 2
        if rho[t + 1] + rho[t + 2] > rho[t - 1] + rho[t]
            rho[t + 1] = (rho[t - 1] + rho[t]) / 2
            rho[t + 2] = rho[t + 1]
        end
    end
    S = m * n
    tau = -1 + 2 * sum(view(rho, 1:max_t)) + rho[max_t + 1]
    tau = max(tau, 1 / log10(S))
    return S / tau
end

# R's type-7 quantile (Statistics.quantile's default) of SORTED values.
# Statistics' own median! / quantile have error paths that print the array
# (repr), which drags the whole array printer into a trimmed binary.
function _quantile7(s::Vector{Float64}, p::Float64)
    n = length(s)
    h = (n - 1) * p + 1
    lo = floor(Int, h)
    lo >= n && return s[n]
    return s[lo] + (h - lo) * (s[lo + 1] - s[lo])
end

function _diagnostics(x::Matrix{Float64})
    all(isfinite, x) || return NaN, NaN, NaN
    s = sort(vec(x))
    sx = _split_chains(x)
    rb = _rhat_basic(_z_scale(sx))
    md = _quantile7(s, 0.5)
    rf = _rhat_basic(_z_scale(_split_chains(abs.(x .- md))))
    rhat = max(rb, rf)
    bulk = _ess_basic(_z_scale(sx))
    q05, q95 = _quantile7(s, 0.05), _quantile7(s, 0.95)
    tail = min(_ess_basic(_split_chains(Float64.(x .<= q05))), _ess_basic(_split_chains(Float64.(x .<= q95))))
    return rhat, bulk, tail
end

function _fmt(x::Float64, digits::Int=2)
    isfinite(x) || return "NA"
    # literal formats: @sprintf compiles each to specialised code
    return digits == 0 ? @sprintf("%.0f", x) : @sprintf("%.2f", x)
end

# padding without Base.lpad / rpad (their repeat(::Char) is an unresolved
# invoke under --trim)
function _pad(io, s::String, n::Int, left::Bool)
    k = n - length(s)
    left && for _ in 1:k; print(io, ' '); end
    print(io, s)
    left || for _ in 1:k; print(io, ' '); end
    return nothing
end

function _print_summary(io, names::Vector{String}, D::Array{Float64,3})
    w = max(8, maximum(length, names))
    _pad(io, "variable", w, false)
    println(io, "      mean        sd        q5       q50       q95  rhat  ess_bulk  ess_tail")
    for k in eachindex(names)
        x = D[:, :, k]
        v = vec(x)
        s = sort(v)
        rhat, bulk, tail = _diagnostics(x)
        _pad(io, names[k], w, false)
        for st in (mean(v), std(v), _quantile7(s, 0.05), _quantile7(s, 0.5), _quantile7(s, 0.95))
            _pad(io, _fmt(st), 10, true)
        end
        _pad(io, _fmt(rhat), 6, true)
        _pad(io, _fmt(bulk, 0), 10, true)
        _pad(io, _fmt(tail, 0), 10, true)
        println(io)
    end
    return nothing
end

# ----------------------------------------------------- run time: the driver

"""
    binary_main(args, recipe, eval_atoms, columns) -> exit code

The body of a model binary's `main`: `eval_atoms(::DataTable) -> AtomVals` is
the generated atom code, `columns` the column descriptions for `--columns`.
"""
function binary_main(args::Vector{String}, rec::ModelRecipe, eval_atoms::F, columns::Vector{String}) where {F}
    out = Core.stdout
    try
        o, cols = _parse_binary_args(args)
        if cols
            foreach(c -> println(out, c), columns)
            return Cint(0)
        end
        if o.help
            print(out, _BINARY_USAGE)
            return Cint(0)
        end
        tab = read_csv(o.data)
        ld, rows = assemble(rec, eval_atoms(tab))
        n0 = _nrows(tab)
        length(rows) < n0 && println(out, "note: $(n0 - length(rows)) rows with missing values were excluded")
        names = _binary_names(ld)
        println(out, "fitting $(ld.N) observations, $(ld.dim) parameters, $(o.chains) chains x ",
                "$(o.iter) iterations ($(o.warmup) warm-up), seed $(o.seed)")
        o.threads && o.chains > 1 && Threads.nthreads() == 1 &&
            println(out, "note: one thread, so chains run one after another; ",
                    "start with JULIA_NUM_THREADS=$(o.chains) to run them in parallel")
        draws, ndiv, times = _binary_sample(ld, o)
        D = _binary_constrain(ld, draws)
        io = open(o.output, "w")
        try
            write_draws_csv(io, names, D)
        finally
            close(io)
        end
        println(out, "chain times (s): ", join([_fmt(t) for t in times], ", "))
        sum(ndiv) > 0 && println(out, "warning: $(sum(ndiv)) divergent transitions after warm-up; ",
                                 "consider --delta 0.95")
        println(out, "draws written to ", o.output)
        o.summary && _print_summary(out, names, D)
        return Cint(0)
    catch e
        # only messages that are plain strings: printing an arbitrary
        # exception is a dynamic call the trimmed binary does not contain
        # (the fields are typed AbstractString: the isa split makes the print concrete)
        if e isa ArgumentError
            m = e.msg
            println(Core.stderr, "error: ", m isa String ? m : "invalid argument")
            return Cint(1)
        elseif e isa SystemError
            p = e.prefix
            println(Core.stderr, "error: ", p isa String ? p : "system error", " (", Libc.strerror(e.errnum), ")")
            return Cint(1)
        end
        rethrow()
    end
end

# ------------------------------------------------------- build time: code

# an atom as straight-line broadcasts over column vectors c_<name>
function _atom_code(ex, colnames::Set{Symbol}, used::Vector{Symbol})
    if ex isa Symbol
        if ex in colnames
            ex in used || push!(used, ex)
            return Symbol("c_", ex)
        end
        return ex
    elseif ex isa Number
        return ex
    elseif ex isa Expr && ex.head == :call
        f = ex.args[1]
        f === :I && return _atom_code(ex.args[2], colnames, used)
        f === :cbind && throw(ArgumentError("matrix responses are not supported in a model binary"))
        return Expr(:call, :broadcast, f, (_atom_code(a, colnames, used) for a in ex.args[2:end])...)
    elseif ex isa Expr && ex.head == :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        return ex                                    # a qualified function name, Base.log
    end
    throw(ArgumentError("unsupported expression `$ex` in a model binary"))
end

_col_kind(cat::Bool) = cat ? "factor" : "numeric"

# generated source of `_eval_atoms(tab)` and the column descriptions
function _atoms_source(atoms::Vector{Atom}, cat::Vector{Bool}, colnames::Set{Symbol})
    io = IOBuffer()
    println(io, "function _eval_atoms(tab::PBR.DataTable)")
    println(io, "    n = PBR._nrows(tab)")
    println(io, "    K = $(length(atoms))")
    println(io, "    num = Vector{Vector{Float64}}(undef, K); str = Vector{Vector{String}}(undef, K); miss = Vector{BitVector}(undef, K)")
    kinds = Dict{Symbol,String}()
    for (k, a) in enumerate(atoms)
        println(io, "    # atom $k: $(a.label)")
        if a.ex isa Symbol
            nm = String(a.ex)
            a.ex in colnames || throw(ArgumentError("`$nm` is not a data column"))
            kinds[a.ex] = cat[k] ? "factor" : get(kinds, a.ex, "numeric")
            println(io, "    let j = PBR._colindex(tab, $(repr(nm)))")
            if cat[k]
                println(io, "        num[$k] = fill(NaN, n)")
            else
                println(io, "        num[$k] = PBR._numeric_col(tab, j)")
            end
            println(io, "        str[$k] = copy(tab.raw[j]); miss[$k] = copy(tab.miss[j])")
            println(io, "    end")
        else
            cat[k] && throw(ArgumentError("$(a.label): a factor must be a data column"))
            used = Symbol[]
            code = _atom_code(a.ex, colnames, used)
            for u in used
                get(kinds, u, "") == "factor" ||
                    (kinds[u] = "numeric")
            end
            println(io, "    let")
            for u in used
                println(io, "        c_$(u) = PBR._numeric_col(tab, PBR._colindex(tab, $(repr(String(u)))))")
            end
            println(io, "        num[$k] = PBR._as_column($code, n)")
            println(io, "        str[$k] = PBR._num_names(num[$k])")
            if isempty(used)
                println(io, "        miss[$k] = falses(n)")
            else
                ms = join(["tab.miss[PBR._colindex(tab, $(repr(String(u))))]" for u in used], " .| ")
                println(io, "        miss[$k] = BitVector($ms)")
            end
            println(io, "    end")
        end
    end
    println(io, "    return PBR.AtomVals(num, str, miss)")
    println(io, "end")
    columns = sort!(["$(c) ($(kinds[c]))" for c in keys(kinds)])
    return String(take!(io)), columns
end

# a column's values as numbers, refusing text in a numeric column
function _numeric_col(t::DataTable, j::Int)
    v = t.num[j]
    for i in eachindex(v)
        if isnan(v[i]) && !t.miss[j][i] && t.raw[j][i] != "NaN"
            throw(ArgumentError("column `$(t.names[j])` must be numeric, but row $i is `$(t.raw[j][i])`"))
        end
    end
    return copy(v)
end
_as_column(v::AbstractVector, n::Int) = Float64[Float64(x) for x in v]
_as_column(v::Real, n::Int) = fill(Float64(v), n)
_num_names(v::Vector{Float64}) = String[isinteger(x) ? string(Int(x)) : string(x) for x in v]

# the training data as CSV text (R's conventions: TRUE / FALSE, NA)
function _table_csv(data)
    cols = Tables.columntable(data)
    nms = collect(keys(cols))
    io = IOBuffer()
    println(io, join(String.(nms), ","))
    cell(x) = ismissing(x) ? "NA" : x isa Bool ? (x ? "TRUE" : "FALSE") :
              x isa AbstractString ? (occursin(',', x) || occursin('"', x) ? "\"" * replace(x, "\"" => "\"\"") * "\"" : x) :
              string(x)
    n = length(first(cols))
    for i in 1:n
        println(io, join((cell(cols[k][i]) for k in nms), ","))
    end
    return String(take!(io))
end

# the generated atom code, evaluated through the CSV path, must match the
# front end's interpreter on every atom
function _check_generated_atoms(src::String, rec::ModelRecipe, atoms::Vector{Atom}, f::BrmsFormula, data)
    m = Module(:_BinaryAtomCheck)
    Core.eval(m, :(import PracticalBayesianRegression as PBR))
    Base.include_string(m, src)
    tab = parse_csv(_table_csv(data))
    gen = Base.invokelatest(Base.invokelatest(getfield, m, :_eval_atoms), tab)
    ref = frontend_atomvals(atoms, f, data)
    for k in eachindex(atoms)
        lab = atoms[k].label
        gen.miss[k] == ref.miss[k] || throw(ArgumentError("internal: missing values of `$lab` differ in the binary"))
        ok = .!ref.miss[k]
        if rec.cat[k]
            gen.str[k][ok] == ref.str[k][ok] ||
                throw(ArgumentError("internal: the levels of `$lab` differ in the binary"))
        else
            isequal(gen.num[k][ok], ref.num[k][ok]) ||
                throw(ArgumentError("`$lab` evaluates differently in a binary (only Base functions of " *
                                    "numeric columns are supported in formulas)"))
        end
    end
    return nothing
end

# --------------------------------------------------- build time: build_binary

function _module_name(name::String)
    s = replace(titlecase(replace(name, r"[^A-Za-z0-9]+" => " ")), " " => "")
    (isempty(s) || isdigit(s[1])) && (s = "Model" * s)
    return s
end

"""
    build_binary(formula, data; family=gaussian(), prior=PriorSpec[], output="model",
                 name="model", juliac=nothing, cc=nothing, verbose=true) -> executable path

Compile a model into a standalone executable that refits it to any CSV file
with the same columns, much as a CmdStan executable refits its model to new
data:

    \$ model/bin/model --data new.csv --output draws.csv --chains 4 --seed 1

`data` is example data: it fixes which columns are numeric and which are
factors, and the build checks the compiled model against `brm` on it. The
executable writes the draws as CSV with brms parameter names and prints a
summary table; `--help` lists its options. It needs no Julia installation:
distribute the whole `output` directory.

Building needs Julia 1.12 or later, a C compiler and JuliaC: by default the
`juliac` app on the `PATH` (`pkg> app add JuliaC`); otherwise pass `juliac`
as a `Cmd`, e.g.
`` `julia --project=/path/to/env -e "using JuliaC; JuliaC.main(ARGS)" --` ``.
`cc` sets the C compiler command. A run with `--seed s` gives the same draws
as `brm(formula, data; family, prior, seed = s)`.

Not every model can be compiled yet; an unsupported feature raises an
`ArgumentError` that names it (see [`model_recipe`](@ref)).
"""
function build_binary(formula, data; family=gaussian(), prior=PriorSpec[], output::AbstractString="model",
                      name::AbstractString="model", juliac=nothing, cc=nothing, verbose::Bool=true)
    VERSION >= v"1.12" || throw(ArgumentError("build_binary needs Julia 1.12 or later (juliac --trim)"))
    f = bf(formula)
    rec, atoms = model_recipe(f, data; family=family, prior=prior)
    colnames = Set{Symbol}(Symbol.(collect(Tables.columnnames(Tables.columntable(data)))))
    src, columns = _atoms_source(atoms, rec.cat, colnames)
    _check_generated_atoms(src, rec, atoms, f, data)

    output = abspath(String(output))
    mod = _module_name(String(name))
    proj = joinpath(output, "build", mod)
    mkpath(joinpath(proj, "src"))
    open(joinpath(proj, "src", "recipe.jls"), "w") do io
        Serialization.serialize(io, rec)
    end
    pbr = pkgdir(@__MODULE__)
    open(joinpath(proj, "Project.toml"), "w") do io
        println(io, "name = \"$mod\"")
        println(io, "uuid = \"$(Base.UUID(rand(Random.RandomDevice(), UInt128)))\"")
        println(io, "version = \"0.1.0\"\n\n[deps]")
        println(io, "PracticalBayesianRegression = \"3622129f-02c2-4cbe-8b24-efc8a7759a86\"")
        println(io, "Serialization = \"9e88b42a-f829-5b0c-bbe9-9e923198166b\"")
    end
    open(joinpath(proj, "src", "$mod.jl"), "w") do io
        println(io, "# Generated by PracticalBayesianRegression.build_binary for")
        println(io, "#     $(f.text)")
        println(io, "#     family $(family_name(rec.family))")
        println(io, "module $mod\n")
        println(io, "import PracticalBayesianRegression as PBR")
        println(io, "import Serialization\n")
        # (no type assertion needed: a const's concrete type is known from its value)
        println(io, "# the model's structure, loaded at BUILD time: a const of a concrete type")
        println(io, "const RECIPE = Serialization.deserialize(joinpath(@__DIR__, \"recipe.jls\"))")
        println(io, "const COLUMNS = $(repr(columns))\n")
        print(io, src)
        println(io, "\nfunction (@main)(args::Vector{String})::Cint")
        println(io, "    return PBR.binary_main(args, RECIPE, _eval_atoms, COLUMNS)")
        println(io, "end\n\nend")
    end
    write(joinpath(output, "training_data.csv"), _table_csv(data))
    open(joinpath(output, "README.txt"), "w") do io
        println(io, "Model: $(f.text)   family: $(family_name(rec.family))\n")
        println(io, "Run:   JULIA_NUM_THREADS=4 bin/$(name) --data FILE.csv --output draws.csv [--chains 4 --iter 2000 --seed 1]")
        println(io, "       bin/$(name) --help")
        println(io, "(JULIA_NUM_THREADS runs the chains in parallel; without it they run one after another.)\n")
        println(io, "FILE.csv needs these columns (other columns are ignored):")
        foreach(c -> println(io, "  ", c), columns)
        println(io, "\ntraining_data.csv is the data the model was built and checked on.")
    end

    julia = Base.julia_cmd()
    env = copy(ENV)
    cc === nothing || (env["JULIA_CC"] = string(cc))
    verbose && println("instantiating $proj")
    run(setenv(`$julia --project=$proj -e "using Pkg; Pkg.develop(path=$(repr(pbr))); Pkg.instantiate()"`, env))
    jc = juliac === nothing ? (Sys.which("juliac") === nothing ? nothing : `juliac`) : juliac
    jc === nothing && throw(ArgumentError("juliac not found: install it with `pkg> app add JuliaC` (and put " *
                                          "~/.julia/bin on PATH), or pass `juliac` as a Cmd"))
    verbose && println("compiling with juliac --trim=safe (a few minutes)")
    run(setenv(`$jc --output-exe $name --bundle $output --trim=safe --experimental $proj`, env))
    exe = joinpath(output, "bin", Sys.iswindows() ? "$name.exe" : name)
    isfile(exe) || throw(ArgumentError("juliac finished but $exe is missing"))
    verbose && println("built $exe")
    return exe
end
