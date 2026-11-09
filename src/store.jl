# =============================================================================
# The model store: compiled model binaries kept on disk and found again.
#
# A binary (binary.jl) takes minutes to build and refits its model to any CSV
# in milliseconds of start-up, so it is worth keeping. The store is a plain
# directory, one subdirectory per model:
#
#     <store>/<name>/model.toml        what the model is, and the keys it is found by
#     <store>/<name>/bin/<name>[.exe]  the juliac bundle (bin/, lib/, ...)
#     <store>/<name>/README.txt, training_data.csv, build/
#
# The R package (PracticalBayesianRegressionR) reads the same directory and
# runs the same executables WITHOUT starting Julia, so model.toml is kept to a
# subset R parses with a few lines of code: top-level `key = "string"` and
# `key = ["a", "b"]` lines, then `[[prior]]` tables (Julia only).
#
# Lookup is by KEY, a canonical description of formula, family, priors and
# column types. Each language computes its own key from its own inputs (an R
# formula and a Julia Expr print differently); a model compiled from R is
# registered under both, and a Julia-side match adds the caller's key, so a
# model is compiled once whichever side asks first.
# =============================================================================

"""
    model_store() -> String

The directory compiled models are kept in: the environment variable
`PBR_MODEL_STORE` if set, else `~/.pbr/models`. The R package uses the same
default, so models compiled from either language are found by both.
"""
model_store() = get(ENV, "PBR_MODEL_STORE", joinpath(homedir(), ".pbr", "models"))

_store_os() = Sys.islinux() ? "linux" : Sys.iswindows() ? "windows" : Sys.isapple() ? "macos" : lowercase(string(Sys.KERNEL))

"""
    CompiledModel

A model binary in the store: made by [`compile_model`](@ref), found again by
[`compiled_model`](@ref) or [`compiled_models`](@ref), and fitted to data
with `brm(model, data)`.
"""
struct CompiledModel
    name::String
    dir::String
    exe::String
    formulas::Vector{String}        # one per bf() formula, re-parsable by bf()
    family::String
    link::String
    priors::Vector{PriorSpec}
    columns::Vector{String}         # "x (numeric)", "g (factor)"
    keys::Vector{String}
    meta::Dict{String,Any}          # the whole model.toml
end

function Base.show(io::IO, cm::CompiledModel)
    print(io, "CompiledModel(\"", cm.name, "\": ", join(cm.formulas, ", "), ", ", cm.family, ")")
end

function Base.show(io::IO, ::MIME"text/plain", cm::CompiledModel)
    println(io, "Compiled model `", cm.name, "`")
    println(io, "  formula: ", join(cm.formulas, ", "))
    println(io, "  family:  ", cm.family, " (link = ", cm.link, ")")
    for p in cm.priors
        println(io, "  prior:   ", _prior_label(p))
    end
    println(io, "  columns: ", join(cm.columns, ", "))
    println(io, "  built:   ", get(cm.meta, "created", "?"), " (PracticalBayesianRegression ",
            get(cm.meta, "package_version", "?"), ", Julia ", get(cm.meta, "julia_version", "?"), ")")
    print(io, "  exe:     ", cm.exe)
end

_prior_label(p::PriorSpec) = string(_prior_string(p.prior), "  class = ", p.class,
                                    p.coef == "" ? "" : ", coef = $(p.coef)", p.group == "" ? "" : ", group = $(p.group)",
                                    p.dpar == "" ? "" : ", dpar = $(p.dpar)")

# a prior as `to_prior` reads it back
_prior_string(p::ScalarPrior) = isflat(p) ? "" : sprint(show, p)
_prior_string(p) = sprint(show, p)

_bound_string(x::Float64) = isfinite(x) ? _g(x) : ""

# The Julia-side lookup key. The formula text is bf's own canonical form
# (spaces stripped, precedence fixed); priors are sorted so their order does
# not matter; column kinds are part of the model (a factor and a numeric
# column give different design matrices).
function _julia_key(f::BrmsFormula, fam::Family, user::Vector{PriorSpec}, columns::Vector{String})
    io = IOBuffer()
    println(io, "julia")
    println(io, "formula: ", f.text)
    println(io, "family: ", family_name(fam), "(", link_name(link(fam)), ")")
    for l in sort([join((string(r.class), r.coef, r.group, r.dpar, _strip_spaces(_prior_string(r.prior)),
                         _bound_string(r.lb), _bound_string(r.ub)), "|") for r in user])
        println(io, "prior: ", l)
    end
    for c in sort(columns)
        println(io, "column: ", c)
    end
    return String(take!(io))
end

# bf's text joins the formulas with ", "; commas inside calls (s(x, k = 5))
# are not separators
function _split_formulas(text::AbstractString)
    out = String[]
    depth = 0
    start = firstindex(text)
    inq = false
    for i in eachindex(text)
        c = text[i]
        if c == '"'
            inq = !inq
        elseif !inq && c in ('(', '[', '{')
            depth += 1
        elseif !inq && c in (')', ']', '}')
            depth -= 1
        elseif !inq && c == ',' && depth == 0
            push!(out, strip(text[start:prevind(text, i)]))
            start = nextind(text, i)
        end
    end
    push!(out, strip(text[start:end]))
    return out
end

# ------------------------------------------------------------- model.toml

function _write_model_toml(path::String, d::Dict{String,Any})
    open(path, "w") do io
        println(io, "# Written by PracticalBayesianRegression.compile_model. The top-level keys are read by")
        println(io, "# the R package too: keep each value on one line.")
        TOML.print(io, d; sorted=true)
    end
end

function _load_compiled(dir::String)
    file = joinpath(dir, "model.toml")
    isfile(file) || throw(ArgumentError("$dir is not a compiled model (no model.toml)"))
    d = TOML.parsefile(file)
    priors = PriorSpec[prior(get(r, "prior", ""); class=Symbol(r["class"]), coef=get(r, "coef", ""),
                             group=get(r, "group", ""), dpar=get(r, "dpar", ""),
                             lb=_parse_bound(get(r, "lb", ""), -Inf), ub=_parse_bound(get(r, "ub", ""), Inf))
                       for r in get(d, "prior", Dict{String,Any}[])]
    return CompiledModel(d["name"], dir, joinpath(dir, d["exe"]), collect(String, d["formulas"]), d["family"],
                         d["link"], priors, collect(String, d["columns"]), collect(String, d["keys"]), d)
end
_parse_bound(s, default) = s == "" ? default : parse(Float64, s)

function _add_keys!(cm::CompiledModel, keys)
    new = [k for k in keys if !(k in cm.keys)]
    isempty(new) && return cm
    append!(cm.keys, new)
    cm.meta["keys"] = cm.keys
    _write_model_toml(joinpath(cm.dir, "model.toml"), cm.meta)
    return cm
end

# ------------------------------------------------------------------ listing

"""
    ModelStore

The compiled models in a store directory, as listed by
[`compiled_models`](@ref). Index it by name or position.
"""
struct ModelStore
    dir::String
    models::Vector{CompiledModel}
end

Base.length(s::ModelStore) = length(s.models)
Base.iterate(s::ModelStore, i...) = iterate(s.models, i...)
Base.getindex(s::ModelStore, i::Integer) = s.models[i]
function Base.getindex(s::ModelStore, name::AbstractString)
    k = findfirst(m -> m.name == name, s.models)
    k === nothing && throw(KeyError(name))
    return s.models[k]
end

function Base.show(io::IO, ::MIME"text/plain", s::ModelStore)
    n = length(s.models)
    print(io, n, " compiled model", n == 1 ? "" : "s", " in ", s.dir)
    n == 0 && return
    println(io)
    rows = [(m.name, m.family, join(m.formulas, ", "), string(get(m.meta, "created", ""))) for m in s.models]
    w = [max(4, maximum(r -> length(r[k]), rows)) for k in 1:3]
    println(io, "  ", rpad("name", w[1]), "  ", rpad("family", w[2]), "  ", rpad("formula", w[3]), "  built")
    for (k, r) in enumerate(rows)
        print(io, "  ", rpad(r[1], w[1]), "  ", rpad(r[2], w[2]), "  ", rpad(r[3], w[3]), "  ", r[4])
        k < n && println(io)
    end
end

"""
    compiled_models(; store = model_store()) -> ModelStore

Every compiled model in the store that runs on this machine (models built on
another operating system are skipped), newest first. Index the result by
name: `compiled_models()["mymodel"]`.
"""
function compiled_models(; store::AbstractString=model_store())
    out = CompiledModel[]
    isdir(store) || return ModelStore(String(store), out)
    for nm in readdir(store)
        dir = joinpath(store, nm)
        (startswith(nm, ".") || !isfile(joinpath(dir, "model.toml"))) && continue
        cm = try
            _load_compiled(dir)
        catch e
            @warn "skipping unreadable compiled model $dir" exception = e
            continue
        end
        _runs_here(cm) && push!(out, cm)
    end
    sort!(out; by=m -> string(get(m.meta, "created", "")), rev=true)
    return ModelStore(String(store), out)
end

_runs_here(cm::CompiledModel) = get(cm.meta, "os", "") == _store_os() && get(cm.meta, "arch", "") == string(Sys.ARCH)

"""
    compiled_model(name; store = model_store()) -> CompiledModel

A compiled model, by its name in the store, the path of its directory, or
the path of its executable (`.../mymod/bin/mymod`). The model is callable:
`m = compiled_model("mymod"); fit = m(data; chains = 4)` is
`brm(m, data; chains = 4)`.
"""
function compiled_model(name::AbstractString; store::AbstractString=model_store())
    dir = _model_dir(String(name), String(store))
    dir === nothing && throw(ArgumentError("no compiled model `$name` in $store (see compiled_models())"))
    cm = _load_compiled(abspath(dir))
    _runs_here(cm) || throw(ArgumentError("`$name` was compiled for $(get(cm.meta, "os", "?")) " *
                                          "$(get(cm.meta, "arch", "?")); recompile it here"))
    return cm
end

# a name in the store, the model's directory, or a path inside it (the
# executable): the nearest ancestor with a model.toml, at most three up
function _model_dir(name::String, store::String)
    if !occursin('/', name) && !occursin('\\', name) && !ispath(name)
        d = joinpath(store, name)
        return isfile(joinpath(d, "model.toml")) ? d : nothing
    end
    ispath(name) || return nothing
    d = isdir(name) ? abspath(name) : dirname(abspath(name))
    for _ in 0:3
        isfile(joinpath(d, "model.toml")) && return d
        d = dirname(d)
    end
    return nothing
end

# m(data; kw...): a compiled model as a function of the data
(cm::CompiledModel)(data; kwargs...) = brm(cm, data; kwargs...)

"""
    remove_compiled_model(name; store = model_store())

Delete a compiled model from the store.
"""
function remove_compiled_model(name::AbstractString; store::AbstractString=model_store())
    cm = compiled_model(name; store=store)
    rm(cm.dir; recursive=true)
    return nothing
end

# --------------------------------------------------------------- compiling

# the model's recipe, column descriptions and Julia key: everything lookup needs
function _model_identity(formula, data, family, prior)
    f = bf(formula)
    fam = family isa Family ? family : brmsfamily(family)
    user = prior isa PriorSpec ? PriorSpec[prior] : collect(PriorSpec, prior)
    rec, atoms = model_recipe(f, data; family=fam, prior=user)
    colnames = Set{Symbol}(Symbol.(collect(Tables.columnnames(Tables.columntable(data)))))
    _, columns = _atoms_source(atoms, rec.cat, colnames)
    return f, fam, user, columns, _julia_key(f, fam, user, columns)
end

function _find_by_keys(store::AbstractString, keys)
    for cm in compiled_models(; store=store)
        any(k -> k in cm.keys, keys) && return cm
    end
    return nothing
end

"""
    find_compiled_model(formula, data; family = gaussian(), prior = PriorSpec[],
                        store = model_store()) -> CompiledModel or nothing

The compiled model for this formula, family, priors and column types, if
one is in the store.
"""
function find_compiled_model(formula, data; family=gaussian(), prior=PriorSpec[], store::AbstractString=model_store())
    _, _, _, _, key = _model_identity(formula, data, family, prior)
    return _find_by_keys(store, [key])
end

function _default_model_name(f::BrmsFormula, fam::Family, key::String)
    resp = replace(f.response.label, r"[^A-Za-z0-9]+" => "_")
    return string(resp, "_", replace(family_name(fam), "." => "_"), "_", string(hash(key) % UInt32; base=16, pad=8))
end

"""
    compile_model(formula, data; family = gaussian(), prior = PriorSpec[], name = nothing,
                  store = model_store(), force = false, juliac = nothing, cc = nothing,
                  keys = String[], verbose = true) -> CompiledModel

Compile a model into a standalone executable (see [`build_binary`](@ref)) and
keep it in the model store, or return the stored one if this formula, family,
priors and column types were compiled before (`force = true` rebuilds). Fit
it to any data with the right columns with `brm(model, data)`; find it again
in a later session with [`compiled_model`](@ref) or [`compiled_models`](@ref),
or by calling `compile_model` with the same arguments.

`name` defaults to `<response>_<family>_<hash>`. `juliac` defaults to the
environment variable `PBR_JULIAC` (a command line, e.g.
`julia --project=/path/env -e "using JuliaC; JuliaC.main(ARGS)" --`), else
the `juliac` app on the PATH. `keys` are extra lookup keys to register (the R
package passes its own).
"""
function compile_model(formula, data; family=gaussian(), prior=PriorSpec[], name=nothing,
                       store::AbstractString=model_store(), force::Bool=false, juliac=nothing, cc=nothing,
                       keys=String[], verbose::Bool=true)
    f, fam, user, columns, key = _model_identity(formula, data, family, prior)
    allkeys = unique(vcat(key, collect(String, keys)))
    if !force
        cm = _find_by_keys(store, allkeys)
        if cm !== nothing && (name === nothing || cm.name == name)
            verbose && println("using compiled model `$(cm.name)` from $(store)")
            return _add_keys!(cm, allkeys)
        end
    end
    nm = name === nothing ? _default_model_name(f, fam, key) : String(name)
    occursin(r"^[A-Za-z][A-Za-z0-9_]*$", nm) ||
        throw(ArgumentError("model name `$nm`: use letters, digits and _, starting with a letter"))
    final = joinpath(store, nm)
    if isdir(final) && !force
        throw(ArgumentError("a different model is already stored as `$nm`; choose another name, or " *
                            "force = true to replace it"))
    end
    if juliac === nothing && haskey(ENV, "PBR_JULIAC")
        juliac = Cmd(Base.shell_split(ENV["PBR_JULIAC"]))
    elseif juliac isa AbstractString
        juliac = Cmd(Base.shell_split(juliac))
    end
    mkpath(store)
    # build beside the store and move in only when complete, so a failed or
    # interrupted build never leaves a half model that lookup would find
    tmp = mktempdir(store; prefix=".build_$(nm)_")
    try
        exe = build_binary(f, data; family=fam, prior=user, output=tmp, name=nm, juliac=juliac, cc=cc,
                           verbose=verbose)
        meta = Dict{String,Any}(
            "name" => nm,
            "exe" => relpath(exe, tmp),
            "formula" => f.text,
            "formulas" => _split_formulas(f.text),
            "family" => family_name(fam),
            "link" => link_name(link(fam)),
            "columns" => columns,
            "keys" => allkeys,
            "os" => _store_os(),
            "arch" => string(Sys.ARCH),
            "julia_version" => string(VERSION),
            "package_version" => string(pkgversion(@__MODULE__)),
            "created" => Libc.strftime("%Y-%m-%d %H:%M:%S", time()),
            # the prior rows again, one line each, for R (which reads no tables)
            "priors" => [join((string(r.class), r.coef, r.group, r.resp, r.dpar, "", _prior_string(r.prior),
                               _bound_string(r.lb), _bound_string(r.ub)), "|") for r in user],
            "prior" => [Dict{String,Any}("prior" => _prior_string(r.prior), "class" => string(r.class),
                                         "coef" => r.coef, "group" => r.group, "dpar" => r.dpar,
                                         "lb" => _bound_string(r.lb), "ub" => _bound_string(r.ub)) for r in user],
        )
        _write_model_toml(joinpath(tmp, "model.toml"), meta)
        isdir(final) && rm(final; recursive=true)
        mv(tmp, final)
    finally
        isdir(tmp) && rm(tmp; recursive=true, force=true)
    end
    verbose && println("compiled model `$nm` stored in $final")
    return compiled_model(nm; store=store)
end

# ------------------------------------------------------------------ running

# what the executable needs from the data: the model's columns, as CSV text
function _check_columns(cm::CompiledModel, data)
    have = Set(String.(collect(Tables.columnnames(Tables.columntable(data)))))
    miss = [c for c in cm.columns if !(first(split(c, " (")) in have)]
    isempty(miss) || throw(ArgumentError("the data lack column(s) the model `$(cm.name)` needs: " * join(miss, ", ")))
    return nothing
end

"""
    run_compiled(model, datafile; output, unconstrained = "", chains = 4, iter = 2000,
                 warmup = iter ÷ 2, seed, delta = 0.8, max_depth = 10, metric = :diag,
                 init = 2.0, threads = true) -> String

Run a compiled model's executable on a CSV file and return what it printed.
Lower level than `brm(model, data)`, which calls it.
"""
function run_compiled(cm::CompiledModel, datafile::AbstractString; output::AbstractString,
                      unconstrained::AbstractString="", chains::Int=4, iter::Int=2000, warmup::Int=iter ÷ 2,
                      seed::Integer, delta=0.8, max_depth::Int=10, metric::Symbol=:diag, init=2.0,
                      threads::Bool=true)
    isfile(cm.exe) || throw(ArgumentError("the executable of `$(cm.name)` is missing: $(cm.exe)"))
    args = ["--data", datafile, "--output", output, "--chains", string(chains), "--iter", string(iter),
            "--warmup", string(warmup), "--seed", string(seed), "--delta", string(Float64(delta)),
            "--max-depth", string(max_depth), "--metric", string(metric), "--init", string(Float64(init)),
            "--no-summary"]
    unconstrained == "" || append!(args, ["--unconstrained", unconstrained])
    threads || push!(args, "--no-threads")
    env = copy(ENV)
    env["JULIA_NUM_THREADS"] = threads ? string(min(chains, Sys.CPU_THREADS)) : "1"
    # a bundle carries its own libraries; an inherited LD_LIBRARY_PATH (a
    # cluster's gcc module) can outrank them and break start-up
    Sys.islinux() && delete!(env, "LD_LIBRARY_PATH")
    out, err = IOBuffer(), IOBuffer()
    p = run(pipeline(ignorestatus(setenv(`$(cm.exe) $args`, env)); stdout=out, stderr=err))
    so, se = String(take!(out)), String(take!(err))
    success(p) || throw(ErrorException("compiled model `$(cm.name)` failed (exit code $(p.exitcode)):\n" * so * se))
    return so
end

# "chain times (s): 0.51, 0.49" -> [0.51, 0.49]
function _parse_chain_times(text::String, chains::Int)
    m = match(r"chain times \(s\): ([^\n]*)", text)
    m === nothing && return fill(NaN, chains)
    t = [something(tryparse(Float64, strip(s)), NaN) for s in split(m.captures[1], ",")]
    return length(t) == chains ? t : fill(NaN, chains)
end

"""
    brm(model::CompiledModel, data; chains = 4, iter = 2000, warmup = iter ÷ 2, seed = nothing,
        delta = 0.8, max_depth = 10, metric = :diag, init = 2.0, threads = true) -> BrmsFit

Fit a compiled model to `data` by running its executable: no compilation, so
a fresh session fits in about the time sampling takes. The result is an
ordinary `BrmsFit` (summaries, predictions, `loo`, `update(fit; newdata)`
all work), and with `seed = s` its draws are those of
`brm(formula, data; family, prior, seed = s)`.
"""
function brm(cm::CompiledModel, data; chains::Int=4, iter::Int=2000, warmup::Int=iter ÷ 2, seed=nothing,
             delta=0.8, max_depth::Int=10, metric::Symbol=:diag, init=2.0, threads::Bool=true)
    iter > warmup >= 0 || throw(ArgumentError("iter must exceed warmup"))
    _check_columns(cm, data)
    seed0 = seed === nothing ? Int(rand(Random.RandomDevice(), UInt32)) : Int(seed)
    f = bf(cm.formulas...)
    fam = brmsfamily(cm.family; link=cm.link)
    # the front end on the same data gives the model the binary sampled (the
    # recipe was checked equal to it at build time); it carries everything
    # post-processing needs
    ld, info = build_model(f, data; family=fam, prior=cm.priors)
    dir = mktempdir()
    try
        datafile, drawsfile, ufile = joinpath(dir, "data.csv"), joinpath(dir, "draws.csv"), joinpath(dir, "theta.csv")
        write(datafile, _table_csv(data))
        out = run_compiled(cm, datafile; output=drawsfile, unconstrained=ufile, chains, iter, warmup, seed=seed0,
                           delta, max_depth, metric, init, threads)
        return _fit_from_binary(cm, f, fam, data, ld, info, read_csv(drawsfile), read_csv(ufile),
                                _parse_chain_times(out, chains);
                                chains, iter, warmup, seed=seed0, delta, max_depth, metric, init, threads)
    finally
        rm(dir; recursive=true, force=true)
    end
end

"""
    fit_from_draws(ld, info, theta, stats, times; sampler, iter, warmup) -> BrmsFit

A `BrmsFit` from draws made elsewhere: `theta` is (parameters x iterations x
chains) on the unconstrained scale, `stats` maps Stan's sampler statistic
names (`lp__`, `divergent__`, ...) to (iterations x chains) matrices. Used for
model binaries' output (from Julia or R).
"""
function fit_from_draws(ld::GLMDensity, info::ModelInfo, theta::Array{Float64,3}, stats::AbstractDict,
                        times::Vector{Float64}; sampler::String="nuts", iter::Int, warmup::Int)
    size(theta, 1) == ld.dim ||
        throw(ArgumentError("the draws have $(size(theta, 1)) parameters but the model has $(ld.dim): " *
                            "was the binary built from a different version of the package?"))
    nch = size(theta, 3)
    results = [(stats=Dict{String,Vector{Float64}}(String(k) => Vector{Float64}(v[:, c]) for (k, v) in stats),
                time=times[c]) for c in 1:nch]
    chain = _make_chain(ld, theta, results)
    return BrmsFit(info, ld, chain, theta, sampler, iter, warmup, 1, times, Dict{Symbol,Any}())
end

function _fit_from_binary(cm, f, fam, data, ld, info, dt::DataTable, ut::DataTable, times;
                          chains, iter, warmup, seed, delta, max_depth, metric, init, threads)
    nkeep = iter - warmup
    col(t, nm) = t.num[_colindex(t, nm)]
    theta = Array{Float64,3}(undef, ld.dim, nkeep, chains)
    for k in 1:(length(ut.names) - 2)
        theta[k, :, :] .= reshape(col(ut, "theta[$k]"), nkeep, chains)
    end
    stats = Dict{String,Matrix{Float64}}(s => reshape(col(dt, s), nkeep, chains) for s in _STAT_NAMES)
    fit = fit_from_draws(ld, info, theta, stats, times; sampler="nuts (compiled model $(cm.name))", iter, warmup)
    # the package's constrain() must agree with what the binary wrote: a
    # binary from an older package version that names or transforms
    # parameters differently fails here instead of giving wrong summaries
    for nm in variables(ld)
        a = vec(Matrix{Float64}(fit.chain[FlexiChains.Parameter(nm)]))
        isapprox(a, col(dt, nm); rtol=1e-8, atol=1e-10) ||
            throw(ErrorException("`$(cm.name)`: the binary's `$nm` differs from this package's; recompile " *
                                 "it (compile_model(...; force = true))"))
    end
    merge!(fit.call, Dict{Symbol,Any}(:compiled => cm, :formula => f, :data => data, :family => fam,
        :prior => cm.priors, :chains => chains, :iter => iter, :warmup => warmup, :thin => 1, :sampler => :nuts,
        :delta => delta, :max_depth => max_depth, :metric => metric, :init => init, :threads => threads,
        :T => Float64, :sample_prior => :no, :algorithm => :sampling))
    return fit
end

# update(fit; newdata) on a compiled fit reruns the binary; changing anything
# the binary cannot (formula, family, priors, ...) refits with brm
const _COMPILED_RUN_KW = (:chains, :iter, :warmup, :seed, :delta, :max_depth, :metric, :init, :threads)
