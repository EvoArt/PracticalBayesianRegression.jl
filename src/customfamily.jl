# =============================================================================
# brms `custom_family`: a user-defined response distribution, written as a
# Julia log density instead of brms's Stan function.
#
#     beta_binom2 = custom_family("beta_binomial2"; dpars = (:mu, :phi), links = (:logit, :log),
#                                 type = :int, ll = (y, mu, phi, n) -> ..., rand = (rng, mu, phi) -> ...)
#     brm(@bf(y | vint(trials) ~ x), d; family = beta_binom2)
#
# `ll(y, dpar values..., vars...)` is the log density of one observation at
# the NATURAL scale of each parameter: the family applies the links (mu's
# link; for the others lb + exp(eta) for "log", lb + (ub - lb) logistic(eta) for
# "logit", eta for "identity", the inverse link otherwise). `vars` are the
# brms addition terms `vreal(...)` / `vint(...)`, per observation. The
# gradient comes from one dual-number evaluation per observation (it is a
# DualFamily), so `ll` must be generic in the number type -- ordinary Julia
# code with SpecialFunctions / Distributions' logpdf of dual-friendly
# distributions is. Optional `rand(rng, values...)` and `mean(values...)`
# (parameter values only) give posterior_predict and posterior_epred.
# A scalar parameter (no formula) gets a flat prior by default, as in brms.
# =============================================================================

struct CustomFamily{F,R,M,N,L<:Link,V} <: DualFamily
    name::String
    dpars::NTuple{N,Symbol}
    links::NTuple{N,Symbol}
    lb::NTuple{N,Float64}
    ub::NTuple{N,Float64}
    ll::F
    rnd::R
    mn::M
    type::Symbol                      # :real or :int (the response)
    vars::Matrix{Float64}             # N x V vreal / vint data (0 x 0 until built)
end

"""
    custom_family(name; dpars = (:mu,), links = (:identity,), lb = nothing, ub = nothing,
                  type = :real, ll, rand = nothing, mean = nothing)

A user-defined response distribution, as brms's `custom_family`, with its log
density written as a Julia function instead of Stan code.

- `dpars`: the parameter names; the first must be `:mu`.
- `links`: one link per parameter (`:identity`, `:log`, `:logit`, or any
  link a built-in family uses for `mu`).
- `lb`, `ub`: per-parameter bounds, `nothing` for none. A `:log` link has
  lower bound 0 and a `:logit` link the interval (0, 1) unless given.
- `type`: `:real` or `:int`, the type of the response.
- `ll(y, mu, other_dpars..., vars...)`: the log density of one observation at
  the natural scale of each parameter. `vars` are the values of the
  `vreal(...)` / `vint(...)` addition terms for that observation. `ll` is
  differentiated with dual numbers, so it must accept any `Real` type (plain
  Julia code, SpecialFunctions.jl and Distributions.jl `logpdf`s do).
- `rand(rng, mu, other_dpars...)` and `mean(mu, other_dpars...)`: optional,
  for [`posterior_predict`](@ref) and [`posterior_epred`](@ref).

Parameters without a formula get flat priors, as in brms; set proper ones
with [`prior`](@ref).

# Examples
```julia
normal2 = custom_family("normal2"; dpars = (:mu, :sigma), links = (:identity, :log),
                        ll = (y, mu, sigma) -> -log(sigma) - (y - mu)^2 / (2 * sigma^2),
                        rand = (rng, mu, sigma) -> mu + sigma * randn(rng),
                        mean = (mu, sigma) -> mu)
fit = brm(@bf(y ~ x), df; family = normal2,
          prior = [prior("exponential(1)"; class = :sigma)])
```
"""
function custom_family(name::AbstractString; dpars=(:mu,), links=(:identity,), lb=nothing, ub=nothing,
                       type=:real, ll, rand=nothing, mean=nothing)
    dp = Tuple(Symbol.(dpars)); lk = Tuple(Symbol.(links))
    N = length(dp)
    length(lk) == N || throw(ArgumentError("custom_family: one link per parameter"))
    dp[1] === :mu || throw(ArgumentError("custom_family: the first parameter must be :mu (brms)"))
    Symbol(type) in (:real, :int) || throw(ArgumentError("custom_family: type must be :real or :int"))
    getb(v, i, d) = v === nothing || v[i] === nothing ? d : Float64(v[i])
    lbs = ntuple(i -> getb(lb, i, lk[i] in (:log, :logit) ? 0.0 : -Inf), N)
    ubs = ntuple(i -> getb(ub, i, lk[i] === :logit ? 1.0 : Inf), N)
    for i in 2:N
        lk[i] in (:identity, :log, :logit) ||
            throw(ArgumentError("custom_family: link `$(lk[i])` for $(dp[i]) (use identity, log or logit)"))
    end
    L = typeof(_link(lk[1]))
    return CustomFamily{typeof(ll),typeof(rand),typeof(mean),N,L,0}(String(name), dp, lk, lbs, ubs, ll, rand, mean,
                                                                    Symbol(type), zeros(0, 0))
end
_with_vars(f::CustomFamily{F,R,M,N,L}, V::Matrix{Float64}) where {F,R,M,N,L} =
    CustomFamily{F,R,M,N,L,size(V, 2)}(f.name, f.dpars, f.links, f.lb, f.ub, f.ll, f.rnd, f.mn, f.type, V)

link(::CustomFamily{F,R,M,N,L}) where {F,R,M,N,L} = L()
dpars(f::CustomFamily) = f.dpars
family_name(f::CustomFamily) = f.name
dpar_link_name(f::CustomFamily, d::Symbol) = string(f.links[findfirst(==(d), f.dpars)])
function aux_bounds(f::CustomFamily, d::Symbol)
    i = findfirst(==(d), f.dpars)
    l = f.links[i]
    return l === :identity ? (:identity, -Inf) : l === :logit ? (:logit, f.lb[i]) : (:log, f.lb[i])
end
aux_ub(f::CustomFamily, d::Symbol) = f.ub[findfirst(==(d), f.dpars)]
_check_response(f::CustomFamily, y, t) =
    f.type === :int ? _chk(all(isinteger, y), "$(f.name): integer responses expected (type = :int)") : nothing
_xdata(f::CustomFamily{F,R,M,N,L,V}, y, trials) where {F,R,M,N,L,V} =
    V == 0 ? Float64[] : [ntuple(j -> f.vars[i, j], Val(V)) for i in axes(f.vars, 1)]
function _default_prior(f::CustomFamily, class::Symbol, dpar::Symbol, y)
    class in f.dpars[2:end] && return flat()          # a scalar parameter of a custom family: flat (brms)
    return invoke(_default_prior, Tuple{Family,Symbol,Symbol,Any}, f, class, dpar, y)
end

# natural values of the parameters from their link-scale etas
@inline function _cf_values(f::CustomFamily{F,R,M,N,L}, eta) where {F,R,M,N,L}
    return ntuple(Val(N)) do i
        e = eta[i]
        i == 1 && return linkinv(L(), e)
        l = f.links[i]
        l === :log ? _pv(e, f.lb[i]) + exp(e) : l === :logit ? _pv(e, f.lb[i]) + _pv(e, f.ub[i] - f.lb[i]) * logistic(e) : e
    end
end
@inline _cf_vars(::CustomFamily{F,R,M,N,L,0}, x) where {F,R,M,N,L} = ()
@inline _cf_vars(::CustomFamily, x::Tuple) = x

@inline _ll(f::CustomFamily, y, x, eta) = f.ll(y, _cf_values(f, eta)..., _cf_vars(f, x)...)
function _mean(f::CustomFamily, eta, t)
    f.mn === nothing && throw(ArgumentError("$(f.name): pass `mean = (values...) -> ...` to custom_family for posterior_epred"))
    return f.mn(_cf_values(f, eta)...)
end
function _rand(rng, f::CustomFamily, eta, t)
    f.rnd === nothing && throw(ArgumentError("$(f.name): pass `rand = (rng, values...) -> ...` to custom_family for posterior_predict"))
    return f.rnd(rng, _cf_values(f, eta)...)
end
