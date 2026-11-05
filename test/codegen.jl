# Model code generation (src/codegen.jl). These tests need neither Turing nor
# PracticalBayes: they check the generated source and data. Evaluating and
# sampling the generated models is test/codegen_backends.jl (needs Turing / PB).

# generated code must at least be valid Julia
_parses(c) = all(ex -> !(ex isa Expr && ex.head in (:error, :incomplete)), Meta.parseall(string(c)).args)

# A minimal stand-in for Turing's `@model`, enough to EVALUATE generated code
# without Turing: `x ~ D` reads parameter x from a NamedTuple and adds
# logpdf(D, x) to the log prior, or, when x is a model argument (data), adds it
# to the log-likelihood; `@addlogprob!` adds to the log-likelihood; `:=` is an
# assignment. The function returns (logprior, loglik).
module MiniPPL
using Distributions, LinearAlgebra
export @model, @addlogprob!, Flat, FlatPos, filldist, arraydist, LogPoisson, BinomialLogit, ordered

# Bijectors' ordered(d): d restricted to increasing vectors; its log density
# at an ordered point is d's own (up to a constant)
ordered(d) = d

struct Flat <: ContinuousUnivariateDistribution end
Distributions.logpdf(::Flat, x::Real) = zero(x)
struct FlatPos{T<:Real} <: ContinuousUnivariateDistribution
    l::T
end
Distributions.logpdf(d::FlatPos, x::Real) = x > d.l ? zero(x) : oftype(float(x), -Inf)
filldist(d, dims...) = product_distribution(fill(d, dims...))
arraydist(ds) = product_distribution(ds)
LogPoisson(eta) = Poisson(exp(eta))
BinomialLogit(n, eta) = Binomial(n, inv(1 + exp(-eta)))

macro addlogprob!(ex)
    # (; logprior = v) adds to the prior, as DynamicPPL; anything else to the likelihood
    if ex isa Expr && ex.head === :tuple && length(ex.args) == 1 && ex.args[1] isa Expr &&
       ex.args[1].head === :parameters && ex.args[1].args[1].args[1] === :logprior
        return esc(:(__lp[] += $(ex.args[1].args[1].args[2])))
    end
    return esc(:(__ll[] += $ex))     # __ll is the @model function's local
end

_rw(ex, data) = ex
function _rw(ex::Expr, data)
    if ex.head === :call && ex.args[1] === :~ && ex.args[2] isa Symbol
        lhs, rhs = ex.args[2], _rw(ex.args[3], data)
        lhs in data && return :(__ll[] += logpdf($rhs, $lhs))
        return :($lhs = getfield(__p, $(QuoteNode(lhs))); __lp[] += logpdf($rhs, $lhs))
    elseif ex.head === :(:=)
        return Expr(:(=), ex.args[1], _rw(ex.args[2], data))
    elseif ex.head === :return          # an early exit (a rejected proposal)
        return :(return (__lp[], __ll[]))
    end
    return Expr(ex.head, map(a -> _rw(a, data), ex.args)...)
end

macro model(fdef)
    sig, body = fdef.args[1], fdef.args[2]
    name, args = sig.args[1], sig.args[2:end]
    ex = :(function $name(__p::NamedTuple, $(args...))
               __lp = Ref(0.0); __ll = Ref(0.0)
               $(_rw(body, Set(args)))
               return (__lp[], __ll[])
           end)
    return esc(ex)
end
end
using .MiniPPL: MiniPPL

@testset "model code generation" begin
    @testset "gaussian, correlated slopes" begin
        f = @bf(yg ~ x + (1 + x | g))
        c = turing_code(f, TD)
        s = string(c)
        @test _parses(c)
        @test startswith(s, "using Turing, LinearAlgebra")
        @test occursin("@model function regression_model(y, X, means_X, J_g, N_g, Z_g)", s)
        @test occursin("LKJCholesky(2, 1)", s)
        @test occursin("r_g = sd_g .* (L_g.L * z_g)", s)
        @test occursin("b_Intercept := Intercept - dot(means_X, b)", s)
        @test occursin("y ~ arraydist(Normal.(eta, sigma; check_args=false))", s)
        # brms defaults: flat b, half student-t sd and sigma
        @test occursin("b ~ filldist(Flat(), 1)", s)
        @test occursin(r"sigma ~ truncated\([^)]*TDist\(3\); lower=0\)", s)
        dat = model_data(f, TD)
        @test keys(dat) == (:y, :X, :means_X, :J_g, :N_g, :Z_g)
        @test size(dat.X) == (300, 1) && dat.N_g == 6
        @test all(isone, dat.Z_g[:, 1])
        # the data are the built model's own (centred design)
        ld, info = build_model(f, TD)
        @test dat.X == ld.preds[1].Xc
        @test abs(sum(dat.X)) < 1e-8
    end

    @testset "count and binary families use the canonical-link distributions" begin
        s = string(turing_code(@bf(yp ~ x + (1 | g)), TD; family=poisson()))
        @test occursin("y ~ arraydist(LogPoisson.(eta))", s)
        @test occursin("r_g = sd_g .* z_g", s) && occursin("eta = Intercept .+ X * b .+ r_g[J_g]", s)
        @test model_data(@bf(yp ~ x + (1 | g)), TD; family=poisson()).y isa Vector{Int}
        s = string(turing_code(@bf(yn | trials(n) ~ x), TD; family=binomial()))
        @test occursin("y ~ arraydist(BinomialLogit.(trials, eta))", s)
        s = string(turing_code(@bf(yb ~ x), TD; family=bernoulli()))
        @test occursin("y ~ arraydist(BernoulliLogit.(eta))", s)
        s = string(turing_code(@bf(yp ~ x), TD; family=negbinomial()))
        @test occursin("NegativeBinomial.(shape, shape ./ (shape .+ mu); check_args=false)", s) && occursin("mu = exp.(eta)", s)
        @test occursin("shape ~ InverseGamma(0.4, 0.3)", s)     # brms's negbinomial default
    end

    @testset "distributional and uncorrelated terms" begin
        s = string(turing_code(@bf(yg ~ x, sigma ~ z), TD))
        @test _parses(s)
        @test occursin("sigma = exp.(eta_sigma)", s) && occursin("Intercept_sigma ~", s)
        @test occursin("y ~ arraydist(Normal.(eta, sigma; check_args=false))", s)
        s = string(turing_code(@bf(yg ~ x + (1 + x || g)), TD))
        @test !occursin("LKJCholesky", s) && occursin("r_g = sd_g .* (z_g)", s)
        s = string(turing_code(@bf(ya ~ x), TD; family=student()))
        @test occursin("nu ~ truncated(Gamma(2, 1 / 0.1); lower=1)", s)
    end

    @testset "PracticalBayes dialect" begin
        c = pb_code(@bf(yg ~ x + (1 | g)), TD)
        @test startswith(string(c), "using PracticalBayes, Distributions, LinearAlgebra")
        # the dialects differ only in the imports (and the generator's name in the comment)
        @test replace(c.body, "pb_code" => "turing_code") == turing_code(@bf(yg ~ x + (1 | g)), TD).body
    end

    @testset "features without explicit code use the density fallback" begin
        # (constant() on a group sd is not written out yet; update the example when it is)
        f = @bf(yg ~ x + (1 | g))
        pri = [prior("constant(1)"; class=:sd, group="g")]
        c = turing_code(f, TD; prior=pri)
        @test !isempty(c.fallback) && _parses(c)
        @test occursin("@addlogprob! model_logdensity(density, θ)", string(c))
        @test startswith(string(c), "using Turing, PracticalBayesianRegression")
        dat = model_data(f, TD; prior=pri)
        @test keys(dat) == (:density,) && dat.density isa GLMDensity
        @test isempty(turing_code(@bf(yg ~ x), TD).fallback)                          # explicit
        @test isempty(turing_code(@bf(yg ~ x + mo(mlev) + (1 | g)), TD).fallback)     # mo() explicit
        @test isempty(turing_code(@bf(yg ~ x) + @bf(ya ~ x), TD).fallback)            # multivariate explicit
        # strict mode refuses instead
        @test_throws ArgumentError turing_code(f, TD; prior=pri, fallback=false)
    end

    @testset "families without a Distributions.jl line use FamilyDist" begin
        s = string(turing_code(@bf(yp ~ x), TD; family=zero_inflated_poisson()))
        @test occursin("using Turing, LinearAlgebra, PracticalBayesianRegression", s)
        @test occursin("y ~ arraydist(FamilyDist.(Ref(family), 0.0, 0, eta, (log(zi) - log1p(-zi))))", s)
        s = string(turing_code(@bf(yg | weights(w) ~ x), TD))
        @test occursin("@addlogprob! sum(weights .* logpdf.(Normal.(eta, sigma; check_args=false), y))", s)
        @test :weights in keys(model_data(@bf(yg | weights(w) ~ x), TD))
    end

    # Every explicit model's likelihood must equal the package's at the
    # package's own parameter values: the guard against a term the generator
    # does not know about (it would otherwise silently drop it). MiniPPL runs
    # the generated code without Turing.
    @testset "explicit code reproduces the package's likelihood" begin
        rng = Xoshiro(9)
        nexpl = 0
        reasons = Dict{String,Vector{String}}()
        cases = vcat([(n, f, fam, [prior("normal(0, 2)"; class=:b)]) for (n, f, fam) in GRAD_CASES],
                     [(n, f, fam, NL_PRIORS) for (n, f, fam) in NL_CASES],
                     [("const_b", @bf(yg ~ x + z), gaussian(), [prior("constant(0.5)"; class=:b, coef="x")]),
                      ("const_sigma", @bf(yg ~ x), gaussian(), [prior("constant(1.2)"; class=:sigma)]),
                      ("mv_rescor", @bf(yg ~ x + (1 | g)) + @bf(yg2 ~ z), gaussian(), PriorSpec[]),
                      ("mv_norescor", @bf(yg ~ x) + @bf(yp ~ z) + set_rescor(false), [gaussian(), poisson()],
                       PriorSpec[])])
        for (name, f, fam, pri) in cases
            # (invokelatest: the string non-linear formula compiles its mean by eval)
            ld, info = PG._is_mv(f) ? build_mv_model(f, TD; family=fam, prior=pri) :
                                      Base.invokelatest(build_model, f, TD; family=fam, prior=pri)
            gen = PG._codegen(:turing, ld, info)
            @test _parses(gen.code)
            if !isempty(gen.code.fallback)
                push!(get!(reasons, gen.code.fallback, String[]), name)
                continue
            end
            nexpl += 1
            m = PG._eval_in_module(gen.code.body, MiniPPL, Distributions, LinearAlgebra, PG)
            fn = Base.invokelatest(getglobal, m, gen.code.name)
            for _ in 1:2
                θ = 0.5 .* randn(rng, ld.dim)
                lp, ll = Base.invokelatest(fn, gen.params(θ), gen.data...)
                ok = isapprox(ll, PG._cg_loglik(ld, θ); rtol=1e-9, atol=1e-8)
                ok || @info "likelihood mismatch" name ll PG._cg_loglik(ld, θ)
                @test ok
                @test isfinite(lp)
            end
        end
        @info "explicit code generation" nexpl total = length(cases)
        for (r, ns) in sort(collect(reasons); by=x -> -length(x[2]))
            println("  fallback ($(length(ns))): $r -- ", join(ns, ", "))
        end
        @test nexpl >= 60
    end

    @testset "evaluation needs the backend loaded" begin
        if !isdefined(Main, :Turing)
            @test_throws ArgumentError turing_model(@bf(yg ~ x), TD)
        end
    end
end
