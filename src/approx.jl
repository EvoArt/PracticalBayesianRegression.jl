# =============================================================================
# brms's `algorithm = "laplace" | "meanfield" | "fullrank" | "pathfinder"`:
# approximate posteriors from the same closed-form-gradient densities, as
# CmdStan computes them (laplace_sample, ADVI, single-path Pathfinder).
#
#   laplace:   the mode of the log density on the unconstrained scale WITH the
#              Jacobian (CmdStan laplace's default), by L-BFGS; the Hessian by
#              central differences of the analytic gradient; draws N(mode, -H^-1).
#   meanfield: ADVI (Kucukelbir et al. 2017, Stan's implementation): q = N(mu,
#              diag(exp(2 omega))); one-draw reparameterisation gradients;
#              Stan's step size eta * k^(-1/2 + 1e-16) / (1 + sqrt(s_k)), s_k a
#              running average of squared gradients (0.1 / 0.9); eta chosen from
#              (100, 10, 1, 0.1, 0.01) by a 50-iteration trial each; convergence
#              when the relative ELBO change (mean or median of the recent ones)
#              falls below tol_rel_obj = 0.01, evaluated every 100 iterations.
#   fullrank:  the same with q = N(mu, L L'), L lower triangular.
#   pathfinder: single-path Pathfinder (Zhang et al. 2022): along the L-BFGS
#              path to the mode, each iterate's normal approximation (its
#              L-BFGS inverse-Hessian estimate) is scored by a Monte Carlo
#              ELBO; draws from the best one, then PSIS-resampled.
# Output: an ordinary fit with one "chain" of `ndraws` draws (brms's 1000).
# =============================================================================

# maximise f with gradient via fg!(g, x) -> f; returns (x, f, (S, Y) history)
function _lbfgs(fg!, x0::Vector{Float64}; m::Int=6, maxiter::Int=2000, gtol=1e-8, ftol=1e-12, keep::Bool=false)
    n = length(x0)
    x = copy(x0); g = zeros(n)
    f = fg!(g, x)
    isfinite(f) || throw(ArgumentError("the log density is not finite at the starting point"))
    S = Vector{Float64}[]; Y = Vector{Float64}[]
    path = keep ? [(copy(x), f, copy(g), Vector{Float64}[], Vector{Float64}[])] : Any[]
    gn = zeros(n); xn = zeros(n)
    for it in 1:maxiter
        # two-loop recursion on -f: direction d = H * g (ascent)
        q = copy(g); α = zeros(length(S))
        for i in length(S):-1:1
            ρ = 1 / dot(Y[i], S[i]); α[i] = ρ * dot(S[i], q); q .-= α[i] .* Y[i]
        end
        γ = isempty(S) ? 1 / max(norm(g), 1.0) : dot(S[end], Y[end]) / dot(Y[end], Y[end])
        d = γ .* q
        for i in eachindex(S)
            ρ = 1 / dot(Y[i], S[i]); β = ρ * dot(Y[i], d); d .+= (α[i] - β) .* S[i]
        end
        dot(d, g) > 0 || (d .= g; empty!(S); empty!(Y))       # not an ascent direction: restart
        # backtracking (Armijo) line search
        t = 1.0; fn = -Inf
        for _ in 1:60
            xn .= x .+ t .* d
            fn = fg!(gn, xn)
            isfinite(fn) && fn >= f + 1e-4 * t * dot(g, d) && break
            t /= 2
        end
        isfinite(fn) || break
        s = xn .- x; yv = g .- gn                                # y for -f: -(gn - g)
        if dot(s, yv) > 1e-10 * dot(yv, yv)
            push!(S, s); push!(Y, yv)
            length(S) > m && (popfirst!(S); popfirst!(Y))
        end
        df = fn - f
        x .= xn; g .= gn; f = fn
        keep && push!(path, (copy(x), f, copy(g), copy.(S), copy.(Y)))
        (maximum(abs, g) < gtol || abs(df) < ftol * max(abs(f), 1.0)) && break
    end
    return x, f, path
end

_approx_ws(dens) = fresh_workspace(dens)
function _fg(dens)
    ws = _approx_ws(dens)
    return (g, x) -> _safe_grad!((gg, xx) -> logdensity_and_gradient!(gg, ws, xx), g, x)
end

function _start_point(dens, rng, init)
    fg! = _fg(dens); g = zeros(LogDensityProblems.dimension(dens))
    for _ in 1:100
        θ0 = init isa AbstractVector ? Float64.(init) : (2rand(rng, length(g)) .- 1) .* Float64(init)
        isfinite(fg!(g, θ0)) && all(isfinite, g) && return θ0
        init isa AbstractVector && break
    end
    throw(ArgumentError("no finite starting point found; try a smaller `init`"))
end

# --- laplace ---------------------------------------------------------------------

# the mode and the Cholesky factor of the precision there
function _laplace_fit(dens, rng, init)
    fg! = _fg(dens)
    mode, _, _ = _lbfgs(fg!, _start_point(dens, rng, init))
    n = length(mode); H = zeros(n, n); gp = zeros(n); gm = zeros(n)
    for j in 1:n
        h = 1e-5 * max(1.0, abs(mode[j]))
        xp = copy(mode); xp[j] += h; xm = copy(mode); xm[j] -= h
        fg!(gp, xp); fg!(gm, xm)
        H[:, j] .= (gp .- gm) ./ 2h
    end
    P = Symmetric(-(H + H') / 2)                            # precision
    C = cholesky(P; check=false)
    issuccess(C) || throw(ArgumentError("laplace: the Hessian at the mode is not negative definite"))
    return mode, C
end
function _laplace(dens, rng; ndraws::Int=1000, init=2.0)
    mode, C = _laplace_fit(dens, rng, init)
    Z = randn(rng, length(mode), ndraws)
    return mode .+ (C.U \ Z)                               # cov = P^-1 = U^-1 U^-T
end

# --- ADVI ---------------------------------------------------------------------------

function _advi(dens, rng; fullrank::Bool=false, ndraws::Int=1000, init=2.0, maxiter::Int=10_000,
               tol_rel_obj=0.01, eval_elbo::Int=100, elbo_draws::Int=100)
    fg! = _fg(dens)
    n = LogDensityProblems.dimension(dens)
    # warm start from the Laplace approximation (mode, and the inverse Hessian's
    # diagonal / Cholesky factor). Stan starts from N(random, I); its relative-ELBO
    # stopping rule then often fires long before q has shrunk to the posterior,
    # so the warm start is what makes the result usable
    μ0, Cp = try
        _laplace_fit(dens, rng, init)
    catch
        (_start_point(dens, rng, init), nothing)
    end
    Σ0 = Cp === nothing ? Matrix(1.0I, n, n) : inv(Cp)
    L0 = fullrank ? Matrix(cholesky(Symmetric(Σ0)).L) : log.(sqrt.(diag(Σ0)))
    g = zeros(n)
    # the ELBO of (μ, scale): E log p(θ) + entropy (constants dropped)
    function elbo(μ, Lw)
        s = 0.0
        for _ in 1:elbo_draws
            ε = randn(rng, n)
            θ = fullrank ? μ .+ LowerTriangular(Lw) * ε : μ .+ exp.(Lw) .* ε
            v = fg!(g, θ)
            isfinite(v) || return -Inf
            s += v
        end
        ent = fullrank ? sum(log(abs(Lw[i, i])) for i in 1:n) : sum(Lw)
        return s / elbo_draws + ent
    end
    # one stochastic-gradient run of `iters` steps with step size η
    function run(η, iters; check=false)
        μ = copy(μ0)
        Lw = copy(L0)
        sμ = zeros(n); sL = zeros(size(Lw))
        hist = Float64[]; best = -Inf; elbo_prev = -Inf
        for k in 1:iters
            ε = randn(rng, n)
            θ = fullrank ? μ .+ LowerTriangular(Lw) * ε : μ .+ exp.(Lw) .* ε
            v = fg!(g, θ)
            isfinite(v) && all(isfinite, g) || return (μ, Lw, -Inf)
            gμ = copy(g)
            gL = fullrank ? LowerTriangular(g * ε') + Diagonal(1 ./ diag(Lw)) : g .* ε .* exp.(Lw) .+ 1
            gL = fullrank ? Matrix(gL) : gL
            # Stan's adaptive step size
            if k == 1
                sμ .= gμ .^ 2; sL .= gL .^ 2
            else
                sμ .= 0.1 .* gμ .^ 2 .+ 0.9 .* sμ; sL .= 0.1 .* gL .^ 2 .+ 0.9 .* sL
            end
            ρ = η * k^(-1 / 2 + 1e-16)
            μ .+= ρ .* gμ ./ (1 .+ sqrt.(sμ))
            Lw .+= ρ .* gL ./ (1 .+ sqrt.(sL))
            fullrank && (Lw .= Matrix(LowerTriangular(Lw)))
            if check && k % eval_elbo == 0
                e = elbo(μ, Lw)
                if isfinite(elbo_prev)
                    push!(hist, abs((e - elbo_prev) / e))
                    rec = hist[max(1, end - 9):end]
                    (mean(rec) < tol_rel_obj || median(rec) < tol_rel_obj) && return (μ, Lw, e)
                end
                elbo_prev = e
            end
        end
        return (μ, Lw, check ? elbo(μ, Lw) : elbo(μ, Lw))
    end
    # eta adaptation: the best ELBO after a short run, Stan's candidate sequence
    best = (-Inf, 1.0)
    for η in (100.0, 10.0, 1.0, 0.1, 0.01)
        _, _, e = run(η, 50)
        isfinite(e) && e > best[1] && (best = (e, η))
    end
    isfinite(best[1]) || throw(ArgumentError("ADVI: every step size diverged"))
    μ, Lw, _ = run(best[2], maxiter; check=true)
    Z = randn(rng, n, ndraws)
    return fullrank ? μ .+ LowerTriangular(Lw) * Z : μ .+ exp.(Lw) .* Z
end

# --- single-path Pathfinder -----------------------------------------------------------

function _pathfinder(dens, rng; ndraws::Int=1000, init=2.0, elbo_draws::Int=30)
    fg! = _fg(dens)
    n = LogDensityProblems.dimension(dens)
    _, _, path = _lbfgs(fg!, _start_point(dens, rng, init); keep=true)
    g = zeros(n)
    # each iterate's normal approximation: mean x + Σ g, Σ the L-BFGS inverse Hessian
    # (dense here: γ I + the rank-2m updates, built by the BFGS recursion)
    function approx(x, gx, S, Y)
        Σ = Matrix((isempty(S) ? 1.0 : dot(S[end], Y[end]) / dot(Y[end], Y[end])) * I, n, n)
        for (s, y) in zip(S, Y)
            ρ = 1 / dot(y, s)
            V = I - ρ * s * y'
            Σ = V * Σ * V' + ρ * s * s'
        end
        Σ = Symmetric((Σ + Σ') / 2)
        C = cholesky(Σ; check=false)
        issuccess(C) || return nothing
        return (x .+ Σ * gx, C)
    end
    best = (-Inf, nothing)
    for (x, _, gx, S, Y) in path[2:end]
        a = approx(x, gx, S, Y)
        a === nothing && continue
        μ, C = a
        s = 0.0; ok = true
        for _ in 1:elbo_draws
            z = randn(rng, n); θ = μ .+ C.L * z
            v = fg!(g, θ)
            isfinite(v) || (ok = false; break)
            s += v + sum(abs2, z) / 2                                # log p - log q (+ const)
        end
        ok || continue
        e = s / elbo_draws + logdet(C) / 2
        e > best[1] && (best = (e, a))
    end
    best[2] === nothing && throw(ArgumentError("pathfinder: no usable approximation along the path"))
    μ, C = best[2]
    # importance resampling of draws from the chosen normal, with PSIS-smoothed weights
    M = 4 * ndraws
    Z = randn(rng, n, M); Θ = μ .+ C.L * Z
    lw = [fg!(g, view(Θ, :, i)) + sum(abs2, view(Z, :, i)) / 2 for i in 1:M]
    lw[.!isfinite.(lw)] .= -Inf
    w = try
        vec(PSIS.psis(lw, 1.0; warn=false).weights)
    catch
        x = exp.(lw .- maximum(lw)); x ./ sum(x)
    end
    idx = [_wsample(rng, w) for _ in 1:ndraws]
    return Θ[:, idx]
end

# --- driver -------------------------------------------------------------------------

function _approx_draws(dens, algorithm::Symbol; seed=nothing, ndraws::Int=1000, init=2.0)
    rng = seed === nothing ? Random.default_rng() : Random.Xoshiro(hash((seed, :approx)))
    t0 = time()
    D = algorithm === :laplace ? _laplace(dens, rng; ndraws, init) :
        algorithm === :meanfield ? _advi(dens, rng; ndraws, init) :
        algorithm === :fullrank ? _advi(dens, rng; fullrank=true, ndraws, init) :
        algorithm === :pathfinder ? _pathfinder(dens, rng; ndraws, init) :
        throw(ArgumentError("algorithm must be :sampling, :laplace, :meanfield, :fullrank or :pathfinder"))
    draws = reshape(D, size(D, 1), size(D, 2), 1)
    _order_draws!(dens, draws)          # ordered intercepts: store tau, not u (as sampling)
    results = [(draws=draws, stats=NamedTuple(), ndiv=0, time=time() - t0)]
    return draws, results, string(algorithm)
end
