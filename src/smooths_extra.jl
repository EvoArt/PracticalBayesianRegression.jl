# =============================================================================
# More mgcv bases, as brms passes them to mgcv (smoothCon then smooth2random):
#
#   s(x, bs = "cp")   cyclic P-spline (smooth.construct.cp): cSplineDes cubic
#                     B-splines on k + 1 even knots over the data range, cyclic
#                     2nd-order difference penalty; x outside wrapped (cwrap)
#   s(x, bs = "bs")   mgcv's B-spline smooth (smooth.construct.bs, m = c(3, 2)):
#                     cubic B-splines on knots padded past the 0.1 %-widened
#                     range, penalty the integrated squared 2nd derivative over
#                     the interior (computed exactly by Gauss-Legendre per knot
#                     interval, where mgcv uses its banded construction: the
#                     same matrix); linear extrapolation outside, as ps
#   s(x, bs = "gp")   mgcv's low-rank GP smooth (smooth.construct.gp, default
#                     m): Matern 3/2 correlation, range = the largest distance
#                     between unique covariate values, the eigenvectors of the k
#                     largest eigenvalues, plus the linear null space (1, x)
#   s(g, bs = "mrf", xt = list(nb = NB))   Markov random field over the levels
#                     of g: indicator basis, penalty from the neighbours (degree
#                     on the diagonal, -1 per neighbour), or xt = list(penalty = P)
#
# All then share smoothCon's penalty scaling, the sum-to-zero constraint and
# smooth2random's eigen split (`_smooth_finish`), exactly as the other bases.
# =============================================================================

# smoothCon (penalty scaling, sum-to-zero constraint) then smooth2random's split:
# (Qc, unpenalised directions, penalised directions scaled by lambda^-1/2, Sc)
function _smooth_finish(Xd, S0; center::Bool=true)
    k = size(Xd, 2)
    S = S0 ./ (opnorm(S0, 1) / opnorm(Xd, Inf)^2)
    Qc = center ? (qr(reshape(vec(sum(Xd; dims=1)), :, 1)).Q * Matrix(I, k, k))[:, 2:k] : Matrix(1.0I, k, k)
    Sc = Symmetric(Qc' * S * Qc)
    es = eigen(Sc)
    lam = es.values
    tol = maximum(abs.(lam)) * 1e-10
    pen = findall(>(tol), lam); nul = findall(<=(tol), lam)
    return Qc, es.vectors[:, nul], es.vectors[:, pen] ./ sqrt.(lam[pen])', Matrix(Sc)
end

# --- cyclic P-spline -----------------------------------------------------------------
struct CPSmooth <: AbstractSmooth
    label::String
    knots::Vector{Float64}
    ord::Int
    Qc::Matrix{Float64}
    Uf::Matrix{Float64}
    Ur::Matrix{Float64}
    S::Matrix{Float64}
end
# mgcv's cSplineDes: B-splines on the knots extended by the last ord - 1 intervals
# copied before the first, the part past knots[nk - ord + 1] wrapped to the start
function _cspline_des(knots, x, ord)
    nk = length(knots); k1 = knots[1]; xc = knots[nk - ord + 1]
    kn2 = vcat(k1 .- (knots[nk] .- knots[(nk - ord + 1):(nk - 1)]), knots)
    return reduce(vcat, map(x) do v
        r = _bsplines(kn2, v, ord)
        v > xc && (r = r .+ _bsplines(kn2, v - maximum(kn2) + k1, ord))
        permutedims(r)
    end)
end
# mgcv's cwrap: values outside [x0, x1] taken round the circle
function _cwrap(x0, x1, x)
    h = x1 - x0
    return [v > x1 ? x0 + mod(v - x1, h) : v < x0 ? x1 - mod(x0 - v, h) : v for v in x]
end
function cps(X::AbstractMatrix{<:Real}; k=nothing, label="s", center::Bool=true)
    size(X, 2) == 1 || throw(ArgumentError("s(..., bs = \"cp\") takes one covariate"))
    x = Float64.(X[:, 1])
    m1, m2 = 2, 2
    np = k === nothing ? 10 : Int(k)
    np > m2 + 1 || throw(ArgumentError("s(..., bs = \"cp\"): k too small"))
    knots = collect(range(minimum(x), maximum(x); length=np + 1))
    Xd = _cspline_des(knots, x, m1 + 2)
    De = Matrix(1.0I, np + m2, np + m2)
    for _ in 1:m2
        De = diff(De; dims=1)
    end
    D = De[:, (m2 + 1):end]
    D[:, (np - m2 + 1):np] .+= De[:, 1:m2]
    Qc, Uf, Ur, Sc = _smooth_finish(Xd, D' * D; center)
    return CPSmooth(label, knots, m1 + 2, Qc, Uf, Ur, Sc)
end
function smooth_basis(t::CPSmooth, X)
    x = _cwrap(first(t.knots), last(t.knots), Float64.(X[:, 1]))
    B = _cspline_des(t.knots, x, t.ord) * t.Qc
    return B * t.Uf, [B * t.Ur]
end

# --- mgcv B-spline smooth ------------------------------------------------------------
struct MBSmooth <: AbstractSmooth
    label::String
    knots::Vector{Float64}
    ord::Int
    lo::Float64; hi::Float64
    Qc::Matrix{Float64}
    Uf::Matrix{Float64}
    Ur::Matrix{Float64}
    S::Matrix{Float64}
end
function mbs(X::AbstractMatrix{<:Real}; k=nothing, label="s", center::Bool=true)
    size(X, 2) == 1 || throw(ArgumentError("s(..., bs = \"bs\") takes one covariate"))
    x = Float64.(X[:, 1])
    m1, m2 = 3, 2
    bsdim = k === nothing ? 10 : Int(k)
    nk = bsdim - m1 + 1
    nk >= 2 || throw(ArgumentError("s(..., bs = \"bs\"): k too small"))
    xl, xu = extrema(x); xr = xu - xl
    xl -= xr * 0.001; xu += xr * 0.001
    dx = (xu - xl) / (nk - 1)
    kn = collect(range(xl - dx * m1, xu + dx * m1; length=nk + 2m1))
    ord = m1 + 1
    Xd = _ps_basis(kn, ord, xl, xu, x)
    # penalty: integral of the m2-th derivative products over the interior knots
    k0 = kn[(m1 + 1):(m1 + nk)]
    gx, gw = _GL01
    S0 = zeros(bsdim, bsdim)
    for j in 1:(nk - 1)
        a, b = k0[j], k0[j + 1]
        for (u, w) in zip(gx, gw)
            d = _bsplines(kn, a + (b - a) * u, ord, m2)
            S0 .+= (w * (b - a)) .* (d * d')
        end
    end
    Qc, Uf, Ur, Sc = _smooth_finish(Xd, S0; center)
    return MBSmooth(label, kn, ord, xl, xu, Qc, Uf, Ur, Sc)
end
function smooth_basis(t::MBSmooth, X)
    B = _ps_basis(t.knots, t.ord, t.lo, t.hi, Float64.(X[:, 1])) * t.Qc
    return B * t.Uf, [B * t.Ur]
end

# --- mgcv GP smooth ------------------------------------------------------------------
struct MGPSmooth <: AbstractSmooth
    label::String
    knt::Matrix{Float64}         # unique covariate rows, shifted
    shift::Vector{Float64}
    rho::Float64
    U::Matrix{Float64}           # eigenvectors of the k largest eigenvalues
    Qc::Matrix{Float64}
    Uf::Matrix{Float64}
    Ur::Matrix{Float64}
    S::Matrix{Float64}
end
_mgp_E(r) = (1 + r) * exp(-r)                     # mgcv gpE type 3 (Matern 3/2)
function _mgp_raw(knt, shift, rho, U, X)
    Xs = Float64.(X) .- shift'
    E = [_mgp_E(sqrt(sum(abs2, view(Xs, i, :) .- view(knt, j, :))) / rho) for i in axes(Xs, 1), j in axes(knt, 1)]
    return hcat(E * U, ones(size(Xs, 1)), Xs)
end
function mgps(X::AbstractMatrix{<:Real}; k=nothing, label="s", center::Bool=true)
    Xf = Float64.(X); d = size(Xf, 2)
    shift = vec(mean(Xf; dims=1))
    knt = reduce(vcat, (permutedims(collect(r)) for r in unique(eachrow(Xf)))) .- shift'
    nk = size(knt, 1)
    bsdim = k === nothing ? d + 1 + (10, 30, 100)[min(d, 3)] : Int(k)
    bsdim >= d + 2 || throw(ArgumentError("s(..., bs = \"gp\"): k must be at least $(d + 2)"))
    kk = bsdim - (d + 1)
    kk <= nk || throw(ArgumentError("s(..., bs = \"gp\"): k exceeds the unique covariate values"))
    Dm = [sqrt(sum(abs2, view(knt, i, :) .- view(knt, j, :))) for i in 1:nk, j in 1:nk]
    rho = maximum(Dm)
    es = eigen(Symmetric(_mgp_E.(Dm ./ rho)))
    ord = sortperm(es.values; rev=true)[1:kk]
    U = es.vectors[:, ord]
    S0 = Matrix(Diagonal(vcat(es.values[ord], zeros(d + 1))))
    Xd = _mgp_raw(knt, shift, rho, U, Xf)
    Qc, Uf, Ur, Sc = _smooth_finish(Xd, S0; center)
    return MGPSmooth(label, knt, shift, rho, U, Qc, Uf, Ur, Sc)
end
function smooth_basis(t::MGPSmooth, X)
    B = _mgp_raw(t.knt, t.shift, t.rho, t.U, X) * t.Qc
    return B * t.Uf, [B * t.Ur]
end

# --- Markov random field -------------------------------------------------------------
struct MRFSmooth <: AbstractSmooth
    label::String
    levels::Vector{String}
    Qc::Matrix{Float64}
    Uf::Matrix{Float64}
    Ur::Matrix{Float64}
    S::Matrix{Float64}
end
function _mrf_X(levels, g)
    pos = Dict(l => j for (j, l) in enumerate(levels))
    X = zeros(length(g), length(levels))
    for (i, v) in enumerate(g)
        j = get(pos, _level_name(v), 0)
        j > 0 && (X[i, j] = 1.0)               # a level not in the model: no effect
    end
    return X
end
# the penalty from xt: list(penalty = P) (levels in sorted order), or list(nb = NB)
# with NB a vector of neighbour index vectors (sorted level order) or a Dict of
# level name => neighbour names
function _mrf_penalty(xt, levels)
    n = length(levels)
    if haskey(xt, :penalty)
        P = Float64.(xt[:penalty])
        size(P) == (n, n) || throw(ArgumentError("s(..., bs = \"mrf\"): the penalty must be $n x $n (the levels, sorted)"))
        return P
    end
    haskey(xt, :nb) || throw(ArgumentError("s(..., bs = \"mrf\") needs xt = list(nb = ...) or xt = list(penalty = ...)"))
    nb = xt[:nb]
    pos = Dict(l => j for (j, l) in enumerate(levels))
    idx(i) = nb isa AbstractDict ? [pos[string(v)] for v in nb[levels[i]]] : Int.(nb[i])
    length(nb) == n || throw(ArgumentError("s(..., bs = \"mrf\"): nb must have one entry per level ($n)"))
    S = zeros(n, n)
    for i in 1:n
        ind = idx(i)
        S[i, i] = length(ind)
        for j in ind
            j == i || (S[i, j] = -1.0)
        end
    end
    issymmetric(S) || throw(ArgumentError("s(..., bs = \"mrf\"): the neighbour list is not symmetric"))
    return S
end
function mrfs(cols, levels, xt; label="s", center::Bool=true)
    length(cols) == 1 || throw(ArgumentError("s(..., bs = \"mrf\") takes one factor"))
    Xd = _mrf_X(levels, cols[1])
    Qc, Uf, Ur, Sc = _smooth_finish(Xd, _mrf_penalty(xt, levels); center)
    return MRFSmooth(label, levels, Qc, Uf, Ur, Sc)
end
function smooth_basis(t::MRFSmooth, cols)
    B = _mrf_X(t.levels, cols[1]) * t.Qc
    return B * t.Uf, [B * t.Ur]
end
