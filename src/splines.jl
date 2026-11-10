# =============================================================================
# Smooth terms, as brms builds them from mgcv, then `smooth2random`'s split
# into an unpenalised part (ordinary coefficients: brms `bs_<label>`, here
# columns of the population design) and penalised parts that are random
# effects, one per penalty:
#
#     eta += Xs * bs + sum_b Zs_b * (sds_b * zs_b),  zs_b ~ N(0, 1)
#
#   s(x, ...)       thin-plate regression spline (mgcv "tp", Wood 2003) of d
#                   covariates, m = floor((d + 1) / 2) + 1, default k = 10 * 3^(d - 1)
#   t2(x, z, ...)   mgcv's t2 tensor product of cubic-regression-spline ("cr")
#                   marginals (default k = 5 each): every marginal is split into
#                   its penalised range and its null space; each product of
#                   pieces containing a range part is one penalty (an identity,
#                   so one random-effect block), the product of the null
#                   spaces is unpenalised
#   s(x, by = g)    one smooth per level of factor g (the basis times the
#                   level's indicator), each with its own sds
#
# The posterior depends on a basis only through span(Xs) and the Zs_b Zs_b',
# not on mgcv's eigenvector signs or QR conventions. Sum-to-zero (centring)
# constraints are absorbed as mgcv does for s(); for t2 the constraint is
# absorbed into the null-space block. The scale of sds differs from brms's
# (mgcv's further reparameterisations), as noted for s(x) before.
# =============================================================================

abstract type AbstractSmooth end

# --- thin-plate regression splines (any d) ----------------------------------------

# mgcv's TPS kernel eta_{m,d}(r)
function _tps_eta(r, d, m)
    r == 0 && return 0.0
    if iseven(d)
        c = (-1)^(m + 1 + d ÷ 2) / (2^(2m - 1) * π^(d / 2) * factorial(m - 1) * factorial(m - d ÷ 2))
        return c * r^(2m - d) * log(r)
    end
    lg, sg = logabsgamma(d / 2 - m)
    c = sg * exp(lg) / (2^(2m) * π^(d / 2) * factorial(m - 1))
    return c * r^(2m - d)
end
# monomials of total degree < m in d variables (the null space), constant first
function _tps_powers(d, m)
    out = Vector{Int}[]
    for deg in 0:(m - 1), p in Iterators.product(ntuple(_ -> 0:deg, d)...)
        sum(p) == deg && push!(out, collect(p))
    end
    return out
end
_tps_T(Xs, pw) = reduce(hcat, [vec(prod(Xs .^ permutedims(p); dims=2)) for p in pw])

struct TPRS <: AbstractSmooth
    label::String              # brms name stem, e.g. "sx" / "sxz"
    knots::Matrix{Float64}     # unique covariate rows
    shift::Vector{Float64}
    m::Int
    powers::Vector{Vector{Int}}
    UkZ::Matrix{Float64}       # nknots x (k - M)
    Qc::Matrix{Float64}        # k x (k - 1): sum-to-zero constraint
    Uf::Matrix{Float64}        # unpenalised directions
    Ur::Matrix{Float64}        # penalised directions scaled by lam^-1/2
    S::Matrix{Float64}
end
function _tprs_raw(t::TPRS, X)
    d = size(X, 2)
    E = [_tps_eta(sqrt(sum(abs2, view(X, i, :) .- view(t.knots, j, :))), d, t.m)
         for i in axes(X, 1), j in axes(t.knots, 1)]
    return hcat(E * t.UkZ, _tps_T(X .- t.shift', t.powers))
end

"""
    tprs(X; k, label) -> TPRS

The thin-plate regression spline of the columns of `X` (a vector for one
covariate), constructed as mgcv does for `s(x1, ..., k = k)` inside brms.
"""
tprs(x::AbstractVector{<:Real}; k=10, label="sx") = tprs(reshape(Float64.(x), :, 1); k, label)
function tprs(X::AbstractMatrix{<:Real}; k=nothing, label="s", center::Bool=true, shrink::Bool=false)
    Xf = Float64.(X)
    d = size(Xf, 2)
    m = (d + 1) ÷ 2 + 1                     # mgcv's default penalty order (2m > d)
    pw = _tps_powers(d, m)
    M = length(pw)
    k = k === nothing ? 10 * 3^(d - 1) : k
    rows = unique(eachrow(Xf))
    knots = reduce(vcat, (permutedims(collect(r)) for r in rows))
    nk = size(knots, 1)
    k <= nk || throw(ArgumentError("s(): k = $k exceeds the number of unique covariate values ($nk)"))
    k > M || throw(ArgumentError("s(): k must exceed the null-space dimension ($M)"))
    shift = vec(mean(Xf; dims=1))
    E = [_tps_eta(sqrt(sum(abs2, view(knots, i, :) .- view(knots, j, :))), d, m) for i in 1:nk, j in 1:nk]
    ev = eigen(Symmetric(E))
    ord = sortperm(abs.(ev.values); rev=true)[1:k]
    Uk = ev.vectors[:, ord]
    Dk = ev.values[ord]
    T = _tps_T(knots .- shift', pw)
    Q = qr(Uk' * T).Q * Matrix(I, k, k)
    Z = Q[:, (M + 1):k]
    UkZ = Uk * Z
    S0 = Z' * Diagonal(Dk) * Z
    tmp = TPRS(label, knots, shift, m, pw, UkZ, zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0))
    Xd = _tprs_raw(tmp, Xf)
    S = zeros(k, k)
    S[1:(k - M), 1:(k - M)] .= S0
    if shrink
        # bs = "ts" (mgcv smooth.construct.tp, shrink = 0.1): the M null-space
        # eigenvalues become 0.1 x the smallest penalised one, so nothing is left
        # unpenalised (no fixed part). Before the scaling, as mgcv.
        es = eigen(Symmetric(S))                    # ascending: 1:M is the null space
        v = copy(es.values); v[1:M] .= v[M + 1] * 0.1
        S = Matrix(Symmetric(es.vectors * Diagonal(v) * es.vectors'))
    end
    S ./= opnorm(S, 1) / opnorm(Xd, Inf)^2           # smoothCon's penalty scaling
    C = vec(sum(Xd; dims=1))                        # sum-to-zero constraint (not for numeric by)
    Qc = center ? (qr(reshape(C, :, 1)).Q * Matrix(I, k, k))[:, 2:k] : Matrix(1.0I, k, k)
    Sc = Symmetric(Qc' * S * Qc)
    es = eigen(Sc)                                  # smooth2random
    lam = es.values
    tol = maximum(abs.(lam)) * 1e-10
    pen = findall(>(tol), lam); nul = findall(<=(tol), lam)
    return TPRS(label, knots, shift, m, pw, UkZ, Qc, es.vectors[:, nul], es.vectors[:, pen] ./ sqrt.(lam[pen])',
                Matrix(Sc))
end

"""
    smooth_basis(t, X) -> (Xs, [Zs_1, ...])

The unpenalised columns and the penalised blocks of smooth `t` at the rows of
`X` (training or new data — mgcv's PredictMat followed by the same transforms).
"""
function smooth_basis(t::TPRS, X)
    Xc = _tprs_raw(t, Float64.(X)) * t.Qc
    return Xc * t.Uf, [Xc * t.Ur]
end

# --- cubic regression splines (mgcv "cr") and t2 tensor products -------------------

struct CRMarginal
    knots::Vector{Float64}
    F::Matrix{Float64}         # values at knots -> second derivatives at knots
    Ur::Matrix{Float64}        # penalised directions scaled by lam^-1/2
    U0::Matrix{Float64}        # null space (constant, linear)
end
# mgcv place.knots: evenly spaced through the order statistics of unique x
function _cr_knots(x, k)
    u = sort(unique(x)); n = length(u)
    n >= k || throw(ArgumentError("k = $k exceeds the number of unique covariate values ($n)"))
    return [begin
                p = (i - 1) * (n - 1) / (k - 1) + 1
                j = min(floor(Int, p), n - 1)
                u[j] + (p - j) * (u[j + 1] - u[j])
            end for i in 1:k]
end
function _cr_FS(kn)
    k = length(kn); h = diff(kn)
    D = zeros(k - 2, k); B = zeros(k - 2, k - 2)
    for i in 1:(k - 2)
        D[i, i] = 1 / h[i]; D[i, i + 1] = -1 / h[i] - 1 / h[i + 1]; D[i, i + 2] = 1 / h[i + 1]
        B[i, i] = (h[i] + h[i + 1]) / 3
        i < k - 2 && (B[i, i + 1] = B[i + 1, i] = h[i + 1] / 6)
    end
    F = vcat(zeros(1, k), B \ D, zeros(1, k))
    return F, D' * (B \ D)
end
# value of the natural cubic spline basis at v inside [kn[1], kn[end]] and its slope
function _cr_row(kn, F, v)
    k = length(kn)
    j = clamp(searchsortedlast(kn, v), 1, k - 1)
    hh = kn[j + 1] - kn[j]
    am, ap = (kn[j + 1] - v) / hh, (v - kn[j]) / hh
    cm = ((kn[j + 1] - v)^3 / hh - hh * (kn[j + 1] - v)) / 6
    cp = ((v - kn[j])^3 / hh - hh * (v - kn[j])) / 6
    row = zeros(k); row[j] += am; row[j + 1] += ap
    row .+= cm .* F[j, :] .+ cp .* F[j + 1, :]
    slope = zeros(k); slope[j] -= 1 / hh; slope[j + 1] += 1 / hh
    slope .+= (-(3 * (kn[j + 1] - v)^2 / hh - hh) / 6) .* F[j, :] .+ ((3 * (v - kn[j])^2 / hh - hh) / 6) .* F[j + 1, :]
    return row, slope
end
function _cr_basis(kn, F, x)
    Bm = zeros(length(x), length(kn))
    for (n, v) in enumerate(x)
        piv = clamp(v, kn[1], kn[end])
        row, slope = _cr_row(kn, F, piv)
        Bm[n, :] = row .+ (v - piv) .* slope        # linear beyond the boundary knots, as mgcv
    end
    return Bm
end
function CRMarginal(x, k)
    kn = _cr_knots(Float64.(x), k)
    F, S = _cr_FS(kn)
    es = eigen(Symmetric(S))
    tol = maximum(abs.(es.values)) * 1e-10
    pen = findall(>(tol), es.values); nul = findall(<=(tol), es.values)
    return CRMarginal(kn, F, es.vectors[:, pen] ./ sqrt.(es.values[pen])', es.vectors[:, nul])
end
_cr_pieces(m::CRMarginal, x) = (B = _cr_basis(m.knots, m.F, x); (B * m.Ur, B * m.U0))

# s(x, bs = "cr"): one cubic regression spline (mgcv smooth.construct.cr), then
# smoothCon's penalty scaling, the sum-to-zero constraint (none for a numeric
# `by`) and smooth2random's split, as for tprs
struct CRSmooth <: AbstractSmooth
    label::String
    knots::Vector{Float64}
    F::Matrix{Float64}
    Qc::Matrix{Float64}
    Uf::Matrix{Float64}
    Ur::Matrix{Float64}
    S::Matrix{Float64}
end
function crs(X::AbstractMatrix{<:Real}; k=nothing, label="s", center::Bool=true, shrink::Bool=false)
    size(X, 2) == 1 || throw(ArgumentError("s(..., bs = \"cr\") takes one covariate"))
    x = Float64.(X[:, 1])
    k = k === nothing ? 10 : Int(k)
    k >= 3 || throw(ArgumentError("s(..., bs = \"cr\"): k must be at least 3"))
    kn = _cr_knots(x, k)
    F, S0 = _cr_FS(kn)
    if shrink
        # bs = "cs" (mgcv smooth.construct.cr, shrink = 0.1): the two null-space
        # eigenvalues become 0.1 and 0.01 x the smallest penalised one. (Which
        # null direction gets which is LAPACK's choice in a degenerate space, in
        # mgcv as here.)
        es = eigen(Symmetric((S0 + S0') / 2))       # ascending: 1, 2 are mgcv's nk, nk - 1
        v = copy(es.values); v[2] = v[3] * 0.1; v[1] = v[2] * 0.1
        S0 = Matrix(Symmetric(es.vectors * Diagonal(v) * es.vectors'))
    end
    Xd = _cr_basis(kn, F, x)
    S = S0 ./ (opnorm(S0, 1) / opnorm(Xd, Inf)^2)    # smoothCon's penalty scaling
    Qc = center ? (qr(reshape(vec(sum(Xd; dims=1)), :, 1)).Q * Matrix(I, k, k))[:, 2:k] : Matrix(1.0I, k, k)
    Sc = Symmetric(Qc' * S * Qc)
    es = eigen(Sc)
    lam = es.values
    tol = maximum(abs.(lam)) * 1e-10
    pen = findall(>(tol), lam); nul = findall(<=(tol), lam)
    return CRSmooth(label, kn, F, Qc, es.vectors[:, nul], es.vectors[:, pen] ./ sqrt.(lam[pen])', Matrix(Sc))
end
function smooth_basis(t::CRSmooth, X)
    B = _cr_basis(t.knots, t.F, Float64.(X[:, 1])) * t.Qc
    return B * t.Uf, [B * t.Ur]
end

# s(x, bs = "ps"): mgcv's P-spline (smooth.construct.ps): cubic B-splines
# (order m1 + 2, m1 = 2) on k + m1 + 2 evenly spaced knots spanning the data
# range widened by 0.1 % and padded by m1 + 1 spacings, and the m2 = 2nd-order
# difference penalty D'D. Outside [xl, xu] mgcv extrapolates linearly. Then
# smoothCon / smooth2random exactly as for cr.
struct PSSmooth <: AbstractSmooth
    label::String
    knots::Vector{Float64}
    ord::Int
    lo::Float64; hi::Float64         # the (widened) data range
    Qc::Matrix{Float64}
    Uf::Matrix{Float64}
    Ur::Matrix{Float64}
    S::Matrix{Float64}
end
function _ps_basis(kn, ord, lo, hi, x)
    return reduce(vcat, permutedims(v < lo ? _bsplines(kn, lo, ord) .+ (v - lo) .* _bsplines(kn, lo, ord, 1) :
                                     v > hi ? _bsplines(kn, hi, ord) .+ (v - hi) .* _bsplines(kn, hi, ord, 1) :
                                     _bsplines(kn, v, ord)) for v in x)
end
function pss(X::AbstractMatrix{<:Real}; k=nothing, label="s", center::Bool=true)
    size(X, 2) == 1 || throw(ArgumentError("s(..., bs = \"ps\") takes one covariate"))
    x = Float64.(X[:, 1])
    m1, m2 = 2, 2
    k = k === nothing ? 10 : Int(k)
    nk = k - m1
    nk >= 2 || throw(ArgumentError("s(..., bs = \"ps\"): k must be at least $(m1 + 2)"))
    xl, xu = extrema(x); xr = xu - xl
    xl -= xr * 0.001; xu += xr * 0.001
    dx = (xu - xl) / (nk - 1)
    kn = collect(range(xl - dx * (m1 + 1), xu + dx * (m1 + 1); length=nk + 2m1 + 2))
    Xd = _ps_basis(kn, m1 + 2, xl, xu, x)
    D = Matrix(1.0I, k, k)
    for _ in 1:m2
        D = diff(D; dims=1)
    end
    S0 = D' * D
    S = S0 ./ (opnorm(S0, 1) / opnorm(Xd, Inf)^2)       # smoothCon's penalty scaling
    Qc = center ? (qr(reshape(vec(sum(Xd; dims=1)), :, 1)).Q * Matrix(I, k, k))[:, 2:k] : Matrix(1.0I, k, k)
    es = eigen(Symmetric(Qc' * S * Qc))
    lam = es.values
    tol = maximum(abs.(lam)) * 1e-10
    pen = findall(>(tol), lam); nul = findall(<=(tol), lam)
    return PSSmooth(label, kn, m1 + 2, xl, xu, Qc, es.vectors[:, nul], es.vectors[:, pen] ./ sqrt.(lam[pen])',
                    Matrix(Symmetric(Qc' * S * Qc)))
end
# s(x, bs = "cc"): mgcv's cyclic cubic regression spline (smooth.construct.cc):
# k knots (place.knots), k - 1 coefficients (the value at the last knot is the
# first's), the cyclic B, D of getBD, penalty D' B^-1 D; x outside the knot
# range is wrapped round (cwrap). The penalty's null space is the constant, so
# after the sum-to-zero constraint everything is penalised (no fixed part).
struct CCSmooth <: AbstractSmooth
    label::String
    knots::Vector{Float64}
    BD::Matrix{Float64}
    Qc::Matrix{Float64}
    Uf::Matrix{Float64}
    Ur::Matrix{Float64}
    S::Matrix{Float64}
end
function _cc_BD(kn)
    n = length(kn) - 1; h = diff(kn)
    B = zeros(n, n); D = zeros(n, n)
    B[1, 1] = (h[n] + h[1]) / 3; B[1, 2] = h[1] / 6; B[1, n] = h[n] / 6
    D[1, 1] = -(1 / h[1] + 1 / h[n]); D[1, 2] = 1 / h[1]; D[1, n] = 1 / h[n]
    for i in 2:(n - 1)
        B[i, i - 1] = h[i - 1] / 6; B[i, i] = (h[i - 1] + h[i]) / 3; B[i, i + 1] = h[i] / 6
        D[i, i - 1] = 1 / h[i - 1]; D[i, i] = -(1 / h[i - 1] + 1 / h[i]); D[i, i + 1] = 1 / h[i]
    end
    B[n, n - 1] = h[n - 1] / 6; B[n, n] = (h[n - 1] + h[n]) / 3; B[n, 1] = h[n] / 6
    D[n, n - 1] = 1 / h[n - 1]; D[n, n] = -(1 / h[n - 1] + 1 / h[n]); D[n, 1] = 1 / h[n]
    return B, D
end
function _cc_basis(kn, BD, x)
    n = length(kn); h = diff(kn); lo, hi = kn[1], kn[n]; w = hi - lo
    X = zeros(length(x), n - 1)
    for (r, v0) in enumerate(x)
        v = v0 > hi ? lo + mod(v0 - hi, w) : v0 < lo ? hi - mod(lo - v0, w) : v0     # cwrap
        j = something(findfirst(i -> v <= kn[i], 2:n), n - 1) + 1                     # smallest i >= 2 with v <= kn[i]
        j1 = j - 1; jj = j == n ? 1 : j
        a, b = kn[j1 + 1] - v, v - kn[j1]; hh = h[j1]
        X[r, :] .= BD[j1, :] .* (a^3 / (6hh) - hh * a / 6) .+ BD[jj, :] .* (b^3 / (6hh) - hh * b / 6)
        X[r, j1] += a / hh; X[r, jj] += b / hh
    end
    return X
end
function ccs(X::AbstractMatrix{<:Real}; k=nothing, label="s", center::Bool=true)
    size(X, 2) == 1 || throw(ArgumentError("s(..., bs = \"cc\") takes one covariate"))
    x = Float64.(X[:, 1])
    k = k === nothing ? 10 : Int(k)
    k >= 4 || throw(ArgumentError("s(..., bs = \"cc\"): k must be at least 4"))
    kn = _cr_knots(x, k)
    B, D = _cc_BD(kn)
    BD = B \ D
    Xd = _cc_basis(kn, BD, x)
    S0 = D' * BD
    S0 = (S0 + S0') / 2
    m = size(Xd, 2)
    S = S0 ./ (opnorm(S0, 1) / opnorm(Xd, Inf)^2)
    Qc = center ? (qr(reshape(vec(sum(Xd; dims=1)), :, 1)).Q * Matrix(I, m, m))[:, 2:m] : Matrix(1.0I, m, m)
    es = eigen(Symmetric(Qc' * S * Qc))
    lam = es.values
    tol = maximum(abs.(lam)) * 1e-10
    pen = findall(>(tol), lam); nul = findall(<=(tol), lam)
    return CCSmooth(label, kn, BD, Qc, es.vectors[:, nul], es.vectors[:, pen] ./ sqrt.(lam[pen])',
                    Matrix(Symmetric(Qc' * S * Qc)))
end
function smooth_basis(t::CCSmooth, X)
    B = _cc_basis(t.knots, t.BD, Float64.(X[:, 1])) * t.Qc
    return B * t.Uf, [B * t.Ur]
end

function smooth_basis(t::PSSmooth, X)
    B = _ps_basis(t.knots, t.ord, t.lo, t.hi, Float64.(X[:, 1])) * t.Qc
    return B * t.Uf, [B * t.Ur]
end

_rowkron(A, B) = reduce(hcat, [A[:, i] .* B[:, j] for i in axes(A, 2) for j in axes(B, 2)])

struct T2Smooth <: AbstractSmooth
    label::String
    margins::Vector{CRMarginal}
    combos::Vector{Vector{Bool}}   # per penalised block: range (true) / null (false) per margin
    Qc::Matrix{Float64}            # sum-to-zero constraint on the null x null block
end
function _t2_blocks(t::T2Smooth, X)
    pieces = [_cr_pieces(m, X[:, d]) for (d, m) in enumerate(t.margins)]
    blk(c) = reduce(_rowkron, [c[d] ? pieces[d][1] : pieces[d][2] for d in eachindex(c)])
    return blk(falses(length(t.margins))), [blk(c) for c in t.combos]
end
function T2Smooth(X::AbstractMatrix; k=nothing, label="t2")
    d = size(X, 2)
    d >= 2 || throw(ArgumentError("t2() needs at least two covariates"))
    kk = k === nothing ? fill(5, d) : (length(k) == 1 ? fill(Int(k[1]), d) : Int.(k))
    margins = [CRMarginal(X[:, j], kk[j]) for j in 1:d]
    combos = [collect(c) for c in Iterators.product(ntuple(_ -> (true, false), d)...) if any(c)]
    tmp = T2Smooth(label, margins, combos, zeros(0, 0))
    nn, _ = _t2_blocks(tmp, Float64.(X))
    C = vec(sum(nn; dims=1))
    Qc = (qr(reshape(C, :, 1)).Q * Matrix(I, length(C), length(C)))[:, 2:end]
    return T2Smooth(label, margins, combos, Qc)
end
function smooth_basis(t::T2Smooth, X)
    nn, Zs = _t2_blocks(t, Float64.(X))
    return nn * t.Qc, Zs
end

struct SmoothBlock{T}
    label::String
    Z::Matrix{T}               # N x r penalised design
    sds::Int                   # θ index of log sds
    zs::UnitRange{Int}         # r standard-normal coefficients
    sds_prior::ScalarPrior
end

function _smooth_forward!(η, sb::SmoothBlock, θ, buf)
    s = exp(θ[sb.sds])
    @inbounds for j in eachindex(sb.zs)
        buf[j] = s * θ[sb.zs[j]]
    end
    mul!(η, sb.Z, buf, true, true)
    return η
end

function _smooth_pullback!(g, sb::SmoothBlock, w, θ, buf)
    T = eltype(g)
    mul!(buf, transpose(sb.Z), w)               # t = Z' w
    u = θ[sb.sds]; s = exp(u)
    dsd = zero(T); lp = zero(T)
    @inbounds for (j, k) in enumerate(sb.zs)
        zk = θ[k]
        dsd += zk * buf[j]
        g[k] += s * buf[j] - zk
        lp -= zk * zk / 2
    end
    v, d = prior_kernel_grad(sb.sds_prior, s)
    g[sb.sds] += s * (dsd + d) + 1
    return lp + v + u
end

function _smooth_generic(sb::SmoothBlock, θ)
    s = exp(θ[sb.sds])
    lp = prior_kernel(sb.sds_prior, s) + θ[sb.sds] - sum(θ[sb.zs] .^ 2) / 2
    return sb.Z * (s .* θ[sb.zs]), lp
end

# s(g, bs = "re"), s(x, g, bs = "re"): mgcv's random-effect smooth. X is the
# model matrix of the terms' interaction with no intercept (a factor coded
# fully, numeric covariates multiplied in; ~ g:h - 1 column order, the first
# factor varying fastest), the penalty is the identity and nothing is
# constrained or left unpenalised. smoothCon's scaling of S = I by ||X||_inf^2
# makes smooth2random's block X / ||X||_inf (checked against brms's standata).
# Covariates arrive as raw columns (factors are not
# numbers), at build time and for new data alike.
struct RESmooth <: AbstractSmooth
    label::String
    levels::Vector{Union{Nothing,Vector{String}}}   # per covariate: factor levels, or nothing (numeric)
    scale::Float64                                  # ||X||_inf of the training design
end
function _re_X(levels, cols)
    N = length(cols[1])
    num = ones(N)
    for (k, c) in enumerate(cols)
        levels[k] === nothing && (num .*= Float64.(c))
    end
    fk = [k for k in eachindex(cols) if levels[k] !== nothing]
    isempty(fk) && return reshape(num, N, 1)
    sizes = [length(levels[k]) for k in fk]
    X = zeros(N, prod(sizes))
    pos = [Dict(l => j for (j, l) in enumerate(levels[k])) for k in fk]
    for i in 1:N
        col, stride, ok = 1, 1, true
        for (q, k) in enumerate(fk)
            j = get(pos[q], _level_name(cols[k][i]), 0)
            j == 0 && (ok = false; break)        # a level not seen in training: no column
            col += (j - 1) * stride; stride *= sizes[q]
        end
        ok && (X[i, col] = num[i])
    end
    return X
end
function res(cols, levels; label="s")
    X = _re_X(levels, cols)
    return RESmooth(label, levels, opnorm(X, Inf))
end
smooth_basis(t::RESmooth, cols) = (zeros(length(cols[1]), 0), [_re_X(t.levels, cols) ./ t.scale])
_smooth_cols(t, cols) = t isa Union{RESmooth,FSSmooth,MRFSmooth} ? cols : reduce(hcat, [Float64.(c) for c in cols])

# s(x, g, bs = "fs"): mgcv's factor-smooth interaction, as brms builds it. The
# base smooth of the numeric covariates (tp, or cr with xt = "cr"), uncentred,
# is reparameterised by mgcv's nat.param(type = 1): X = QR, R^-T S R^-1 = U D U',
# basis X R^-1 U with the penalised columns divided by sqrt(D), each group of
# columns (penalised / null space) scaled to unit mean square. That basis is
# replicated per level of the factor (zero elsewhere). One penalty for all the
# penalised columns, one per null-space column, all identities after smoothCon's
# scaling, so smooth2random gives 1 + null-dim random blocks, each the columns
# divided by the fs design's infinity norm; nothing is unpenalised. As with cs,
# mgcv's null-space basis is whatever LAPACK returns for an exactly-zero
# eigenvalue pair (an arbitrary rotation of e.g. {1, x}); here it is ours.
struct FSSmooth{B<:AbstractSmooth} <: AbstractSmooth
    label::String
    base::B                       # uncentred tp / cr smooth of the numeric covariates
    P::Matrix{Float64}            # raw base basis -> nat.param basis
    rank::Int
    fpos::Int                     # which covariate is the factor
    levels::Vector{String}
    scale::Float64                # ||X_fs||_inf of the training design
end
_fs_raw(b::TPRS, X) = _tprs_raw(b, X)
_fs_raw(b::CRSmooth, X) = _cr_basis(b.knots, b.F, Float64.(X[:, 1]))
_fs_rawS(b::Union{TPRS,CRSmooth}) = b.S          # uncentred: Qc = I
function _fs_split(cols, fpos)
    num = [Float64.(c) for (k, c) in enumerate(cols) if k != fpos]
    return reduce(hcat, num), [_level_name(v) for v in cols[fpos]]
end
function fss(cols, fpos, levels; k=nothing, base=:tp, label="s")
    Xn, _ = _fs_split(cols, fpos)
    b = base === :cr ? crs(Xn; k, label, center=false) : tprs(Xn; k, label, center=false)
    X = _fs_raw(b, Xn)
    S = _fs_rawS(b)
    # nat.param(X, S, type = 1, unit.fnorm = TRUE)
    F = qr(X)
    R = Matrix(F.R); Q = Matrix(F.Q)
    RSR = R' \ (R' \ S')'                         # R^-T S R^-1
    es = eigen(Symmetric((RSR + RSR') / 2); sortby=v -> -v)
    tol = maximum(es.values) * eps()^0.8
    rank = count(>(tol), es.values)
    U = es.vectors
    Xr = Q * U
    P = R \ U
    E = vcat(sqrt.(es.values[1:rank]), ones(size(X, 2) - rank))
    Xr ./= E'; P ./= E'
    sc = 1 / sqrt(mean(abs2, Xr[:, 1:rank]))
    Xr[:, 1:rank] .*= sc; P[:, 1:rank] .*= sc
    if rank < size(X, 2)
        ind = (rank + 1):size(X, 2)
        scf = 1 / sqrt(mean(abs2, Xr[:, ind]))
        Xr[:, ind] .*= scf; P[:, ind] .*= scf
    end
    # each row has one level's block: the fs design's infinity norm is the largest row sum
    scale = maximum(sum(abs, Xr; dims=2))
    return FSSmooth(label, b, P, rank, fpos, levels, scale)
end
function smooth_basis(t::FSSmooth, cols)
    Xn, gv = _fs_split(cols, t.fpos)
    Xr = _fs_raw(t.base, Xn) * t.P ./ t.scale
    N, kb = size(Xr); nf = length(t.levels)
    pos = Dict(l => j for (j, l) in enumerate(t.levels))
    Zpen = zeros(N, t.rank * nf)
    Znul = [zeros(N, nf) for _ in 1:(kb - t.rank)]
    for i in 1:N
        j = get(pos, gv[i], 0)
        j == 0 && continue                          # a level not in training: no effect
        Zpen[i, ((j - 1) * t.rank + 1):(j * t.rank)] .= view(Xr, i, 1:t.rank)
        for (m, Zm) in enumerate(Znul)
            Zm[i, j] = Xr[i, t.rank + m]
        end
    end
    return zeros(N, 0), vcat([Zpen], Znul)
end
