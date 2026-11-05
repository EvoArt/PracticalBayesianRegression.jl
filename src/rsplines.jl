# =============================================================================
# R's regression-spline bases in formulas: splines::bs() and splines::ns().
#
# Both are design-matrix expansions of one numeric variable (like poly(), see
# design.jl), so they live in the atom's AtomInfo: the training data fixes the
# knots (and for ns the constraint rotation), and new data is coded with the
# stored ones. Following R's source:
#
#   bs(x, df, degree = 3): ord = degree + 1; df - degree interior knots at
#     quantiles (type 7) of x; Aknots = sort(c(rep(range(x), ord), knots));
#     basis = splineDesign(Aknots, x, ord) without its first column. Outside
#     the boundary: Taylor expansion of degree `degree` at the boundary knot.
#   ns(x, df): df - 1 interior knots; cubic B-splines on Aknots as above,
#     first column dropped, then rotated by the trailing columns of Q from
#     qr(t(const)), const = the basis' second derivatives at the two boundary
#     knots (also first column dropped) — so the fit is linear beyond the
#     boundaries, where it is extrapolated linearly.
#
# R's qr() (LINPACK dqrdc2) and LAPACK reflect x onto -sign(x1) |x| e1; a
# Householder reflector onto a given vector is unique, so Julia's full Q is
# R's Q and the ns columns match R's, not just their span.
# =============================================================================

struct SplineBasis
    natural::Bool              # ns (true) or bs (false)
    knots::Vector{Float64}     # R's Aknots (boundary knots repeated ord times)
    ord::Int
    lo::Float64
    hi::Float64
    Q::Matrix{Float64}         # ns: rotation onto the constraint null space
end

# All B-splines of order k on knots t at x (Cox-de Boor), or their d-th
# derivative. Length length(t) - k. At the right boundary R takes the last
# non-empty interval (the basis is right-continuous there).
function _bsplines(t, x, k, d=0)
    n = length(t)
    if d > 0
        B = _bsplines(t, x, k - 1, d - 1)
        out = zeros(n - k)
        for i in 1:(n - k)
            a = t[i + k - 1] - t[i]
            b = t[i + k] - t[i + 1]
            out[i] = (k - 1) * ((a > 0 ? B[i] / a : 0.0) - (b > 0 ? B[i + 1] / b : 0.0))
        end
        return out
    end
    B = zeros(n - 1)
    hit = findlast(i -> t[i] <= x < t[i + 1], 1:(n - 1))
    if hit === nothing
        # x on the right boundary: last interval with positive length
        hit = findlast(i -> t[i] < t[i + 1] && t[i] < x <= t[i + 1], 1:(n - 1))
    end
    hit === nothing || (B[hit] = 1.0)
    for m in 2:k
        Bm = zeros(n - m)
        for i in 1:(n - m)
            a = t[i + m - 1] - t[i]
            b = t[i + m] - t[i + 1]
            Bm[i] = (a > 0 ? (x - t[i]) / a * B[i] : 0.0) + (b > 0 ? (t[i + m] - x) / b * B[i + 1] : 0.0)
        end
        B = Bm
    end
    return B
end

# one row of R's bs / ns basis before the ns rotation (first column dropped)
function _spline_row(sb::SplineBasis, x)
    t, k = sb.knots, sb.ord
    if sb.lo <= x <= sb.hi
        return _bsplines(t, x, k)[2:end]
    end
    piv = x < sb.lo ? sb.lo : sb.hi
    if sb.natural
        # linear extrapolation from the boundary
        return (_bsplines(t, piv, k) .+ (x - piv) .* _bsplines(t, piv, k, 1))[2:end]
    end
    row = zeros(length(t) - k)
    for j in 0:(k - 1)
        row .+= _bsplines(t, piv, k, j) .* ((x - piv)^j / factorial(j))
    end
    return row[2:end]
end

function SplineBasis(x::AbstractVector{<:Real}, natural::Bool, df::Int, degree::Int)
    xf = Float64.(x)
    lo, hi = extrema(xf)
    ord = natural ? 4 : degree + 1
    nik = natural ? df - 1 : df - degree
    nik >= 0 || throw(ArgumentError("$(natural ? "ns" : "bs")(): df = $df is too small" *
                                    (natural ? "" : " for degree $degree")))
    probs = range(0, 1; length=nik + 2)[2:(end - 1)]
    inner = isempty(probs) ? Float64[] : quantile(xf, collect(probs))
    knots = sort(vcat(fill(lo, ord), inner, fill(hi, ord)))
    Q = zeros(0, 0)
    if natural
        C = reduce(vcat, [_bsplines(knots, b, ord, 2)[2:end]' for b in (lo, hi)])   # 2 x K
        Q = Matrix(qr(Matrix(C')).Q * Matrix(I, size(C, 2), size(C, 2)))
    end
    return SplineBasis(natural, knots, ord, lo, hi, Q)
end

function _spline_block(sb::SplineBasis, x)
    B = reduce(vcat, [_spline_row(sb, Float64(v))' for v in x])
    return sb.natural ? B * sb.Q[:, 3:end] : B
end

_basis_block(sb::SplineBasis, x) = _spline_block(sb, x)
_basis_ncol(sb::SplineBasis) = sb.natural ? size(sb.Q, 2) - 2 : length(sb.knots) - sb.ord - 1

# bs(x, df = 5) / bs(x, 5, degree = 2) / ns(x, df = 3) / ns(x, 3)
function _spline_args(ex)
    (ex isa Expr && ex.head == :call && ex.args[1] in (:bs, :ns)) || return nothing
    natural = ex.args[1] === :ns
    df = nothing; degree = 3
    pos = [a for a in ex.args[3:end] if !(a isa Expr && a.head == :kw)]
    length(pos) <= 1 || throw(ArgumentError("$(ex.args[1])(): give df positionally or as `df = `; knots are not supported yet"))
    isempty(pos) || (df = pos[1])
    for a in ex.args[3:end]
        a isa Expr && a.head == :kw || continue
        if a.args[1] === :df
            df = a.args[2]
        elseif a.args[1] === :degree && !natural
            degree = a.args[2]
        else
            throw(ArgumentError("$(ex.args[1])(): unsupported argument `$(a.args[1])` (supported: df$(natural ? "" : ", degree"))"))
        end
    end
    df isa Integer || throw(ArgumentError("$(ex.args[1])() needs a literal integer df, e.g. $(ex.args[1])(x, df = 4)"))
    degree isa Integer || throw(ArgumentError("bs(): degree must be a literal integer"))
    return ex.args[2], natural, Int(df), Int(degree)
end
