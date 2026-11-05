# =============================================================================
# Unordered categorical responses: brms `categorical()` (multinomial logit).
#
# Category 1 is the reference; categories k = 2..K each have their own linear
# predictor `mu<k>` (brms's names: b_mu2_Intercept, b_mu2_x, sd_g__mu2_...),
# built from the main formula unless `mu<k> ~ ...` is given:
#
#     log P(y = k) = eta_k - log(1 + sum_j exp(eta_j)),   eta_1 = 0
#     d/d eta_k    = [y == k] - softmax_k
#
# The number of categories is a type parameter, set from the data at build
# time (`categorical()` is a placeholder with K = 0 until then).
# =============================================================================

dpars(::Categorical{L,K}) where {L,K} = K >= 2 ? ntuple(k -> Symbol("mu", k + 1), K - 1) : ()
dpar_link_name(::Categorical, d::Symbol) = "logit"
_with_ncat(::Categorical{L}, K) where {L} = Categorical{L,K}()

# log(1 + sum exp(eta)) and the softmax probabilities of categories 2..K
@inline function _cat_lse(eta::NTuple{M,T}) where {M,T}
    m = max(zero(T), maximum(eta))
    s = exp(-m) + sum(e -> exp(e - m), eta)
    return m + log(s), map(e -> exp(e - m) / s, eta)
end

@inline function _ll(::Categorical, y, x, eta)
    lse, _ = _cat_lse(eta)
    k = Int(y)
    return (k == 1 ? zero(lse) : eta[k - 1]) - lse
end
@inline function _ll_grad(::Categorical, y, x, eta)
    lse, p = _cat_lse(eta)
    k = Int(y)
    v = (k == 1 ? zero(lse) : eta[k - 1]) - lse
    return v, ntuple(j -> (k == j + 1 ? one(lse) : zero(lse)) - p[j], length(eta))
end
# two categories: one predictor, which takes the density's scalar fast path
@inline _ll_grad_pre(c::Categorical, y, x, m, p) = _ll_grad(c, y, x, (m,))
_ll_const(::Categorical, y, t) = 0.0
_check_response(::Categorical{L,K}, y, t) where {L,K} =
    _chk(all(v -> isinteger(v) && 1 <= v <= K, y), "categorical responses must be categories 1..$K")

function _cat_probs(::Categorical, eta)
    _, p = _cat_lse(Tuple(eta))
    return [1 - sum(p); collect(p)]
end
_mean(fam::Categorical, eta, t) = sum(k * p for (k, p) in enumerate(_cat_probs(fam, eta)))
function _rand(rng, fam::Categorical, eta, t)
    p = _cat_probs(fam, eta)
    u = rand(rng); c = 0.0
    for k in eachindex(p)
        c += p[k]
        u <= c && return Float64(k)
    end
    return Float64(length(p))
end
