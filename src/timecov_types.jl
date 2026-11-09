# Types of timecov.jl, defined before density.jl (LinPred holds LatentAC terms).

struct TimeCov{F<:Family,R<:Real} <: Family
    fam::F
    kind::Symbol               # :arma_cov, :cosy, :fcor, :lagsar, :errorsar
    p::Int                     # arma_cov: ar present (0/1)
    q::Int                     # arma_cov: ma present (0/1)
    perm::Vector{Int}          # observations in (gr, time) order (1:N for fcor / sar)
    beg::Vector{Int}           # group k is perm[beg[k]:(beg[k+1] - 1)]
    nmax::Int                  # the largest group
    M::Matrix{R}               # fcor: lower Cholesky factor of M; sar: W
    eig::Vector{R}             # sar: eigenvalues of W
    Wy::Vector{R}              # lagsar: W * y
    # (R: Float64 as built; the density's own copy is in its sampling precision,
    # `_prec_family`, so a Float32 fit reads them per observation as Float32)
    ldM::Float64               # fcor: log det M (a data constant)
    tlev::Vector{Int}          # unstr: time level of each perm position (else empty)
    tlabels::Vector{String}    # unstr: the time levels (names cortime__t1__t2)
end
TimeCov(fam, kind, p, q, perm, beg, nmax, M, eig, Wy, ldM) =
    TimeCov(fam, kind, p, q, perm, beg, nmax, M, eig, Wy, ldM, Int[], String[])

# the family as the density holds it: per-observation data in the sampling precision
_prec_family(::Type{T}, f) where {T} = f
_prec_family(::Type{T}, f::TimeCov) where {T} =
    TimeCov(f.fam, f.kind, f.p, f.q, f.perm, f.beg, f.nmax, Matrix{T}(f.M), Vector{T}(f.eig), Vector{T}(f.Wy),
            f.ldM, f.tlev, f.tlabels)

struct LatentAC{T}
    tc::TimeCov{Gaussian{IdentityLink},Float64}   # ordering, groups and the R recipe
    sd::Int                               # log sderr
    rho::UnitRange{Int}                   # ar / ma / cosy (bounded, logit scale)
    lb::Vector{Float64}
    ub::Vector{Float64}
    z::UnitRange{Int}                     # N, in (gr, time) order
    sd_prior::ScalarPrior
    rho_priors::Vector{ScalarPrior}
    names::Vector{String}                 # "ar[1]", "ma[1]", "cosy"
end
