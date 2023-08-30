# SDE parameter estimation with Kalman likelihood in Turing.jl
# Likelihood is approximated by extended Kalman Filter

using BayesScoreCal
using Distributions
using LinearAlgebra
using Turing
using SparseArrays
using DifferentialEquations
using Plots

include("direct-ekf.jl")

# approximate/true model settings
N_samples = 1000

# optimisation settings
N_importance = 200
N_energy = 1000
energyβ = 1.0
vmultiplier = 2.0

checkprobs = range(0.1,0.95,step=0.05)


# Turing helpers
function getsamples(chains::Chains, sym::Symbol, sample_chain::Union{Iterators.ProductIterator{Tuple{UnitRange{Int64}, UnitRange{Int64}}},Vector{Tuple{Int64, Int64}}})
    smpls = [vec(chains[sc[1], namesingroup(chains, sym), sc[2]].value) for sc in sample_chain]
    convert.(Vector{Float64}, smpls)
end
function getsamples(chains::Chains, sym::Symbol, N::Int64)
    # down sample
    sample_chain = sample([Iterators.product(1:length(chains), 1:size(chains, 3))...], N, replace = false)
    getsamples(chains, sym, sample_chain)
end
function getsamples(chains::Chains, sym::Symbol)
    # full sample
    sample_chain = Iterators.product(1:length(chains), 1:size(chains, 3))
    getsamples(chains, sym, sample_chain)
end
getparams(m::DynamicPPL.Model) = DynamicPPL.syms(DynamicPPL.VarInfo(m))

# setup Lotka Volterra
u0 = [1.0, 1.0]
tspan = (0.0, 10.0)
function multiplicative_noise!(du, u, p, t)
    x, y = u
    du[1] = p[5] * x
    return du[2] = p[6] * y
end

true_p = [1.5, 1.0, 3.0, 1.0, 0.1, 0.1]

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, γ, δ = p
    du[1] = dx = α * x - β * x * y
    return du[2] = dy = δ * x * y - γ * y
end

prob_sde = SDEProblem(lotka_volterra!, multiplicative_noise!, u0, tspan, true_p)

data = solve(prob_sde, SOSRI(), saveat=0.01)
plot(data)

function lvdrift(u,p)
    du = zeros(eltype(p), 2)
    lotka_volterra!(du, u, p, 0.0)
    return du
end
function lvnoise(u,p)
    du = zeros(eltype(p), 2)
    multiplicative_noise!(du, u, p, 0.0)
    return du
end

function lvjac(u,p)
    x, y = u
    α, β, γ, δ = p
    j11 = α - β * y
    j12 = - β * x
    j21 = δ * y
    j22 = - γ
    return [j11 j12; j21 j22]
end

otimes = 0.1:0.1:tspan[2]
obsv = [Observation(data(t),t) for t in otimes]

lvem = KalmanEM(lvdrift, lvnoise, lvjac)

x = GaussianFilter(u0, Diagonal(0.01 * ones(2)))
kfsde = KalmanApproxSDE(lvem, obsv, 0.0, 0.05, Diagonal(ones(2)), x)

@model function lv_kalman(kfsde::KalmanApproxSDE)

    pars = zeros(6)
    # Prior distributions.
    pars[1] ~ Uniform(0.1,5) # α
    pars[2] ~ Uniform(0.1,5) # β
    pars[3] ~ Uniform(0.1,5) # γ
    pars[4] ~ Uniform(0.1,5) # δ
    pars[5] = 0.1 #~ Uniform(0.01, 0.5) # ϕ[1]
    pars[6] = 0.1 #~ Uniform(0.01, 0.5) # ϕ[2]

    σ ~ InverseGamma(10, 1)

    sol = kfsde(pars, Diagonal(I * σ^2, 2))
    
    if sol.info == 0
        Turing.@addlogprob! sol.logZ
    else
        # Early exit if simulation could not be computed successfully.
        Turing.@addlogprob! -Inf
        return nothing
    end

    return nothing
end

approxmodel = lv_kalman(kfsde)


ch_approx = sample(approxmodel, NUTS(), N_samples; progress=true, drop_warmup=true)
