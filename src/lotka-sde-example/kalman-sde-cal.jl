# SDE parameter estimation with Kalman likelihood in Turing.jl
# Likelihood is approximated by extended Kalman Filter

using BayesScoreCal
using Distributions
using LinearAlgebra
using Turing
using SparseArrays
using DifferentialEquations
using Plots
using ProgressMeter
using Distributed
using SharedArrays

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


## Calibration

function multiplyscale(x::Vector{Vector{Float64}}, scale::Float64) 
    μ = mean(x)
    scale .* (x .- [μ]) .+ [μ]
end

# transform
bij_all = bijector(approxmodel)
bij = Stacked(bij_all.bs[1:4]...)

# select and transform approx samples
select_approx_samples = getsamples(ch_approx, :pars, N_importance)

cal_points = inverse(bij).(multiplyscale(bij.(select_approx_samples), vmultiplier))

    # newx approx models: pre-allocate
    tr_approx_samples_newx = SharedArray{Float64}(length(cal_points[1]), N_energy, N_importance)

    pmeter = Progress(length(cal_points))

    Threads.@threads for t in eachindex(cal_points)
        # new data
        new_prob = remake(prob_sde, p = [cal_points[t]..., 0.1, 0.1])
        newdata = solve(new_prob, SOSRI(), saveat=0.1)
        # (model|new data)
        mod_obsv = [Observation(newdata(t),t) for t in otimes]
        mod_kfsde = KalmanApproxSDE(lvem, mod_obsv, 0.0, 0.05, Diagonal(ones(2)), x)
        mod_newdata = lv_kalman(mod_kfsde)
        # mcmc(model|new data)
        approx_samples_newdata = sample(mod_newdata, NUTS(), N_samples; progress=false, drop_warmup=false)

        # samples from vi(model|new data)
        tr_approx_samples_newx[:,:,t] = hcat(bij.(getsamples(approx_samples_newdata, :pars))...)
            
        ProgressMeter.next!(pmeter)
    end
    ProgressMeter.finish!(pmeter)