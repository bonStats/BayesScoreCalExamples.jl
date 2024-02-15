# SDE parameter estimation with Kalman likelihood in Turing.jl
# Likelihood is approximated by extended Kalman Filter

using BayesScoreCal
using Distributions
using LinearAlgebra
using Turing
using SparseArrays
using DifferentialEquations
using StatsPlots
using Distributed
using SharedArrays
using JLD2
using DataFrames
using CSV

include("direct-ekf.jl")
include("turing-helpers.jl")

# approximate/true model settings
N_samples = 1000

# optimisation settings
N_importance = 200
N_energy = 1000
energyβ = 1.0
vmultiplier = 2.0
dt = 0.2 # time step observations
dtk = 0.05 # time step kalman

# dt = 0.1 # time step observations
# dtk = 0.02 # time step kalman

# diagnostic settings
checkprobs = range(0.1,0.95,step=0.05)

# calibration helpers
function multiplyscale(x::Vector{Vector{Float64}}, scale::Float64) 
    μ = mean(x)
    scale .* (x .- [μ]) .+ [μ]
end

# setup Lotka Volterra
u0 = [1.0, 1.0]
tspan = (0.0, 6.0)
true_p = [1.5, 1.0, 3.0, 1.0, 0.1, 0.1]

function multiplicative_noise!(du, u, p, t)
    x, y = u
    du[1] = p[5] * x
    return du[2] = p[6] * y
end

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, γ, δ = p # gets p[1:4]
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

otimes = dt:dt:tspan[2]
obsv = [Observation(data(t),t) for t in otimes]

lvem = KalmanEM(lvdrift, lvnoise, lvjac)

H = Diagonal(ones(2)) # map latent to obs
Σ₀ = Diagonal(0.01 * ones(2))
x = GaussianFilter(u0, Σ₀)
kfsde = KalmanApproxSDE(lvem, obsv, 0.0, dtk, H, x)

@model function lv_kalman(kfsde::KalmanApproxSDE)

    pars = zeros(4) # 'pars' must be first named in turing model
    ϕ = zeros(2)
    # Prior distributions.
    pars[1] ~ Uniform(0.1,4) # α
    pars[2] ~ Uniform(0.1,4) # β
    pars[3] ~ Uniform(0.1,4) # γ
    pars[4] ~ Uniform(0.1,4) # δ
    ϕ[1] ~ Uniform(0.01, 0.25) # 
    ϕ[2] ~ Uniform(0.01, 0.25) # ϕ[2]

    σ ~ Truncated(Normal(0, 0.05), 0.0, Inf)

    sol = kfsde([pars; ϕ], Diagonal(I * σ^2, 2))
    
    if sol.info == 0
        Turing.@addlogprob! sol.logZ
    else
        # Early exit if simulation could not be computed successfully.
        Turing.@addlogprob! -Inf
        return nothing
    end

    return nothing
end

# try InverseGamma(10, 1) again, is this sensitive to initial conditions?
# try uniform (0.01,0.1)?

approxmodel = lv_kalman(kfsde)

ch_approx = sample(approxmodel, NUTS(), N_samples; progress=true, drop_warmup=true)


## Calibration

# get vector of samples
approx_samples = getsamples(ch_approx, :pars)

# transform
npars = length(approx_samples[1])
true_pars = true_p[1:npars]
bij_all = bijector(approxmodel)
bij = Stacked(bij_all.bs[1:npars]...)

# select and transform approx samples

select_id = sample(1:length(ch_approx), N_importance) # manual because we need them to line up
select_approx_samples = approx_samples[select_id,1]
additional_approx_samples = getsamples(ch_approx, :ϕ)[select_id,1]

# on transformed scale
cal_points = inverse(bij).(multiplyscale(bij.(select_approx_samples), vmultiplier))

    # newx approx models: pre-allocate
    cal_samples_array = SharedArray{Float64}(length(cal_points[1]), N_energy, N_importance)

    Threads.@threads for t in eachindex(cal_points)
        # new data
        new_prob = remake(prob_sde, p = [cal_points[t]; additional_approx_samples[t]])
        newdata = solve(new_prob, SOSRI(), saveat=0.01)
        # (model|new data)
        mod_obsv = [Observation(newdata(t),t) for t in otimes]
        mod_kfsde = KalmanApproxSDE(lvem, mod_obsv, 0.0, dtk, H, x)
        mod_newdata = lv_kalman(mod_kfsde)
        # mcmc(model|new data)
        mod_approx_samples_newdata = sample(mod_newdata, NUTS(), N_samples; progress=false, drop_warmup=false)

        # samples from mcmc(model|new data)
        cal_samples_array[:,:,t] = hcat(getsamples(mod_approx_samples_newdata, :pars)...)
            
    end

    # transform data to  Matrix{Vector}
    cal_samples = [cal_samples_array[:,i,j] for i in 1:N_energy, j in 1:N_importance]
    
    # save calibration data

    cal = Calibration(bij.(cal_points), bij.(cal_samples))



# jldsave("src/lotka-sde-example/kalman-sde-cal.jld2"; cal, approx_samples, true_pars, bij)

cal = load("src/lotka-sde-example/kalman-sde-cal.jld2", "cal")
approx_samples = load("src/lotka-sde-example/kalman-sde-cal.jld2", "approx_samples")
true_pars = load("src/lotka-sde-example/kalman-sde-cal.jld2", "true_pars")
bij = load("src/lotka-sde-example/kalman-sde-cal.jld2", "bij")

# approximate weights
is_weights = ones(N_importance)

d = BayesScoreCal.dimension(cal)[1]
tf = CholeskyAffine(d)
M = inv(Diagonal(std(cal.μs)))
res = energyscorecalibrate!(tf, cal, is_weights, scaling = M, penalty = (0.0, 0.05))

tf.L
tf.b

# no adjustment
calcheck_approx = coverage(cal, checkprobs)
plot(checkprobs, hcat(calcheck_approx...)',  label = ["α"  "β"  "γ"  "δ"], title = "Approx. posterior calibration coverage")
plot!(checkprobs, checkprobs, label = "target", colour = "black")

# adjustment
calcheck_adjust = coverage(cal, tf, checkprobs)
plot(checkprobs, hcat(calcheck_adjust...)',  label = ["α"  "β"  "γ"  "δ"], title = "Adjusted posterior calibration coverage")
plot!(checkprobs, checkprobs, label = "target", colour = "black")

rmse(cal)
rmse(cal,tf)

# get adjusted samples
tr_approx_samples = bij.(approx_samples)
tf_samples = inverse(bij).(tf.(tr_approx_samples, [mean(tr_approx_samples)]))

[mean(tf_samples) std(tf_samples)]

# store results
parnames = [Symbol("beta$i") for i in 1:4]

samples = DataFrame[]
push!(samples, 
    DataFrame(
        [parnames[i] => getindex.(approx_samples, i)[:,1] for i in 1:4]...,
        :method => "Approx-post",
        :alpha => -1.0
    )
)

push!(samples, 
    DataFrame(
        [parnames[i] => getindex.(tf_samples, i)[:,1] for i in 1:4]...,
        :method => "Adjust-post",
        :alpha => 1.0
    )
)

push!(samples, 
    DataFrame(
        [parnames[i] => true_p[i] for i in 1:4]...,
        :method => "True-vals",
        :alpha => -1.0
    )
)

check = DataFrame[]

push!(check,
    DataFrame(
        [parnames[i] => getindex.(calcheck_approx, i) for i in 1:4]...,
        :prob => checkprobs,
        :method => "Approx-post",
        :alpha => -1.0
    )
)

push!(check,
    DataFrame(
        [parnames[i] => getindex.(calcheck_adjust, i) for i in 1:4]...,
        :prob => checkprobs,
        :method => "Adjust-post",
        :alpha => 1.0
    )
)


#CSV.write("src/lotka-sde-example/kalman-sde-samples.csv", samples)
#CSV.write("src/lotka-sde-example/kalman-sde-covcheck.csv", check)