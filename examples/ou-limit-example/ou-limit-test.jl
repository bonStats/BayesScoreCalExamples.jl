# Univariate OU process
# Approx model: Limiting/stationary distribution of data
# Approx samples: MCMC via Turing.jl 
# True model samples: MCMC via Turing.jl  

using Turing
using Distributions
using Optim
import DataFrames: transform as dtransform, transform! as dtransform!, DataFrame, nrow
# import LinearAlgebra: PosDefException
using BayesScoreCal
using BayesScoreCalExamples

vexp(x) = exp.(x)
vlog(x) = log.(x)
vms(x::Vector{<:Real}, y::Real) = x .- y

function rescale(x, s::Real)
    mx = mean(x)
    (s .* (x .- [mx])) .+ [mx]
end

function rescaler(s::Real)
    function(x) 
        mx = mean(x)
        (s .* (x .- [mx])) .+ [mx]
    end
end

## setup data
T = 1.0
N = 100

# generate settings data
x₀= 10.0
μ = 1.0
γ = 2.0
σ = sqrt(20)   
D = σ^2 / 2

truevals = Dict([:μ => μ, :D => D, :logD => log(D)])

# approximate/true model settings
N_samples = 2_000
n_adapt = 1000

# optimisation settings
N_importance = 100
N_energy = 1000
energyβ = 1.0
vmultiplier = 2.0
alphalevels = [0.0, 0.25, 0.5, 0.9, 1.0]

reps = 100

# params helper
params = [:μ, :logD] # use these (possibly wrong order)
paramstring = Dict([:μ => "mu", :D => "D", :logD => "logD"])

ouprocess(T::Float64, x₀::Float64, μ::Float64, γ::Float64, σ::Float64) = 
Normal(μ + (x₀ - μ) * exp(-γ * T), sqrt((σ^2) * (1 - exp(-2*γ*T)) / (2 * γ)))

ou = ouprocess(T, x₀, μ, γ, σ)

## setup model
@model function approxmodel(x)
    D ~ Exponential(10) # scale = 10
    μ ~ Normal(0.0, 10.0)
    for i in eachindex(x)
        x[i] ~ Normal(μ, sqrt(D / γ))
    end
end

@model function truemodel(x)
    D ~ Exponential(10) # scale = 10
    μ ~ Normal(0.0, 10.0)
    for i in eachindex(x)
        x[i] ~ Normal(μ + (x₀ - μ) * exp(-γ * T), sqrt(D * (1 - exp(-2*γ*T)) / γ))
    end
end

# storage

function testfun(ou::Normal, N::Int64, N_samples::Int64, vmultiplier::Float64, alphalevels::Vector{Float64}, iter::Int64,  options::Optim.Options = Optim.Options())

    dfsamples  = DataFrame[]

    # X_T generate true data
    x = rand(ou, N)

    # Instantiate models
    amod = approxmodel(x)
    tmod = truemodel(x)

    # "true" posterior
    true_samples = sample(tmod, NUTS(n_adapt, 0.65), N_samples)

    # "approx" posterior 
    approx_samples = sample(amod, NUTS(n_adapt, 0.65), N_samples)

    
    # dataframe and add logD
    df_approx_samples = dtransform(DataFrame(approx_samples), :D => vlog => :logD)
    df_true_samples = dtransform(DataFrame(true_samples), :D => vlog => :logD)
    dtransform!(df_approx_samples, :D => vlog => :logD)
    dtransform!(df_true_samples, :D => vlog => :logD)

    caldist = df_approx_samples[sample(1:nrow(df_approx_samples), N_importance), :]
    dtransform!(caldist, :μ => rescaler(vmultiplier), :μ)
    dtransform!(caldist, :logD => rescaler(vmultiplier), :logD)
    dtransform!(caldist, :logD => vexp => :D, :logD)

    calpoints = [collect(values(rw[[:logD,:μ]])) for rw in eachrow(caldist)]

    # log prior evaluated on importance distribution samples
    ℓprior = [logprior(amod, v) for v in eachrow(caldist)] 

    # log density evaluated on caldist samples (inverse transform and transform cancel)
    ℓimport = caldist[!,:lp]

    # IS weights, jacobian of log transform in numerator and denominator cancel
    is_weights = ℓprior .- ℓimport

    # new data generation
    approx_informed_ou = ouprocess.([T], [x₀], caldist.μ, [γ], sqrt.(2*caldist.D))
    newx = rand.(approx_informed_ou, [N])

    # new approx models: pre-allocate
    tr_appmod_samples = Matrix{Vector{Float64}}(undef, N_energy, N_importance)

    for t in eachindex(newx) #

        amod_newx = approxmodel(newx[t])
        newx_samples = dtransform(DataFrame(sample(amod_newx, NUTS(n_adapt, 0.65), N_samples)), :D => vlog => :logD)
        dtransform!(newx_samples, :D => vlog => :logD)
        downsampleid = sample(1:nrow(newx_samples), N_energy)
        
        tr_appmod_samples[:,t] = [collect(values(newx_samples[rw, params])) for rw in downsampleid]

    end

    # true and approx
    for pr in params 
        samplecomp = DataFrame(
            samples = df_true_samples[!,pr],
            method = "True-post",
            iter = iter,
            param = paramstring[pr],
            alpha = -1.0,
            trueval = truevals[pr]
        )

        push!(dfsamples, samplecomp)

        samplecomp = DataFrame(
            samples = df_approx_samples[!,pr],
            method = "Approx-post",
            iter = iter,
            param = paramstring[pr],
            alpha = -1.0,
            trueval = truevals[pr]
        )
        push!(dfsamples, samplecomp)
    end

    # approx cal
    for alpha in alphalevels

        trimval = quantile(is_weights, 1 - alpha)
        w = [is_weights[i] > trimval ? trimval : is_weights[i] for i in eachindex(is_weights)]
        w = exp.(w .- maximum(w))

        for pr in params
            prid = findfirst(pr .== params)
            # μ, logD order == params
            cpoints = map(x -> getindex(x,prid), calpoints)
            csamples = map(x -> getindex(x,prid), tr_appmod_samples)

            cal = Calibration(cpoints, csamples)
            tf = UnivariateAffine()
            # updates tf
            res = energyscorecalibrate!(tf, cal, w; β = energyβ, options = options)

            # transform original approx samples
            # and save
            samplecomp = DataFrame(
                samples = tf.(map(x-> x, df_approx_samples[!,pr]), [mean(df_approx_samples[!,pr])]),
                method = "Adjust-post",
                iter = iter,
                param = paramstring[pr],
                alpha = alpha,
                trueval = truevals[pr]
            )

            push!(dfsamples, samplecomp)

        end

    end

    return dfsamples
    
end



options = Optim.Options(f_tol = 0.00001)
rr = testfun(ou, N, N_samples, vmultiplier, alphalevels, 1, options)


allres = testfun.([ou], [N], [N_samples], [vmultiplier], [alphalevels], 1:reps, [options])

#results
allres = vcat(reduce(vcat, allres)...)
