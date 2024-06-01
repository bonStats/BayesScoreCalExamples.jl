using BayesScoreCal
using BayesScoreCalExamples
using Distributions
using ModelingToolkit
using Catalyst
using LinearAlgebra
using DifferentialEquations
using Turing
using SparseArrays
using Distributed
using SharedArrays
using JLD2
using StatsPlots
using DataFrames
using CSV

getparams(m::DynamicPPL.Model) = DynamicPPL.syms(DynamicPPL.VarInfo(m))
getstatesymbol(x::SymbolicUtils.BasicSymbolic) = x.metadata[Symbolics.VariableSource][2]

# diagnostic settings
checkprobs = range(0.1,0.95,step=0.05)

# calibration helpers
function multiplyscale(x::Vector{Vector{Float64}}, scale::Float64) 
    μ = mean(x)
    scale .* (x .- [μ]) .+ [μ]
end

# approximate/true model settings
N_samples = 1000

# optimisation settings
N_importance = 200
N_energy = 1000
energyβ = 1.0
vmultiplier = 1.5
dtk = 1.0 # time step kalman

mapk_2step = @reaction_network begin
    k1, X + E --> XE
    k2, XE --> X + E
    k3, XE --> Xᵃ + E
    k4, Xᵃ + P₁ --> XᵃP₁
    k5, XᵃP₁ --> Xᵃ + P₁
    k6, XᵃP₁ --> X + P₁
    k7, Xᵃ + Y --> XᵃY
    k8, XᵃY --> Xᵃ + Y
    k9, XᵃY --> Xᵃ + Yᵃ
    k10, Yᵃ + P₂ --> YᵃP₂
    k11, YᵃP₂ --> Yᵃ + P₂
    k12, YᵃP₂ --> Y + P₂
end

par = Dict(
    :k1 => 0.001, 
    :k3 => 0.18, 
    :k4 => 0.001, 
    :k6 => 0.3,
    :k7 => 0.0001,
    :k9 => 0.2,
    :k10 => 0.001,
    :k12 => 0.3,
)
par = merge(par,
    Dict(
        :k2 => par[:k1]/120, 
        :k5 => par[:k4]/22,
        :k8 => par[:k7]/110, 
        :k11 => par[:k10]/22
    )
)

# activated are observed only
observed = Symbol.(["Xᵃ(t)", "Yᵃ(t)"])
observedid = [Symbol(x) ∈ observed for x in states(mapk_2step)]
nobs = length(observed)
nstates = length(states(mapk_2step))
npars = length(parameters(mapk_2step))

# identifiable parameters
idpar = [:k3, :k6, :k9, :k12]

σ = 1.
R = Diagonal(repeat([σ^2], nobs))
H = Matrix(hcat([I[1:nstates, j] for j in findall(observedid)]...)')

u0 = Dict(
    :E => 94,
    :X => 757,
    :Y => 567,
    :P₁ => 32,
    :P₂ => 32,
    :Xᵃ => 0,
    :Yᵃ => 0,
    :XE => 0,
    :XᵃP₁ => 0,
    :XᵃY => 0,
    :YᵃP₂ => 0
)


ord_u0 = Float64.([u0[getstatesymbol(x)] for x in states(mapk_2step)])

times = 0:4:200
otimes = times[2:end]
tspan = (Float64(times[1]),Float64(times[end]))

ord_parameters = [Symbol(x) for x in parameters(mapk_2step)]
p = get.([par], ord_parameters, -Inf)

par_id = Dict(ord_parameters .=> 1:length(ord_parameters))

dprob = DiscreteProblem(mapk_2step, u0, tspan, p)
jprob = JumpProblem(mapk_2step, dprob, Direct())
data = solve(jprob, SSAStepper())

obsv = [Observation(data(t)[observedid], t) for t in otimes]
gstates = GaussianFilter(data(0.0)[observedid], R)

mapk_sde_sys = convert(SDESystem, mapk_2step; combinatoric_ratelaws=true)
mapk_sde = SDEFunction(mapk_sde_sys, jac = true)

mapk_sde_sys.eqs
mapk_sde_sys.noiseeqs

modelsymdrift = getfield.(mapk_sde_sys.eqs, :rhs)
modelsymjac = Symbolics.jacobian(modelsymdrift, states(mapk_2step))
modelsymnoise = mapk_sde_sys.noiseeqs
ss = simplify(modelsymnoise * modelsymnoise')

ss[1,1]

function mapk2drift(u::AbstractVector, p::AbstractVector)
    [p[2]*u[3] + p[6]*u[6] - p[1]*u[2]*u[1],  # X
    p[2]*u[3] + p[3]*u[3] - p[1]*u[2]*u[1],  # E
    p[1]*u[2]*u[1] - p[2]*u[3] - p[3]*u[3],  # XE
    p[3]*u[3] + p[5]*u[6] + p[8]*u[8] + p[9]*u[8] - p[4]*u[5]*u[4] - p[7]*u[4]*u[7], # Xᵃ
    p[5]*u[6] + p[6]*u[6] - p[4]*u[5]*u[4],  # P₁
    p[4]*u[5]*u[4] - p[5]*u[6] - p[6]*u[6],  # XᵃP₁
    p[12]*u[11] + p[8]*u[8] - p[7]*u[4]*u[7], # Y
    p[7]*u[4]*u[7] - p[8]*u[8] - p[9]*u[8],  # XᵃY
    p[9]*u[8] + p[11]*u[11] - p[10]*u[10]*u[9],    # Yᵃ
    p[11]*u[11] + p[12]*u[11] - p[10]*u[10]*u[9], # P₂
    p[10]*u[10]*u[9] - p[11]*u[11] - p[12]*u[11]] # YᵃP₂
end

function mapk2noise(u::AbstractVector, p::AbstractVector)
    [-sqrt(abs(p[1]*u[2]*u[1])) sqrt(abs(p[2]*u[3])) 0. 0. 0. sqrt(abs(p[6]*u[6])) 0. 0. 0. 0. 0. 0.; 
    -sqrt(abs(p[1]*u[2]*u[1])) sqrt(abs(p[2]*u[3])) sqrt(abs(p[3]*u[3])) 0. 0. 0. 0. 0. 0. 0. 0. 0.;
    sqrt(abs(p[1]*u[2]*u[1])) -sqrt(abs(p[2]*u[3])) -sqrt(abs(p[3]*u[3])) 0. 0. 0. 0. 0. 0. 0. 0. 0.; 
    0. 0. sqrt(abs(p[3]*u[3])) -sqrt(abs(p[4]*u[5]*u[4])) sqrt(abs(p[5]*u[6])) 0. -sqrt(abs(p[7]*u[4]*u[7])) sqrt(abs(p[8]*u[8])) sqrt(abs(p[9]*u[8])) 0. 0. 0.;
    0. 0. 0. -sqrt(abs(p[4]*u[5]*u[4])) sqrt(abs(p[5]*u[6])) sqrt(abs(p[6]*u[6])) 0. 0. 0. 0. 0. 0.; 
    0. 0. 0. sqrt(abs(p[4]*u[5]*u[4])) -sqrt(abs(p[5]*u[6])) -sqrt(abs(p[6]*u[6])) 0. 0. 0. 0. 0. 0.;
    0. 0. 0. 0. 0. 0. -sqrt(abs(p[7]*u[4]*u[7])) sqrt(abs(p[8]*u[8])) 0. 0. 0. sqrt(abs(p[12]*u[11])); 
    0. 0. 0. 0. 0. 0. sqrt(abs(p[7]*u[4]*u[7])) -sqrt(abs(p[8]*u[8])) -sqrt(abs(p[9]*u[8])) 0. 0. 0.;
    0. 0. 0. 0. 0. 0. 0. 0. sqrt(abs(p[9]*u[8])) -sqrt(abs(p[10]*u[10]*u[9])) sqrt(abs(p[11]*u[11])) 0.; 
    0. 0. 0. 0. 0. 0. 0. 0. 0. -sqrt(abs(p[10]*u[10]*u[9])) sqrt(abs(p[11]*u[11])) sqrt(abs(p[12]*u[11]));
    0. 0. 0. 0. 0. 0. 0. 0. 0. sqrt(abs(p[10]*u[10]*u[9])) -sqrt(abs(p[11]*u[11])) -sqrt(abs(p[12]*u[11]))]
end

function mapk2noise_sparse(u::AbstractVector, p::AbstractVector)
    rw = [1, 2, 3, 1, 2, 3, 2, 3, 4, 4, 5, 6, 4, 5, 6, 1, 5, 6, 4, 7, 8, 4, 7, 8, 4, 8, 9, 9, 10, 11, 9, 10, 11, 7, 10, 11]
    cl = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12]

    z = [
        sqrt(abs(p[1]*u[2]*u[1])),
        sqrt(abs(p[2]*u[3])), 
        sqrt(abs(p[3]*u[3])), 
        sqrt(abs(p[4]*u[5]*u[4])), 
        sqrt(abs(p[5]*u[6])),
        sqrt(abs(p[6]*u[6])),
        sqrt(abs(p[7]*u[4]*u[7])),
        sqrt(abs(p[8]*u[8])),
        sqrt(abs(p[9]*u[8])),
        sqrt(abs(p[10]*u[10]*u[9])),
        sqrt(abs(p[11]*u[11])),
        sqrt(abs(p[12]*u[11]))
    ]

    el = [-z[1],-z[1],z[1],z[2],z[2],-z[2],z[3],-z[3],z[3],-z[4],-z[4],z[4],z[5],z[5],-z[5],z[6],z[6],-z[6],-z[7],-z[7],z[7],z[8],z[8],-z[8],z[9],-z[9],z[9],-z[10],-z[10],z[10],z[11],z[11],-z[11],z[12],z[12],-z[12]]

    return sparse(rw,cl,el)

end

# check
all(mapk2noise_sparse(ord_u0 .+0.01, p) .== mapk2noise(ord_u0 .+0.01, p))


function mapk2jacobian(u::AbstractVector, p::AbstractVector)
    [-p[1]*u[2] -p[1]*u[1] p[2] 0. 0. p[6] 0. 0. 0. 0. 0.; 
    -p[1]*u[2] -p[1]*u[1] p[2] + p[3] 0. 0. 0. 0. 0. 0. 0. 0.; 
    p[1]*u[2] p[1]*u[1] -p[2] - p[3] 0. 0. 0. 0. 0. 0. 0. 0.; 
    0. 0. p[3] -p[4]*u[5] - p[7]*u[7] -p[4]*u[4] p[5] -p[7]*u[4] p[8] + p[9] 0. 0. 0.; 
    0. 0. 0. -p[4]*u[5] -p[4]*u[4] p[5] + p[6] 0. 0. 0. 0. 0.; 
    0. 0. 0. p[4]*u[5] p[4]*u[4] -p[5] - p[6] 0. 0. 0. 0. 0.; 
    0. 0. 0. -p[7]*u[7] 0. 0. -p[7]*u[4] p[8] 0. 0. p[12]; 
    0. 0. 0. p[7]*u[7] 0. 0. p[7]*u[4] -p[8] - p[9] 0. 0. 0.; 
    0. 0. 0. 0. 0. 0. 0. p[9] -p[10]*u[10] -p[10]*u[9] p[11]; 
    0. 0. 0. 0. 0. 0. 0. 0. -p[10]*u[10] -p[10]*u[9] p[11] + p[12]; 
    0. 0. 0. 0. 0. 0. 0. 0. p[10]*u[10] p[10]*u[9] -p[11] - p[12]]
end

function mapk2jacobian_sparse(u::AbstractVector, p::AbstractVector)
    rw = [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4, 5, 6, 7, 8, 4, 5, 6, 1, 4, 5, 6, 4, 7, 8, 4, 7, 8, 9, 9,  10, 11, 9,  10, 11, 7,  9,  10, 11]
    cl = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 11]

    z = [p[1]*u[2], p[1]*u[1], p[4]*u[5], p[7]*u[7], p[4]*u[4], p[7]*u[4], p[10]*u[10], p[10]*u[9]]

    el = [-z[1], -z[1], z[1], -z[2], -z[2], z[2], p[2], p[2] + p[3], -p[2] - p[3], p[3], -z[3] - z[4], -z[3], z[3], -z[4], z[4], 
    -z[5], -z[5], z[5], p[6], p[5], p[5] + p[6], -p[5] - p[6], -z[6], -z[6], z[6], p[8] + p[9], p[8], -p[8] - p[9], p[9], -z[7], 
    -z[7], z[7], -z[8], -z[8], z[8], p[12], p[11], p[11] + p[12], -p[11] - p[12]]
    return sparse(rw,cl,el)
end

all(mapk2jacobian_sparse(ord_u0 .+0.01, p) .== mapk2jacobian(ord_u0 .+0.01, p))


mapk2kem = KalmanEM(mapk2drift, mapk2noise_sparse, mapk2jacobian_sparse)

#mapk2kem.drift(ord_u0, p)
#mapk2kem.noise(ord_u0, p)
#mapk2kem.jac(ord_u0, p)

x = GaussianFilter(ord_u0 .+ 0.001, Diagonal(0.5 * sqrt.(max.(ord_u0, 1.0))))
kfsde = KalmanApproxSDE(mapk2kem, obsv, 0.0, dtk, H, x)

kfsde(p, R).logZ
kfsde(p, R).info

param_fixed = Dict(
    :k1 => 0.001,
    :k4 => 0.001,
    :k7 => 0.0001,
    :k10 => 0.001
)

padfixed(p::Vector{<:Real}, k1::Real, k4::Real, k7::Real, k10::Real) = [k1; p[1:2]; k4; p[6]; p[3]; k7; p[7]; p[4]; k10; p[8]; p[5]]

@model kalman_model(kfsde::KalmanApproxSDE) = begin
    
    κ = zeros(4)

    k1 = par[:k1] # fixed
    k2 ~ Uniform(0.0, par[:k1])
    κ[1] ~ Uniform(0.0, 1.0)   # κ[1] = k3
    k4 = par[:k4] # fixed
    k5 ~ Uniform(0.0, par[:k4])
    κ[2] ~ Uniform(0.0, 1.0) # κ[2] = k6
    k7 = par[:k7] # fixed
    k8 ~ Uniform(0.0, par[:k7])
    κ[3] ~ Uniform(0.0, 1.0) # κ[3] = k9
    k10 = par[:k10] # fixed    
    k11 ~ Uniform(0.0, par[:k10])
    κ[4] ~ Uniform(0.0, 1.0) # κ[4] = k12      

    k = [k1, k2, κ[1], k4, k5, κ[2], k7, k8, κ[3], k10, k11, κ[4]]

    σ = 0.01
    sol = kfsde(k, Diagonal(I * σ^2, nobs))
    
    if sol.info == 0
        Turing.@addlogprob! sol.logZ
    else
        # Early exit if simulation could not be computed successfully.
        Turing.@addlogprob! -Inf
        return nothing
    end
    
    return nothing
end

approxmodel = kalman_model(kfsde)
ch_approx = sample(approxmodel, NUTS(), 1000) #  init_params = [par[:k3], par[:k6], par[:k9], par[:k12]]


## Calibration

# get vector of samples
approx_samples_all = getsamples.([ch_approx], getparams(approxmodel))
initial_par = vcat(mean.(approx_samples_all)...)

# get bijection used by turing
κ_sel = findfirst(:κ .== getparams(approxmodel))
npars = length(approx_samples_all[κ_sel][1])
true_pars = getindex.([par], idpar)
bij_all = bijector(approxmodel)
bij = Stacked(bij_all.bs[(0:npars-1) .+ κ_sel]...)

select_id = sample(1:length(ch_approx), N_importance) 
select_approx_samples_all = hcat(approx_samples_all...)[select_id,:]

cal_samples_all = select_approx_samples_all
cal_points = inverse(bij).(multiplyscale(bij.(select_approx_samples_all[:,κ_sel]), vmultiplier))
cal_samples_all[:,κ_sel] = cal_points

# newx approx models: pre-allocate
cal_samples_array = SharedArray{Float64}(length(cal_points[1]), N_energy, N_importance)

Threads.@threads for t in eachindex(cal_points)
    # new data
    new_p = padfixed(vcat(cal_samples_all[t,:]...), par[:k1], par[:k4], par[:k7], par[:k10])
    new_prob = remake(jprob, p = new_p)
    newdata = solve(new_prob, SSAStepper())
    # (model|new data)
    mod_obsv = [Observation(newdata(t)[observedid],t) for t in otimes]
    mod_kfsde = KalmanApproxSDE(mapk2kem, mod_obsv, 0.0, dtk, H, x)
    mod_newdata = kalman_model(mod_kfsde)
    # mcmc(model|new data)
    mod_approx_samples_newdata = sample(mod_newdata, NUTS(), N_samples; progress=false, drop_warmup=false, initial_params = initial_par)

    # samples from mcmc(model|new data)
    cal_samples_array[:,:,t] = hcat(getsamples(mod_approx_samples_newdata, :κ)...)
        
end

# transform data to  Matrix{Vector}
cal_samples = [cal_samples_array[:,i,j] for i in 1:N_energy, j in 1:N_importance]

# save calibration data
cal = Calibration(bij.(cal_points), bij.(cal_samples))

#jldsave("examples/reaction-network-ekf/rn-cal-20240526.jld2"; cal, ch_approx, par, bij, data)


cal = load("examples/reaction-network-ekf/rn-cal-20240526.jld2", "cal")
ch_approx = load("examples/reaction-network-ekf/rn-cal-20240526.jld2", "ch_approx")
par = load("examples/reaction-network-ekf/rn-cal-20240526.jld2", "par")
bij = load("examples/reaction-network-ekf/rn-cal-20240526.jld2", "bij")
data = load("examples/reaction-network-ekf/rn-cal-20240526.jld2", "data")


# approximate weights
is_weights = ones(N_importance)

d = BayesScoreCal.dimension(cal)[1]
tf = EigenAffine(d, 0.5)
M = inv(Diagonal(std(cal.μs)))
res = energyscorecalibrate!(tf, cal, is_weights, scaling = M, penalty = (10.0, 0.0, 0.0))

tf.V * Diagonal(tf.d .+ tf.dmin)
tf.b
tf.V * tf.V'

# no adjustment
calcheck_approx = coverage(cal, checkprobs)
plot(checkprobs, hcat(calcheck_approx...)',  label = ["κ₁"  "κ₂"  "κ₃"  "κ₄"], title = "Approx. posterior calibration coverage")
plot!(checkprobs, checkprobs, label = "target", colour = "black")
plot!(checkprobs, checkprobs .- 0.1, label = "", colour = "black", linestyle = :dash)
plot!(checkprobs, checkprobs .+ 0.1, label = "", colour = "black", linestyle = :dash)

# adjustment
calcheck_adjust = coverage(cal, tf, checkprobs)
plot(checkprobs, hcat(calcheck_adjust...)',  label = ["κ₁"  "κ₂"  "κ₃"  "κ₄"], title = "Adjusted posterior calibration coverage")
plot!(checkprobs, checkprobs, label = "target", colour = "black")
plot!(checkprobs, checkprobs .- 0.1, label = "", colour = "black", linestyle = :dash)
plot!(checkprobs, checkprobs .+ 0.1, label = "", colour = "black", linestyle = :dash)


rmse(cal)
rmse(cal,tf)

# get adjusted samples
approx_samples = vec(approx_samples_all[κ_sel])
tr_approx_samples = bij.(approx_samples)
tf_samples = inverse(bij).(tf.(tr_approx_samples, [mean(tr_approx_samples)]))

[mean(tf_samples) std(tf_samples) true_pars]
[mean(approx_samples) std(approx_samples) true_pars]

# store results
parnames = [:k3, :k6, :k9, :k12]

samples = 
    DataFrame(
        [parnames[i] => getindex.(approx_samples, i) for i in 1:4]...,
        :method => "Approx-post",
        :alpha => -1.0
    )

append!(samples, 
    DataFrame(
        [parnames[i] => getindex.(tf_samples, i) for i in 1:4]...,
        :method => "Adjust-post",
        :alpha => 1.0
    )
)

append!(samples, 
    DataFrame(
        [parnames[i] => true_pars[i] for i in 1:4]...,
        :method => "True-vals",
        :alpha => -1.0
    )
)

check = 
    DataFrame(
        [parnames[i] => getindex.(calcheck_approx, i) for i in 1:4]...,
        :prob => checkprobs,
        :method => "Approx-post",
        :alpha => -1.0
    )

append!(check,
    DataFrame(
        [parnames[i] => getindex.(calcheck_adjust, i) for i in 1:4]...,
        :prob => checkprobs,
        :method => "Adjust-post",
        :alpha => 1.0
    )
)

CSV.write("examples/reaction-network-ekf/kalman-rn-samples.csv", samples)
CSV.write("examples/reaction-network-ekf/kalman-rn-covcheck.csv", check)