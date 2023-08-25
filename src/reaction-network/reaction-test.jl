using BayesScoreCal
using Distributions
using ModelingToolkit
using Catalyst
using LinearAlgebra
using DifferentialEquations
using Turing

getparams(m::DynamicPPL.Model) = DynamicPPL.syms(DynamicPPL.VarInfo(m))
getstatesymbol(x::SymbolicUtils.BasicSymbolic) = x.metadata[Symbolics.VariableSource][2]

include("ekf.jl")

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


ord_u0 = [u0[getstatesymbol(x)] for x in states(mapk_2step)]

times = 0:4:200
otimes = times[2:end]
tspan = (Float64(times[1]),Float64(times[end]))

ord_parameters = [Symbol(x) for x in parameters(mapk_2step)]
p = get.([par], ord_parameters, -Inf)

par_id = Dict(ord_parameters .=> 1:length(ord_parameters))

dprob = DiscreteProblem(mapk_2step, u0, tspan, p)
jprob = JumpProblem(mapk_2step, dprob, Direct())
jsol = solve(jprob, SSAStepper())

obsv = [Observation(jsol(t)[observedid], t) for t in otimes]
gstates = GaussianFilter(jsol(0.0)[observedid], R)

mapk_sde_sys = convert(SDESystem, mapk_2step)
mapk_sde = SDEFunction(mapk_sde_sys, jac = true)



# u = rand(length(states(mapk_2step)))
# t = 0
# mapk_sde.f(u, p, t)
# L = mapk_sde.g(u, p, t)

# mapk_sde.jac(u, p, t)

# generate_jacobian(mapk_sde_sys)

# inv(L * L' + 0.01*I)



g = GaussianFilter(zeros(nstates), Diagonal(ones(nstates)))

predict!(g, mapk_sde, p, 0.01, nugget = 0.01)

ll = update!(g, rand(nstates), Diagonal(ones(nstates)), Diagonal(ones(nstates)))


g = GaussianFilter(ord_u0, Diagonal(0.5 * sqrt.(max.(ord_u0, 1.0))))

kalman!(g, 0.0, mapk_sde, H, R, p.+ 0.000001, obsv, 0.1, nugget = (0.0, 0.0))


x = GaussianFilter(ord_u0, Diagonal(0.5 * sqrt.(max.(ord_u0, 1.0))))
kfsde = KalmanApproxSDE(mapk_sde, obsv, 0.0, 0.1, H, x)

kfsde(p, R).logZ
kfsde(p, R).info

param_fixed = Dict(
    :k1 => 0.001,
    :k4 => 0.001,
    :k7 => 0.0001,
    :k10 => 0.0001
)


# collect and order parameters
reorder_id = [get(par_id, v, 0) for v in ord_parameters]

@model kalman_model(kfsde::KalmanApproxSDE) = begin
     
    k = tzeros(12)
    k[1] = param_fixed[:k1]
    k[2] = par[:k2] #~ Uniform(0.0, par[:k1]) 
    k[3] ~ Uniform(0.0, 1.0)    
    k[4] = param_fixed[:k4]  
    k[5] = par[:k5] #~ Uniform(0.0, par[:k4])  
    k[6] ~ Uniform(0.0, 1.0)
    k[7] = param_fixed[:k7]  
    k[8] = par[:k8] #~ Uniform(0.0, par[:k7])   
    k[9] ~ Uniform(0.0, 1.0)
    k[10] = param_fixed[:k10]     
    k[11] = par[:k11] #~ Uniform(0.0, par[:k10]) 
    k[12] ~ Uniform(0.0, 1.0)          

    σ = 1.0
    sol = kfsde(k[reorder_id], Diagonal(I * σ^2, nobs))
    
    if sol.info == 0
        Turing.@addlogprob! sol.logZ
    else
        # Early exit if simulation could not be computed successfully.
        Turing.@addlogprob! -Inf
        return nothing
    end
    
end

approx_mod = kalman_model(kfsde)
ch_approx_mod = sample(approx_mod, NUTS(), 1000, init_params = [par[:k3], par[:k6], par[:k9], par[:k12]])

DynamicPPL.VarInfo(approx_mod)

# try laplace approximation instead