using BayesScoreCal
using BayesScoreCalExamples
using Distributions
using ModelingToolkit
using Catalyst
using LinearAlgebra
using DifferentialEquations
using Turing
using SparseArrays

getparams(m::DynamicPPL.Model) = DynamicPPL.syms(DynamicPPL.VarInfo(m))
getstatesymbol(x::SymbolicUtils.BasicSymbolic) = x.metadata[Symbolics.VariableSource][2]

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


ord_u0 = Float64.([u0[getstatesymbol(x)] for x in states(mapk_2step)])

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
kfsde = KalmanApproxSDE(mapk2kem, obsv, 0.0, 1.0, H, x)

kfsde(p, R).logZ
kfsde(p, R).info

param_fixed = Dict(
    :k1 => 0.001,
    :k4 => 0.001,
    :k7 => 0.0001,
    :k10 => 0.001
)


# collect and order parameters
reorder_id = [get(par_id, v, 0) for v in ord_parameters]

@model kalman_model(kfsde::KalmanApproxSDE) = begin
     
    k1 = param_fixed[:k1]
    k2 = par[:k2] #~ Uniform(0.0, par[:k1]) 
    k3 ~ Uniform(0.0, 1.0)    
    k4 = param_fixed[:k4]  
    k5 = par[:k5] #~ Uniform(0.0, par[:k4])  
    k6 ~ Uniform(0.0, 1.0)
    k7 = param_fixed[:k7]  
    k8 = par[:k8] #~ Uniform(0.0, par[:k7])   
    k9 ~ Uniform(0.0, 1.0)
    k10 = param_fixed[:k10]     
    k11 = par[:k11] #~ Uniform(0.0, par[:k10]) 
    k12 ~ Uniform(0.0, 1.0)          

    k = [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12]

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

approx_mod = kalman_model(kfsde)
ch_approx_mod = sample(approx_mod, NUTS(), 1000, init_params = [par[:k3], par[:k6], par[:k9], par[:k12]])

DynamicPPL.VarInfo(approx_mod)

# try laplace approximation instead

reorder_id


k1 = p
k1[3] = 0.18 - 0.005
sol = kfsde(k1[reorder_id], Diagonal(I * 1.0, nobs))
sol.logZ

@profile kfsde(p, Diagonal(I * 1.0, nobs))

using ForwardDiff

f(x) = sum(mapk2drift(ord_u0, x) .* ones(11))
ForwardDiff.gradient(x -> sum(mapk2drift(ord_u0, x) .* ones(11)), p)


ForwardDiff.gradient(x -> sum(mapk2noise(ord_u0 .+ 0.001, x) * ones(12)), p)
ForwardDiff.gradient(x -> sum(mapk2jacobian(ord_u0, x) * ones(11)), p)
ForwardDiff.gradient(x -> kfsde(x, R).logZ, p)

pp = remake(sdeprob, tspan = (0.0,1.0), u0 = uu)

sol = solve(pp, EM(), dt = 0.1)
sol(0.1)

SciMLSensitivity.SDEAdjointProblem(sol, BacksolveAdjoint(), )


generate_jacobian(sdeprob, u0, k1)


zz = zeros(Real, 11)
function f(x) 
    vv = merge(Dict(parameters(mapk_2step) .=> x), Dict(states(mapk_2step) .=> uu))
    zz = Symbolics.value.(substitute(modelsymjac, vv))
    sum(zz)
end

ForwardDiff.gradient(f, k1)

(zz, uu, k1)


solve(prob, EM(), dt = dt)

# check if differentiable 



zz = mapk2noise(ord_u0 .+ 0.01, p)

indzz = CartesianIndices(zz)[.! iszero.(zz)]

repr(map(x -> x.I[1], indzz))
repr(map(x -> x.I[2], indzz))


["-sqrt(abs(p[1]*u[2]*u[1]))" "sqrt(abs(p[2]*u[3]))" "0" "0" "0" sqrt(abs(p[6]*u[6])) "0" "0" "0" "0" "0" "0"; 
"-sqrt(abs(p[1]*u[2]*u[1]))" "sqrt(abs(p[2]*u[3]))" "sqrt(abs(p[3]*u[3]))" "0" "0" "0" "0" "0" "0" "0" "0" "0";
"sqrt(abs(p[1]*u[2]*u[1]))" "-sqrt(abs(p[2]*u[3]))" "-sqrt(abs(p[3]*u[3]))" "0" "0" "0" "0" "0" "0" "0" "0" "0"; 
"0" "0" "sqrt(abs(p[3]*u[3]))" "-sqrt(abs(p[4]*u[5]*u[4]))" sqrt(abs(p[5]*u[6])) "0" -sqrt(abs(p[7]*u[4]*u[7])) sqrt(abs(p[8]*u[8])) sqrt(abs(p[9]*u[8])) "0" "0" "0";
"0" "0" "0" -sqrt(abs(p[4]*u[5]*u[4])) sqrt(abs(p[5]*u[6])) sqrt(abs(p[6]*u[6])) "0" "0" "0" "0" "0" "0"; 
"0" "0" "0" sqrt(abs(p[4]*u[5]*u[4])) -sqrt(abs(p[5]*u[6])) -sqrt(abs(p[6]*u[6])) "0" "0" "0" "0" "0" "0";
"0" "0" "0" "0" "0" "0" -sqrt(abs(p[7]*u[4]*u[7])) sqrt(abs(p[8]*u[8])) "0" "0" "0" sqrt(abs(p[12]*u[11])); 
"0" "0" "0" "0" "0" "0" sqrt(abs(p[7]*u[4]*u[7])) -sqrt(abs(p[8]*u[8])) -sqrt(abs(p[9]*u[8])) "0" "0" "0";
"0" "0" "0" "0" "0" "0" "0" "0" sqrt(abs(p[9]*u[8])) -sqrt(abs(p[10]*u[10]*u[9])) sqrt(abs(p[11]*u[11])) "0"; 
"0" "0" "0" "0" "0" "0" "0" "0" "0" -sqrt(abs(p[1"0"]*u[10]*u[9])) sqrt(abs(p[11]*u[11])) sqrt(abs(p[12]*u[11]));
"0" "0" "0" "0" "0" "0" "0" "0" "0" sqrt(abs(p[10]*u[10]*u[9])) -sqrt(abs(p[11]*u[11])) -sqrt(abs(p[12]*u[11]))]

[-sqrt(abs(p[1]*u[2]*u[1])), -sqrt(abs(p[1]*u[2]*u[1])), sqrt(abs(p[1]*u[2]*u[1])), 
sqrt(abs(p[2]*u[3])), sqrt(abs(p[2]*u[3])), -sqrt(abs(p[2]*u[3])),
sqrt(abs(p[3]*u[3])), -sqrt(abs(p[3]*u[3])), sqrt(abs(p[3]*u[3])),
-sqrt(abs(p[4]*u[5]*u[4]))
]



@profile mapk2noise(ord_u0 .+ 0.01 , p)






M = [
-a b  0. 0. 0. f  0. 0. 0. 0. 0. 0.; 
-a b  c  0. 0. 0. 0. 0. 0. 0. 0. 0.;
a -b -c  0. 0. 0. 0. 0. 0. 0. 0. 0.; 
0. 0. c -d  e  0 -g  h  i  0. 0. 0.;
0. 0. 0 -d  e  f  0. 0. 0. 0. 0. 0.; 
0. 0. 0. d -e -f  0. 0. 0. 0. 0. 0.;
0. 0. 0. 0. 0. 0 -g  h  0. 0. 0. l ; 
0. 0. 0. 0. 0. 0. g -h -i  0. 0. 0.;
0. 0. 0. 0. 0. 0. 0. 0. i -j  k  0.; 
0. 0. 0. 0. 0. 0. 0. 0. 0 -j  k  l ;
0. 0. 0. 0. 0. 0. 0. 0. 0. j -k -l
]

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