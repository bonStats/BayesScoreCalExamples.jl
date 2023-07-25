using BayesScoreCal
using Distributions
using ModelingToolkit
using Catalyst
using GaussianFilters

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

mapk_sde_sys = convert(SDESystem, mapk_2step)
mapk_sde = SDEFunction(mapk_sde_sys)

u = rand(length(states(mapk_2step)))
p = rand(length(parameters(mapk_2step)))
t = 0
mapk_sde.f(u, p, t)
mapk_sde.g(u, p, t)



J[1]

