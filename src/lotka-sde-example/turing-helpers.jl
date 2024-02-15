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