
mutable struct GaussianFilter
    μ::Vector{Float64}
    Σ::AbstractMatrix{Float64}
end

function predict!(x::GaussianFilter, model::SDEFunction, p::Vector{<:Real}, dt::Float64; t::Real = 0.0, nugget::Real = 0.0)

    μp = x.μ + model.f(x.μ, p, t) .* dt
    F = model.jac(x.μ, p, t)

    L = model.g(x.μ, p, t) 
    Q = L * L'
    Σp = x.Σ + ((F * x.Σ + x.Σ * F') + Q ).* dt + (nugget * I)
    
    x.μ = μp
    x.Σ = Symmetric(Σp)

end

function update!(x::GaussianFilter, y::Vector{<:Real}, H::AbstractMatrix{<:Real}, R::AbstractMatrix{<:Real}; nugget::Real = 0.0)

    yres = y - H * x.μ
    S = H * x.Σ * H' + R
    ll = logpdf(MvNormal(zero(yres), S), yres)

    K = x.Σ * H' / S
    μu = x.μ + K * yres
    Σu = (I - K*H) * x.Σ + (nugget * I)

    x.μ = μu
    x.Σ = Symmetric(Σu)

    return ll

end

struct Observation
    y::Vector{<:Real}
    t::Float64
end


function kalman!(x::GaussianFilter, t0::Float64, model::SDEFunction, H::AbstractMatrix{<:Real}, R::AbstractMatrix{<:Real}, p::Vector{<:Real}, observations::Vector{Observation}, dt0::Float64; nugget::Float64 = 0.0)
    # takes step sizes dt or to the nearest observation time if closer
    # t = t0, t0 + dt, ..., t1, t1 + dt, t1 + 2dt, ... , tn
    ll = 0.0

    # initial time
    t = t0
    for obs in observations
        # predict at dt increments
        while t < obs.t
            # next increment
            dt = min(dt0, obs.t - t)

            # update to time t + dt
            predict!(x, model, p, dt, t = t, nugget = nugget)
            t += dt
        end

        # update
        ll = ll + update!(x, obs.y, H, R, nugget = nugget)
        #println("$t")
    end

    return ll

end