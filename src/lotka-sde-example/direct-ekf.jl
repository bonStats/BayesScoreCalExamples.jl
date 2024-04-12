
mutable struct GaussianFilter
    μ::Vector{<:Real}
    Σ::AbstractMatrix{<:Real}
end

struct KalmanEM
    drift::Function
    noise::Function
    jac::Function
    # sdrift::Vector{<:Real}
    # snoise::AbstractMatrix{<:Real}
    # sjac::AbstractMatrix{<:Real}
end

function predict!(x::GaussianFilter, kem::KalmanEM, p::Vector{<:Real}, dt::Float64; t::Real = 0.0, nugget::Real = 0.0)
    
    μp = x.μ + kem.drift(x.μ, p) .* dt
    F = kem.jac(x.μ, p)

    L = kem.noise(x.μ, p) 
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


function kalman!(x::GaussianFilter, t0::Float64, kem::KalmanEM, H::AbstractMatrix{<:Real}, R::AbstractMatrix{<:Real}, p::Vector{<:Real}, observations::Vector{Observation}, dt0::Float64; nugget::Tuple{Float64, Float64} = (0.0,0.0))
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
            predict!(x, kem, p, dt, t = t, nugget = nugget[1])
            t += dt
        end

        # update
        ll = ll + update!(x, obs.y, H, R, nugget = nugget[2])
        #println("$t")
    end

    return ll

end

struct KalmanApproxSDE
    model::KalmanEM
    observations::Vector{Observation}
    t0::Float64 # initial time
    dt0::Float64 # (max) time increments
    H::AbstractMatrix{<:Real} # y = Hx + ε
    x0::GaussianFilter # initial distirbution
    xscratch::GaussianFilter # scratch space for filter
end

KalmanApproxSDE(model::KalmanEM, observations::Vector{Observation}, t0::Float64, dt0::Float64, H::AbstractMatrix{<:Real}, x0::GaussianFilter) = KalmanApproxSDE(model, observations, t0, dt0, H, x0, GaussianFilter(similar(x0.μ), similar(x0.Σ)))


struct KalmanSolution
    x::GaussianFilter # final distribution
    logZ::Real
    info::Int64
end

# unknown prior mean covariance
function (k::KalmanApproxSDE)(p::Vector{<:Real}, R::AbstractMatrix{<:Real}, μ0::AbstractVector{<:Real}, Σ0::AbstractMatrix{<:Real}; nugget::Tuple{Float64, Float64} = (0.0,0.0))
    # initialise scratch
    k.xscratch.μ = μ0
    k.xscratch.Σ = Σ0
    
    # run Kalman Filter
    info = 0
    loglike = 0.0
    
    try
        loglike = kalman!(k.xscratch, k.t0, k.model, k.H, R, p, k.observations, k.dt0; nugget = nugget)
    catch err
        if err isa PosDefException
            info = 1
        elseif err isa SingularException
            info = 2
        elseif err isa ArgumentError
            info = 3
        else 
            info = 99
            throw(err)
        end
    end

    return KalmanSolution(deepcopy(k.xscratch), loglike, info)
end

# fixed prior mean covariance
(k::KalmanApproxSDE)(p::Vector{<:Real}, R::AbstractMatrix{<:Real}; nugget::Tuple{Float64, Float64} = (0.0,0.0)) = k(p, R, k.x0.μ, k.x0.Σ, nugget = nugget)