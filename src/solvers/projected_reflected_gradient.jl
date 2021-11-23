"""
> Yura Malitsky. Projected reflected gradient methods for monotone variational inequalities. 
> SIAM Journal on Optimization, 25(1):502–520, 2015.
"""
function ProjectedReflectedGradientVI(stp::AbstractStopping; rho0::Float64 = 0.5)
    xk = stp.current_state.x
    xkp = similar(xk)
    yk  = copy(xk)
    rho = rho0
    Fx = similar(xk)
  
    OK = update_and_start!(stp)
    while !OK
      abcresidual!(stp.pb, xk, rho, yk, Fx)
      project!(stp.pb, Fx, xk) # possible failure here
      yk .= 2 .* xkp .- xk
  
      if norm(xk - xkp, Inf) < stp.meta.atol * rho
        stp.meta.optimal = true
      end
      OK = update_and_stop!(stp, x = xk)
    end
  
    return xk
  end
  
export ProjectedReflectedGradientVI

function ProjectedReflectedGradientVI(model::AbstractVIModel, x0::AbstractVector; rho0::Float64 = 0.5, kwargs...)
  stp = GenericStopping(model, x0; kwargs...)
  return ProjectedReflectedGradientVI(stp, rho0 = rho0)
end

function abcresidual!(model, xk, rho, yk, Fx) # xk - rho * F(yk)
  residual!(model, yk, Fx)
  Fx .*= -rho
  Fx .+= xk
  return Fx
end
