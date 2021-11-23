export ProjectionVI

function ProjectionVI(model::AbstractVIModel, x0::AbstractVector; rho0::Float64 = 0.5, kwargs...)
  stp = GenericStopping(model, x0; kwargs...)
  return ProjectionVI(stp, rho0 = rho0)
end

function abresidual!(model, xk, rho, Fx) # xk + rho * F(xk)
  residual!(model, xk, Fx)
  Fx .*= rho
  Fx .+= xk
  return Fx
end

function ProjectionVI(stp::AbstractStopping; rho0::Float64 = 0.5)
  xk = stp.current_state.x
  xkp = similar(xk)
  rho = rho0
  Fx = similar(xk)

  OK = update_and_start!(stp)
  while !OK
    abresidual!(stp.pb, xk, rho, Fx)
    project!(stp.pb, Fx, xk) # possible failure here

    if norm(xk - xkp, Inf) < stp.meta.atol * rho
      stp.meta.optimal = true
    end
    OK = update_and_stop!(stp, x = xk)
  end

  return xk
end
