export ProjectionVI

function ProjectionVI(
  model::AbstractVIModel,
  x0::AbstractVector;
  prec :: Float64 = 1e-6,
  rho0 :: Float64 = 0.5,
  itmax :: Int64 = 10000,
)
  xk = copy(x0)
  xkp = similar(xk)
  rho = rho0
  i=0
  OK = false
  Fx = similar(xk)
  residual!(model, xk, Fx)
  Fx .*= rho
  Fx .+= xk # xk + rho * F(xk)

  #main loop
  while !OK
    residual!(model, xk, Fx)
    Fx .*= rho
    Fx .+= xk # xk + rho * F(xk)
    proj!(model, Fx, xkp) # possible failure here

    i=i+1
    OK=(i>=itmax) || (norm(xk-xkp,Inf)<prec*rho )
    xk = copy(xkp)
  end #end of main loop

  return xk
end