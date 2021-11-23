using VariationalInequalitySolver
using ADNLPModels, NLPModels
using Random, Test

Random.seed!(1234)

@testset "VariationalInequalitySolver.jl" begin
  nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
  vi = NLSVIModel(nls)
  xr = rand(2)
  @test project(vi, xr) == xr
  sol = ProjectionVI(vi, xr)
end

@testset "Test API AbstractVIModel" begin
  for T in [Float16, Float32, Float64]
    nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], T[-1.2; 1.0], 2)
    vi = NLSVIModel(nls)

    @test get_meta(vi) == vi.meta

    xr = ones(T, 2)
    @test residual(vi, xr) == T[0; 0]
    J = T[ 1 0; -20 10]
    @test jac_residual(vi, xr) == J
    v = ones(T, 2)
    @test jprod_residual(vi, xr, v) == T[1; -10]
    @test jtprod_residual(vi, xr, v) == T[-19; 10]
    J = jac_op_residual(vi, xr)
    @test J * v == T[1; -10]
    @test J' * v == T[-19; 10]
    @test hess_residual(vi, xr, xr) == T[-20 0; 0 0]
    @test hprod_residual(vi, xr, 2, v) == T[-20; 0]
    H = hess_op_residual(vi, xr, 2)
    @test H * v == T[-20; 0]
  end
end
