using VariationalInequalitySolver
using ADNLPModels, NLPModels
using Random, Test

Random.seed!(1234)

@testset "VariationalInequalitySolver.jl" begin
  nls = ADNLSModel(x -> [x[1] - 1; 10 * (x[2] - x[1]^2)], [-1.2; 1.0], 2)
  vi = NLSVIModel(nls)
  xr = rand(2)
  @test project(vi, xr) == xr
end