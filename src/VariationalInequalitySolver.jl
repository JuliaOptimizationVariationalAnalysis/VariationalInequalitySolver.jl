module VariationalInequalitySolver

using FastClosures, LinearAlgebra, Logging, NLPModels

abstract type AbstractVIModel{T, S} end

abstract type AbstractVIMeta{T, S} end

mutable struct VIMeta{T, S} <: AbstractVIMeta{T, S}
  x0::S
  nvar
  nnzj
  nnzh
end

#largely inspired from https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/main/src/nls/api.jl
include("projector/proj_nls.jl")
include("model/api.jl")

export NLSVIModel

mutable struct NLSVIModel{S, T, NLP <: AbstractNLSModel{T, S}} <: AbstractVIModel{T, S}
  meta::VIMeta{T, S}
  nls::NLP
  function NLSVIModel(nls::AbstractNLSModel{T, S}) where {T, S}
    @lencheck nls.nls_meta.nequ nls.meta.x0 #test that nls.meta.nvar == nls.nls_meta.nequ
    return new{S, T, typeof(nls)}(
      VIMeta{T, S}(nls.meta.x0, nls.meta.nvar, nls.nls_meta.nnzj, nls.nls_meta.nnzh),
      nls
    )
  end
end

for meth in [:residual!, :jac_structure_residual!, :jac_coord_residual!, :jprod_residual!, :jtprod_residual!, :jac_op_residual!, :hess_structure_residual!, :hess_coord_residual!, :hprod_residual!, :hess_op_residual!]
  @eval begin
    $meth(model::NLSVIModel, args...; kwargs...) = $meth(model.nls, args...; kwargs...)
  end
end

using CaNNOLeS

function project!(model::NLSVIModel, d::AbstractVector{T}, Px::AbstractVector{T}) where {T, S}
  # Here we need to solve the optimization problem
  proj = NLSProjector(model.nls, d)
  stats = with_logger(NullLogger()) do
    cannoles(proj)
  end
  if stats.status != :first_order
    @warn "There was an error in the projection computation"
  end
  Px .= stats.solution
  return Px
end

#=
include("projector/ProjNLP.jl")
include("solvers/penalizedVI.jl")
=#

end
