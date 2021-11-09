module VariationalInequalitySolver

using FastClosures, LinearAlgebra, NLPModels

abstract type AbstractVIModel{S, T} end

abstract type AbstractVIMeta{S, T} end

mutable struct VIMeta{S, T} <: AbstractVIMeta{S, T}
  x0::S
  nvar
  nnzj
  nnzh
end

#largely inspired from https://github.com/JuliaSmoothOptimizers/NLPModels.jl/blob/main/src/nls/api.jl
include("api.jl")

#=
mutable struct VIModel{S, T} <: AbstractVIModel{S, T}
  meta::AbstractVIMeta{S, T}
  F::Function # in-place function
  JF # in-place Jacobian-function
  ProjX::Function # in-place function
end

function residual!(Fx::AbstractVector{T}, model::VIModel, x::AbstractVector{T}) where {T}
  @lencheck model.n x Fx
  VIModel.F(Fx, x)
  return Fx
end

function project!(Px::AbstractVector{T}, model::AbstractVIModel, d::AbstractVector{T}) where {T}
  return Px
end

mutable struct NLSVIModel{T <: AbstractNLSModel} <: AbstractVIModel
  nls::NLP
end

include("projector/ProjNLP.jl")
include("solvers/penalizedVI.jl")
=#

end
