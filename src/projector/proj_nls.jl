export residual, residual!, jac_residual, jac_structure_residual, jac_structure_residual!
export jac_coord_residual!, jac_coord_residual, jprod_residual, jprod_residual!
export jtprod_residual, jtprod_residual!, jac_op_residual, jac_op_residual!
export hess_residual, hess_structure_residual, hess_structure_residual!
export hess_coord_residual!, hess_coord_residual, jth_hess_residual
export hprod_residual, hprod_residual!, hess_op_residual, hess_op_residual!

mutable struct NLSProjector{T, S} <: AbstractNLSModel{T, S}
  model::AbstractNLSModel{T, S}
  meta::NLPModelMeta{T, S}
  nls_meta::NLSMeta{T, S}
  counters::NLSCounters
  d::S
  function NLSProjector(model::AbstractNLSModel{T, S}, d::S; name = "Projection over $(model.meta.name)") where {T, S}
    nvar = length(d)
    x0 = d #fill!(S(undef, nvar), zero(T))
    meta = NLPModelMeta{T, S}(nvar, x0 = x0, name = name)
    nls_meta = NLSMeta{T, S}(
      nvar,
      nvar,
      nnzj = nvar,
      nnzh = 0,
    )
    return new{T, S}(model, meta, nls_meta, NLSCounters(), d)
  end
end

function NLPModels.residual!(model::NLSProjector, x, Fx)
  Fx .= x - model.d
  return Fx
end

function NLPModels.jac_structure_residual!(
  model::NLSProjector,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
)
  @lencheck model.nls_meta.nnzj rows cols
  for i=1:model.meta.nvar
    rows[i] = i
    cols[i] = i
  end
  return rows, cols
end

function NLPModels.jac_coord_residual!(model::NLSProjector{T, S}, x::AbstractVector, vals::AbstractVector) where {T, S}
  @lencheck model.meta.nvar x
  @lencheck model.nls_meta.nnzj vals
  increment!(model, :neval_jac_residual)
  vals .= one(T)
  return vals
end

function NLPModels.jprod_residual!(
  model::NLSProjector,
  x::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck model.meta.nvar x v
  @lencheck model.nls_meta.nequ Jv
  increment!(model, :neval_jprod_residual)
  Jv .= v
  return Jv
end

function NLPModels.jtprod_residual!(
  model::NLSProjector,
  x::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck model.meta.nvar x Jtv
  @lencheck model.nls_meta.nequ v
  increment!(model, :neval_jtprod_residual)
  Jtv .= v
  return Jtv
end

function NLPModels.hess_structure_residual!(::NLSProjector, rows, cols)
  return rows, cols
end

function NLPModels.hess_coord_residual!(::NLSProjector, x, v, vals)
  return vals
end

function NLPModels.hprod_residual!(::NLSProjector, x, i, v, Hiv)
  return Hiv
end
