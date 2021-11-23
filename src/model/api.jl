export residual, residual!, jac_residual, jac_structure_residual, jac_structure_residual!
export jac_coord_residual!, jac_coord_residual, jprod_residual, jprod_residual!
export jtprod_residual, jtprod_residual!, jac_op_residual, jac_op_residual!
export hess_residual, hess_structure_residual, hess_structure_residual!
export hess_coord_residual!, hess_coord_residual, jth_hess_residual
export hprod_residual, hprod_residual!, hess_op_residual, hess_op_residual!
export project!, project, get_meta

"""
    Fx = residual(model, x)
Computes ``F(x)``, the residual at x.
"""
function NLPModels.residual(model::AbstractVIModel{T, S}, x::AbstractVector{T}) where {T, S}
  @lencheck model.meta.nvar x
  Fx = S(undef, model.meta.nvar)
  residual!(model, x, Fx)
end

"""
    Fx = residual!(model, x, Fx)
Computes ``F(x)``, the residual at x.
"""
function residual! end

"""
    Jx = jac_residual(model, x)
Computes ``J(x)``, the Jacobian of the residual at x.
"""
function NLPModels.jac_residual(model::AbstractVIModel, x::AbstractVector)
  @lencheck model.meta.nvar x
  rows, cols = jac_structure_residual(model)
  vals = jac_coord_residual(model, x)
  sparse(rows, cols, vals, model.meta.nvar, model.meta.nvar)
end

"""
    (rows,cols) = jac_structure_residual!(model, rows, cols)
Returns the structure of the constraint's Jacobian in sparse coordinate format in place.
"""
function jac_structure_residual! end

"""
    (rows,cols) = jac_structure_residual(model)
Returns the structure of the constraint's Jacobian in sparse coordinate format.
"""
function NLPModels.jac_structure_residual(model::AbstractVIModel)
  rows = Vector{Int}(undef, model.meta.nnzj)
  cols = Vector{Int}(undef, model.meta.nnzj)
  jac_structure_residual!(model, rows, cols)
end

"""
    vals = jac_coord_residual!(model, x, vals)
Computes the Jacobian of the residual at `x` in sparse coordinate format, rewriting
`vals`. `rows` and `cols` are not rewritten.
"""
function jac_coord_residual! end

"""
    (rows,cols,vals) = jac_coord_residual(model, x)
Computes the Jacobian of the residual at `x` in sparse coordinate format.
"""
function NLPModels.jac_coord_residual(model::AbstractVIModel, x::AbstractVector)
  @lencheck model.meta.nvar x
  vals = Vector{eltype(x)}(undef, model.meta.nnzj)
  jac_coord_residual!(model, x, vals)
end

"""
    Jv = jprod_residual(model, x, v)
Computes the product of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)v``.
"""
function NLPModels.jprod_residual(
  model::AbstractVIModel{T, S},
  x::AbstractVector{T},
  v::AbstractVector,
) where {T, S}
  @lencheck model.meta.nvar x v
  Jv = S(undef, model.meta.nvar)
  jprod_residual!(model, x, v, Jv)
end

"""
    Jv = jprod_residual!(model, x, v, Jv)
Computes the product of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)v``, storing it in `Jv`.
"""
function jprod_residual! end

"""
    Jv = jprod_residual!(model, rows, cols, vals, v, Jv)
Computes the product of the Jacobian of the residual given by `(rows, cols, vals)`
and a vector, i.e.,  ``J(x)v``, storing it in `Jv`.
"""
function NLPModels.jprod_residual!(
  model::AbstractVIModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jv::AbstractVector,
)
  @lencheck model.meta.nnzj rows cols vals
  @lencheck model.meta.nvar v Jv
  increment!(model, :neval_jprod_residual)
  coo_prod!(rows, cols, vals, v, Jv)
end

"""
    Jtv = jtprod_residual(model, x, v)
Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)^Tv``.
"""
function NLPModels.jtprod_residual(
  model::AbstractVIModel{T, S},
  x::AbstractVector{T},
  v::AbstractVector,
) where {T, S}
  @lencheck model.meta.nvar x v
  Jtv = S(undef, model.meta.nvar)
  jtprod_residual!(model, x, v, Jtv)
end

"""
    Jtv = jtprod_residual!(model, x, v, Jtv)
Computes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  ``J(x)^Tv``, storing it in `Jtv`.
"""
function jtprod_residual! end

"""
    Jtv = jtprod_residual!(model, rows, cols, vals, v, Jtv)
Computes the product of the transpose of the Jacobian of the residual given by `(rows, cols, vals)`
and a vector, i.e.,  ``J(x)^Tv``, storing it in `Jv`.
"""
function NLPModels.jtprod_residual!(
  model::AbstractVIModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  v::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck model.meta.nnzj rows cols vals
  @lencheck model.meta.nvar v Jtv
  increment!(model, :neval_jtprod_residual)
  coo_prod!(cols, rows, vals, v, Jtv)
end

"""
    Jx = jac_op_residual(model, x)
Computes ``J(x)``, the Jacobian of the residual at x, in linear operator form.
"""
function NLPModels.jac_op_residual(model::AbstractVIModel{T, S}, x::AbstractVector{T}) where {T, S}
  @lencheck model.meta.nvar x
  Jv = S(undef, model.meta.nvar)
  Jtv = S(undef, model.meta.nvar)
  return jac_op_residual!(model, x, Jv, Jtv)
end

"""
    Jx = jac_op_residual!(model, rows, cols, vals, Jv, Jtv)
Computes ``J(x)``, the Jacobian of the residual given by `(rows, cols, vals)`, in linear operator form. The
vectors `Jv` and `Jtv` are used as preallocated storage for the operations.
"""
function NLPModels.jac_op_residual!(
  model::AbstractVIModel,
  rows::AbstractVector{<:Integer},
  cols::AbstractVector{<:Integer},
  vals::AbstractVector,
  Jv::AbstractVector,
  Jtv::AbstractVector,
)
  @lencheck model.meta.nnzj rows cols vals
  @lencheck model.meta.nvar Jv Jtv
  prod! = @closure (res, v, α, β) -> begin
    jprod_residual!(model, rows, cols, vals, v, Jv)
    if β == 0
      @. res = α * Jv
    else
      @. res = α * Jv + β * res
    end
    return res
  end
  ctprod! = @closure (res, v, α, β) -> begin
    jtprod_residual!(model, rows, cols, vals, v, Jtv)
    if β == 0
      @. res = α * Jtv
    else
      @. res = α * Jtv + β * res
    end
    return res
  end
  return LinearOperator{eltype(vals)}(
    model.meta.nvar,
    model.meta.nvar,
    false,
    false,
    prod!,
    ctprod!,
    ctprod!,
  )
end

"""
    H = hess_residual(model, x, v)
Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v`.
A `Symmetric` object wrapping the lower triangle is returned.
"""
function NLPModels.hess_residual(model::AbstractVIModel, x::AbstractVector, v::AbstractVector)
  @lencheck model.meta.nvar x v
  rows, cols = hess_structure_residual(model)
  vals = hess_coord_residual(model, x, v)
  Symmetric(sparse(rows, cols, vals, model.meta.nvar, model.meta.nvar), :L)
end

"""
    (rows,cols) = hess_structure_residual(model)
Returns the structure of the residual Hessian.
"""
function NLPModels.hess_structure_residual(model::AbstractVIModel)
  rows = Vector{Int}(undef, model.meta.nnzh)
  cols = Vector{Int}(undef, model.meta.nnzh)
  hess_structure_residual!(model, rows, cols)
end

"""
    hess_structure_residual!(model, rows, cols)
Returns the structure of the residual Hessian in place.
"""
function hess_structure_residual! end

"""
    vals = hess_coord_residual!(model, x, v, vals)
Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v` in sparse coordinate format, rewriting `vals`.
"""
function hess_coord_residual! end

"""
    vals = hess_coord_residual(model, x, v)
Computes the linear combination of the Hessians of the residuals at `x` with coefficients
`v` in sparse coordinate format.
"""
function NLPModels.hess_coord_residual(model::AbstractVIModel, x::AbstractVector, v::AbstractVector)
  @lencheck model.meta.nvar x v
  vals = Vector{eltype(x)}(undef, model.meta.nnzh)
  hess_coord_residual!(model, x, v, vals)
end

"""
    Hj = jth_hess_residual(model, x, j)
Computes the Hessian of the j-th residual at x.
"""
function NLPModels.jth_hess_residual(model::AbstractVIModel, x::AbstractVector, j::Int)
  @lencheck model.meta.nvar x
  increment!(model, :neval_jhess_residual)
  decrement!(model, :neval_hess_residual)
  v = [i == j ? one(eltype(x)) : zero(eltype(x)) for i = 1:(model.meta.nvar)]
  return hess_residual(model, x, v)
end

"""
    Hiv = hprod_residual(model, x, i, v)
Computes the product of the Hessian of the i-th residual at x, times the vector v.
"""
function NLPModels.hprod_residual(
  model::AbstractVIModel{T, S},
  x::AbstractVector{T},
  i::Int,
  v::AbstractVector,
) where {T, S}
  @lencheck model.meta.nvar x
  Hv = S(undef, model.meta.nvar)
  hprod_residual!(model, x, i, v, Hv)
end

"""
    Hiv = hprod_residual!(model, x, i, v, Hiv)
Computes the product of the Hessian of the i-th residual at x, times the vector v, and stores it in vector Hiv.
"""
function hprod_residual! end

"""
    Hop = hess_op_residual(model, x, i)
Computes the Hessian of the i-th residual at x, in linear operator form.
"""
function NLPModels.hess_op_residual(
  model::AbstractVIModel{T, S},
  x::AbstractVector{T},
  i::Int,
) where {T, S}
  @lencheck model.meta.nvar x
  Hiv = S(undef, model.meta.nvar)
  return hess_op_residual!(model, x, i, Hiv)
end

get_meta(model::AbstractVIModel) = model.meta

"""
    project!(Px::AbstractVector{T}, model::VIModel, x::AbstractVector{T}) where {T}

Compute the projection of d over X in-place, i.e.,
```math
    min_x 0.5 | d - x |²₂ s.t. x ∈ X
```
"""
function project! end

"""
    project(model::VIModel, x::AbstractVector{T}) where {T}

Compute the projection of d over X, i.e.,
```math
    min_x 0.5 | d - x |²₂ s.t. x ∈ X
```
```
"""
function project(model::AbstractVIModel{T, S}, x::AbstractVector{T}) where {T, S}
  @lencheck model.meta.nvar x
  Px = S(undef, model.meta.nvar)
  project!(model, x, Px)
end
