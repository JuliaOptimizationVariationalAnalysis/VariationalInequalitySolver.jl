var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = VariationalInequalitySolver","category":"page"},{"location":"#VariationalInequalitySolver","page":"Home","title":"VariationalInequalitySolver","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for VariationalInequalitySolver.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [VariationalInequalitySolver]","category":"page"},{"location":"#NLPModels.hess_coord_residual!","page":"Home","title":"NLPModels.hess_coord_residual!","text":"vals = hess_coord_residual!(model, x, v, vals)\n\nComputes the linear combination of the Hessians of the residuals at x with coefficients v in sparse coordinate format, rewriting vals.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModels.hess_coord_residual-Tuple{VariationalInequalitySolver.AbstractVIModel, AbstractVector{T} where T, AbstractVector{T} where T}","page":"Home","title":"NLPModels.hess_coord_residual","text":"vals = hess_coord_residual(model, x, v)\n\nComputes the linear combination of the Hessians of the residuals at x with coefficients v in sparse coordinate format.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.hess_op_residual-Union{Tuple{S}, Tuple{T}, Tuple{VariationalInequalitySolver.AbstractVIModel{T, S}, AbstractVector{T}, Int64}} where {T, S}","page":"Home","title":"NLPModels.hess_op_residual","text":"Hop = hess_op_residual(model, x, i)\n\nComputes the Hessian of the i-th residual at x, in linear operator form.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.hess_residual-Tuple{VariationalInequalitySolver.AbstractVIModel, AbstractVector{T} where T, AbstractVector{T} where T}","page":"Home","title":"NLPModels.hess_residual","text":"H = hess_residual(model, x, v)\n\nComputes the linear combination of the Hessians of the residuals at x with coefficients v. A Symmetric object wrapping the lower triangle is returned.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.hess_structure_residual!","page":"Home","title":"NLPModels.hess_structure_residual!","text":"hess_structure_residual!(model, rows, cols)\n\nReturns the structure of the residual Hessian in place.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModels.hess_structure_residual-Tuple{VariationalInequalitySolver.AbstractVIModel}","page":"Home","title":"NLPModels.hess_structure_residual","text":"(rows,cols) = hess_structure_residual(model)\n\nReturns the structure of the residual Hessian.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.hprod_residual!","page":"Home","title":"NLPModels.hprod_residual!","text":"Hiv = hprod_residual!(model, x, i, v, Hiv)\n\nComputes the product of the Hessian of the i-th residual at x, times the vector v, and stores it in vector Hiv.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModels.hprod_residual-Union{Tuple{S}, Tuple{T}, Tuple{VariationalInequalitySolver.AbstractVIModel{T, S}, AbstractVector{T}, Int64, AbstractVector{T} where T}} where {T, S}","page":"Home","title":"NLPModels.hprod_residual","text":"Hiv = hprod_residual(model, x, i, v)\n\nComputes the product of the Hessian of the i-th residual at x, times the vector v.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jac_coord_residual!","page":"Home","title":"NLPModels.jac_coord_residual!","text":"vals = jac_coord_residual!(model, x, vals)\n\nComputes the Jacobian of the residual at x in sparse coordinate format, rewriting vals. rows and cols are not rewritten.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModels.jac_coord_residual-Tuple{VariationalInequalitySolver.AbstractVIModel, AbstractVector{T} where T}","page":"Home","title":"NLPModels.jac_coord_residual","text":"(rows,cols,vals) = jac_coord_residual(model, x)\n\nComputes the Jacobian of the residual at x in sparse coordinate format.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jac_op_residual!-Tuple{VariationalInequalitySolver.AbstractVIModel, AbstractVector{var\"#s5\"} where var\"#s5\"<:Integer, AbstractVector{var\"#s3\"} where var\"#s3\"<:Integer, AbstractVector{T} where T, AbstractVector{T} where T, AbstractVector{T} where T}","page":"Home","title":"NLPModels.jac_op_residual!","text":"Jx = jac_op_residual!(model, rows, cols, vals, Jv, Jtv)\n\nComputes J(x), the Jacobian of the residual given by (rows, cols, vals), in linear operator form. The vectors Jv and Jtv are used as preallocated storage for the operations.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jac_op_residual-Union{Tuple{S}, Tuple{T}, Tuple{VariationalInequalitySolver.AbstractVIModel{T, S}, AbstractVector{T}}} where {T, S}","page":"Home","title":"NLPModels.jac_op_residual","text":"Jx = jac_op_residual(model, x)\n\nComputes J(x), the Jacobian of the residual at x, in linear operator form.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jac_residual-Tuple{VariationalInequalitySolver.AbstractVIModel, AbstractVector{T} where T}","page":"Home","title":"NLPModels.jac_residual","text":"Jx = jac_residual(model, x)\n\nComputes J(x), the Jacobian of the residual at x.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jac_structure_residual!","page":"Home","title":"NLPModels.jac_structure_residual!","text":"(rows,cols) = jac_structure_residual!(model, rows, cols)\n\nReturns the structure of the constraint's Jacobian in sparse coordinate format in place.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModels.jac_structure_residual-Tuple{VariationalInequalitySolver.AbstractVIModel}","page":"Home","title":"NLPModels.jac_structure_residual","text":"(rows,cols) = jac_structure_residual(model)\n\nReturns the structure of the constraint's Jacobian in sparse coordinate format.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jprod_residual!","page":"Home","title":"NLPModels.jprod_residual!","text":"Jv = jprod_residual!(model, x, v, Jv)\n\nComputes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)v, storing it in Jv.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModels.jprod_residual!-Tuple{VariationalInequalitySolver.AbstractVIModel, AbstractVector{var\"#s3\"} where var\"#s3\"<:Integer, AbstractVector{var\"#s8\"} where var\"#s8\"<:Integer, AbstractVector{T} where T, AbstractVector{T} where T, AbstractVector{T} where T}","page":"Home","title":"NLPModels.jprod_residual!","text":"Jv = jprod_residual!(model, rows, cols, vals, v, Jv)\n\nComputes the product of the Jacobian of the residual given by (rows, cols, vals) and a vector, i.e.,  J(x)v, storing it in Jv.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jprod_residual-Union{Tuple{S}, Tuple{T}, Tuple{VariationalInequalitySolver.AbstractVIModel{T, S}, AbstractVector{T}, AbstractVector{T} where T}} where {T, S}","page":"Home","title":"NLPModels.jprod_residual","text":"Jv = jprod_residual(model, x, v)\n\nComputes the product of the Jacobian of the residual at x and a vector, i.e.,  J(x)v.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jth_hess_residual-Tuple{VariationalInequalitySolver.AbstractVIModel, AbstractVector{T} where T, Int64}","page":"Home","title":"NLPModels.jth_hess_residual","text":"Hj = jth_hess_residual(model, x, j)\n\nComputes the Hessian of the j-th residual at x.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jtprod_residual!","page":"Home","title":"NLPModels.jtprod_residual!","text":"Jtv = jtprod_residual!(model, x, v, Jtv)\n\nComputes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)^Tv, storing it in Jtv.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModels.jtprod_residual!-Tuple{VariationalInequalitySolver.AbstractVIModel, AbstractVector{var\"#s5\"} where var\"#s5\"<:Integer, AbstractVector{var\"#s3\"} where var\"#s3\"<:Integer, AbstractVector{T} where T, AbstractVector{T} where T, AbstractVector{T} where T}","page":"Home","title":"NLPModels.jtprod_residual!","text":"Jtv = jtprod_residual!(model, rows, cols, vals, v, Jtv)\n\nComputes the product of the transpose of the Jacobian of the residual given by (rows, cols, vals) and a vector, i.e.,  J(x)^Tv, storing it in Jv.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.jtprod_residual-Union{Tuple{S}, Tuple{T}, Tuple{VariationalInequalitySolver.AbstractVIModel{T, S}, AbstractVector{T}, AbstractVector{T} where T}} where {T, S}","page":"Home","title":"NLPModels.jtprod_residual","text":"Jtv = jtprod_residual(model, x, v)\n\nComputes the product of the transpose of the Jacobian of the residual at x and a vector, i.e.,  J(x)^Tv.\n\n\n\n\n\n","category":"method"},{"location":"#NLPModels.residual!","page":"Home","title":"NLPModels.residual!","text":"Fx = residual!(model, x, Fx)\n\nComputes F(x), the residual at x.\n\n\n\n\n\n","category":"function"},{"location":"#NLPModels.residual-Union{Tuple{S}, Tuple{T}, Tuple{VariationalInequalitySolver.AbstractVIModel{T, S}, AbstractVector{T}}} where {T, S}","page":"Home","title":"NLPModels.residual","text":"Fx = residual(model, x)\n\nComputes F(x), the residual at x.\n\n\n\n\n\n","category":"method"},{"location":"#VariationalInequalitySolver.project!","page":"Home","title":"VariationalInequalitySolver.project!","text":"project!(Px::AbstractVector{T}, model::VIModel, x::AbstractVector{T}) where {T}\n\nCompute the projection of d over X in-place, i.e.,\n\n    min_x 05  d - x ²₂ st x  X\n\n\n\n\n\n","category":"function"},{"location":"#VariationalInequalitySolver.project-Union{Tuple{S}, Tuple{T}, Tuple{VariationalInequalitySolver.AbstractVIModel{T, S}, AbstractVector{T}}} where {T, S}","page":"Home","title":"VariationalInequalitySolver.project","text":"project(model::VIModel, x::AbstractVector{T}) where {T}\n\nCompute the projection of d over X, i.e.,\n\n    min_x 05  d - x ²₂ st x  X\n\n```\n\n\n\n\n\n","category":"method"}]
}
