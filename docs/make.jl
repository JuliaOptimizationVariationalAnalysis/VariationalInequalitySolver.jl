using VariationalInequalitySolver
using Documenter

DocMeta.setdocmeta!(
  VariationalInequalitySolver,
  :DocTestSetup,
  :(using VariationalInequalitySolver);
  recursive = true,
)

makedocs(;
  modules = [VariationalInequalitySolver],
  authors = "Tangi Migot tangi.migot@gmail.com",
  repo = "https://github.com/JuliaOptimizationVariationalAnalysis/VariationalInequalitySolver.jl/blob/{commit}{path}#{line}",
  sitename = "VariationalInequalitySolver.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://JuliaOptimizationVariationalAnalysis.github.io/VariationalInequalitySolver.jl",
    assets = String[],
  ),
  pages = ["Home" => "index.md"],
)

deploydocs(;
  repo = "github.com/JuliaOptimizationVariationalAnalysis/VariationalInequalitySolver.jl",
  devbranch = "main",
)
