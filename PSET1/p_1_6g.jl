using Pkg
Pkg.activate(".")

using LinearAlgebra, StatsBase, Statistics, Plots, StatsPlots, Random, Distributions

function expectation_exact(p::Real, n::Integer)
    expectation = 0.0
    for i in 1:n
        expectation += 1/i * binomial(n, i) * p^(i) * (1-p)^(n-i)
    end
    return expectation
end


function expectation_approximate(p::Real, n::Integer)
    return 1 / (n * p)
end

plot(1:50, [expectation_exact(0.5, i) for i in 1:50], label="Exact", linewidth=2)
plot!(1:50, [expectation_approximate(0.5, i) for i in 1:50], label="Approximate", linewidth=2)
savefig("PSET1/expectation_exact_05.pdf")

plot(1:50, [expectation_exact(0.02, i) for i in 1:50], label="Exact", linewidth=2)
plot!(1:50, [expectation_approximate(0.02, i) for i in 1:50], label="Approximate", linewidth=2)
savefig("PSET1/expectation_exact_002.pdf")

plot(1:50, [expectation_exact(0.1, i) for i in 1:50], label="Exact", linewidth=2)
plot!(1:50, [expectation_approximate(0.1, i) for i in 1:50], label="Approximate", linewidth=2)
savefig("PSET1/expectation_exact_01.pdf")