using Pkg
Pkg.activate(".")

using LinearAlgebra, StatsBase, Statistics, Plots, StatsPlots, Random, Distributions, DataFrames, StatFiles

# Number of MC simulations
M = 100000

function B(y::Real, Y::AbstractVector{<:Real})
    return mean(Int.(Y .<= y))
end 

function B_std(y::Real, Y::AbstractVector{<:Real})
    return std(Int.(Y .<= y))
end 

function T_stat(β_0::Real, B_est::Real, σ_est::Real, N::Integer)
    return (B_est - β_0) / (σ_est / sqrt(N))
end

data = DataFrame(load("cps.dta"))
Y = data.wage
Y = collect(skipmissing(Y))
Y = Float64.(Y)

y_values = [30, 50, 100]
N_values = [100, 500, 1000]
B_values = zeros(M, length(y_values), length(N_values))
σ_values = zeros(M, length(y_values), length(N_values))
for m in 1:M
    for i in eachindex(y_values)
        for j in eachindex(N_values)
            sample = rand(Y, N_values[j])
            B_values[m, i, j] = B(y_values[i], sample)
            σ_values[m, i, j] = B_std(y_values[i], sample)
        end
    end
end

for i in eachindex(y_values)
    for j in eachindex(N_values)
        histogram(B_values[:, i, j], label="", normalize=:pdf, bins = LinRange(minimum(B_values[:, i, :]), maximum(B_values[:, i, :]), 150))
        vline!([mean(B_values[:, i, j])], label="Mean", style=:dash)
        vline!([B(y_values[i], Y)], label="True")
        plot!(x -> pdf(Normal(mean(B_values[:, i, j]), std(B_values[:, i, j])), x), label="Normal Approximation")
        savefig("PSET1/hist_$(y_values[i])_$(N_values[j]).pdf")
    end
end

β_range = LinRange(minimum(Y), maximum(Y), 100)
for i in eachindex(y_values)
    for j in eachindex(N_values)
        plot(β_range, [mean([T_stat(β, B_values[:, i, j], B_std[:, i, j], N_values[j]) for i in eachindex(y_values), j in eachindex(N_values)]) for β in β_range], label="")
        savefig("PSET1/T_stat_$(y_values[i])_$(N_values[j]).pdf")
    end
end

