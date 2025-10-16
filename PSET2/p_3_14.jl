using Pkg
Pkg.activate(".")

using LinearAlgebra, StatsBase, Statistics, Plots, StatsPlots, Random, Distributions, DataFrames, StatFiles

##### Key statistical functions
function Φ(x)
    return cdf(Normal(), x)
end

function Φ_inv(x)
    return quantile(Normal(), x)
end

function B(x::Real, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real})
    return mean(Y[X .== x])
end

function B(x::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real})
    return [B(x, Y, X) for x in x]
end

function B_var(x::Real, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real})
    return var(Y[X .== x])
end

function B_var(x::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real})
    return Diagonal([B_var(x, Y, X) for x in x])
end

function B_std(x::Real, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real})
    return sqrt(B_var(x, Y, X))
end

function B_std(x::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real})
    return Diagonal([B_std(x, Y, X) for x in x])
end

function σ_hat(B::AbstractVector{<:Real})
    return sqrt(mean((B .- mean(B)).^2))
end

function ∇σ_hat(B::AbstractVector{<:Real})
    return [(B[i] - mean(B)) / (length(B) * σ_hat(B)) for i in eachindex(B)]
end


# Number of MC simulations
M = 10

data = DataFrame(load("cps.dta"))
U = data.wage
U = collect(skipmissing(U))
U = Float64.(U)

state = data.state

state_counts = countmap(state)
top_states = sort(collect(keys(state_counts)), by=x->-state_counts[x])[1:10]
data_10 = data[in.(data.state, Ref(top_states)), :]

state_freqs = countmap(data_10.state)
sorted_states = sort(collect(keys(state_freqs)), by=x -> -state_freqs[x])
state_to_rank = Dict(s => r for (r, s) in enumerate(sorted_states))
data_10.state_rank = [state_to_rank[s] for s in data_10.state]
state_rank_counts = countmap(data_10.state_rank)

F_X = DiscreteNonParametric(collect(keys(state_rank_counts)), values(state_rank_counts) ./ sum(values(state_rank_counts)))


σ_values = collect(LinRange(0.0, 2.0, 100))
N_values = [Int64(round(length(U)/ 2)), Int64(round(length(U))), Int64(round(length(U) * 2))]
@time for m in 1:M
    sample_X = rand(F_X, N_values[1])
    sample_U = rand(U, N_values[1])

    sample_Y = (-1).^(sample_X) * σ_values[50] .+ sample_U

    B_n = B(collect(1:10), sample_Y, sample_X)
    S_n = σ_hat(B_n)

    display(S_n)
end