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
    return var(Y[X .== x]) / mean(X .== x)
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

function σ_hat_var(x::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real}; 
    B::AbstractVector{<:Real} = B(x, Y, X), 
    B_var::AbstractMatrix{<:Real} = B_var(x, Y, X))
    return ∇σ_hat(B)' * B_var * ∇σ_hat(B)
end

# Number of MC simulations
M = 1000
α = 0.05
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


σ_values = collect(LinRange(0.0, 3.0, 50))
N_values = [Int64(round(length(U)/ 2)), Int64(round(length(U))), Int64(round(length(U) * 2))]
S_n_values = zeros(M, length(σ_values), length(N_values))
S_n_vars = zeros(M, length(σ_values), length(N_values))
S_n_CI_low = zeros(M, length(σ_values), length(N_values))
S_n_CI_high = zeros(M, length(σ_values), length(N_values))

for i in eachindex(σ_values)
    for j in eachindex(N_values)
        for m in 1:M
            sample_X = rand(F_X, N_values[j])
            sample_U = rand(U, N_values[j])

            sample_Y = (-1).^(sample_X) * σ_values[i] .+ sample_U

            B_n = B(collect(1:10), sample_Y, sample_X)
            B_n_vars = B_var(collect(1:10), sample_Y, sample_X)
            S_n_values[m, i, j] = σ_hat(B_n)
            S_n_vars[m, i, j] = σ_hat_var(collect(1:10), sample_Y, sample_X, B=B_n, B_var=B_n_vars)

            S_n_CI_low[m, i, j] = S_n_values[m, i, j] - Φ_inv(1 - α/2) * sqrt(S_n_vars[m, i, j] / N_values[j])
            S_n_CI_high[m, i, j] = S_n_values[m, i, j] + Φ_inv(1 - α/2) * sqrt(S_n_vars[m, i, j] / N_values[j])
        end
    end
end

mean_S_n = mean(S_n_values, dims=1) 
mean_S_n = dropdims(mean_S_n; dims=1)
mean_S_n_vars = mean(S_n_vars, dims=1)
mean_S_n_vars = dropdims(mean_S_n_vars; dims=1)
mean_S_n_CI_low = mean(S_n_CI_low, dims=1)
mean_S_n_CI_low = dropdims(mean_S_n_CI_low; dims=1)
mean_S_n_CI_high = mean(S_n_CI_high, dims=1)
mean_S_n_CI_high = dropdims(mean_S_n_CI_high; dims=1)

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values, mean_S_n[:,j], ribbon=(mean_S_n_CI_high[:,j] - mean_S_n[:,j], mean_S_n[:,j] - mean_S_n_CI_low[:,j]), fillalpha=0.2, label = "N = $(N_values[j])")
end
plot!(σ_values, σ_values, label = "True")
xlabel!(plt, "σ")
ylabel!(plt, "Mean Sₙ")
title!(plt, "Mean Sₙ by σ for different N")
savefig(plt, "PSET2/mean_S_n.pdf")

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values, mean_S_n_vars[:,j], label = "N = $(N_values[j])")
end
xlabel!(plt, "σ")
ylabel!(plt, "Mean Sₙ")
title!(plt, "Mean Sₙ by σ for different N")
savefig(plt, "PSET2/mean_S_n_vars.pdf")


display(S_n_CI_low[:,end,end] .<= σ_values[end])
display(σ_values[end] .<= S_n_CI_high[:,end,end])

display(mean(S_n_CI_low[:,end,end] .<= σ_values[end]))
display(mean(σ_values[end] .<= S_n_CI_high[:,end,end]))


true_in_CI = [mean((S_n_CI_low[:,i,j] .<= σ_values[i]) .&& (σ_values[i] .<= S_n_CI_high[:,i,j])) for i in eachindex(σ_values), j in eachindex(N_values)]
display(true_in_CI)

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values, true_in_CI[:,j], label = "N = $(N_values[j])")
end
xlabel!(plt, "σ")
ylabel!(plt, "True in CI")
title!(plt, "True in CI by σ for different N")
hline!([[1 - α]], label = "1 - α", color = :black, style=:dash)
savefig(plt, "PSET2/true_in_CI.pdf")