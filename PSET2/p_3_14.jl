using Pkg
Pkg.activate(".")

using LinearAlgebra, StatsBase, Statistics, Plots, StatsPlots, Random, Distributions, DataFrames, StatFiles, LaTeXStrings, KernelDensity

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

function σ2_hat(B::AbstractVector{<:Real})
    return mean((B .- mean(B)).^2)
end

function ∇σ_hat(B::AbstractVector{<:Real})
    return [(B[i] - mean(B)) / (length(B) * σ_hat(B)) for i in eachindex(B)]
end

function ∇σ2_hat(B::AbstractVector{<:Real})
    return [2 * (B[i] - mean(B)) / length(B) for i in eachindex(B)]
end

function σ_hat_var(x::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real}; 
    B::AbstractVector{<:Real} = B(x, Y, X), 
    B_var::AbstractMatrix{<:Real} = B_var(x, Y, X))
    return ∇σ_hat(B)' * B_var * ∇σ_hat(B)
end

function σ2_hat_var(x::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real}; 
    B::AbstractVector{<:Real} = B(x, Y, X), 
    B_var::AbstractMatrix{<:Real} = B_var(x, Y, X))
    return ∇σ2_hat(B)' * B_var * ∇σ2_hat(B)
end

function tilde_σ2_hat(x::AbstractVector{<:Real}, Y::AbstractVector{<:Real}, X::AbstractVector{<:Real})
    B_n = B(x, Y, X)
    N = [length(Y[X .== x[i]]) for i in eachindex(x)]
    σ2_hat_n = σ2_hat(B_n)

    S2_n = [sum( Int64.(X .== x[i]) .* (Y .- B_n[i]).^2 ) / (N[i] - 1) for i in eachindex(x)]
    return σ2_hat_n - (length(B_n) - 1) / (length(B_n)^2) * sum(S2_n ./ N)
end

# Number of MC simulations
M = 100
α = 0.05
states = 10
data = DataFrame(load("cps.dta"))
U = data.wage
U = collect(skipmissing(U))
U = Float64.(U)

state = data.state

state_counts = countmap(state)
top_states = sort(collect(keys(state_counts)), by=x->-state_counts[x])[1:states]
data = data[in.(data.state, Ref(top_states)), :]

state_freqs = countmap(data.state)
sorted_states = sort(collect(keys(state_freqs)), by=x -> -state_freqs[x])
state_to_rank = Dict(s => r for (r, s) in enumerate(sorted_states))
data.state_rank = [state_to_rank[s] for s in data.state]
state_rank_counts = countmap(data.state_rank)

F_X = DiscreteNonParametric(collect(keys(state_rank_counts)), values(state_rank_counts) ./ sum(values(state_rank_counts)))


σ_values = collect(LinRange(0.0, 3.0, 50))
N_values = [Int64(round(length(U)/ 2)), Int64(round(length(U))), Int64(round(length(U) * 2))]

B_n_values = zeros(M, states, length(σ_values), length(N_values))
B_n_vars = zeros(M, states, length(σ_values), length(N_values))

S_n_values = zeros(M, length(σ_values), length(N_values))
S_n_vars = zeros(M, length(σ_values), length(N_values))
S_n_CI_low = zeros(M, length(σ_values), length(N_values))
S_n_CI_high = zeros(M, length(σ_values), length(N_values))

V_n_values = zeros(M, length(σ_values), length(N_values))
V_n_vars = zeros(M, length(σ_values), length(N_values))
V_n_CI_low = zeros(M, length(σ_values), length(N_values))
V_n_CI_high = zeros(M, length(σ_values), length(N_values))

tilde_V_n_values = zeros(M, length(σ_values), length(N_values))
tilde_V_n_vars = zeros(M, length(σ_values), length(N_values))
tilde_V_n_CI_low = zeros(M, length(σ_values), length(N_values))
tilde_V_n_CI_high = zeros(M, length(σ_values), length(N_values))

for i in eachindex(σ_values)
    for j in eachindex(N_values)
        for m in 1:M
            sample_X = rand(F_X, N_values[j])
            sample_U = rand(U, N_values[j])

            sample_Y = (-1).^(sample_X) * σ_values[i] .+ sample_U

            B_n = B(collect(1:states), sample_Y, sample_X)
            B_n_var = B_var(collect(1:states), sample_Y, sample_X)

            B_n_values[m, :, i, j] = B_n
            B_n_vars[m, :, i, j] = diag(B_n_var)

            S_n_values[m, i, j] = σ_hat(B_n)
            S_n_vars[m, i, j] = σ_hat_var(collect(1:states), sample_Y, sample_X, B=B_n, B_var=B_n_var)
            S_n_CI_low[m, i, j] = S_n_values[m, i, j] - Φ_inv(1 - α/2) * sqrt(S_n_vars[m, i, j] / N_values[j])
            S_n_CI_high[m, i, j] = S_n_values[m, i, j] + Φ_inv(1 - α/2) * sqrt(S_n_vars[m, i, j] / N_values[j])

            V_n_values[m, i, j] = σ2_hat(B_n)
            V_n_vars[m, i, j] = σ2_hat_var(collect(1:states), sample_Y, sample_X, B=B_n, B_var=B_n_var)
            V_n_CI_low[m, i, j] = V_n_values[m, i, j] - Φ_inv(1 - α/2) * sqrt(V_n_vars[m, i, j] / N_values[j])
            V_n_CI_high[m, i, j] = V_n_values[m, i, j] + Φ_inv(1 - α/2) * sqrt(V_n_vars[m, i, j] / N_values[j])

            tilde_V_n_values[m, i, j] = tilde_σ2_hat(collect(1:states), sample_Y, sample_X)
            tilde_V_n_vars[m, i, j] = V_n_vars[m, i, j]
            tilde_V_n_CI_low[m, i, j] = tilde_V_n_values[m, i, j] - Φ_inv(1 - α/2) * sqrt(tilde_V_n_vars[m, i, j] / N_values[j])
            tilde_V_n_CI_high[m, i, j] = tilde_V_n_values[m, i, j] + Φ_inv(1 - α/2) * sqrt(tilde_V_n_vars[m, i, j] / N_values[j])
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

mean_V_n = mean(V_n_values, dims=1)
mean_V_n = dropdims(mean_V_n; dims=1)
mean_V_n_vars = mean(V_n_vars, dims=1)
mean_V_n_vars = dropdims(mean_V_n_vars; dims=1)
mean_V_n_CI_low = mean(V_n_CI_low, dims=1)
mean_V_n_CI_low = dropdims(mean_V_n_CI_low; dims=1)
mean_V_n_CI_high = mean(V_n_CI_high, dims=1)
mean_V_n_CI_high = dropdims(mean_V_n_CI_high; dims=1)

mean_tilde_V_n = mean(tilde_V_n_values, dims=1)
mean_tilde_V_n = dropdims(mean_tilde_V_n; dims=1)
mean_tilde_V_n_vars = mean(tilde_V_n_vars, dims=1)
mean_tilde_V_n_vars = dropdims(mean_tilde_V_n_vars; dims=1)
mean_tilde_V_n_CI_low = mean(tilde_V_n_CI_low, dims=1)
mean_tilde_V_n_CI_low = dropdims(mean_tilde_V_n_CI_low; dims=1)
mean_tilde_V_n_CI_high = mean(tilde_V_n_CI_high, dims=1)
mean_tilde_V_n_CI_high = dropdims(mean_tilde_V_n_CI_high; dims=1)

#### Estimator's means and confidence bands

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values, mean_S_n[:,j], ribbon=(mean_S_n_CI_high[:,j] - mean_S_n[:,j], mean_S_n[:,j] - mean_S_n_CI_low[:,j]), fillalpha=0.2, label = "N = $(N_values[j])")
end
plot!(σ_values, σ_values, label = "True")
xlabel!(plt, L"$\sigma$")
title!(plt, L"$S_n$ by $\sigma$ for different N")
savefig(plt, "PSET2/mean_S_n_$(states).pdf")

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values.^2, mean_V_n[:,j], ribbon=(mean_V_n_CI_high[:,j] - mean_V_n[:,j], mean_V_n[:,j] - mean_V_n_CI_low[:,j]), fillalpha=0.2, label = "N = $(N_values[j])")
end
plot!(σ_values.^2, σ_values.^2, label = "True")
xlabel!(plt, L"$\sigma^2$")
title!(plt, L"$S_n^2$ by $\sigma^2$ for different N")
savefig(plt, "PSET2/mean_V_n_$(states).pdf")

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values.^2, mean_tilde_V_n[:,j], ribbon = (mean_tilde_V_n_CI_high[:,j] - mean_tilde_V_n[:,j], mean_tilde_V_n[:,j] - mean_tilde_V_n_CI_low[:,j]), fillalpha=0.2, label = "N = $(N_values[j])")
end
plot!(σ_values.^2, σ_values.^2, label = "True")
xlabel!(plt, L"$\sigma^2$")
title!(plt, L"$\tilde{S}_n^2$ by $\sigma^2$ for different N")
savefig(plt, "PSET2/mean_tilde_V_n_$(states).pdf")

##### CIs Test

true_in_CI = [mean((S_n_CI_low[:,i,j] .<= σ_values[i]) .&& (σ_values[i] .<= S_n_CI_high[:,i,j])) for i in eachindex(σ_values), j in eachindex(N_values)]
display(true_in_CI)

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values, true_in_CI[:,j], label = "N = $(N_values[j])")
end
xlabel!(plt, L"$\sigma$")
title!(plt, L"True $\sigma$ in CI by $\sigma$ for different N")
hline!([1 - α], label = L"1 - \alpha", color = :black, style=:dash)
savefig(plt, "PSET2/true_in_CI_$(states).pdf")

true_in_CI_V = [mean((V_n_CI_low[:,i,j] .<= σ_values[i].^2) .&& (σ_values[i].^2 .<= V_n_CI_high[:,i,j])) for i in eachindex(σ_values), j in eachindex(N_values)]
display(true_in_CI_V)

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values.^2, true_in_CI_V[:,j], label = "N = $(N_values[j])")
end
xlabel!(plt, L"$\sigma^2$")
title!(plt, L"True $\sigma^2$ in CI by $\sigma^2$ for different N")
hline!([1 - α], label = L"1 - \alpha", color = :black, style=:dash)
savefig(plt, "PSET2/true_in_CI_V_$(states).pdf")

true_in_CI_tilde_V = [mean((tilde_V_n_CI_low[:,i,j] .<= σ_values[i].^2) .&& (σ_values[i].^2 .<= tilde_V_n_CI_high[:,i,j])) for i in eachindex(σ_values), j in eachindex(N_values)]
display(true_in_CI_tilde_V)

plt = plot()
for j in 1:length(N_values)
    plot!(plt, σ_values.^2, true_in_CI_tilde_V[:,j], label = "N = $(N_values[j])")
end
xlabel!(plt, "σ")
title!(plt, L"True $\sigma^2$ in CI by $\sigma^2$ for different N")
hline!([1 - α], label = L"1 - \alpha", color = :black, style=:dash)
savefig(plt, "PSET2/true_in_CI_tilde_V_$(states).pdf")

##### (f): The χ^2 Test

χ_2_test_stat = zeros(M, length(N_values))
for j in eachindex(N_values)
    for m in 1:M
        P = zeros(states, states)

        for k in 1:states
            P[k, k] = ((states - 1) / states)^2 * B_n_vars[m, k, 1, j] + (1 / states)^2 * sum([B_n_vars[m, l, 1, j] for l in 1:states if l != k])
            
            for l in (k+1):states
                P[k, l] = (1 / states)^2 * (B_n_vars[m, k, 1, j] + B_n_vars[m, l, 1, j])
                P[l, k] = P[k, l]
            end
        end

        P = Symmetric(P)
        B_norm = B_n_values[m, :, 1, j] .- mean(B_n_values[m, :, 1, j])

        χ_2_test_stat[m, j] = (B_norm' * P * B_norm) / N_values[j]
    end

    density(χ_2_test_stat[:, j], label = "N = $(N_values[j])")
    savefig("PSET2/density_χ_2_test_stat_$(states)_$(N_values[j]).pdf")
end

plot(0:0.01:30, x -> pdf(Chisq(states), x), label = L"χ^2_m")
savefig("PSET2/density_χ_2_true_$(states).pdf")