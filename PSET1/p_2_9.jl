using Pkg
Pkg.activate(".")

using LinearAlgebra, StatsBase, Statistics, Plots, StatsPlots, Random, Distributions, DataFrames, StatFiles

# Number of MC simulations
M = 10_000

##### Key statistical functions

function Φ(x)
    return cdf(Normal(), x)
end

function Φ_inv(x)
    return quantile(Normal(), x)
end

function max_wald_cv(α::Real, k::Integer)
    dist = rand(MvNormal(zeros(k), I), 10_000)
    sup = [maximum(abs.(dist[:, i])) for i in 1:10_000]
    return quantile(sup, 1 - α)
end 

function kolmogorov_cv(α::Real)
    return sqrt(-1/2 * log(α / 2))
end

function B(y::Real, Y::AbstractVector{<:Real})
    return mean(Int.(Y .<= y))
end 

function B_var(y::Real, Y::AbstractVector{<:Real})
    return 1/length(Y) * B(y, Y) * (1 - B(y, Y))
end 

function B_std(y::Real, Y::AbstractVector{<:Real})
    return sqrt(B_var(y, Y))
end 

function B_cov(y_1::Real, y_2::Real, Y::AbstractVector{<:Real})
    return 1/length(Y) * (B(min(y_1, y_2), Y) - B(y_1, Y) * B(y_2, Y))
end 

function V(y::AbstractVector{<:Real}, Y::AbstractVector{<:Real})
    @assert issorted(y) "Input vector y must be sorted."

    vcov = zeros(length(y), length(y))
    for i in eachindex(y)
        for j in eachindex(y)
            if i == j
                vcov[i, j] = B_var(y[i], Y)
            else
                vcov[i, j] = B_cov(y[i], y[j], Y)
            end
        end
    end
    return Symmetric(vcov)
end

function S(y::AbstractVector{<:Real}, Y::AbstractVector{<:Real})
    return sqrt(V(y, Y))
end

function T_stat(ν_0::Real, B_est::Real, σ_est::Real)
    return (B_est - ν_0) / (σ_est)
end

function T_test(α::Real, ν_0::Real, B_est::Real, σ_est::Real)
    return abs(T_stat(ν_0, B_est, σ_est)) > Φ_inv(1 - α/2)
end

function max_wald_stat(ν_0::AbstractVector{<:Real}, B_est::AbstractVector{<:Real}, S::AbstractMatrix{<:Real})
    return maximum(abs.(S \ (ν_0 - B_est)))
end

function max_wald_test(α::Real, ν_0::AbstractVector{<:Real}, B_est::AbstractVector{<:Real}, S::AbstractMatrix{<:Real})
    return max_wald_stat(ν_0, B_est, S) > max_wald_cv(α, length(ν_0))
end

function kolmogorov_stat(ν_0::AbstractVector{<:Real}, B_est::AbstractVector{<:Real}, N::Integer)
    return sqrt(N) * maximum(abs.(ν_0 - B_est))
end

function kolmogorov_test(α::Real, ν_0::AbstractVector{<:Real}, B_est::AbstractVector{<:Real}, N::Integer)
    return kolmogorov_stat(ν_0, B_est, N) > kolmogorov_cv(α)
end

##### Loading the data

data = DataFrame(load("cps.dta"))
Y = data.wage
Y = collect(skipmissing(Y))
Y = Float64.(Y)

##### (b): The 1d MC simulation
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

plots = []
for j in eachindex(N_values)
    for i in eachindex(y_values)
        p = histogram(B_values[:, i, j], label="", normalize=:pdf, bins = LinRange(minimum(B_values[:, i, :]), maximum(B_values[:, i, :]), 150))
        vline!([mean(B_values[:, i, j])], label="Mean", style=:dash)
        vline!([B(y_values[i], Y)], label="True")
        plot!(x -> pdf(Normal(mean(B_values[:, i, j]), std(B_values[:, i, j])), x), label="Normal Approximation")
        push!(plots, p)
    end
end

plot(plots..., layout=(length(y_values), length(N_values)), size=(1000, 1000))
savefig("PSET1/hist_1d_mc.pdf")

##### (c): The 1d MC power curves
α = 0.1
ν_range = LinRange(0, 1, 1000)
plots = []
for j in eachindex(N_values)
    for i in eachindex(y_values)
        p = plot(ν_range, [mean([T_test(α, ν, B_values[m, i, j], σ_values[m, i, j]) for m in 1:M]) for ν in ν_range], label="")
        vline!([B(y_values[i], Y)], label="True")
        hline!([α], label = "α", color = :black, style=:dash)
        push!(plots, p)
    end
end

plot(plots..., layout=(length(y_values), length(N_values)), size=(1000, 1000))
savefig("PSET1/T_test_1d_mc.pdf")

##### (e-f): The size of multidimensional Wald-max test and Kolmogorov test
K = collect(1:30)
y_k = []
for k in K
    push!(y_k, [quantile(Y, q) for q in collect(LinRange(0, 1, k+2)[2:(end-1)])])
end

α = 0.1
N_values = [500, 1000, 4000]
test_results_wald = zeros(length(K), length(N_values), M)
test_results_kolmogorov = zeros(length(K), length(N_values), M)

Y_supp = sort(unique(Y))
for k in eachindex(K)
    ν_0 = [B(y, Y) for y in y_k[k]]
    for j in eachindex(N_values)
        for m in 1:M
            sample = rand(Y, N_values[j])
            B_est = [B(y, sample) for y in y_k[k]]
            S_est = S(y_k[k], sample)
            try
                test_results_wald[k, j, m] = max_wald_test(α, ν_0, B_est, S_est)
            catch SingularException
                test_results_wald[k, j, m] = true
            end
            test_results_kolmogorov[k, j, m] = kolmogorov_test(α, ν_0, B_est, N_values[j])
        end
    end
end

test_sizes_wald = [mean(test_results_wald[k, j, :]) for k in eachindex(K), j in eachindex(N_values)]
test_sizes_kolmogorov = [mean(test_results_kolmogorov[k, j, :]) for k in eachindex(K), j in eachindex(N_values)]
display(test_sizes_wald)
display(test_sizes_kolmogorov)


plots = []
for j in eachindex(N_values)
    p = plot(K, test_sizes_wald[:, j], label="Wald")
    plot!(K, test_sizes_kolmogorov[:, j], label="Kolmogorov")
    hline!([α], label = "α", color = :black, style=:dash)
    push!(plots, p)
end

plot(plots..., layout=(length(N_values)), size=(1000, 1000))
savefig("PSET1/wald_kolmogorov_test_sizes.pdf")
