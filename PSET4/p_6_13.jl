using Pkg
Pkg.activate(".")

using LinearAlgebra, StatsBase, Statistics, Plots, StatsPlots, Random, Distributions, DataFrames, StatFiles, LaTeXStrings, KernelDensity, Econometrics

##### Key statistical functions
function basic_OLS(X::Matrix{<:Real}, Y::Vector{<:Real})
    return inv(X' * X) * X' * Y
end

function V_OLS_HC0(X::Matrix{<:Real}, Y::Vector{<:Real}; B::Vector{<:Real} = basic_OLS(X, Y))
    N = length(Y)
    U = Y - X * B
    U_dg = diagm(U.^2)
    return N * inv(X' * X) * X' * U_dg * X * inv(X' * X)
end

function V_OLS_HC1(X::Matrix{<:Real}, Y::Vector{<:Real}; B::Vector{<:Real} = basic_OLS(X, Y))
    N = length(Y)
    U = Y - X * B
    U_dg = diagm(U.^2)
    return (N^2 / (N - size(X, 2))) * inv(X' * X) * X' * U_dg * X * inv(X' * X)
end

function V_OLS_hom(X::Matrix{<:Real}, Y::Vector{<:Real}; B::Vector{<:Real} = basic_OLS(X, Y))
    U = Y - X * B
    return mean(U.^2) * inv(X' * X)
end

data = DataFrame(load("cps.dta"))

data = filter(row -> !ismissing(row.wage) && !ismissing(row.sex), data)
data.sex = Int.(data.sex .== "Female")

data_filtered = data[data.state .== "Illinois", :]
data_filtered = data_filtered[data_filtered.age .== 35, :]
data_filtered = data_filtered[data_filtered.educ .== "High school diploma or equivalent", :]

X = [ones(size(data_filtered, 1)) data_filtered.sex]
Y = Float64.(log.(data_filtered.wage))

display(length(Y))

B = basic_OLS(X, Y)
display(B)

V = V_OLS_HC0(X, Y, B=B)
S = sqrt.(diag(V) ./ length(Y))
display(S)

V = V_OLS_HC1(X, Y, B=B)
S = sqrt.(diag(V) ./ length(Y))
display(S)

model = fit(EconometricModel, @formula(log(wage) ~ sex), data_filtered, vce=HC1)
display(model)

#model = fit(EconometricModel, @formula(log(wage) ~ sex + educ), data)
#display(model)

#model = fit(EconometricModel, @formula(log(wage) ~ sex + educ + exper), data)
#display(model)

#model = fit(EconometricModel, @formula(log(wage) ~ sex + educ + exper + tenure), data)
#display(model)