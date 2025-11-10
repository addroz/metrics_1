using Pkg
Pkg.activate(".")

using LinearAlgebra, StatsBase, Statistics, Plots, StatsPlots, Random, Distributions, DataFrames, StatFiles, LaTeXStrings, KernelDensity, Econometrics

##### Key statistical functions
function OLS(X::Matrix{<:Real}, Y::Vector{<:Real})
    return inv(X' * X) * X' * Y
end

function Φ(x)
    return cdf(Normal(), x)
end

function Φ_inv(x)
    return quantile(Normal(), x)
end

function cv_normal(α::Real)
    return Φ_inv(1 - α/2)
end

function cv_t(α::Real, df::Integer)
    return quantile(TDist(df), 1 - α/2)
end

function cv_χ2(α::Real, df::Integer)
    return quantile(Chisq(df), 1 - α)
end

function V_OLS_HC0(X::Matrix{<:Real}, Y::Vector{<:Real}; B::Vector{<:Real} = OLS(X, Y))
    N = length(Y)
    U = Y - X * B
    U_dg = diagm(U.^2)
    return N * inv(X' * X) * X' * U_dg * X * inv(X' * X)
end

function V_OLS_HC1(X::Matrix{<:Real}, Y::Vector{<:Real}; B::Vector{<:Real} = OLS(X, Y))
    N = length(Y)
    U = Y - X * B
    U_dg = diagm(U.^2)
    return (N^2 / (N - size(X, 2))) * inv(X' * X) * X' * U_dg * X * inv(X' * X)
end

function V_OLS_HOM(X::Matrix{<:Real}, Y::Vector{<:Real}; B::Vector{<:Real} = OLS(X, Y))
    U = Y - X * B
    return mean(U.^2) * inv(X' * X)
end

function T_stat(β_0::Real, B::Real, σ::Real)
    return (B - β_0) / σ
end

function T_test(α::Real, β_0::Real, B::Real, σ::Real)
    return abs(T_stat(β_0, B, σ)) > cv_normal(α)
end

function P_value_t(β_0::Real, B::Real, σ::Real)
    return 2 * (1 - Φ(abs(T_stat(β_0, B, σ))))
end

##### Some helper functions
function create_fe_matrix(X::AbstractVector{<:Any})
    X_vals = unique(X)
    X_base = X_vals[1]
    X_fe = zeros(length(X), length(X_vals) - 1)
    for i in 2:length(X_vals)
        X_fe[:, i-1] = Int.(X .== X_vals[i])
    end
    return X_fe, X_base
end

α = 0.05

data = DataFrame(load("cps.dta"))

data = filter(row -> !ismissing(row.wage) && !ismissing(row.sex) && !ismissing(row.age) && !ismissing(row.educ) && !ismissing(row.state), data)
data.sex = Int.(data.sex .== "Female")
data.age = Int.(data.age)


#### (a)

data_filtered = data[data.state .== "Illinois", :]
data_filtered = data_filtered[data_filtered.age .== 35, :]
data_filtered = data_filtered[data_filtered.educ .== "High school diploma or equivalent", :]

Y = Float64.(log.(data_filtered.wage))
X = [data_filtered.sex ones(size(data_filtered, 1))]

display(length(Y))

B = OLS(X, Y)
display(B)

V_HOM = V_OLS_HOM(X, Y, B=B)
S_HOM = sqrt.(diag(V_HOM) ./ length(Y))
display(S_HOM)

V_HC0 = V_OLS_HC0(X, Y, B=B)
S_HC0 = sqrt.(diag(V_HC0) ./ length(Y))
display(S_HC0)

V_HC1 = V_OLS_HC1(X, Y, B=B)
S_HC1 = sqrt.(diag(V_HC1) ./ length(Y))
display(S_HC1)
ci_HOM = (B[1] - cv_normal(α) * sqrt(V_HOM[1,1] / length(Y)), B[1] + cv_normal(α) * sqrt(V_HOM[1,1] / length(Y)))
ci_HC0 = (B[1] - cv_normal(α) * sqrt(V_HC0[1,1] / length(Y)), B[1] + cv_normal(α) * sqrt(V_HC0[1,1] / length(Y)))
ci_HC1 = (B[1] - cv_normal(α) * sqrt(V_HC1[1,1] / length(Y)), B[1] + cv_normal(α) * sqrt(V_HC1[1,1] / length(Y)))
ci_T = (B[1] - cv_t(α, length(Y) - size(X, 2)) * sqrt(V_HOM[1,1] / length(Y)), B[1] + cv_t(α, length(Y) - size(X, 2)) * sqrt(V_HOM[1,1] / length(Y)))

display(ci_HOM)
display(ci_HC0)
display(ci_HC1)
display(ci_T)


#### (b)
Y = Float64.(log.(data.wage))
X_educ, X_educ_base = create_fe_matrix(data.educ)
X_state, X_state_base = create_fe_matrix(data.state)
X = [data.sex data.age data.age.^2 X_educ X_state ones(size(data, 1))]
display(size(Y))
display(size(X))
display(X_educ_base)
display(X_state_base)

B = OLS(X, Y)
display(B[1:3])

V_HOM = V_OLS_HOM(X, Y, B=B)
V_HC0 = V_OLS_HC0(X, Y, B=B)
V_HC1 = V_OLS_HC1(X, Y, B=B)

se_sex_HOM = sqrt(V_HOM[1,1] / length(Y))
se_sex_HC0 = sqrt(V_HC0[1,1] / length(Y))
se_sex_HC1 = sqrt(V_HC1[1,1] / length(Y))

display(se_sex_HOM)
display(se_sex_HC0)
display(se_sex_HC1)

ci_sex_HOM = (B[1] - cv_normal(α) * se_sex_HOM, B[1] + cv_normal(α) * se_sex_HOM)
ci_sex_HC0 = (B[1] - cv_normal(α) * se_sex_HC0, B[1] + cv_normal(α) * se_sex_HC0)
ci_sex_HC1 = (B[1] - cv_normal(α) * se_sex_HC1, B[1] + cv_normal(α) * se_sex_HC1)

display(P_value_t(0, B[1], se_sex_HOM))
display(P_value_t(0, B[1], se_sex_HC0))
display(P_value_t(0, B[1], se_sex_HC1))

display(ci_sex_HOM)
display(ci_sex_HC0)
display(ci_sex_HC1)