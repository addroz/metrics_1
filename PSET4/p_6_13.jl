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
    inv_X_X = inv(X' * X)
    N = length(Y)
    U = Y - X * B
    U_dg = diagm(U.^2)
    return N * inv_X_X * X' * U_dg * X * inv_X_X
end

function V_OLS_HC1(X::Matrix{<:Real}, Y::Vector{<:Real}; B::Vector{<:Real} = OLS(X, Y))
    inv_X_X = inv(X' * X)
    N = length(Y)
    U = Y - X * B
    U_dg = diagm(U.^2)
    return (N^2 / (N - size(X, 2))) * inv_X_X * X' * U_dg * X * inv_X_X
end

function leverage(X::Matrix{<:Real}; inv_X_X::Matrix{<:Real} = inv(X' * X))
    return diag(X * inv_X_X * X')
end

function V_OLS_HC3(X::Matrix{<:Real}, Y::Vector{<:Real}; B::Vector{<:Real} = OLS(X, Y))
    inv_X_X = inv(X' * X)
    N = length(Y)
    U = Y - X * B
    U_dg = diagm(U.^2 ./ (1 .- leverage(X; inv_X_X=inv_X_X)).^2)
    return N * inv_X_X * X' * U_dg * X * inv_X_X
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

function T_test_t(α::Real, β_0::Real, B::Real, σ::Real, df::Integer)
    return abs(T_stat(β_0, B, σ)) > cv_t(α, df)
end

function P_value_normal(β_0::Real, B::Real, σ::Real)
    return 2 * (1 - Φ(abs(T_stat(β_0, B, σ))))
end

function P_value_t(β_0::Real, B::Real, σ::Real, df::Integer)
    return 2 * (1 - cdf(TDist(df), abs(T_stat(β_0, B, σ))))
end

function P_value_χ2(T::Real, df::Integer)
    return 1 - cdf(Chisq(df), T)
end

##### Some helper functions
function create_fe_matrix(X::AbstractVector{<:Any}; create_base::Bool = true)
    X_vals = unique(X)
    if create_base
        X_base = X_vals[1]
    else
        X_base = nothing
    end
    X_fe = zeros(length(X), length(X_vals) - Int(create_base))
    for i in eachindex(X_vals)
        if i == 1 && create_base
            continue
        end
        X_fe[:, i - Int(create_base)] = [Int(X[j] == X_vals[i]) for j in eachindex(X)]
    end
    return X_fe, X_base
end

α = 0.1

data = DataFrame(load("cps.dta"))

data = filter(row -> !ismissing(row.wage) && !ismissing(row.sex) && !ismissing(row.age) && !ismissing(row.educ) && !ismissing(row.state), data)
data.sex = Int.(data.sex .== "Female")
data.age = Int.(data.age)


#### (a)
display("Part (a)")
data_filtered = data[data.state .== "Illinois", :]
data_filtered = data_filtered[data_filtered.age .== 35, :]
data_filtered = data_filtered[data_filtered.educ .== "High school diploma or equivalent", :]

Y = Float64.(log.(data_filtered.wage))
X = [data_filtered.sex ones(size(data_filtered, 1))]

display(length(Y))

B = OLS(X, Y)
display(B)

V_HOM = V_OLS_HOM(X, Y, B=B)
V_HC0 = V_OLS_HC0(X, Y, B=B)
V_HC1 = V_OLS_HC1(X, Y, B=B)

ci_HOM = (B[1] - cv_normal(α) * sqrt(V_HOM[1,1] / length(Y)), B[1] + cv_normal(α) * sqrt(V_HOM[1,1] / length(Y)))
ci_HC0 = (B[1] - cv_normal(α) * sqrt(V_HC0[1,1] / length(Y)), B[1] + cv_normal(α) * sqrt(V_HC0[1,1] / length(Y)))
ci_HC1 = (B[1] - cv_normal(α) * sqrt(V_HC1[1,1] / length(Y)), B[1] + cv_normal(α) * sqrt(V_HC1[1,1] / length(Y)))
ci_T = (B[1] - cv_t(α, length(Y) - size(X, 2)) * sqrt(V_HOM[1,1] / length(Y)), B[1] + cv_t(α, length(Y) - size(X, 2)) * sqrt(V_HOM[1,1] / length(Y)))

display(ci_HOM)
display(ci_HC1)
display(ci_T)

#### (b)
display("Part (b)")
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
V_HC1 = V_OLS_HC1(X, Y, B=B)
V_HC3 = V_OLS_HC3(X, Y, B=B)

se_sex_HOM = sqrt(V_HOM[1,1] / length(Y))
se_sex_HC1 = sqrt(V_HC1[1,1] / length(Y))
se_sex_HC3 = sqrt(V_HC3[1,1] / length(Y))

# HOM & T
display("HOM & T")
display(se_sex_HOM)
display(T_stat(0, B[1], se_sex_HOM))
display(P_value_t(0, B[1], se_sex_HOM, length(Y) - size(X, 2)))

# HC1 & Normal
display("HC1 & Normal")
display(se_sex_HC1)
display(T_stat(0, B[1], se_sex_HC1))
display(P_value_normal(0, B[1], se_sex_HC1))

# HC3 & Normal
display("HC3 & Normal")
display(se_sex_HC3)
display(T_stat(0, B[1], se_sex_HC3))
display(P_value_normal(0, B[1], se_sex_HC3))

#### (c)
display("Part (c)")
data_filtered = data[data.state .== "Illinois", :]
data_filtered = data_filtered[data_filtered.educ .== "High school diploma or equivalent", :]
Y = Float64.(log.(data_filtered.wage))
X = [data_filtered.sex data_filtered.age data_filtered.age.^2 ones(size(data_filtered, 1))]
display(size(Y))
display(size(X))

B = OLS(X, Y)
display(B[1:3])

V_HOM = V_OLS_HOM(X, Y, B=B)
V_HC1 = V_OLS_HC1(X, Y, B=B)
V_HC3 = V_OLS_HC3(X, Y, B=B)

se_sex_HOM = sqrt(V_HOM[1,1] / length(Y))
se_sex_HC1 = sqrt(V_HC1[1,1] / length(Y))
se_sex_HC3 = sqrt(V_HC3[1,1] / length(Y))

# HC1 & Normal
display("HC1 & Normal")
display(se_sex_HC1)
display(T_stat(0, B[1], se_sex_HC1))
display(P_value_normal(0, B[1], se_sex_HC1))


#### (d)
display("Part (d)")
Y = Float64.(log.(data.wage))
X_educ, X_educ_base = create_fe_matrix(data.educ)
sex_state = collect(zip(data.sex, data.state))
X_state, X_state_base = create_fe_matrix(data.state)
X_state_female = X_state .* data.sex
X = [X_state_female X_state data.sex data.age data.age.^2 X_educ ones(size(data, 1))]
display(X_state_base)

d_r = size(X_state_female, 2)
d_x = size(X, 2)

r = [I(d_r) zeros(d_r, d_x - d_r)]

B = OLS(X, Y)
V_HC1 = V_OLS_HC1(X, Y, B=B)


B_r = r * B
V_r = r * V_HC1 * r'

display(size(B_r))
display(size(V_r))

T_r = norm(sqrt(length(Y)) * inv(sqrt(V_r)) * B_r)^2
display(T_r)

display(cv_χ2(α, d_r))
display(P_value_χ2(T_r, d_r))
