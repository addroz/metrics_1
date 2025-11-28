using Pkg
Pkg.activate(".")

using LinearAlgebra, StatsBase, Statistics, Plots, StatsPlots, Random, Distributions, DataFrames, StatFiles, LaTeXStrings, KernelDensity, FixedEffectModels

# Helper functions
function weighted_mean_skipmissing(vals, wts)
    mask = .!ismissing.(vals) .& .!ismissing.(wts)
    return mean(Float64.(vals[mask]), weights(Float64.(wts[mask])))
end

mean_skipmissing(vals) = mean(Float64.(vals[.!ismissing.(vals)]))

lead_if_consecutive(vals, years) = [i == length(vals) || years[i + 1] - years[i] != 1 ? missing : vals[i + 1] for i in 1:length(vals)]

function create_event_study_plot(weighted_panel, unweighted_panel, var, label; base_year=2005, treatment_year=2005.5)
    weighted_treated = combine(groupby(filter(row -> row.treated, weighted_panel), :year), var => mean_skipmissing => :mean)
    weighted_control = combine(groupby(filter(row -> !row.treated, weighted_panel), :year), var => mean_skipmissing => :mean)
    unweighted_treated = combine(groupby(filter(row -> row.treated, unweighted_panel), :year), var => mean_skipmissing => :mean)
    unweighted_control = combine(groupby(filter(row -> !row.treated, unweighted_panel), :year), var => mean_skipmissing => :mean)
    
    weighted_diff = weighted_treated.mean .- weighted_control.mean
    unweighted_diff = unweighted_treated.mean .- unweighted_control.mean
    
    base_year_idx = findfirst(weighted_treated.year .== base_year)
    weighted_did = weighted_diff .- weighted_diff[base_year_idx]
    unweighted_did = unweighted_diff .- unweighted_diff[base_year_idx]
    
    p = plot(weighted_treated.year, weighted_did, 
        label="Weighted", linewidth=2, marker=:circle,
        xlabel="Year", ylabel="Difference-in-Differences (from $base_year)", title=label,
        legend=:best, size=(800, 500))
    plot!(p, unweighted_treated.year, unweighted_did, 
        label="Unweighted", linewidth=2, marker=:square, linestyle=:dash)
    hline!(p, [0], label="", linestyle=:solid, color=:gray, linewidth=1, alpha=0.5)
    vline!(p, [treatment_year], label="", linestyle=:dot, color=:black, linewidth=1)
    
    return p
end

# Load data
data = DataFrame(load("tenncare-micro.dta"))
data_agg = DataFrame(load("tenncare.dta"))

# Construct weighted aggregate panel
agg_panel = combine(groupby(data, [:year, :state])) do df
    DataFrame(
        any_public = weighted_mean_skipmissing(df.any_public, df.hinswt),
        any_empl = weighted_mean_skipmissing(df.any_empl, df.hinswt),
        working = weighted_mean_skipmissing(df.working, df.wtsupp),
        hrs_lw_lt20 = weighted_mean_skipmissing(df.hrs_lw_lt20, df.wtsupp),
        hrs_lw_2035 = weighted_mean_skipmissing(df.hrs_lw_2035, df.wtsupp),
        hrs_lw_ge35 = weighted_mean_skipmissing(df.hrs_lw_ge35, df.wtsupp),
        hrs_lw_ge20 = weighted_mean_skipmissing(df.hrs_lw_ge20, df.wtsupp)
    )
end

# Construct unweighted aggregate panel
agg_panel_unweighted = combine(groupby(data, [:year, :state])) do df
    DataFrame(
        any_public = mean_skipmissing(df.any_public),
        any_empl = mean_skipmissing(df.any_empl),
        working = mean_skipmissing(df.working),
        hrs_lw_lt20 = mean_skipmissing(df.hrs_lw_lt20),
        hrs_lw_2035 = mean_skipmissing(df.hrs_lw_2035),
        hrs_lw_ge35 = mean_skipmissing(df.hrs_lw_ge35),
        hrs_lw_ge20 = mean_skipmissing(df.hrs_lw_ge20)
    )
end

# Process both panels
for df in [agg_panel, agg_panel_unweighted]
    sort!(df, [:year, :state])
    transform!(groupby(df, :state), 
        [:any_public, :year] => lead_if_consecutive => :any_public,
        [:any_empl, :year] => lead_if_consecutive => :any_empl)
end

filter!(row -> row.year >= 2000, agg_panel)
sort!(data_agg, [:year, :state])

# Event Study Plots
agg_panel_filtered = filter(row -> 2000 <= row.year <= 2007, agg_panel)
agg_panel_unweighted_filtered = filter(row -> 2000 <= row.year <= 2007, agg_panel_unweighted)

agg_panel_filtered.treated = agg_panel_filtered.state .== "Tennessee"
agg_panel_unweighted_filtered.treated = agg_panel_unweighted_filtered.state .== "Tennessee"

vars_to_plot = [:any_public, :any_empl, :working, :hrs_lw_lt20, :hrs_lw_2035, :hrs_lw_ge35, :hrs_lw_ge20]
var_labels = ["Any Public Insurance", "Any Employment-Based Insurance", "Working", 
              "Hours Worked < 20", "Hours Worked 20-35", "Hours Worked ≥ 35", "Hours Worked ≥ 20"]

for (i, var) in enumerate(vars_to_plot)
    p = create_event_study_plot(agg_panel_filtered, agg_panel_unweighted_filtered, var, var_labels[i])
    savefig(p, "PSET5/event_study_$(var).pdf")
end

# TWFE Difference-in-Differences Regression
data_filtered = filter(row -> 2000 <= row.year <= 2007, data)
data_filtered.treated = data_filtered.state .== "Tennessee"
data_filtered.post = data_filtered.year .>= 2006

twfe_model = reg(data_filtered, @formula(working ~ treated & post + fe(state) + fe(year)))

println("\n" * "="^80)
println("TWFE DiD Regression: working ~ treated × post + state FE + year FE")
println("="^80)
display(twfe_model)
println("="^80)

