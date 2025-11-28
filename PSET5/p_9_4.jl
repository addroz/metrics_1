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

function create_event_study_plot(weighted_panel, unweighted_panel, microdata, var, label; base_year=2005, shift_micro=false, year_min=2000, year_max=2007)
    # Shift microdata years by 1 if requested (for lagged variables)
    microdata_shifted = copy(microdata)
    microdata_shifted.year = shift_micro ? microdata.year .- 1 : microdata.year
    
    # Filter by date range after shifting
    weighted_panel_filtered = filter(row -> year_min <= row.year <= year_max, weighted_panel)
    unweighted_panel_filtered = filter(row -> year_min <= row.year <= year_max, unweighted_panel)
    microdata_filtered = filter(row -> year_min <= row.year <= year_max, microdata_shifted)

    weighted_treated = combine(groupby(filter(row -> row.treated, weighted_panel_filtered), :year), var => mean_skipmissing => :mean)
    weighted_control = combine(groupby(filter(row -> !row.treated, weighted_panel_filtered), :year), var => mean_skipmissing => :mean)
    
    unweighted_agg_treated = combine(groupby(filter(row -> row.treated, unweighted_panel_filtered), :year), var => mean_skipmissing => :mean)
    unweighted_agg_control = combine(groupby(filter(row -> !row.treated, unweighted_panel_filtered), :year), var => mean_skipmissing => :mean)
    
    micro_treated = combine(groupby(filter(row -> row.treated, microdata_filtered), :year), var => mean_skipmissing => :mean)
    micro_control = combine(groupby(filter(row -> !row.treated, microdata_filtered), :year), var => mean_skipmissing => :mean)
    
    weighted_diff = weighted_treated.mean .- weighted_control.mean
    unweighted_agg_diff = unweighted_agg_treated.mean .- unweighted_agg_control.mean
    micro_diff = micro_treated.mean .- micro_control.mean
    
    base_year_idx = findfirst(weighted_treated.year .== base_year)
    weighted_did = weighted_diff .- weighted_diff[base_year_idx]
    unweighted_agg_did = unweighted_agg_diff .- unweighted_agg_diff[base_year_idx]
    micro_did = micro_diff .- micro_diff[base_year_idx]
    
    p = plot(weighted_treated.year, weighted_did, 
        label="Weighted Aggregated", linewidth=2, marker=:circle,
        xlabel="Year", ylabel="Difference-in-Differences (from $base_year)", title=label,
        legend=:best, size=(800, 500))
    plot!(p, unweighted_agg_treated.year, unweighted_agg_did, 
        label="Unweighted Aggregated", linewidth=2, marker=:square, linestyle=:dash)
    plot!(p, micro_treated.year, micro_did, 
        label="Unaggregated Micro", linewidth=2, marker=:diamond, linestyle=:dot)
    hline!(p, [0], label="", linestyle=:solid, color=:gray, linewidth=1, alpha=0.5)
    vline!(p, [base_year + 0.5], label="", linestyle=:dot, color=:black, linewidth=1)
    
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
agg_panel.treated = agg_panel.state .== "Tennessee"
agg_panel_unweighted.treated = agg_panel_unweighted.state .== "Tennessee"
data.treated = data.state .== "Tennessee"

vars_to_plot = [:any_public, :any_empl, :working, :hrs_lw_lt20, :hrs_lw_2035, :hrs_lw_ge35, :hrs_lw_ge20]
var_labels = ["Any Public Insurance", "Any Employment-Based Insurance", "Working", 
              "Hours Worked < 20", "Hours Worked 20-35", "Hours Worked ≥ 35", "Hours Worked ≥ 20"]

for (i, var) in enumerate(vars_to_plot)
    shift_micro = var in [:any_public, :any_empl]
    p = create_event_study_plot(agg_panel, agg_panel_unweighted, data, var, var_labels[i], shift_micro=shift_micro)
    savefig(p, "PSET5/event_study_$(var).pdf")
end

# TWFE Difference-in-Differences Regression
data_filtered = filter(row -> 2005 <= row.year <= 2006, data)
data_filtered.treated = data_filtered.state .== "Tennessee"
data_filtered.post = data_filtered.year .>= 2006

twfe_model = reg(data_filtered, @formula(working ~ treated & post + fe(state) + fe(year)))

println("\n" * "="^80)
println("TWFE DiD Regression: working ~ treated × post + state FE + year FE")
println("="^80)
display(twfe_model)
println("="^80)

