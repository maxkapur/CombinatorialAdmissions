using Random
using Plots, Plots.Measures
using JuMP
using Ipopt
using LinearAlgebra
using Distributions: Beta, Uniform
using KernelDensity
using Dates: now
using DelimitedFiles
using Libdl
using StatsBase: median



const ν_fix = 1//10
const γ_fix = 5//10
const beta_fix = (15, 5)
const n_markets = 600
const n = 100
const q = 15



try
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)
    global const linearsolvername = "pardiso"
catch e
    @warn "Couldn't load Pardiso, using ma27 instead"
    global const linearsolvername = "ma27"
end

struct FailedToConvergeException <: Exception end


function admitresult_allapply(
    a::Vector{T},
    b::Vector{T},
    q::Int,
    γ::Real) where T<:Real

    n = length(b)

    mdl = Model(Ipopt.Optimizer)
    set_optimizer_attribute(mdl, "linear_solver", linearsolvername)
    set_silent(mdl)

    @variable(mdl, 0 ≤ y[1:n] ≤ 1)
    @variable(mdl, minby)

    @objective(mdl, Max, (1 - γ) * dot(a, y) + n * γ * minby)

    @constraint(mdl, Leonteif[i in 1:n], minby ≤ b[i] * y[i])
    @constraint(mdl, Capacity, sum(y) ≤ q)

    optimize!(mdl)

    return value.(y)
end


function admitresult_someapply(
    a::Vector{T},
    b::Vector{T},
    q::Int,
    γ::Real,
    x::Vector{Bool}) where {T<:Real}

    n = length(b)

    n_applicants = sum(x)

    if n_applicants < q
        return ones(n)
    end

    y_res = zeros(n)

    # Container for optimal admit policy wrt current applicants
    y_current = zeros(n)

    # Compute current admissions probabilities
    mdl = Model(Ipopt.Optimizer)
    set_optimizer_attribute(mdl, "linear_solver", linearsolvername)
    set_silent(mdl)

    @variable(mdl, 0 ≤ y[1:n])

    @variable(mdl, minby)
    # Dummy upper bound for minby when a certain student doesn't apply
    M = 2*maximum(b)

    # Only admit applicants
    for i in 1:n
        if x[i]
            set_upper_bound(y[i], 1)
        else
            fix(y[i], 0; force=true)
        end
    end

    @objective(mdl, Max, (1 - γ) * dot(a, y) + n_applicants * γ * minby)
    @constraint(mdl, Leonteif[i in 1:n], minby - b[i] * y[i] ≤ (x[i] ? 0 : M))
    @constraint(mdl, Capacity, sum(y) ≤ q)

    optimize!(mdl)

    y_current[:] = value.(y)

    # Warm start for subsequent problems
    for i in 1:n
        set_start_value(y[i], y_current[i])
    end

    # In subsequent problems, an extra student is applying
    @objective(mdl, Max, (1 - γ) * dot(a, y) + (n_applicants + 1) * γ * minby)

    for i in 1:n
        if x[i]
            # If student is already applying, then the relevant LP is the one solved above
            y_res[i] = y_current[i]
        else
            # Now we need to solve the modified LP in which y[i] can go to one
            unfix(y[i])
            set_lower_bound(y[i], 0)
            set_upper_bound(y[i], 1)
            set_normalized_rhs(Leonteif[i], 0)

            optimize!(mdl)
            
            y_res[i] = value(y[i])

            # Fix back
            fix(y[i], 0; force=true)
            set_normalized_rhs(Leonteif[i], M)
        end
    end

    return y_res
end






function equilibrate(
    a::Vector{T},
    b::Vector{T},
    q::Int,
    γ::Real,
    t::AbstractVector;
    nit = 10,
    tries = 3) where T<:Real

    n = length(b)
    y = admitresult_allapply(a, b, q, γ)

    x_seensofar = Set{Vector{Bool}}()
    x = zeros(Bool, n)
    x_next = zeros(Bool, n)

    f = zeros(n)

    for j in 1:tries
        println("  Attempt $j of $tries")
        x[:] = rand(Bool, n)

        for k in 1:nit
            print("    BR iteration $k of $nit: ")

            f[:] = admitresult_someapply(a, b, q, γ, x)
            x_next[:] = (f .≥ t)

            if all(x .== x_next)
                println("Equilibrium found")
                # Stationary point: equilibrium found
                return x, f, y
            elseif x_next in x_seensofar
                println("Cycled")
                # Point already tried, so we are cycling
                # Proceed to next j
                break
            else
                println("New BR")
                # Keep iterating
                push!(x_seensofar, x_next)
                x[:] = x_next
            end

        end
    end

    throw(FailedToConvergeException)
end



"""
    alphas(β, ν)

Generates parameters `α1` and `α2` for a beta distribution
having mean `β` and variance `ν`.
"""
function alphas(β::AbstractFloat, ν::Rational)::Tuple{Float64,Float64}
    α1 = β^2 * (1 - β) / ν - β
    α2 = (1 - β)^2 * β / ν - 1 + β
    return α1, α2
end





function experiment_tbar()
    ϵ = 1e-6

    # The range of possible means for a beta dist having νinv
    tbar_dist = Uniform(
        (1 - sqrt(1 - 4 * ν_fix)) / 2 + ϵ,
        (1 + sqrt(1 - 4 * ν_fix)) / 2 - ϵ,
    )

    tbar = rand(tbar_dist, n_markets)
    S = zeros(n_markets)

    # Allocate memory
    t = zeros(n)

    a = zeros(n)
    b = zeros(n)

    x = zeros(Bool, n)
    f = zeros(n)
    y = zeros(n)

    for m in 1:n_markets
        println("Market $m")

        dist = Beta(alphas(tbar[m], ν_fix)...)
        t[:] = rand(dist, n)

        a[:] = randexp(n)
        a[:] ./= sum(a)
        b[:] = randexp(n)
        b[:] ./= sum(b)

        try
            x[:], f[:], y[:] = equilibrate(a, b, q, γ_fix, t)
            S[m] = dot(x, y) / q
        catch e
            if e <: FailedToConvergeException
                println("  Failed to converge")
                S[m] = 2         # Obvious because S ∈ [0,1] by design
            else
                throw(e)
            end
        end
    end

    return tbar, S
end

@time tbar, S = experiment_tbar()
open("./results/$(now())_tbar_result.txt", "a") do io
    writedlm(io, [tbar S])
end






converged_idx = S .< 1.5
kdensity = kde((tbar[converged_idx], S[converged_idx]), boundary = ((-1, 2), (-1, 2)))

pl = plot(size(600, 350),
    xlabel = "Average risk aversion t̄",
    ylabel = "Stability of equilibrium S(x)",
    # xscale = :log10,
    xlim = (minimum(tbar) - 0.01, maximum(tbar) + 0.01),
    ylim = (minimum(S) - 0.02, 1.02),
    legend = nothing)

contourf!(pl, kdensity.x, kdensity.y, kdensity.density', color = :bamako, lw = 0, la = 0)

scatter!(pl, tbar[converged_idx], S[converged_idx], msw = 0, c = :white, alpha = 0.5, ms = 3, shape = :x)

annotate!(pl, [(minimum(tbar),
    minimum(S),
    text(
        "ν = $(ν_fix.num)/$(ν_fix.den), γ = $(γ_fix.num)/$(γ_fix.den)\n" *
        "$(sum(.!converged_idx))/$(length(S)) did not converge",
        :bottom,
        :left,
        :white,
        10)
)])

savefig(pl, "tbar-stability-density.pdf")
savefig(pl, "tbar-stability-density.png")














bins = range((1 - sqrt(1 - 4 * ν_fix)) / 2,
    (1 + sqrt(1 - 4 * ν_fix)) / 2,
    length = 21)

thresh = round(median(S[converged_idx]), digits=2)
efficient_idx = thresh .< S .< 1.5

pm = plot(size(600, 350), xlim = (bins[1], bins[end]), ylim = (0, :auto), xlabel = "Average risk aversion t̄", ylabel = "Count", label = nothing)

bw = step(bins)
dw = 0.0003
histogram!(pm, tbar, label = "did not converge",
    bins = bins, color = :lightsteelblue, lw = 0, la=0, bar_width = bw + dw)
histogram!(pm, tbar[converged_idx], label = "S(x) ≤ $thresh (inefficient)",
    bins = bins, color = :firebrick, lw = 0, la=0, bar_width = bw + 2dw)
histogram!(pm, tbar[efficient_idx], label = "S(x) > $thresh (efficient)",
    bins = bins, color = :olivedrab, lw = 0, la=0, bar_width = bw + 3dw)



savefig(pm, "tbar-stability-histogram.pdf")
savefig(pm, "tbar-stability-histogram.png")













function experiment_gamma()
    t_dist = Beta(beta_fix...)

    S = zeros(n_markets)
    γ = rand(n_markets)

    # Allocate memory
    t = zeros(n)

    a = zeros(n)
    b = zeros(n)

    x = zeros(Bool, n)
    f = zeros(n)
    y = zeros(n)

    for m in 1:n_markets
        println("Market $m")

        t[:] = rand(t_dist, n)

        a[:] = randexp(n)
        a[:] ./= sum(a)
        b[:] = randexp(n)
        b[:] ./= sum(b)

        try
            x[:], f[:], y[:] = equilibrate(a, b, q, γ[m], t)
            S[m] = dot(x, y) / q
        catch e
            if e <: FailedToConvergeException
                println("  Failed to converge")
                S[m] = 2         # Obvious because S ∈ [0,1] by design
            else
                throw(e)
            end
        end
    end

    return γ, S
end














@time γ, S = experiment_gamma()
open("./results/$(now())_gamma_result.txt", "a") do io
    writedlm(io, [γ S])
end






converged_idx = S .< 1.5
kdensity = kde((γ[converged_idx], S[converged_idx]), boundary = ((-1, 2), (-1, 2)))

pn = plot(size(600, 350),
    xlabel = "Evaluative complementarity γ",
    ylabel = "Stability of equilibrium S(x)",
    # xscale = :log10,
    xlim = (minimum(γ) - 0.01, maximum(γ) + 0.01),
    ylim = (minimum(S) - 0.02, 1.02),
    legend = nothing)

contourf!(pn, kdensity.x, kdensity.y, kdensity.density', color = :bamako, lw = 0, la = 0)

scatter!(pn, tbar[converged_idx], S[converged_idx], msw = 0, c = :white, alpha = 0.5, ms = 3, shape = :x)

annotate!(pn, [(minimum(γ),
    minimum(S),
    text(
        "ν = $(ν_fix.num)/$(ν_fix.den), t ~ Beta($(beta_fix[1]), $(beta_fix[2]))\n" *
        "$(sum(.!converged_idx))/$(length(S)) did not converge",
        :bottom,
        :left,
        :white,
        10)
)])

savefig(pn, "gamma-stability-density.pdf")
savefig(pn, "gamma-stability-density.png")










bins = range(0, 1, length = 21)

thresh = round(median(S[converged_idx]), digits=2)
efficient_idx = thresh .< S .< 1.5

po = plot(size(600, 350), xlim = (bins[1], bins[end]), ylim = (0, :auto), xlabel = "Evaluative complementarity γ", ylabel = "Count", label = nothing)

bw = step(bins)
dw = 0.0003
histogram!(po, γ, label = "did not converge",
    bins = bins, color = :lightsteelblue, lw = 0, la=0, bar_width = bw + dw)
histogram!(po, γ[converged_idx], label = "S(x) ≤ $thresh (inefficient)",
    bins = bins, color = :firebrick, lw = 0, la=0, bar_width = bw + 2dw)
histogram!(po, γ[efficient_idx], label = "S(x) > $thresh (efficient)",
    bins = bins, color = :olivedrab, lw = 0, la=0, bar_width = bw + 3dw)



savefig(po, "gamma-stability-histogram.pdf")
savefig(po, "gamma-stability-histogram.png")


















function experiment_nu()
    S = zeros(n_markets)

    ϵ = 1e-8

    # Range of possible variances for a beta dist with mean 0.5
    ν_dist = Uniform(ϵ, 0.25 - ϵ)
    ν = 0.25 * rand(n_markets)
    β = 1 ./(8*ν) .- 1/2

    # Allocate memory
    t = zeros(n)

    a = zeros(n)
    b = zeros(n)

    x = zeros(Bool, n)
    f = zeros(n)
    y = zeros(n)

    for m in 1:n_markets
        println("Market $m")

        t_dist = Beta(β[i], β[i])
        t[:] = rand(t_dist, n)

        a[:] = randexp(n)
        a[:] ./= sum(a)
        b[:] = randexp(n)
        b[:] ./= sum(b)

        try
            x[:], f[:], y[:] = equilibrate(a, b, q, γ_fix, t)
            S[m] = dot(x, y) / q
        catch e
            if e <: FailedToConvergeException
                println("  Failed to converge")
                S[m] = 2         # Obvious because S ∈ [0,1] by design
            else
                throw(e)
            end
        end
    end

    return ν, S
end














@time ν, S = experiment_gamma()
open("./results/$(now())_nu_result.txt", "a") do io
    writedlm(io, [ν S])
end






converged_idx = S .< 1.5
kdensity = kde((ν[converged_idx], S[converged_idx]), boundary = ((-1, 2), (-1, 2)))

pp = plot(size(600, 350),
    xlabel = "Variance ν in student risk aversion",
    ylabel = "Stability of equilibrium S(x)",
    # xscale = :log10,
    xlim = (minimum(ν) - 0.01, maximum(ν) + 0.01),
    ylim = (minimum(S) - 0.02, 1.02),
    legend = nothing)

contourf!(pp, kdensity.x, kdensity.y, kdensity.density', color = :bamako, lw = 0, la = 0)

scatter!(pp, ν[converged_idx], S[converged_idx], msw = 0, c = :white, alpha = 0.5, ms = 3, shape = :x)

annotate!(pp, [(minimum(tbar),
    minimum(S),
    text(
        "γ = $(γ_fix.num)/$(γ_fix.den), t ~ Beta($(beta_fix[1]), $(beta_fix[2]))\n" *
        "$(sum(.!converged_idx))/$(length(S)) did not converge",
        :bottom,
        :left,
        :white,
        10)
)])

savefig(pp, "nu-stability-density.pdf")
savefig(pp, "nu-stability-density.png")










bins = range(0, 0.25, length = 21)

thresh = round(median(S[converged_idx]), digits=2)
efficient_idx = thresh .< S .< 1.5

pq = plot(size(600, 350), xlim = (bins[1], bins[end]), ylim = (0, :auto), xlabel = "Variance ν in student risk aversion", ylabel = "Count", label = nothing)

bw = step(bins)
dw = 0.0003
histogram!(pq, ν, label = "did not converge",
    bins = bins, color = :lightsteelblue, lw = 0, la=0, bar_width = bw + dw)
histogram!(pq, ν[converged_idx], label = "S(x) ≤ $thresh (inefficient)",
    bins = bins, color = :firebrick, lw = 0, la=0, bar_width = bw + 2dw)
histogram!(pq, ν[efficient_idx], label = "S(x) > $thresh (efficient)",
    bins = bins, color = :olivedrab, lw = 0, la=0, bar_width = bw + 3dw)



savefig(pq, "nu-stability-histogram.pdf")
savefig(pq, "nu-stability-histogram.png")
