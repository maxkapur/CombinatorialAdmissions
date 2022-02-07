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


try
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)
    global linearsolvername = "pardiso"
catch e
    @warn "Couldn't find pardiso, using ma27 instead"
    global linearsolvername = "ma27"
end

struct FailedToConvergeException <: Exception end

function makedata(n)
    # Complementarity matrix
    A = UpperTriangular(-randn(n, n))
    for i in 1:n
        A[i, i] = 0
    end
    A

    # Additive component
    b = n * randexp(n)

    # Proportionality grouping
    g = randn(n)

    # Risk aversion
    t = rand(n)

    return A, b, g, t
end


function admitresult_allapply(
    A::AbstractMatrix,
    b::AbstractVector,
    g::AbstractVector,
    q::Int)::Vector{Float64}

    n = length(b)

    # mdl = Model(SCIP.Optimizer)
    # set_optimizer_attribute(mdl, SCIP.Param("parallel/maxnthreads"), Threads.nthreads())
    mdl = Model(Ipopt.Optimizer)
    set_optimizer_attribute(mdl, "linear_solver", linearsolvername)
    set_silent(mdl)

    @variable(mdl, 0 ≤ y[1:n] ≤ 1)
    # z[i, j] = y[i] ∧ y[j]
    @variable(mdl, z[i in 1:n, j in i+1:n] ≥ 0)

    @objective(mdl, Max, sum(A[i, j] * z[i, j] for i in 1:n, j in i+1:n) + dot(b, y) / 2)

    @constraint(mdl, Capacity, sum(y) ≤ q)
    @constraint(mdl, Proportionality, dot(g, y) ≤ 0)

    @constraint(mdl, Interaction1[i in 1:n, j in i+1:n], z[i, j] ≤ y[i])
    @constraint(mdl, Interaction2[i in 1:n, j in i+1:n], z[i, j] ≤ y[j])
    @constraint(mdl, Interaction3[i in 1:n, j in i+1:n], z[i, j] ≥ y[i] + y[j] - 1)

    optimize!(mdl)

    return value.(y)
end

function admitresult_someapply(
    A::AbstractMatrix,
    b::AbstractVector,
    g::AbstractVector,
    q::Int,
    x::Vector{Bool};
    tol = 1e-6::AbstractFloat)::Vector{Float64}

    n = length(b)

    if sum(x) < q
        return ones(n)
    end

    Y_res = zeros(n)

    # Container for optimal admit policy wrt current applicants
    y_current = zeros(n)

    # Current admissions probabilities
    mdl = Model(Ipopt.Optimizer)
    set_optimizer_attribute(mdl, "linear_solver", linearsolvername)

    set_silent(mdl)

    @variable(mdl, 0 ≤ y[1:n])
    @variable(mdl, z[i in 1:n, j in i+1:n] ≥ 0)
    # z[i, j] = y[i] ∧ y[j]

    @objective(mdl, Max, sum(A[i, j] * z[i, j] for i in 1:n, j in i+1:n) + dot(b, y) / 2)

    @constraint(mdl, Capacity, sum(y) ≤ q)
    @constraint(mdl, Proportionality, dot(g, y) ≤ 0)

    @constraint(mdl, Interaction1[i in 1:n, j in i+1:n], z[i, j] ≤ y[i])
    @constraint(mdl, Interaction2[i in 1:n, j in i+1:n], z[i, j] ≤ y[j])
    @constraint(mdl, Interaction3[i in 1:n, j in i+1:n], z[i, j] ≥ y[i] + y[j] - 1)

    @constraint(mdl, OnlyAdmitApplicants[i in 1:n], y[i] ≤ x[i])

    optimize!(mdl)

    y_current[:] = value.(y)

    d = shadow_price.(OnlyAdmitApplicants)
    # @show [x d]

    delete(mdl, OnlyAdmitApplicants)
    unregister(mdl, :OnlyAdmitApplicants)

    for i in 1:n
        set_start_value(y[i], y_current[i])

        if x[i]
            set_upper_bound(y[i], 1)
        else
            fix(y[i], 0; force=true)
            for j in i+1:n
                fix(z[i, j], 0; force=true)
            end
        end
    end

    for s in 1:n
        if x[s]
            # If s was already applying, then the relevant IP is the one we solved above
            Y_res[s] = y_current[s]
        elseif d[s] < tol
            # Dual variable associated with s's nonapplication constraint had value zero
            # Therefore y[s] will remain zero even if we relax the constraint
            Y_res[s] = 0
        else
            # Now we know that increasing y[s] will increase the objective value,
            # and we need to actually solve the modified LP to see the new value of y[s]
        
            # Suffices to unfix y and corresp. z and solve again
            unfix(y[s])
            set_lower_bound(y[s], 0)
            set_upper_bound(y[s], 1)
        
            for j in s+1:n
                unfix(z[s, j])
                set_lower_bound(z[s, j], 0)
            end
        
            optimize!(mdl)
        
            Y_res[s] = value(y[s])
        
            # And fix back so that it doesn't interfere with subsequent computations
            fix(y[s], 0; force=true)

            for j in s+1:n
                fix(z[s, j], 0; force=true)
            end
        end
    end

    return Y_res
end


function equilibrate(
    A::AbstractMatrix,
    b::AbstractVector,
    g::AbstractVector,
    q::Int,
    t::AbstractVector;
    nit = 10,
    tries = 3)

    n = length(b)
    y = admitresult_allapply(A, b, g, q)
    # println(y)

    x_seensofar = Set{Vector{Bool}}()
    x = zeros(Bool, n)
    x_next = zeros(Bool, n)

    f = zeros(n)

    for j in 1:tries
        println("  Attempt $j of $tries")
        x[:] = rand(Bool, n)

        for k in 1:nit
            print("    BR iteration $k of $nit: ")

            f[:] = admitresult_someapply(A, b, g, q, x)
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


function experiment()
    n_markets = 600

    n = 50
    q = 10

      ϵ = 1e-6

    # The range of possible means for a beta dist having νinv
    tbar_dist = Uniform(
        (1 - sqrt(1 - 4 * ν)) / 2 + ϵ,
        (1 + sqrt(1 - 4 * ν)) / 2 - ϵ,
    )

    tbar = rand(tbar_dist, n_markets)
    S = zeros(n_markets)

    # Allocate memory for A, b, g, x, f, y
    t = zeros(n)

    A = zeros(n, n)
    b = zeros(n)
    g = zeros(n)

    x = zeros(Bool, n)
    f = zeros(n)
    y = zeros(n)

    # Unused t output from makedata
    dummy = zeros(n)

    for m in 1:n_markets
        println("Market $m")

        dist = Beta(alphas(tbar[m], ν)...)
        t[:] = rand(dist, n)

        A[:], b[:], g[:], dummy[:] = makedata(n)

        try
            x[:], f[:], y[:] = equilibrate(A, b, g, q, t)
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

const ν = 1 // 10
@time tbar, S = experiment()








converged_idx = S .< 1.5
kdensity = kde((tbar[converged_idx], S[converged_idx]), boundary = ((-1, 2), (-1, 2)))

pl = plot(size(600, 350),
    xlabel = "Average risk aversion t̄",
    ylabel = "Stability of equilibrium S(x)",
    # xscale = :log10,
    xlim = (minimum(tbar) - 0.01, maximum(tbar) + 0.01),
    ylim = (minimum(S) - 0.01, 1.01),
    legend = nothing)

contourf!(pl, kdensity.x, kdensity.y, kdensity.density', color = :bamako, lw = 0)

scatter!(pl, tbar[converged_idx], S[converged_idx], msw = 0, c = :white, alpha = 0.5, ms = 3, shape = :x)

annotate!(pl, [(minimum(tbar),
    minimum(S),
    text(
        "ν = $(ν.num)/$(ν.den)\n" *
        "$(sum(.!converged_idx))/$(length(S)) did not converge",
        :bottom,
        :left,
        :white,
        10)
)])

savefig(pl, "tbar-stability-density.pdf")
savefig(pl, "tbar-stability-density.png")














bins = range((1 - sqrt(1 - 4 * ν)) / 2,
    (1 + sqrt(1 - 4 * ν)) / 2,
    length = 21)

thresh = 0.95
efficient_idx = thresh .< S .< 1.5

pm = plot(size(600, 350), xlim = (bins[1], bins[end]), ylim = (0, :auto), xlabel = "Average risk aversion t̄", ylabel = "Count", label = nothing)

bw = step(bins)
dw = 0.0003
histogram!(pm, tbar, label = "did not converge",
    bins = bins, color = :lightsteelblue, lw = 0, bar_width = bw + dw)
histogram!(pm, tbar[converged_idx], label = "S(x) ≤ $thresh (inefficient)",
    bins = bins, color = :firebrick, lw = 0, bar_width = bw + 2dw)
histogram!(pm, tbar[efficient_idx], label = "S(x) > $thresh (efficient)",
    bins = bins, color = :olivedrab, lw = 0, bar_width = bw + 3dw)



savefig(pm, "tbar-stability-histogram.pdf")
savefig(pm, "tbar-stability-histogram.png")







open("./results/$(now())_result.txt", "a") do io
    writedlm(io, [tbar S])
end


# A, b, g, t = makedata(40)
# q = 10
# equilibrate(A, b, g, q, t)
