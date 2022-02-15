using Random
using LinearAlgebra
using Plots, Plots.Measures
using JuMP
using Ipopt
using Libdl
using DataStructures
using Distributions
using DelimitedFiles
using DataFrames
using CSV
using HypothesisTests: UnequalVarianceTTest
import Base.copy
import Base.length

try
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)
    Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)
    global linearsolvername = "pardiso"
catch e
    @warn "Couldn't find pardiso, using ma27 instead"
    global linearsolvername = "ma27"
end


"""
Contains static information about a two-school admissions market with
combinatorial school preferences.
"""
struct Market
    a::Vector{Float64}   # Additive utility component
    b::Vector{Float64}   # Complementarity component
    t::Vector{Float64}   # Student preferences
    n::Int64
    
    # Infer n from a, b, t
    function Market(a::Vector{Float64}, b::Vector{Float64}, t::Vector{Float64})
        n = length(a)
        @assert n == length(b) == length(t)
        
        return new(a, b, t, n)
    end
    
    # Given t, generates a and b randomly
    # Useful for simulations
    function Market(t::Vector{Float64})
        n = length(t)
        a = randexp(n)
        a .*= n/sum(a)
        b = randexp(n)
        b .*= n/sum(b)
        
        return new(a, b, t, n)
    end
    
    # Random market with t from given distibution
    function Market(dist::Sampleable, n)
        t = rand(dist, n)
        return Market(t)
    end
end


"""
Contains information about the equilibrium of a two-school admissions market 
with combinatorial school preferences.
"""
struct Equilibrium
    market::Market
    x::Vector{Float64}   # Additive utility component
    f::Vector{Float64}   # Complementarity component
    y::Vector{Float64}   # Student preferences
    γ::Float64           # Complementarity ratio
    q::Int64
    err::Float64         # Disequilibrium measure
end


"""
Lighter version of `Equilibrium` to use during iteration.
"""
mutable struct EquilibriumCandidate
    x::Vector{Float64}   # Additive utility component
    dx::Vector{Float64}  # Error vector
    f::Vector{Float64}   # Complementarity component
    err::Float64         # Disequilibrium measure
    k::Int64             # Number of iterations it has been alive for

    function EquilibriumCandidate(x, dx, f, err, k)
        return new(x, dx, f, err, k)
    end

    function EquilibriumCandidate(n::Int64)
        return new(zeros(n), zeros(n), zeros(n), 100*n, 1)
    end

    function EquilibriumCandidate(m::Market, γ::Float64, q::Int64)
        x = randexp(m.n)
        x[:] .*= q / (mean(m.t) * sum(x))
        
        f = admitresult(m, x, γ, q)
        dx = f - m.t
        err = norm(x - clamp.(x + dx, 0, 1))

        return new(x, dx, f, err, 1)
    end
end


Base.isless(A::EquilibriumCandidate, B::EquilibriumCandidate) = A.err ≤ B.err
copy(A::EquilibriumCandidate) = EquilibriumCandidate(copy(A.x), copy(A.dx), copy(A.f), copy(A.err), copy(A.k))


"""
Contains the effiency measures associated with a given equilibrium. 
"""
struct EfficiencyMeasures
    equilibrium::Equilibrium
    S::Float64   # Stability measure
    T::Float64   # Alignment measure
    U::Float64   # Aggregate student welfare
    k::Float64   # Size of equilibrium
    
    function EfficiencyMeasures(e::Equilibrium)
        xdoty = dot(e.x, e.y)
        k = sum(e.x)
        
        S = xdoty/k
        T = xdoty/e.q
        U = e.q + dot(1 .- e.x, e.market.t)
        
        return new(e, S, T, U, k)
    end
end


"""
    admitresult(m, x, γ, q; tol=1e-6)

For a given admissions market `m`, complementarity weight `γ`, and capacity `q`,
compute the probability `f[i]` that student `i` is admitted if she applies given
other applicants' application probabilities `x`. 
"""
function admitresult(m::Market, x::Union{BitVector, Vector{<:Float64}}, γ::Float64, q::Int64; tol=1e-8)::Vector{Float64}
    n_applicants = sum(x)
    
    if n_applicants < q
        return ones(m.n)
    end
    
    mdl = Model(Ipopt.Optimizer)
    set_optimizer_attribute(mdl, "linear_solver", linearsolvername)
    set_silent(mdl)
    
    if n_applicants == m.n
        @variable(mdl, 0 ≤ f[1:m.n] ≤ 1)
        @objective(mdl, Max, (1-γ) * dot(m.a, f) - γ * dot(m.b - f, m.b - f))
        @constraint(mdl, Capacity, sum(f) ≤ q)
        
        optimize!(mdl)

        return value.(f)
    
    else # Between q and n applicants
        f_res = zeros(m.n)
        
        @variable(mdl, 0 ≤ f[1:m.n])
        @objective(mdl, Max, (1-γ) * dot(m.a, f) - γ * dot(m.b - f, m.b - f))
        @constraint(mdl, Capacity, sum(f) ≤ q)
        
        @constraint(mdl, OnlyAdmitApplicants[i in 1:m.n], f[i] ≤ x[i])
        
        optimize!(mdl)

        f_current = value.(f)
        d = shadow_price.(OnlyAdmitApplicants)
        
        # Remove the overall application constraint
        # Now we will check, student-by-student, their admit probability if they apply wp 1
        delete(mdl, OnlyAdmitApplicants)
        unregister(mdl, :OnlyAdmitApplicants)
        
        # Rewrite the constraint as bounds so that we can modify indiv rows below
        for i in 1:m.n
            set_start_value(f[i], f_current[i])
            set_upper_bound(f[i], x[i])
        end

        for i in 1:m.n
            if x[i] > 1 - tol || d[i] < tol
                # If i was already applying, then the relevant LP is the one we solved above
                
                # Likewise, if dual variable associated with i's nonapplication constraint had
                # value zero, then f[i] will remain unchanged even if we relax the constraint
                f_res[i] = f_current[i]
            else
                # Now we know that increasing f[i] will increase the objective value,
                # and we need to actually solve the modified LP to see the new value of f[i]

                # Suffices to modify bounds on f[i] as follows
                set_upper_bound(f[i], 1)

                optimize!(mdl)

                f_res[i] = value(f[i])
                
                # And fix back so that it doesn't interfere with subsequent computations
                set_upper_bound(f[i], x[i])
            end
        end
        
        return f_res
    end
end


"""
    equilibrate(m, γ, q)

Compute an equilibrium for the market `m` with complementarity parameter `γ` and capacity `q`
using a myopic adjustment step rule.
"""
function equilibrate(
    m::Market,
    γ::Float64,
    q::Int64;
    tol = 1e-5::Float64,  # Permissible equilibrium error
    nit = 50::Int64,    # Number of root nodes to inspect
    nseeds = 3::Int64,  # Number of initial seeds
    Q = 10::Int64, # Maximum size of the tabu queue
    steps = [1.0, 10.0]::Vector{Float64},
    β = 0.9::Float64
)

    y = admitresult(m, trues(m.n), γ, q)
    k = 0

    seen = Set{UInt64}()

    cand = EquilibriumCandidate(m.n)
    best_cand = copy(cand)
    curr_cand = copy(cand)

    candidates = Deque{EquilibriumCandidate}()

    for j in 1:nseeds
        push!(candidates, EquilibriumCandidate(m, γ, q))
        push!(seen, hash(last(candidates).x))
    end
    # sort!(candidates, rev=true)

    for k in 1:nit
        curr_cand.x[:] = last(candidates).x
        curr_cand.dx[:] = last(candidates).dx
        curr_cand.f[:] = last(candidates).f
        curr_cand.err = last(candidates).err
        curr_cand.k = pop!(candidates).k
        println("  BR iteration $k of $nit: Candidate error = $(round(curr_cand.err, digits=24)), " *
                "$(length(candidates)) candidate(s) in queue")

        # Generate neighbor from best response
        cand.x[:] = curr_cand.f .> m.t
        h = hash(cand.x)
        if !(h in seen)
            push!(seen, h)
            cand.f[:] = admitresult(m, cand.x, γ, q)
            cand.dx[:] = cand.f - m.t
            cand.err = norm(cand.x - clamp.(cand.x + cand.dx, 0, 1))
            cand.k = 1

            if cand.err ≤ tol
                println("    Equilibrium found by pure-strategy best response at iteration $k")
                return Equilibrium(m, cand.x, cand.f, y, γ, q, cand.err)
            end

            if cand.err < best_cand.err
                best_cand.x[:] = cand.x
                best_cand.dx[:] = cand.dx
                best_cand.f[:] = cand.f
                best_cand.err = cand.err
                best_cand.k = cand.k
                push!(candidates, copy(cand))
            elseif length(candidates) < Q
                pushfirst!(candidates, copy(cand))
            end
        end

        # Generate neighbors using gradient
        for α in steps
            cand.x[:] = clamp.(curr_cand.x + curr_cand.dx * α / curr_cand.k^β, 0, 1)
            h = hash(cand.x)
            if !(h in seen)
                push!(seen, h)
                cand.f[:] = admitresult(m, cand.x, γ, q)
                cand.dx[:] = cand.f - m.t
                cand.err = norm(cand.x - clamp.(cand.x + cand.dx, 0, 1))
                cand.k = curr_cand.k + 1
            
                if cand.err ≤ tol
                    println("    Equilibrium found by gradient at iteration $k")
                    return Equilibrium(m, cand.x, cand.f, y, γ, q, cand.err)
                end
            
                if cand.err < best_cand.err
                    best_cand.x[:] = cand.x
                    best_cand.dx[:] = cand.dx
                    best_cand.f[:] = cand.f
                    best_cand.err = cand.err
                    best_cand.k = cand.k
                    push!(candidates, copy(cand))
                elseif length(candidates) < Q
                    pushfirst!(candidates, copy(cand))
                end
            end
        end
    
        # Generate neighbors by changing just one student to her BR
        i_new = findfirst(i -> curr_cand.x[i] < 1 && curr_cand.f[i] > m.t[i], 1:m.n)
        if !isnothing(i_new)
            cand.x[:] = curr_cand.x
            h = hash(cand.x)
            if !(h in seen)
                push!(seen, h)
                cand.x[i_new] = 1
                cand.f[:] = admitresult(m, cand.x, γ, q)
                cand.dx[:] = cand.f - m.t
                cand.err = norm(cand.x - clamp.(cand.x + cand.dx, 0, 1))
                cand.k = 1
            
                if cand.err ≤ tol
                    println("    Equilibrium found by single swap to 1 at iteration $k")
                    return Equilibrium(m, cand.x, cand.f, y, γ, q, cand.err)
                end
            
                if cand.err < best_cand.err
                    best_cand.x[:] = cand.x
                    best_cand.dx[:] = cand.dx
                    best_cand.f[:] = cand.f
                    best_cand.err = cand.err
                    best_cand.k = cand.k
                    push!(candidates, copy(cand))
                elseif length(candidates) < Q
                    pushfirst!(candidates, copy(cand))
                end
            end
        end
    
        i_new = findfirst(i -> curr_cand.x[i] > 0 && curr_cand.f[i] < m.t[i], 1:m.n)
        if !isnothing(i_new)
            cand.x[:] = curr_cand.x
            h = hash(cand.x)
            if !(h in seen)
                push!(seen, h)
                cand.x[i_new] = 0
                cand.f[:] = admitresult(m, cand.x, γ, q)
                cand.dx[:] = cand.f - m.t
                cand.err = norm(cand.x - clamp.(cand.x + cand.dx, 0, 1))
                cand.k = 1
            
                if cand.err ≤ tol
                    println("    Equilibrium found by single swap to 0 at iteration $k")
                    return Equilibrium(m, cand.x, cand.f, y, γ, q, cand.err)
                end
            
                if cand.err < best_cand.err
                    best_cand.x[:] = cand.x
                    best_cand.dx[:] = cand.dx
                    best_cand.f[:] = cand.f
                    best_cand.err = cand.err
                    best_cand.k = cand.k
                    push!(candidates, copy(cand))
                elseif length(candidates) < Q
                    pushfirst!(candidates, copy(cand))
                end
            end
        end
    end

    println("    Failed to converge to equilibrium, returning best solution found")
    return Equilibrium(m, best_cand.x, best_cand.f, y, γ, q, best_cand.err)
end


function experiment_heterogeneity(n_markets = 5::Int, n = 20::Int, q = 5::Int)
    Binv = rand(n_markets)
    B = inv.(Binv)
    EQ = Equilibrium[]
    EM_ = EfficiencyMeasures[]
    err = zeros(n_markets)
    converged = falses(n_markets)

    for p in 1:n_markets
        println("Market $p of $n_markets")
        m = Market(Beta(B[p], B[p]), n)
        eq = equilibrate(m, 0.5, q)

        push!(EQ, eq)
        push!(EM_, EfficiencyMeasures(eq))
        err[p] = eq.err
        converged[p] = eq.err < 1e-5
    end

    EM = DataFrame(
        Binv = Binv,
        err = err,
        converged = converged,
        S = [EM_[p].S for p in 1:n_markets],
        T = [EM_[p].T for p in 1:n_markets],
        U = [EM_[p].U for p in 1:n_markets],
        k = [EM_[p].k for p in 1:n_markets],
    )

    return (; EQ, EM, n_markets)
end


function plots_heterogeneity(res)
    P = Plots.Plot[plot() for i in 1:4]
    
    plot!(P[1], size=(600, 350), xlabel="Student preference heterogeneity 1/B", ylabel="Stability index S̄(x)", legend=nothing)
    scatter!(res.EM.Binv[res.EM.converged], res.EM.S[res.EM.converged], ms=5, ma=0.75, c=:dodgerblue, msw=0, msa=0)
    
    plot!(P[2], size=(600, 350), xlabel="Student preference heterogeneity 1/B", ylabel="Alignment index T̄(x)", legend=nothing)
    scatter!(res.EM.Binv[res.EM.converged], res.EM.T[res.EM.converged], ms=5, ma=0.75, c=:crimson, msw=0, msa=0)
    
    plot!(P[3], size=(600, 350), xlabel="Student preference heterogeneity 1/B", ylabel="Aggregate student welfare Ū(x)", legend=nothing)
    scatter!(res.EM.Binv[res.EM.converged], res.EM.U[res.EM.converged], ms=5, ma=0.75, c=:olivedrab, msw=0, msa=0)
    
    plot!(P[4], size=(600, 350), xlabel="Student preference heterogeneity 1/B", ylabel="Size of equilibrium k(x)", legend=nothing)
    scatter!(res.EM.Binv[res.EM.converged], res.EM.k[res.EM.converged], ms = 5, ma = 0.75, c = :darkorange, msw = 0, msa = 0)
    
    return P
end


function experiment_complementarity(n_markets = 5::Int, n = 20::Int, q = 5::Int)
    Γ = rand(n_markets)
    EQ = Equilibrium[]
    EM_ = EfficiencyMeasures[]
    err = zeros(n_markets)
    converged = falses(n_markets)

    for p in 1:n_markets
        println("Market $p of $n_markets")
        m = Market(Beta(10, 10), n)
        eq = equilibrate(m, Γ[p], q)

        push!(EQ, eq)
        push!(EM_, EfficiencyMeasures(eq))
        err[p] = eq.err
        converged[p] = eq.err < 1e-5
    end

    EM = DataFrame(
        γ = Γ,
        err = err,
        converged = converged,
        S = [EM_[p].S for p in 1:n_markets],
        T = [EM_[p].T for p in 1:n_markets],
        U = [EM_[p].U for p in 1:n_markets],
        k = [EM_[p].k for p in 1:n_markets],
    )

    return (; EQ, EM, n_markets)
end


function plots_complementarity(res)
    P = Plots.Plot[plot() for i in 1:4]
    
    plot!(P[1], size=(600, 350), xlabel="Complementarity γ in college preferences", ylabel="Stability index S̄(x)", legend=nothing)
    scatter!(res.EM.γ[res.EM.converged], res.EM.S[res.EM.converged], ms=5, ma=0.75, c=:dodgerblue, msw=0, msa=0)
    
    plot!(P[2], size=(600, 350), xlabel="Complementarity γ in college preferences", ylabel="Alignment index T̄(x)", legend=nothing)
    scatter!(res.EM.γ[res.EM.converged], res.EM.T[res.EM.converged], ms=5, ma=0.75, c=:crimson, msw=0, msa=0)
    
    plot!(P[3], size=(600, 350), xlabel="Complementarity γ in college preferences", ylabel="Aggregate student welfare Ū(x)", legend=nothing)
    scatter!(res.EM.γ[res.EM.converged], res.EM.U[res.EM.converged], ms=5, ma=0.75, c=:olivedrab, msw=0, msa=0)
    
    plot!(P[4], size=(600, 350), xlabel="Complementarity γ in college preferences", ylabel="Size of equilibrium k(x)", legend=nothing)
    scatter!(res.EM.γ[res.EM.converged], res.EM.k[res.EM.converged], ms=5, ma=0.75, c=:darkorange, msw=0, msa=0)
    
    return P
end


"""
    everything(n_markets, n, q)

Runs the heterogeneity and complementarity experiments, exports results and plots,
displays results of hypothesis tests on whether converge depends on experimental variables.
"""
function everything(n_markets=5::Int, n=20::Int, q=5::Int, write=false::Bool)
    measures_names = split("S T U k")

    println("==== Student heterogeneity experiment ==== ")
    res_heterogeneity = experiment_heterogeneity(n_markets, n, q)
    P_heterogeneity = plots_heterogeneity(res_heterogeneity)
    
    # Test that the instances that converge are not significally
    # different from those that didn't in the experimental variable
    if 1 < sum(res_heterogeneity.EM.converged) < n_markets - 1
    	println("Some markets failed to converge. The following hypothesis test checks whether\nconvergence is dependent on 1/B.\n")
        display(UnequalVarianceTTest(
            res_heterogeneity.EM.Binv[res_heterogeneity.EM.converged],
            res_heterogeneity.EM.Binv[.!res_heterogeneity.EM.converged]))
    end

    if write
        CSV.write("heterogeneity_results.csv", res_heterogeneity.EM)
        for (i, pl) in enumerate(P_heterogeneity)
            savefig(pl, "heterogeneity-$(measures_names[i]).png")
            savefig(pl, "heterogeneity-$(measures_names[i]).pdf")
        end
    else
        display(res_heterogeneity.EM)
        display.(P_heterogeneity)
    end


    println("\n==== School complementarity experiment ==== ")
    res_complementarity = experiment_complementarity(n_markets, n, q)
    P_complementarity = plots_complementarity(res_complementarity)

    if 1 < sum(res_complementarity.EM.converged) < n_markets - 1
        println("Some markets failed to converge. The following hypothesis test checks whether\nconvergence is dependent on γ.\n")
        display(UnequalVarianceTTest(
            res_complementarity.EM.γ[res_complementarity.EM.converged],
            res_complementarity.EM.γ[.!res_complementarity.EM.converged]))
    end

    if write
        CSV.write("complementarity_results.csv", res_complementarity.EM)
        for (i, pl) in enumerate(P_complementarity)
            savefig(pl, "complementarity-$(measures_names[i]).png")
            savefig(pl, "complementarity-$(measures_names[i]).pdf")
        end
    else
        display(res_complementarity.EM)
        display.(P_complementarity)
    end
end

@time everything(600, 60, 20, true)

