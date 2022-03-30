using TopOpt, Parameters, Makie, FiniteDifferences, Ipopt, Juniper, JuMP, Statistics, Ferrite
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
import GLMakie, Plots
include("./utils.jl")
# Nonconvex.NonconvexCore.show_residuals[] = true

@with_kw mutable struct FEAparameters
    meshSize::Tuple{Int, Int} = (70, 30) # Size of rectangular mesh
    elementIDs::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists nodeIDs
    problem::Any = 0.0
end

FEAparams = FEAparameters()
nels = prod(FEAparams.meshSize)

#

    # nodeCoords = Vector of tuples with node coordinates
    # cells = Vector of tuples of integers. Each line refers to an element
    # and lists the IDs of its nodes
    nodeCoords, cells = mshData(FEAparams.meshSize)
    # Type of element (CPS4 = linear quadrilateral)
    cellType = "CPS4"
    # toy grid
    grid = generate_grid(Quadrilateral, FEAparams.meshSize)
    numCellNodes = length(grid.cells[1].nodes) # number of nodes per cell/element
    # integer matrix representing displacement boundary conditions (supports):
    # 0: free element
    # 1: element restricted in the x direction ("roller")
    # 2: element restricted in the y direction ("roller")
    # 3: element restricted in both directions ("pinned"/"clamped")
    dispBC = zeros(Int, (3,3))
    
    # Dictionary mapping strings to vectors of integers. The vector groups node IDs that can be later
        # referenced by the name in the string
    # Clamp left boundary of rectangular domain
    nodeSets, dispBC = simplePins!("left", dispBC, FEAparams)
    # Similar to nodeSets, but refers to groups of cells (FEA elements) 
    cellSets = Dict(
        "SolidMaterialSolid" => FEAparams.elementIDs,
        "Eall"               => FEAparams.elementIDs,
        "Evolumes"           => FEAparams.elementIDs
    )
    # Dictionary mapping strings to vectors of tuples of Int and Float. The string contains a name. It refers to
        # a group of nodes defined in nodeSets. The tuples inform the displacement (Float) applied to a
        # a certain DOF (Int) of the nodes in that group. This is used to apply
        # Dirichlet boundary conditions.
    nodeDbcs = Dict("supps" => [(1, 0.0), (2, 0.0)])
    # lpos has the IDs of the loaded nodes.
    # each line in "forces" contains [forceLine forceCol forceXcomponent forceYcomponent]
    # lpos, forces = loadPos(nels, dispBC, FEAparams, grid)
    lpos = [709 710 781 780 1420 1419 1491 1490]
    forces = [
        21 70 1.0 1.0
        11 70 1.0 1.0
    ]
    # Dictionary mapping integers to vectors of floats. The vector
    # represents a force applied to the node with
    # the respective integer ID.
    cLoads = Dict(lpos[1] => forces[1,3:4])
    [merge!(cLoads, Dict(lpos[c] => forces[1,3:4])) for c in 2:numCellNodes];
    if length(lpos) > numCellNodes+1
        for pos in (numCellNodes+1):length(lpos)
            pos == (numCellNodes+1) && (global ll = 2)
            merge!(cLoads, Dict(lpos[pos] => forces[ll,3:4]))
            pos % numCellNodes == 0 && (global ll += 1)
        end
    end

    # Create struct with FEA input data
    FEAparams.problem = InpStiffness(InpContent(nodeCoords, cellType, cells, nodeSets, cellSets, 1.0, 0.3,
    0.0, nodeDbcs, cLoads, Dict("uselessFaces" => [(1,1)]), Dict("uselessFaces" => 0.0)))

#

# function TOBS(FEAparams, VF)
    VF = 0.7
    nl_solver = optimizer_with_attributes(Ipopt.Optimizer, "print_level"=>1)
    minlp_solver = optimizer_with_attributes(Juniper.Optimizer, "nl_solver"=>nl_solver)
    solver = FEASolver(Direct, FEAparams.problem; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0)) # instancing of FEA solver
    numVars = length(solver.vars)
    @show numVars
    β = 0.05 # parameter that limits volume change per iteration
    count = 1 # iteration counter
    N = 3 # number of past iterations to include in convergence criterion calculation
    er = 0 # initialize error
    τ = 0.001 # convergence parameter (upper bound of error)
    comps = zeros(N-1)

    # iterate until convergence
    while τ > er

        @show mean(solver.vars)
        solver()
        
        m = JuMP.Model(minlp_solver)
        comp = Compliance(FEAparams.problem, solver)
        filter = DensityFilter(solver; rmin=3.0)
        obj = x -> comp(filter(x))
        
        count != 1 && (sensPast = copy(sensNew))
        # Update sensitivities
        # sensNew = ForwardDiff.gradient(obj, solver.vars)
        # sensNew = ReverseDiff.gradient(obj, solver.vars)
        println("grads")
        sensNew = grad(central_fdm(2, 1), obj, solver.vars)[1]
        count == 1 && (sensPast = copy(sensNew))
        # Sensitivities history averaging
        sensNew = (sensPast+sensNew)/2
        
        # linearized objective
        # linObj = deltaX -> sensNew*deltaX
        
        # Volume constraint
        # volfrac = TopOpt.Volume(problem, solver)
        # volConstr = x -> volfrac(filter(x)) - FEAparams.V[1]
        # linVolConstr = deltaX -> ForwardDiff.gradient(volConstr, zeros(nels))*deltaX # linearized constraint
        
        # https://www.juliaopt.org/packages/
        println("setup model")
        # Define optimization variables (change in each pseudo-density for this iteration)
        @variable(m, deltaX[1:numVars], Int)
        [@constraint(m, deltaX[k] ∈ [-solver.vars[k] 1-solver.vars[k]]) for k ∈ 1:numVars]
        # Constrain volume of structure
        @constraint(m, volfrac(filter(solver.vars+deltaX)) <= VF)
        # Constrain volume change per iteration
        @constraint(m, sum(deltaX) <= β*nels)
        # @constraint(m, ForwardDiff.gradient(volfrac(filter(solver.vars+deltaX)), solver.vars)*deltaX)
        @objective(m, Min, deltaX -> sensNew*deltaX)
        # Optimize linearized problem
        println("optimize")
        optimize!(m)
        println("optimize done")
        # Perform step (update pseudo-densities)
        solver.vars += JuMP.value.(deltaX)
        # Store recent history of objectives
        [comps[i] = comps[i+1] for i ∈ 1:length(comps)-1]
        comps[end] = JuMP.objective_value(m)
        
        #   Convergence check
        #=
            convergence criterion
            k = 10
            N = 5
            ((10-5) + (9-4) + (8-3) + (7-2) + (6-1)) / (10+9+8+7+6)
            k > 2*N - 1
        =#
        if count > 2*N - 1
            global er = abs(sum([comps[end-i+1] - comps[end-N-i+1] for i in 1:N]))/sum([comps[end-i+1] for i in 1:N])
        end
      
        print("$count    comps: ")
        [print("$(round(comps[f];digits=3)) ") for f in 1:length(comps)]
        print("er = $(round(er;digits=4))   ")
        @show mean(solver.vars)

        global count += 1
        
    end
# end

# TOBS(FEAparams, 0.7)