using TopOpt, Parameters, Makie, Zygote, Cbc, Juniper, JuMP, Statistics, Ferrite
using TopOpt.TopOptProblems.InputOutput.INP.Parser: InpContent
import GLMakie, Plots
include("./utils.jl")

@with_kw mutable struct FEAparameters
    meshSize::Tuple{Int, Int} = (70, 30) # Size of rectangular mesh
    elementIDs::Array{Int} = [i for i in 1:prod(meshSize)] # Vector that lists nodeIDs
    problem::Any = 0.0
end

FEAparams = FEAparameters()

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

function TOBS(FEAparams, VF)
    milp_solver = optimizer_with_attributes(Cbc.Optimizer)
    solver = FEASolver(Direct, FEAparams.problem; xmin=1e-6, penalty=TopOpt.PowerPenalty(3.0)) # instancing of FEA solver
    solver.vars .= 1
    numVars = length(solver.vars)
    β = 1 # parameter that limits volume change per iteration
    count = 1 # iteration counter
    N = 3 # number of past iterations to include in convergence criterion calculation
    er = 1 # initialize error
    τ = 0.001 # convergence parameter (upper bound of error)
    comps = zeros(N-1)
    ϵ = 0.01
    x = ones(numVars)

    comp = Compliance(FEAparams.problem, solver)
    filter = DensityFilter(solver; rmin=3.0)
    obj = x -> comp(filter(x))
    volfrac = TopOpt.Volume(FEAparams.problem, solver)
    currentVF = volfrac(filter(x))

    # iterate until convergence
    while τ < er

        solver()
        
        m = JuMP.Model(milp_solver)
        set_optimizer_attribute(m, "logLevel", 1)

        count != 1 && (sensPast = copy(sensNew))
        # Update sensitivities
        global sensNew = Zygote.gradient(obj, x)[1]
        gradVF = Zygote.gradient(x -> volfrac(filter(x)), x)[1]
        count == 1 && (sensPast = copy(sensNew))
        # Sensitivities history averaging
        global sensNew = (sensPast+sensNew)/2
        
        # https://www.juliaopt.org/packages/
        # Define optimization variables (change in each pseudo-density for this iteration)
        @variable(m, deltaX[1:numVars], Int)
        set_lower_bound.(deltaX,-x)
        set_upper_bound.(deltaX,1 .- x)
        # Constrain volume of structure
        @constraint(m, currentVF+gradVF'*deltaX <= VF)
        # Constrain volume change per iteration
        @constraint(m, sum(deltaX) <= β*numVars)
        # Constraint relaxation
        if VF < (1-ϵ)*currentVF
            ΔV = -ϵ*currentVF
        elseif VF > (1+ϵ)*currentVF
            ΔV = ϵ*currentVF
        else
            ΔV = VF - currentVF
        end
        @constraint(m, gradVF'*deltaX <= ΔV)
        # Define optimization objective
        @objective(m, Min, sensNew'*deltaX)
        # Optimize linearized problem
        optimize!(m)
        # Perform step (update pseudo-densities)
        x += JuMP.value.(deltaX)
        # Store recent history of objectives
        [comps[i] = comps[i+1] for i ∈ 1:length(comps)-1]
        comps[end] = obj(x)
        
        #   Convergence check
        #=
            convergence criterion
            k = 10
            N = 5
            ((10-5) + (9-4) + (8-3) + (7-2) + (6-1)) / (10+9+8+7+6)
            k > 2*N - 1
        =#
        if count > 2*N - 1
            er = abs(sum([comps[end-i+1] - comps[end-N-i+1] for i in 1:N]))/sum([comps[end-i+1] for i in 1:N])
            println(abs(sum([comps[end-i+1] - comps[end-N-i+1] for i in 1:N]))/sum([comps[end-i+1] for i in 1:N]))
        end
        
        print("$count    comps: ")
        [print("$(round(comps[f];digits=3)) ") for f in 1:length(comps)]
        print("er = $(round(er;digits=4))   ")
        @show mean(x)

        currentVF = volfrac(filter(x))

        count += 1
        
    end
end

TOBS(FEAparams, 0.7)