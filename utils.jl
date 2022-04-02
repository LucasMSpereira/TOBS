using Ferrite, Parameters, HDF5, LinearAlgebra

# Generate nodeIDs used to position point loads
# However, original article "applied loads and supports to elements", not nodes
function loadPos(nels, dispBC, FEAparams, grid)
  # Random ID(s) to choose element(s) to be loaded
  global loadElements = randDiffInt(2, nels)
  # Matrices to indicate position and component of load
  forces = zeros(2,4)'
  # i,j mesh positions of chosen elements
  global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
  
  # Verify if load will be applied on top of support.
  # Randomize positions again if that's the case
  while true
    if dispBC[1,3] > 3
      
      
      if dispBC[1,3] == 4
        # left
        if prod([loadPoss[i][2] != 1 for i in keys(loadPoss)])
          break
        else
          global loadElements = randDiffInt(2, nels)
          global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 5
        # bottom
        if prod([loadPoss[i][1] != FEAparams.meshSize[2] for i in keys(loadPoss)])
          break
        else
          global loadElements = randDiffInt(2, nels)
          global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 6
        # right
        if prod([loadPoss[i][2] != FEAparams.meshSize[1] for i in keys(loadPoss)])
          break
        else
          global loadElements = randDiffInt(2, nels)
          global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      elseif dispBC[1,3] == 7
        # top
        if prod([loadPoss[i][1] != 1 for i in keys(loadPoss)])
          break
        else
          global loadElements = randDiffInt(2, nels)
          global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
        end
      else
        println("\nProblem with dispBC\n")
      end
      
      
    else
      
      
      global boolPos = true
      for i in keys(loadPoss)
        global boolPos *= !in([loadPoss[i][k] for k in 1:2], [dispBC[h,1:2] for h in 1:size(dispBC)[1]])
      end
      if boolPos
        break
      else
        global loadElements = randDiffInt(2, nels)
        global loadPoss = findall(x->in(x, loadElements), FEAparams.elementIDmatrix)
      end
      
      
    end
  end
  # Generate point load component values
  randLoads = (-ones(length(loadElements),2) + 2*rand(length(loadElements),2))*90
  # Build matrix with positions and components of forces
  forces = [
  loadPoss[1][1] loadPoss[1][2] randLoads[1,1] randLoads[1,2]
  loadPoss[2][1] loadPoss[2][2] randLoads[2,1] randLoads[2,2]
  ]
  # Get vector with IDs of loaded nodes
  myCells = [grid.cells[g].nodes for g in loadElements]
  pos = reshape([myCells[ele][eleNode] for eleNode in 1:length(myCells[1]), ele in keys(loadElements)], (:,1))
  return pos, forces, randLoads
end

function mshData(meshSize)
  
  # Create vector of (float, float) tuples with node coordinates for "node_coords"
  # Supposes rectangular elements with unit sides staring at postition (0.0, 0.0)
  # meshSize = (x, y) = quantity of elements in each direction
  
  coordinates = Array{Tuple{Float64, Float64}}(undef, (meshSize[1]+1)*(meshSize[2]+1))
  for line in 1:(meshSize[2] + 1)
    coordinates[(line + (line - 1)*meshSize[1]):(line*(1 + meshSize[1]))] .= [((col - 1)/1, (line - 1)/1) for col in 1:(meshSize[1] + 1)]
  end
  
  # Create vector of tuples of integers for "cells"
  # Each line refers to a cell/element and lists its nodes in counter-clockwise order
  
  g_num = Array{Tuple{Vararg{Int, 4}}, 1}(undef, prod(meshSize))
  for elem in 1:prod(meshSize)
    dd = floor(Int32, (elem - 1)/meshSize[1]) + elem
    g_num[elem] = (dd, dd + 1, dd + meshSize[1] + 2, dd + meshSize[1] + 1)
  end
  
  return coordinates, g_num
  
end

# reshape vectors with element quantity to reflect mesh shape
function quad(nelx,nely,vec)
  # nelx = number of elements along x axis (number of columns in matrix)
  # nely = number of elements along y axis (number of lines in matrix)
  # vec = vector of scalars, each one associated to an element.
  # this vector is already ordered according to element IDs
  global quadd=zeros(nely,nelx)
  for i in 1:nely
    for j in 1:nelx
      global quadd[nely-(i-1),j] = vec[(i-1)*nelx+1+(j-1)]
    end
  end
  return quadd
end

function randPins!(nels, FEAparams, dispBC, grid)
  # generate random element IDs
  randEl = randDiffInt(3, nels)
  # get "matrix position (i,j)" of elements chosen
  suppPos = findall(x->in(x,randEl), FEAparams.elementIDmatrix)
  # build compact dispBC with pin positions chosen
  for pin in 1:length(unique(randEl))
    dispBC[pin,1] = suppPos[pin][1]
    dispBC[pin,2] = suppPos[pin][2]
    dispBC[pin,3] = 3
  end
  # get node positions of pins
  myCells = [grid.cells[g].nodes for g in randEl]
  pos = vec(reshape([myCells[ele][eleNode] for eleNode in 1:length(myCells[1]), ele in keys(randEl)], (:,1)))
  nodeSets = Dict("supps" => pos)
  return nodeSets, dispBC
end

# generate vector with n random and different integer values between 1 and val
function randDiffInt(n, val)
  global randVec = zeros(Int, n)
  randVec[1] = rand(1:val)
  for ind in 2:n
    global randVec[ind] = rand(1:val)
    while in(randVec[ind], randVec[1:ind-1])
      global randVec[ind] = rand(1:val)
    end
  end
  return randVec
end

# Create the node set necessary for specific and well defined support conditions
function simplePins!(type, dispBC, FEAparams)
  type == "rand" && (type = rand(["left" "right" "top" "bottom"]))
  if type == "left"
    # Clamp left boundary of rectangular domain.
    fill!(dispBC, 4)
    # clamped nodes
    firstCol = [(n-1)*(FEAparams.meshSize[1]+1) + 1 for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .+ 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "right"
    # Clamp right boundary of rectangular domain.
    fill!(dispBC, 6)
    # clamped nodes
    firstCol = [(FEAparams.meshSize[1]+1)*n for n in 1:(FEAparams.meshSize[2]+1)]
    secondCol = firstCol .- 1
    nodeSet = Dict("supps" => vcat(firstCol, secondCol))
  elseif type == "bottom"
    # Clamp bottom boundary of rectangular domain
    fill!(dispBC, 5)
    # clamped nodes
    nodeSet = Dict("supps" => [n for n in 1:(FEAparams.meshSize[1]+1)*2])
  elseif type == "top"
    # Clamp top boundary of rectangular domain
    fill!(dispBC, 7)
    # clamped nodes
    # (first node of second highest line of nodes) : nodeQuant
    # (ndx)*(ndy-2)+1 : (ndx)*(ndy)
    nodeSet = Dict("supps" => [n for n in ((FEAparams.meshSize[1]+1)*(FEAparams.meshSize[2]-1)+1):((FEAparams.meshSize[1]+1)*((FEAparams.meshSize[2]+1)))])
  end
  return nodeSet, dispBC
end

function plotData(dens, meshSize, fullVF, fullComps, fullError)
  fig = Figure(resolution = (1200, 600));
  display(fig)
  colSize = 500
  # labels
  Label(fig[1, 1], "Topology", textsize = 20)
  colsize!(fig.layout, 1, Fixed(colSize))
  Label(fig[1, 2], "VF history", textsize = 20)
  colsize!(fig.layout, 2, Fixed(colSize))
  Label(fig[3, 1], "Objective history", textsize = 20)
  Label(fig[3, 2], "Error history", textsize = 20)
  # plot topology
  heatmap(fig[2,1], 1:meshSize[1], meshSize[2]:-1:1, quad(meshSize..., dens)')
  # plot history of VF
  lines(fig[2,2], 1:length(fullVF), fullVF)
  # plot history of compliances
  lines(fig[4,1], 1:length(fullComps), fullComps)
  # plot history of error
  lines(fig[4,2], 1:length(fullError), fullError)
end