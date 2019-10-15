############################################################
# This file makes 2 isomorphic graphs w/ vertex attributes that can be used
# as input to GeniePath
############################################################
require(igraph)
require(hashmap)


#  swap rows in a matrix
rowSwaps <- function(mat, swapsMat){
  # swapsMat holds the rows we want to switch
  # do input checking on it
  stopifnot(class(swapsMat)=="matrix" && ncol(swapsMat) == 2)
  #
  for(rr in 1:nrow(swapsMat)){
    ii <- swapsMat[rr, 1] # first index in swap
    jj <- swapsMat[rr, 2] # second index
    tmpRow <- mat[ii, ]
    mat[ii, ] <- mat[jj, ]
    mat[jj, ] <- tmpRow
  }
  return(mat)
}

make.toy.data <- function(swaps){
  #
  # Create random graph from igraph
  #
  while(TRUE){ # If R had "do-while", that's what I'd use
    G <- sample_smallworld(1, N, 3, 0.01)
    if(is.connected(G)){  
      break; 
    }
  }
  A <- as.matrix(as_adj(G))
  #
  # Create an isomorphism manually for illustration purposes
  #   > can construct a permutation matrix by swapping rows of an identity matrix.s
  P <- diag(N)
  P <- rowSwaps(mat = P, swapsMat = swaps)
  
  A.prime <- P %*% A %*% t(P)
  G.prime <- graph_from_adjacency_matrix(A.prime, "undirected")
  # -----------------------------
  # Initialize attributes in the graphs
  # -----------------------------
  # Create a dictionary of labels in the isomorphic graphs
  #  (if f is the isomorphism, this maps f(u) = v)
  dict.iso <- hashmap(1:N, 1:N)
  dict.iso[[ swaps[,1] ]] <- swaps[,2]
  dict.iso[[ swaps[,2] ]] <- swaps[,1]
  # Simulate a "hidden" matrix for each node
  for(vv in V(G)){
    # Create H
    # I am ignoring h^(0) = WX, and just simulating h^(0) directly..
    Hmat <- matrix(nrow = HIDDEN_DIM, ncol = TOTAL_LAYERS, data = NA)
    Hmat[,1] <- rnorm(HIDDEN_DIM, 0, 3)
    # Add to G
    V(G)[vv]$vertex_attr <- list(Hmat) # igraph does not accept a matrix-attribute unless you specify it as a list
    # Add to G.prime
    uu <- dict.iso[[vv]]
    V(G.prime)[uu]$vertex_attr <- list(Hmat) 
  }
  return(list(G=G, G.prime=G.prime))
}
