############################################################
# <<<<<<<<<<<<INCLUDE FILE>>>>>>>>>>>>>>>>>>>>
# GenieMakeData.R 
# <<< In >>> 
# GenieImpl1.R
# Ryan Murphy
# Quck and dirty implementation of adaptive path layer from GeniePath
# "GeniePath: Graph Neural Networks with Adaptive Receptive Paths"
# Ziqi Liu et al, currently on arXiv.
#
# This file makes 2 isomorphic graphs w/ vertex attributes that can be used
#
############################################################
# ========================================
# Create toy data
# ========================================
set.seed(17)
N <- 20
TOTAL_LAYERS <- 3 # I am including h^(0).  So 3 layers means 2 updates
UPDATES <- TOTAL_LAYERS - 1 # capital T from the paper
HIDDEN_DIM <- 4 # assume for now that hidden dim is same as data dim
#
# Create random graph from igraph
#
while(TRUE){ # If R had "do-while", that's what I'd use
  G <- erdos.renyi.game(n = N, p = "0.4", mode = "undirected")
  if(components(G)$no == 1){
    break; 
  }
}
A <- as.matrix(as_adj(G))
# -----------------------------
# Create isomorphism
#  we could just apply "swaps" function to rows and cols
#  but instead I'll use a permutation matrix to reinforce my knowledge that
#  A and PAP' lead to isomorphic graphs
# -----------------------------
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
# Make permutation matrix
P <- diag(N)
swaps <- rbind(c(2,4), c(3, 9))
P <- rowSwaps(mat = P, swapsMat = swaps)

A.prime <- P %*% A %*% t(P)
G.prime <- graph_from_adjacency_matrix(A.prime, "undirected")
isomorphic(G, G.prime)
# -----------------------------
# Initialize attribute lists
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
#### test isomorphism still holds
#### i.e. vertex attrs same
# Assume that no two attributes in the same graph are the same
#  which is almost impossible to be violated given our construction
for(jj in 1:N){
  # See if this is permuted
  isPermuted <- jj %in% as.numeric(swaps)
  # Compare jj and jj to see if they are the same
  if(isPermuted == FALSE){
    attrsEqual <- as.logical(all.equal(V(G)[jj]$vertex_attr,
                                       V(G.prime)[jj]$vertex_attr
    ))# will check for non-true soon
  }
  if(isPermuted == TRUE){
    kk <- dict.iso[[jj]]
    attrsEqual <- as.logical(all.equal(V(G)[jj]$vertex_attr,
                                       V(G.prime)[kk]$vertex_attr
    ))
  }
  attrsEqual <- ifelse(is.na(attrsEqual), FALSE, TRUE)
  #
  if(attrsEqual != TRUE){
    print(paste("problem at", jj))
  }
}
