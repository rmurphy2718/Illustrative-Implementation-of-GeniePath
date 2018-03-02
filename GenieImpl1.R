############################################################
# GenieImpl1.R
# Ryan Murphy
# Quck and dirty implementation of adaptive path layer from GeniePath
# "GeniePath: Graph Neural Networks with Adaptive Receptive Paths"
# Ziqi Liu et al, currently on arXiv.
############################################################
library(igraph)
# ========================================
# Create toy data
# ========================================
set.seed(17)
N <- 20
TOTAL_LAYERS <- 3 # I am including h^(0).  So 3 layers means 2 updates
UPDATES <- TOTAL_LAYERS - 1 # capital T from the paper
HIDDEN_DIM <- 4 # assume for now that hidden dim is same as data dim
# Create random graphs from igraph
G <- erdos.renyi.game(n = N, p = "0.4", mode = "undirected")
A <- as.matrix(as_adj(G))
#
# Initialize attribute lists
# 
for(vv in V(G)){
  Hmat <- matrix(nrow = HIDDEN_DIM, ncol = TOTAL_LAYERS, data = NA)
  Hmat[,1] <- rnorm(HIDDEN_DIM, 0, 3)
  V(G)[vv]$vertex_attr <- list(Hmat) # igraph does not accept a matrix-attribute unless you specify it as a list
}
# ========================================
# Algo: Compute all H using adaptive path layers
# ========================================
sigmoid <- function(X){
  sigmoid.scalar <- function(x){
    if(x >  99){ return(1) }
    if(x < -99){ return(0) }
    return( 1/(1+exp(-x)) )    
  }
  # input check
  if(class(X) == "matrix"){ # The W'h operations will result in a matrix.
      #
      # We need to check if the input number to sigmoid is too large, and threshold if so.
      # Since we need an IF condition, we cannot apply a function to a matrix, as in f(matrix);
      #  instead, we must apply over the matrix's appropriate dimensions
      # Hence, the first step is to figure out the dimensions
      validDims <- which(dim(X) > 1) 
      print(validDims)
      retMat <- apply(X, validDims, sigmoid.scalar)
      return(retMat)
  }
  #
  #
  #
  if(class(X) == "numeric"){
      return(sigmoid.scalar(X))
  }
}

initMatrices <- function(numUpdates, numrows, numcols){
  ll <- list()
  for(ii in 1:numUpdates){
    ll[[ii]] <- matrix(nrow = numrows, ncol = numcols, data = rnorm(numrows*numcols, 0, 5))
  }
  return(ll)
}
runAlgo <- function(){
  # Generate weight matrix lists
  # Assume weight matrices are square for now
  # >> This means (number units in) = (number units out)
  # Equiv, we don't need a new programming parameter, just "NUMBER HIDDEN"
  W <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
}