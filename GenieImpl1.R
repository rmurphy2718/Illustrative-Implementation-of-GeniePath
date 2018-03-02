############################################################
# GenieImpl1.R
# Ryan Murphy
# Quck and dirty implementation of adaptive path layer from GeniePath
# "GeniePath: Graph Neural Networks with Adaptive Receptive Paths"
# Ziqi Liu et al, currently on arXiv.
############################################################
setwd("/media/TI10716100D/_Files/1_Purdue/_Research/Network_Brain/CONVOLUTIONAL_NNET_GRAPHS/GeniePath/GeniePath_BasicImpl/GeniePath_BasicImpl")
library(igraph)
library(hashmap)
source("GenieMakeData.R")
# ========================================
# Algo: Compute all H using adaptive path layers
# I am ignoring h^(0) = WX, and just simulating h^(0) directly..
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

runAlgo <- function(N, updates, hiddenDim){
  # Generate weight matrix lists
  # Assume weight matrices are square for now
  # >> This means (number units in) = (number units out)
  # Equiv, we don't need a new programming parameter, just "NUMBER HIDDEN"
  W <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wi <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wf <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wo <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wc <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wd <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Ws <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  v.weight <- matrix(ncol = 1, data = rnorm(HIDDEN_DIM, 0, 3))
  #
  #
  #
}
#
# Softmax function
# 
softmax.vec <- function(X){
    if(class(X) == "matrix" && max(dim(X)) > 1 ){
        stop("Enter a vector")
    }
    return(
        exp(X - log(sum(exp(X))))
    )
}

runAlgo(N=N, updates = UPDATES, hiddenDim = HIDDEN_DIM)
