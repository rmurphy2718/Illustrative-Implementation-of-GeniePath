#!/usr/bin/R
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
# ~~~ 
# Helper functions
# ~~~ 
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
    ll[[ii]] <- matrix(nrow = numrows, ncol = numcols, data = rnorm(numrows*numcols, 0, 0.4))
  }
  return(ll)
}
#
# Softmax function
# 
softmax.vec <- function(X){
    if(class(X) == "matrix" && all(dim(X) > 1 ) ){
        stop("Enter a vector")
    }
    return(
        exp(X - log(sum(exp(X))))
    )
}

#
# Alpha function
#  >> Attention function
# alpha(v, neighbors(v)) -> a numeric (probability) vector giving importance of each neighbor
#
# Explanation of this function, alpha(x,y) from the paper
# (1) Project x and y into some embedding with Wx and Wy
# (2) Squash them into the range -1 to 1 with tanh
# (3) Apply another dot product that "learns" how to interpret the embedding
#       and determine importance
# 
alpha <- function(v, G, tt, Ws, Wd, v.weight){ # v.weight is just a weight parameter...nothing to do w/ vertex v
    nbers.v <- neighbors(G, vv)
    num.nbers <- length(nbers.v)
    ##### A matricized implementation
    # Get hidden vector h_i for v at time t
    h.i <- v$vertex_attr[[1]][, tt]
    # Put in matrix so that each row is the same
    X <- matrix(nrow = num.nbers, byrow = TRUE,
                 data = rep(h.i, num.nbers))
    # Get hidden vectors of each neighbor, then put into a matrix
    Y <- matrix(nrow = num.nbers, ncol = ncol(X))
    ro <- 1
    for(uu in nbers.v){
        Y[ro, ] <- V(G)[uu]$vertex_attr[[1]][, tt]
        ro <- ro + 1
    }
    #
    # Compute the heart of equation (8)
    EMBED <- tanh(X%*%Ws + Y%*%Wd) # squashed embedding of these two hidden vectors
    return(
      list(
        attn = softmax.vec(EMBED %*% v.weight),
        h.nber.mat = Y
      )
    )
}
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Main
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
runAlgo <- function(N, updates, hiddenDim){
  # Generate weight matrix lists
  # Assume weight matrices are square for now
  # >> This means (number units in) = (number units out)
  # Equiv, we don't need a new programming parameter, just "NUMBER HIDDEN"
  W  <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wi <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wf <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wo <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wc <- initMatrices(numUpdates = UPDATES, numrows = HIDDEN_DIM, numcols = HIDDEN_DIM)
  Wd <- matrix(nrow = HIDDEN_DIM, ncol = HIDDEN_DIM, data = rnorm(HIDDEN_DIM*HIDDEN_DIM, 0, 0.4))
  Ws <- matrix(nrow = HIDDEN_DIM, ncol = HIDDEN_DIM, data = rnorm(HIDDEN_DIM*HIDDEN_DIM, 0, 0.4))
  v.weight <- matrix(ncol = 1, data = rnorm(HIDDEN_DIM, 0, 3))
  #
  #
  CC <- list()
  CC[[1]] <- matrix(nrow = N, ncol = hiddenDim, data = 0)
  #
  for(tt in 1:updates){
      #
      CC[[tt + 1]] <- matrix(nrow = N, ncol = hiddenDim)
      #
      for(ii in 1:N){
          # Compute alpha masks
          vv <- V(G)[ii]
          alph.ret <- alpha(vv, G, tt, Ws, Wd, v.weight)
          attn <- alph.ret$attn
          h.nber.mat <- alph.ret$h.nber.mat # rows are for each vector
          #
          # Apply mask to each neighbor attribute
          # (
          #   note:
          #   using as.numeric makes sure we get
          #   element 1 attn * row 1 h.nber.mat
          # )
          stopifnot(nrow(attn) == nrow(h.nber.mat))
          nber.mask <- as.numeric(attn) * h.nber.mat
          # 
          # Sum up nbers
          summedNbers <- colSums(nber.mask) 
          # finall, get htemp
          h.temp <- summedNbers %*% W[[tt]]
          #
          # LSTM Updates
          #
          i.gate <- sigmoid(h.temp %*% Wi[[tt]])
          f.gate <- sigmoid(h.temp %*% Wf[[tt]])
          o.gate <- sigmoid(h.temp %*% Wo[[tt]])
          C.tilde <- tanh  (h.temp %*% Wc[[tt]])
          #
          # Update cell
          CC[[tt + 1]][ii,] <- f.gate * CC[[tt]][ii,] + i.gate * C.tilde
          #
          # Update hidden state
          V(G)[ii]$vertex_attr[[1]][, tt + 1] <- o.gate * tanh(CC[[tt + 1]][ii,])
    }
  }
}

runAlgo(N=N, updates = UPDATES, hiddenDim = HIDDEN_DIM)
