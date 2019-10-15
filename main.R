#!/usr/bin/R
############################################################
# GenieImpl1.R
# Ryan Murphy
# Very simple implementation of forward-pass of adaptive path layer from GeniePath paper
# AAAI 2019: https://arxiv.org/pdf/1802.00910.pdf
#
# Assuming intput is an igraph object with vertex features.
#
############################################################

setwd("/media/TI10716100D/_Files/1_Purdue/_Research/Network_Brain/CONVOLUTIONAL_NNET_GRAPHS/GeniePath/GeniePath_BasicImpl/GeniePath_BasicImpl")
library(igraph)
library(hashmap)
source("GenieMakeData.R")  

N <- 100
TOTAL_LAYERS <- 6 # I am including h^(0).  So 3 layers means 2 updates
UPDATES <- TOTAL_LAYERS - 1 # capital T from the paper
HIDDEN_DIM <- 8 # assume for now that hidden dim is same as data dim

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
  }else if(class(X) == "numeric"){
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
# (1) Project x and y into some representation with Wx and Wy
# (2) Squash them into the range -1 to 1 with tanh
# (3) Apply another dot product that "learns" how to interpret the representation
#       and determine importance
# 
alpha <- function(v, G, tt, Ws, Wd, v.weight){ # v.weight is just a weight parameter...nothing to do w/ vertex v
    nbers.v <- neighbors(G, v)
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
    EMBED <- tanh(X%*%Ws + Y%*%Wd) # squashed representation of these two hidden vectors
    return(
      list(
        attn = softmax.vec(EMBED %*% v.weight),
        h.nber.mat = Y
      )
    )
}
# ---------------------------------------------------------------
# Main Algo: Compute all H (hidden layers) using adaptive path layers
# I am ignoring h^(0) = WX, and just simulating h^(0) directly..
# ---------------------------------------------------------------
runAlgo <- function(G, updates, hiddenDim){
  set.seed(3)  # use same weight matrices
  N <- length(V(G))
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
  return(G)
}

# Compare vertex features at indices i, j in two graphs
compare <- function(i, j){
  # G.new and G.prime.new are read from the global...
  # ...namespace for simplicity
  all.equal(V(G.new)[i]$vertex_attr,
            V(G.prime.new)[j]$vertex_attr)
}

# =============================================================================
# Main logic begins
# =============================================================================
# Create an igraph object with features, G, and an isomorphic graph G.prime
# >> User can select which rows to swap, to check isomorphic invariance
# >> Can also swap random rows using sample.
swaps <- rbind(c(2,4), c(3,9))

print("Making a toy graph and applying an isomorphism to it")
graph.dat <- make.toy.data(swaps)  
G <- graph.dat$G
G.prime <- graph.dat$G.prime 

# run through adaptive path layer.
print("Forwarding both graphs through the GeniePath layer")
G.new <- runAlgo(G=G, updates = UPDATES, hiddenDim = HIDDEN_DIM)
G.prime.new <- runAlgo(G=G.prime, updates = UPDATES, hiddenDim = HIDDEN_DIM)
#
# Test of isomorphic invariance.
#
# Manually check that the vertex attributes are the same in the two graphs
# (before and after isomorphism),
# up to relabeled indices.
#
print("Testing that vertex representations are the same between two graphs, up to relabeling.")
comparisons <- list()
for(ii in 1:N){
    if(!(ii %in% swaps)){
      res <- compare(ii, ii)
      comparisons <- append(comparisons, res)
    }
}
for(rr in 1:nrow(swaps)){
  res <- compare(swaps[rr, 1], swaps[rr, 2])
  comparisons <- append(comparisons, res) 
}
print("Test whether the vertex representations are the same between the original and permuted graphs")
print(all(unlist(comparisons)))
