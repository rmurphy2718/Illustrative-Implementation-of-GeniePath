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
D <- 3 # number features
TOTAL_LAYERS <- 3 
HIDDEN_DIM <- 4
# Create random graphs from igraph
G <- erdos.renyi.game(n = N, p = "0.4", mode = "undirected")
A <- as.matrix(as_adj(G))
#
# Initialize attribute lists
# 
for(vv in V(G)){
  Hmat <- matrix(nrow = D, ncol = TOTAL_LAYERS, data = NA)
  Hmat[,1] <- rnorm(D, 0, 3)
  V(G)[vv]$vertex_attr <- list(Hmat) # igraph does not accept a matrix-attribute unless you specify it as a list
}
