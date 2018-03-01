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
N <- 20
D <- 3 # number features
totalHops <- 3 
hiddenDim <- 4
# Create random graphs from igraph
G <- erdos.renyi.game(n = N, p = "0.4", mode = "undirected")
A <- as.matrix(as_adj(G))
# Insert attribute vectors
G <- set_vertex_attr(G, "node_attrs", index = V(G), list())
#
# Enter raw data values


