#include <Rcpp.h>
using namespace Rcpp;

//// [[Rcpp::export]]
int weightedsample(NumericVector nodes, NumericVector weights) {
	// Normalise weights
	int n = weights.length();
	int sumOfWeights = 0;
	for(int i=0; i<n; i++) {
		sumOfWeights += weights(i);
	}
	for(int i=0; i<n; i++) {
		weights(i) = weights(i)/sumOfWeights;
	}
	// Choose random weighted sample
	NumericVector randomNumber = runif(1);
	for(int i=0; i<n; i++) {
		if(randomNumber(0) < weights(i)) {
			return nodes(i);
		}
		else {
			randomNumber(0) -= weights(i);
		}
	}
	return nodes(n-1); // Shouldn't need this
}

//// [[Rcpp::export]]
NumericVector findnodes(NumericMatrix adjMatrix, int currentNode) {
	int numNodes = adjMatrix.nrow();
	NumericVector tmp(numNodes);
	int numNeighbours = 0;
	// std::cout<<adjMatrix;
	for(int j=0; j<numNodes; j++) {
		if(adjMatrix(currentNode, j) > 0) {
			tmp(numNeighbours) = j;
			numNeighbours++;

		}

	}
	NumericVector nextNodes(numNeighbours);
	for(int j=0; j<numNeighbours; j++) {
		nextNodes(j) = tmp(j);
	}
	return nextNodes;
}

//// [[Rcpp::export]]
bool isuniquepath(NumericMatrix paths, int i, NumericVector keep) {
	int numNodes = paths.ncol();
	for(int j=0; j<i; j++) {
		if(keep(j)>0) {
			int sum = 0;
			for(int k=0; k<numNodes; k++) {
				if(paths(i, k) == paths(j, k)) {
					sum++;
				}
			}
			if(sum == numNodes) {
				return false;
			}
		}
	}
	return true;
}

//// [[Rcpp::export]]
bool containsendnode(NumericMatrix paths, int i, int endNode) {
	int numNodes = paths.ncol();
	for(int j=0; j<numNodes; j++) {
		if(paths(i, j) == endNode) {
			return true;
		}
	}
	return false;
}

NumericMatrix clean(NumericMatrix paths, int endNode) {
  int numPaths = paths.nrow();
  int numNodes = paths.ncol();
  // std::cout<<numPaths;
  // Keep path?
  NumericVector keep(numPaths);
  for (int z=0; z<numPaths; z++){keep(z)=0;}  
  int numUniquePaths = 0;
  for(int i=0; i<numPaths; i++) {
    // if(containsendnode(paths, i, endNode) && isuniquepath(paths, i, keep)) 
      if(containsendnode(paths, i, endNode)){
      keep(i) = 1;
      numUniquePaths++;
    }
  }
  // Remove unwanted paths
  NumericMatrix goodPaths(numUniquePaths, numNodes);
  int n = 0;
  for(int i=0; i<numPaths; i++) {
    if(keep(i)>0) {
      for(int k=0; k<numNodes; k++) {
        goodPaths(n, k) = paths(i, k);
      }
      n++;
    }
  }
  return goodPaths;
}


NumericVector clean_g(NumericMatrix paths, NumericVector g_in, int endNode) {
  int numPaths = paths.nrow();
  int numNodes = paths.ncol();
  // std::cout<<numPaths;
  // Keep path?
  NumericVector keep(numPaths);
  for (int z=0; z<numPaths; z++){keep(z)=0;}  
  int numUniquePaths = 0;
  for(int i=0; i<numPaths; i++) {
    // if(containsendnode(paths, i, endNode) && isuniquepath(paths, i, keep)) 
      if(containsendnode(paths, i, endNode))
      {
      keep(i) = 1;
      numUniquePaths++;
    }
  }
  // Remove unwanted paths
  NumericVector goodg(numUniquePaths);
  int n = 0;
  for(int i=0; i<numPaths; i++) {
    if(keep(i)>0) {
      for(int k=0; k<numNodes; k++) {
        goodg(n) = g_in(i);
      }
      n++;
    }
  }
  return goodg;
}

//// [[Rcpp::export]]
NumericVector clean2(NumericMatrix paths, int endNode) {
	int numPaths = paths.nrow();
	int numNodes = paths.ncol();
	// std::cout<<numPaths;
	// Keep path?
	NumericVector keep(numPaths);
	for (int z=0; z<numPaths; z++){keep(z)=0;}  
	int numUniquePaths = 0;
	for(int i=0; i<numPaths; i++) {
		// if(containsendnode(paths, i, endNode) && isuniquepath(paths, i, keep)) 
		if(containsendnode(paths, i, endNode)){
			keep(i) = 1;
			numUniquePaths++;
		}
	}
	std::cout<<numUniquePaths<<'\n';
	// Remove unwanted paths
	NumericMatrix goodPaths(numUniquePaths, numNodes);
	int n = 0;
	for(int i=0; i<numPaths; i++) {
		if(keep(i)>0) {
			for(int k=0; k<numNodes; k++) {
				goodPaths(n, k) = paths(i, k);
			}
			n++;
		}
	}
	return keep;
}

//// [[Rcpp::export]]
NumericVector naivepaths(NumericMatrix adjMatrix, int startNode, int endNode, int nPilot) {
	int numNodes = adjMatrix.nrow();
	NumericMatrix naivePaths(nPilot, numNodes);
	NumericVector gn(nPilot);
	for(int i=0; i<nPilot; i++) {
		for(int j=0; j<numNodes; j++) {
			naivePaths(i, j) = -1;
		}
	}
	for(int i=0; i<nPilot; i++) {
		NumericMatrix adjMatrixCopy = clone(adjMatrix);
		naivePaths(i, 0) = startNode;
		int pathLength = 1;
		gn(i)=1;
		int currentNode = startNode;
		while(currentNode != endNode) {
			NumericVector nextNodes = findnodes(adjMatrixCopy, currentNode);

			if(nextNodes.length() == 0) {
				break;
			}
			else {
			  gn(i) = gn(i)/nextNodes.length();
				for(int j=0; j<numNodes; j++) {
					// adjMatrixCopy(currentNode, j) = 0;
					adjMatrixCopy(j, currentNode) = 0;
				}
				NumericVector weights(nextNodes.length(), 1.0);
				currentNode = weightedsample(nextNodes, weights);
				naivePaths(i, pathLength) = currentNode;
				pathLength++;
			}
		}
	}
	// std::cout<<naivePaths<<'\n';
	NumericMatrix goodPaths = clean(naivePaths, endNode);
	NumericVector goodg = clean_g(naivePaths, gn, endNode);
	
	// int numNodes = adjMatrix.nrow();
	int numUniquePaths = goodPaths.nrow();
	// std::cout<<"num:"<<numUniquePaths;
	// Get the lengths of all paths
	NumericVector pathLengths(numUniquePaths);
	for(int i=0; i<numUniquePaths; i++) {
	  int sum = 0;
	  for(int j=1; j<numNodes; j++) {
	    if(goodPaths(i, j) > -1) {
	      sum++;
	    }
	  }
	  pathLengths(i) = sum;
	}
	// std::cout<<pathLengths<<'\n';
	// Calculate the length-distribution vector, p
	NumericVector p(numNodes-1);
	for(int k=1; k<numNodes; k++) {
	  double n1 = 0.0;
	  double n2 = 0.0;
	  for(int i=0; i<numUniquePaths; i++) {
	    if(pathLengths(i) == k) {
	      n1 = n1 + 1.0/goodg(i);
	    }
	    if(pathLengths(i) >= k and adjMatrix(goodPaths(i,k-1),endNode)==1) {
	      n2 = n2 + 1.0/goodg(i);
	    }
	  }
	  p(k-1) = n1/n2;
	  std::cout<<n1<<' '<<n2<<'\n';
	  float pn_count=0;
	  for (int s=0;s<numUniquePaths;s++)
	  {pn_count = pn_count + 1.0/goodg(s);
	    // std::cout<<pn_count<<'\n';
	  }
	  pn_count = pn_count/nPilot;
	  // std::cout<<pn_count<<'\n';
	  if(p(k-1) == 0 or (n1==0 and n2==0)) {
	    p(k-1) = 1.0/pn_count;
	  }
	  if(p(k-1) == 1) {
	    p(k-1) = 1.0 - 1.0/pn_count;
	  }
	  // if(p(k) > 1) {
	  //   p(k) = 0.00000001 ;
	  // }
	}
	std::cout<<p<<'\n';
	return p;
	// return goodPaths;
}

//// [[Rcpp::export]]
// NumericVector lengthdistribution(NumericMatrix adjMatrix, NumericMatrix goodPaths) {
// 	int numNodes = adjMatrix.nrow();
// 	int numUniquePaths = goodPaths.nrow();
// 	// std::cout<<"num:"<<numUniquePaths;
// 	// Get the lengths of all paths
// 	NumericVector pathLengths(numUniquePaths);
// 	for(int i=0; i<numUniquePaths; i++) {
// 		int sum = 0;
// 		for(int j=0; j<numNodes; j++) {
// 			if(goodPaths(i, j) > -1) {
// 				sum++;
// 			}
// 		}
// 		pathLengths(i) = sum;
// 	}
// 	// Calculate the length-distribution vector, p
// 	NumericVector p(numNodes-1);
// 	for(int k=0; k<numNodes; k++) {
// 		double n1 = 0.0;
// 		double n2 = 0.0;
// 		for(int i=0; i<numUniquePaths; i++) {
// 			if(pathLengths(i) == k) {
// 				n1 = n1 + 1.0;
// 			}
// 			if(pathLengths(i) >= k) {
// 				n2 = n2 + 1.0;
// 			}
// 		}
// 		p(k) = n1/n2;
// 		// std::cout<<n1<<' '<<n2<<'\n';
// 		if(p(k) == 0) {
// 			p(k) = 1.0/(double)numUniquePaths;
// 		}
// 		if(p(k) == 1) {
// 			p(k) = 1.0 - 1.0/(double)numUniquePaths;
// 		}
// 	}
// 	return p;
// 	
// }

//// [[Rcpp::export]]
NumericVector estimatedpaths(NumericMatrix adjMatrix, int startNode, int endNode, int maxDepth, int nEstimation, NumericVector p) {
	int numNodes = adjMatrix.nrow();
	NumericMatrix paths(nEstimation, numNodes);
	NumericVector g(nEstimation);
	for(int i=0; i<nEstimation; i++) {
		for(int j=0; j<numNodes; j++) {
			paths(i, j) = -1;
		}
	}
	for(int i=0; i<nEstimation; i++) {
		NumericMatrix adjMatrixCopy = clone(adjMatrix);
		paths(i, 0) = startNode;
		int pathLength = 1;
		int currentNode = startNode;
		g(i) = 1;
		while(currentNode != endNode) {
			if(adjMatrixCopy(currentNode, endNode) == 0) {
				NumericVector nextNodes = findnodes(adjMatrixCopy, currentNode);
			  
				if(nextNodes.length() == 0) {
					break;
				}
				else {
					for(int j=0; j<numNodes; j++) {
						adjMatrixCopy(currentNode, j) = 0;
						adjMatrixCopy(j, currentNode) = 0;
					}
					NumericVector weights(nextNodes.length(), 1.0);
					currentNode = weightedsample(nextNodes, weights);
					g(i) = g(i)/nextNodes.length();
					paths(i, pathLength) = currentNode;
					pathLength++;
					if(pathLength >= maxDepth) {
						break;
					}
				}
			}
			else {
				NumericVector nextNodes = findnodes(adjMatrixCopy, currentNode);
				if(nextNodes.length() == 1) {
					paths(i, pathLength) = endNode;
					break;
				}
				else {
					NumericVector randomNumber = runif(1);
					if(randomNumber(0) < p(pathLength)) {
						paths(i, pathLength) = endNode;
					  g(i) = g(i) * p(pathLength-1);
						break;
					}
					else {
						adjMatrixCopy(currentNode, endNode) = 0;
						adjMatrixCopy(endNode, currentNode) = 0;
						g(i) = g(i) * (1-p(pathLength-1));
					}
				}
			}
		}
	}
	NumericVector allPaths = clean2(paths, endNode);
	// std::cout<<g;
	float p_count=0;
	for (int v=0;v<allPaths.length();v++)
	{p_count = p_count + allPaths(v)/g(v);
	 // std::cout<<g(v)<<'\n';
	}
	// std::cout<<p_count;
	  // std::cout<<'\n';}
	// std::cout<<allPaths(0);
	// std::cout<<allPaths(0)/g(0);
	std::cout<<p_count/10000;
	return allPaths;
}

// [[Rcpp::export]]
NumericVector allpaths(NumericMatrix adjMatrix, int startNode, int endNode, int maxDepth, int nPilot, int nEstimation) {

	// One to zero indexing (R to Cpp)
	startNode--;
	endNode--;
	
	// Naive path generation
	NumericVector p = naivepaths(adjMatrix, startNode, endNode, nPilot);	
	
	// Calculate the length-distribution vector
	// NumericVector p = lengthdistribution(adjMatrix, goodPaths);

	// Length-distribution method
	NumericVector allPaths = estimatedpaths(adjMatrix, startNode, endNode, maxDepth, nEstimation, p);
	
 	// Zero to one indexing (Cpp to R)
 	// for(int i=0; i<allPaths.nrow(); i++) {
 	// 	for(int j=0; j<adjMatrix.nrow(); j++) {
 	// 		allPaths(i, j) += 1;
 	// 	}
 	// }
    return allPaths;
}
