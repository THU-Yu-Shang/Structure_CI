#' Social correlation matrix
#'
#' Calculates the social correlation matrix for a given network
#'
#' @param A a (possibly weighted) adjacency matrix.
#' @param max.depth the maximum length of the paths to the use.
#' @param n.pilot parameter to be passed to \code{all.paths}: the number of naive paths to generate.
#' @param n.estimate parameter to be passed to \code{all.paths}: the number of paths to generate.
#'
#' @return The calculated social correlation matrix.
#'
#' @examples
#' A = matrix(c(0,1,0,1,0,
#'              1,0,0,1,1,
#'              0,0,0,1,1,
#'              1,1,1,0,0,
#'              0,1,1,0,0), nrow=5)
#' S = social.cor.matrix(A)
#'
#' @export
social.cor.matrix = function(A, max.depth=nrow(A), n.pilot=5000, n.estimate=10000) {
	n = nrow(A)
	S = matrix(1, nrow=n, ncol=n)
	for(i in 1:(n-1)) {
		for(j in (i+1):n) {
			paths = social.all.paths(A, i, j, max.depth, n.pilot, n.estimate)
			rho = 0
			if(nrow(paths) > 0) {
				for(t in 1:nrow(paths)) {
					path = paths[t, paths[t,]>0]
					p = c()
					for(k in 1:(length(path)-1)) {
						w = A[path[k], path[k+1]]
						k1 = sum(A[path[k],])
						k2 = sum(A[path[k+1],])
						p = c(p, (w^2)/(k1*k2))
					}
					rho = rho + prod(p)
				}
			}
			S[i,j] = rho
			S[j,i] = rho
		}
	}
	return(S)
}
