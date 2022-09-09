#' Socially-correlated data
#'
#' Generate data with a given social signal
#'
#' @param S a social correlation matrix.
#' @param target.rho the target social signal of the returned data.
#' @param x0 a numeric vector of data to use as the basis for the optimisation.
#' 
#' @return A numeric vector of data with a given social signal.
#'
#' @examples
#' A = matrix(c(0,1,0,1,0,
#'              1,0,0,1,1,
#'              0,0,0,1,1,
#'              1,1,1,0,0,
#'              0,1,1,0,0), nrow=5)
#' S = social.cor.matrix(A)
#' x = social.random(S, 0.7)
#' s = social.signal(x, S)
#'
social.random = function(S, target.rho, x0=rnorm(nrow(S))) {
	ssfcn = function(x) {
		x = x - mean(x)
		I = social.signal(x, S)$Is
		return(abs(target.rho - I))
	}
	out = optim(x0, ssfcn, control=list(maxit=1000))
	val = out$par
	return(val)
}
