#' Social scatterplot
#'
#' A plot of social data against its socially lagged values
#'
#' @param x a numeric vector of social data.
#' @param S a social correlation matrix.
#' @param ... further arguments to be passed to \code{\link{[graphics]{plot}}.
#'
#' @return None
#'
#' @examples
#' A = matrix(c(0,1,0,1,0,
#'              1,0,0,1,1,
#'              0,0,0,1,1,
#'              1,1,1,0,0,
#'              0,1,1,0,0), nrow=5)
#' S = social.cor.matrix(A)
#' x = rnorm(nrow(A))
#' social.plot(x, S, ylim=c(min(x),max(x)), xlab="x", ylab="Socially lagged x")
#' abline(0, 1, lty=2)
#'
#' @export
social.plot = function(x, S, ...) {
	n = nrow(S)
	diag(S) = 0
	rowSumS = rowSums(S)
	rowSumS[rowSumS==0] = 1
	S = S/rowSumS
	y = S %*% x
	plot(x, y, ...)
}
