#' Social signal
#'
#' Calculates the social signal for a given variable (essentially
#'     just Moran's I, but using the social correlation matrix 
#'     as the weights)
#'
#' @param x a numeric vector of social data.
#' @param S a social correlation matrix.
#' 
#' @return A list containing the computed global social signal (\code{Is}), 
#'     the P-value of a test of the null hypothesis that there
#'     is no social autocorrelation (\code{p.value}), and the 
#'     local social signal for each node (\code{I.local}).
#'
#' @examples
#' A = matrix(c(0,1,0,1,0,
#'              1,0,0,1,1,
#'              0,0,0,1,1,
#'              1,1,1,0,0,
#'              0,1,1,0,0), nrow=5)
#' S = social.cor.matrix(A)
#' x = rnorm(nrow(A))
#' s = social.signal(x, S)
#'
#' @export
social.signal = function(x, S) {
	n = nrow(S)
	diag(S) = 0
	rowSumS = rowSums(S)
	rowSumS[rowSumS==0] = 1
	S = S/rowSumS
	y = x - mean(x)
	obs = (n/sum(S))*(sum(S*y%o%y)/sum(y^2))
	S1 = 0.5*sum((S + t(S))^2)
	S2 = sum((apply(S, 1, sum) + apply(S, 2, sum))^2)
	s.sq = sum(S)^2
	k = (sum(y^4)/n)/(sum(y^2)/n)^2
	sigma = sqrt((n*((n^2 - 3*n + 3)*S1 - n*S2 + 3*s.sq) - 
		    k*(n*(n - 1)*S1 - 2*n*S2 + 6*s.sq))/
		    ((n - 1)*(n - 2)*(n - 3)*s.sq) - 1/((n - 1)^2))
	exp = -1/(n - 1)
	pv = pnorm(obs, mean=exp, sd=sigma)
	if (obs <= exp) {
		pv = 2*pv
	} else {
		pv = 2*(1 - pv)
	}
	I.local = rep(0, n)
	for(i in 1:n) {
		I.local[i] = (y[i]/(sum(y^2)/n))*sum(S[i,]*y)
	}
	return(list(Is=obs, p.value=pv, I.local=I.local))
}
