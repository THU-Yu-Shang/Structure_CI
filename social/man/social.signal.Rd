% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/social.signal.R
\name{social.signal}
\alias{social.signal}
\title{Social signal}
\usage{
social.signal(x, S)
}
\arguments{
\item{x}{a numeric vector of social data.}

\item{S}{a social correlation matrix.}
}
\value{
A list containing the computed global social signal (\code{Is}), 
    the p-value of a test of the null hypothesis that there
    is no social autocorrelation under the assumption of normality (\code{p.value}), and the 
    local social signal for each node (\code{I.local}).
}
\description{
Calculates the social signal for a given variable (essentially
    just Moran's I, but using the social correlation matrix 
    as the weights)
}
\examples{
A = matrix(c(0,1,0,1,0,
             1,0,0,1,1,
             0,0,0,1,1,
             1,1,1,0,0,
             0,1,1,0,0), nrow=5)
S = social.cor.matrix(A)
x = rnorm(nrow(A))
s = social.signal(x, S)

}
