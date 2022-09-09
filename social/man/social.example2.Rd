\name{social.example2}
\alias{social.example2}
\docType{data}
\title{Example dataset 2}
\description{
An example dataset for demonstrating the functions available in the social package.
}
\usage{data(social.example2)}
\format{
The dataset consists of a list with 3 items: \code{A}, a 30x30 adjacency matrix; \code{S}, a 30x30 social correlation matrix derived from \code{A} using \code{S = social.cor.matrix(A, max.depth=5)}; and \code{social.data}, a 30-row data frame containing two columns of numeric data, \code{x} and \code{y}, and a column of node IDs (\code{node.id}, corresponding to the row and column names of \code{A} and \code{S}). 
}
\examples{
data(social.example2)
}
\keyword{datasets}
