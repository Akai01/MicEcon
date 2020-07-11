\name{equivalent_variation}
\alias{equivalent_variation}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Equivalent variation
}
\description{
Equivalent variation (EV) is a measure of economic welfare changes associated with changes in prices. John Hicks (1939) is attributed with introducing the concept of compensating and equivalent variation.

The equivalent variation is the change in wealth, at current prices, that would have the same effect on consumer welfare as would the change in prices, with income unchanged. It is a useful tool when the present prices are the best place to make a comparison.
}
\usage{
equivalent_variation(he_m, he_p, alpha, Y_0)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{he_m}{
Market rent
}
  \item{he_p}{
Program rent
}
  \item{alpha}{
Rent/income
}
  \item{Y_0}{
Income}
}
\details{
%%  ~~ If necessary, more details than the description above ~~
}
\value{
%%  ~Describe the value returned
%%  If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
}
\references{
%% ~put references to the literature/web site here ~
}
\author{
Res Akay
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
equivalent_variation(he_m, he_p, alpha, Y_0)
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory (show via RShowDoc("KEYWORDS")):
% \keyword{ ~kwd1 }
% \keyword{ ~kwd2 }
% Use only one keyword per line.
% For non-standard keywords, use \concept instead of \keyword:
% \concept{ ~cpt1 }
% \concept{ ~cpt2 }
% Use only one concept per line.
