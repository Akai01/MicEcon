% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/tools.R
\name{plot_varimp}
\alias{plot_varimp}
\title{Catboost variable importance plot}
\usage{
plot_varimp(varimp)
}
\arguments{
\item{varimp}{A dataframe class of varimp, e.g. from get_var_imp}
}
\value{
variable importance plot
}
\description{
\Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
}
\examples{
\dontrun{
varimp <- get_var_imp(fit)

plot.varimp(varimp)
}
}
\author{
Resul Akay
}
