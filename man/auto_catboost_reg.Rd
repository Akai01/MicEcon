\name{auto_catboost_reg}
\alias{auto_catboost_reg}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
Automatic model tununig using Bayesian optimazation}
\description{

\Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}

Automatic model tununig using Bayesian optimazation
}
\usage{
auto_catboost_reg(data, label_col_name = NULL, cat_features = NULL, has_time = FALSE,
fold_count = 10, type = "Classical", partition_random_seed = 0, shuffle = TRUE, 
stratified = FALSE, early_stopping_rounds = NULL, iterations = list(lower = 500, 
upper = 1000), learning_rate = list(lower = 0.001, upper = 0.05), 
l2_leaf_reg = list(lower = 0, upper = 5), depth = list(lower = 1, upper = 10), 
bagging_temperature = list(lower = 0, upper = 100), rsm = list(lower = 0, upper = 1), 
border_count = list(lower = 1, upper = 254), bo_iters = 10)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{data}{
Input data
}
  \item{label_col_name}{
Depandent variable names, as a string
}
  \item{cat_features}{
Name of the categorical variables,
}
  \item{has_time}{
Boolean, does data have time con
}
  \item{fold_count}{
  Number of cross-validation folds
}
  \item{type}{
The type of cross-validation
}
  \item{partition_random_seed}{
The random seed used for splitting pool into folds.
}
  \item{shuffle}{
Shuffle the dataset objects before splitting into folds.
}
  \item{stratified}{Perform stratified sampling.
  }
  \item{early_stopping_rounds}{
Activates Iter overfitting detector with od_wait set to early_stopping_rounds.
}
  \item{iterations}{
  
  }
  \item{learning_rate}{
The maximum number of trees that can be built when solving machine learning problems. When using other parameters that limit the number of iterations, the final number of trees may be less than the number specified in this parameter. Default value: 1000}
  \item{l2_leaf_reg}{
L2 regularization coefficient. Used for leaf value calculation. Any positive values are allowed. Default value: 3
}
  \item{depth}{
Depth of the tree. The value can be any integer up to 16. It is recommended to use values in the range [1; 10]. Default value: 6
}
  \item{bagging_temperature}{
Controls intensity of Bayesian bagging. The higher the temperature the more aggressive bagging is. Typical values are in the range [0, 1] (0 is for no bagging). Possible values are in the range [0, +∞). Default value: 1
}
  \item{rsm}{
	
Random subspace method. The percentage of features to use at each iteration of building trees. At each iteration, features are selected over again at random. The value must be in the range [0;1]. Default value: 1
}
  \item{border_count}{
Maximum number of borders used in target binarization for categorical features that need it. If TargetBorderCount is specified in 'simple_ctr', 'combinations_ctr' or 'per_feature_ctr' option it overrides this value. Default value: 1}
  \item{bo_iters}{
aximum iteration for Bayesian optimazation.Default value: 10
}
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
Resul Akay
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
}
\examples{
TO do
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
