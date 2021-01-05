#' Automatic model tununig using Bayesian optimazation
#'
#' @description
#'
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#' Automatic model tununig using Bayesian optimazation
#' @param x Indepandent variables (Features)
#' @param y Depandent variable
#' @param cat_features Name of the categorical variables
#' @param iterations The maximum number of trees that can be built when solving
#' machine learning problems.When using other parameters that limit the number
#' of iterations, the final number of trees may be less than the number
#' specified in this parameter. Default value: 1000
#' @param learning_rate The learning rate. Used for reducing the gradient step.
#' Default value: 0.03
#' @param l2_leaf_reg L2 regularization coefficient. Used for leaf value
#' calculation. Any positive values are allowed. Default value: 3
#'
#' @param depth Depth of the tree. The value can be any integer up to 16.
#' It is recommended to use values in the range \code{[1; 10]}. Default value: 6
#'
#' @param bagging_temperature Controls intensity of Bayesian bagging.
#' The higher the temperature the more aggressive bagging is.
#' Typical values are in the range \code{[0, 1]} (0 is for no bagging).
#' Possible values are in the range \code{0<= < +∞)}. Default value: 1
#'
#' @param rsm Random subspace method.
#' The percentage of features to use at each iteration of building trees.
#' At each iteration, features are selected over again at random.
#' The value must be in the range \code{[0;1]}. Default value: 1
#'
#' @param border_count Maximum number of borders used in target binarization
#' for categorical features that need it. If TargetBorderCount is
#' specified in 'simple_ctr', 'combinations_ctr' or 'per_feature_ctr' option
#' it overrides this value. Default value: 1
#' @param verbose Possible values: 'Silent', 'Verbose', 'Info', 'Debug'
#' Default value: 'Silent'
#' @param bo_iters Maximum iteration for Bayesian optimazation.Default value: 10
#' @param init_design Length of the initial design
#' @param validation_error_metric Later
#' @param k An integer to specify the number of folds.
#' @param \dots Other parameters passed to catboost.train's param argument.
#' @author Resul Akay
#' @source \url{https://github.com/Akai01/MicEcon}
#' @references
#' \url{https://github.com/Akai01/MicEcon}
#'
#' W. Powell and I. Ryzhov (2012), Optimal Learning.
#'
#' @examples
#' \dontrun{
#' # A toy example
#' data(iris, package = "datasets")
#'
#' y <- iris$Petal.Length
#'
#' x <- iris
#'
#' x$Petal.Length <- NULL
#'
#' fit <- auto_catboost( x = x, y = y,
#'                       iterations = list(lower = 10, upper = 11),
#'                       learning_rate = list(lower = 0.001, upper = 0.05),
#'                       l2_leaf_reg = list(lower = 0, upper = 5),
#'                       depth = list(lower = 1, upper = 10),
#'                       bagging_temperature = list(lower = 0, upper = 100),
#'                       rsm = list(lower = 0, upper = 1),
#'                       border_count = list(lower = 1, upper = 254),
#'                       bo_iters = 2, verbose = "Verbose")
#' }
#' @import ParamHelpers
#' @import mlrMBO
#' @importFrom dplyr select
#' @export
auto_catboost <- function(x,
                          y,
                          cat_features = NULL,
                          iterations = list(lower = 500, upper = 1000),
                          learning_rate = list(lower = 0.001, upper = 0.05),
                          l2_leaf_reg = list(lower = 0,     upper = 5),
                          depth = list(lower= 1,      upper = 10),
                          bagging_temperature = list(lower= 0, upper = 100),
                          rsm = list(lower = 0,   upper = 1),
                          border_count = list(lower = 1,   upper = 254),
                          verbose = 'Silent',
                          k = 6,
                          bo_iters = 10,
                          init_design = 20,
                          validation_error_metric = "RMSE", ...){

  catboost2 <- loadNamespace(package = "catboost")

  objective_function <- smoof::makeSingleObjectiveFunction(
    name = "catboost",
    fn =   function(par){

      acc <- catboost_cv(x = x, y = y, cat_features = cat_features,
                         params = list(
                           logging_level = verbose,
                           iterations =             par["iterations"],
                           depth =                  par["depth"],
                           learning_rate =          par["learning_rate"],
                           rsm =                    par["rsm"],
                           l2_leaf_reg =            par["l2_leaf_reg"],
                           bagging_temperature =    par["bagging_temperature"],
                           border_count =           par["border_count"],
                           ...),
                         k = k, verbose = verbose)

      a <- - mean(acc[,validation_error_metric])
      if(verbose != "Silent"){
        message(paste0("The mean error of ", k, " folds is ", - a))
      }
      return(a)

    },
    par.set = makeParamSet(
      makeIntegerParam("iterations",
                       lower = iterations[["lower"]],
                       upper = iterations[["upper"]]),
      makeNumericParam("learning_rate",
                       lower = learning_rate[["lower"]],
                       upper = learning_rate[["upper"]]),
      makeNumericParam("l2_leaf_reg",
                       lower = l2_leaf_reg[["lower"]],
                       upper = l2_leaf_reg[["upper"]]),
      makeIntegerParam("depth",
                       lower = depth[["lower"]], upper = depth[["upper"]]),
      makeIntegerParam("bagging_temperature",
                       lower = bagging_temperature[["lower"]],
                       upper = bagging_temperature[["upper"]]),
      makeNumericParam("rsm",
                       lower = rsm[["lower"]], upper = rsm[["upper"]]),
      makeIntegerParam("border_count",
                       lower = border_count[["lower"]],
                       upper = border_count[["upper"]])
    ),
    minimize = FALSE
  )
  des <- ParamHelpers::generateDesign(
    n= init_design,
    par.set = ParamHelpers::getParamSet(objective_function),
    fun = lhs::randomLHS)

  control = makeMBOControl()
  control = setMBOControlTermination(control, iters = bo_iters)
  show_info <- FALSE
  if(verbose != "Silent"){
    show_info <- TRUE
  }
  mlrmbo_result = mbo(fun = objective_function,
                      design = des,
                      control = control,
                      show.info = show_info)

  learn_pool_final <- catboost2$catboost.load_pool(data = x, label = y,
                                                   cat_features = cat_features)

  params <- mlrmbo_result[["x"]]
  params[["logging_level"]] <- verbose

  model_final <- catboost2$catboost.train(learn_pool = learn_pool_final,
                                          params = params)

  object <- list("model"= list("model" = model_final,
                               "cat_features" = cat_features),
                 "mlrmbo_result" = mlrmbo_result)
  class(object) <- "auto_catboost"
  return(object)
}
