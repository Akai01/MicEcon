#' Automatic model tununig using Bayesian optimazation
#'
#' @description
#'
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#' Automatic model tuning using Bayesian optimization
#' @param x Independent variables (features)
#' @param y Dependent variable
#' @param cat_features Name of the categorical variables
#' @param has_time Boolean, does data have time con
#' @param fold_count Number of cross-validation folds
#' @param type The type of cross-validation
#' @param partition_random_seed The random seed used for splitting pool into folds.
#' @param shuffle Shuffle the dataset objects before splitting into folds.
#' @param stratified Perform stratified sampling.
#' @param early_stopping_rounds Activates Iter over fitting detector with
#' od_wait set to early_stopping_rounds.
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
#' @param logging_level Possible values: 'Silent', 'Verbose', 'Info', 'Debug'
#' Default value: 'Silent'
#'
#' @param bo_iters Maximum iteration for Bayesian optimazation.Default value: 10
#' @author Resul Akay
#' @examples
#' \dontrun{
#' # A toy example
#'
#' data(iris, package = "datasets")
#'
#' fit <- auto_catboost_reg(
#'   iris,
#'   label_col_name = "Petal.Length",
#'   cat_features = "Species",
#'   has_time = FALSE,
#'   fold_count = 3,
#'   type = "Classical",
#'   partition_random_seed = 0,
#'   shuffle = TRUE,
#'   stratified = FALSE,
#'   early_stopping_rounds = NULL,
#'   iterations = list(lower = 100, upper = 110),
#'   learning_rate = list(lower = 0.001, upper = 0.05),
#'   l2_leaf_reg = list(lower = 0, upper = 5),
#'   depth = list(lower = 1, upper = 10),
#'   bagging_temperature = list(lower = 0, upper = 100),
#'   rsm = list(lower = 0, upper = 1),
#'   border_count = list(lower = 1, upper = 254),
#'   bo_iters = 2
#' )
#'
#'
#' varimp <- get_var_imp(fit$model)
#'
#' plot_varimp(varimp)
#' }
#' @import ParamHelpers
#' @import mlrMBO
#' @importFrom dplyr select
#' @export
auto_catboost_reg <- function(x,
                              y,
                              cat_features = NULL,
                              has_time = FALSE,
                              fold_count = 10,
                              type = "Classical",
                              partition_random_seed = 0,
                              shuffle = TRUE,
                              stratified = FALSE,
                              early_stopping_rounds = NULL,
                              iterations = list(lower = 500, upper = 1000),
                              learning_rate = list(lower = 0.001, upper = 0.05),
                              l2_leaf_reg = list(lower = 0,     upper = 5),
                              depth = list(lower= 1,      upper = 10),
                              bagging_temperature = list(lower= 0,      upper = 100),
                              rsm = list(lower = 0,   upper = 1),
                              border_count = list(lower = 1,   upper = 254),
                              logging_level = 'Silent',
                              bo_iters = 10){

  catboost2 <- loadNamespace(package = "catboost")

  obj.fun  <- smoof::makeSingleObjectiveFunction(
    name = "catboost",
    fn =   function(par){

      cv <- catboost2$catboost.cv(pool = learn_pool,
                        params = list(
                          logging_level = logging_level,
                          iterations =             par["iterations"],
                          depth =                  par["depth"],
                          learning_rate =          par["learning_rate"],
                          rsm =                    par["rsm"],
                          l2_leaf_reg =            par["l2_leaf_reg"],
                          has_time =               has_time,
                          bagging_temperature =    par["bagging_temperature"],
                          border_count =           par["border_count"]),
                        fold_count = fold_count,
                        type = type,
                        partition_random_seed = partition_random_seed,
                        shuffle = shuffle,
                        stratified = stratified,
                        early_stopping_rounds = early_stopping_rounds)

      a <- - mean(cv$test.RMSE.mean)
      return(a)

    },
    par.set = makeParamSet(
      makeIntegerParam("iterations",
                       lower = iterations[["lower"]], upper = iterations[["upper"]]),
      makeNumericParam("learning_rate",
                       lower = learning_rate[["lower"]], upper = learning_rate[["upper"]]),
      makeNumericParam("l2_leaf_reg",
                       lower = l2_leaf_reg[["lower"]], upper = l2_leaf_reg[["upper"]]),
      makeIntegerParam("depth",
                       lower = depth[["lower"]], upper = depth[["upper"]]),
      makeIntegerParam("bagging_temperature",
                       lower = bagging_temperature[["lower"]], upper = bagging_temperature[["upper"]]),
      makeNumericParam("rsm",
                       lower = rsm[["lower"]], upper = rsm[["upper"]]),
      makeIntegerParam("border_count",
                       lower = border_count[["lower"]], upper = border_count[["upper"]])
    ),
    minimize = FALSE
  )


 learn_pool <- catboost2$catboost.load_pool(data = x, label = y , cat_features = cat_features)

  control = makeMBOControl()
  control = setMBOControlTermination(control, iters = bo_iters)

  if(logging_level=='Verbose'){
    show_info <- TRUE
  } else {
    show_info <- FALSE
  }

 BO = suppressWarnings(mbo(fun = obj.fun,
           control = control,
           show.info = show_info))

  params <- BO$x

  params[["logging_level"]] <- logging_level

  model_final <- catboost2$catboost.train(learn_pool = learn_pool, params = params)

  return(list("model"= model_final, "BO" = BO))
}
