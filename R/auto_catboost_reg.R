#' Automatic model tununig using Bayesian optimazation
#' 
#' @description 
#' 
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#' Automatic model tununig using Bayesian optimazation
#' @param data Input data 
#' @param label_col_name Depandent variable names, as a string
#' @param cat_features Name of the categorical variables
#' @param has_time Boolean, does data have time con
#' @param fold_count Number of cross-validation folds
#' @param type The type of cross-validation
#' @param partition_random_seed The random seed used for splitting pool into folds.
#' @param shuffle Shuffle the dataset objects before splitting into folds.
#' @param stratified Perform stratified sampling. 
#' @param early_stopping_rounds Activates Iter overfitting detector with 
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
#' It is recommended to use values in the range [1; 10]. Default value: 6
#'
#' @param bagging_temperature Controls intensity of Bayesian bagging. 
#' The higher the temperature the more aggressive bagging is. 
#' Typical values are in the range [0, 1] (0 is for no bagging). 
#' Possible values are in the range [0, +∞). Default value: 1
#'
#' @param rsm Random subspace method. 
#' The percentage of features to use at each iteration of building trees. 
#' At each iteration, features are selected over again at random. 
#' The value must be in the range [0;1]. Default value: 1
#'
#' @param border_count Maximum number of borders used in target binarization 
#' for categorical features that need it. If TargetBorderCount is 
#' specified in 'simple_ctr', 'combinations_ctr' or 'per_feature_ctr' option 
#' it overrides this value. Default value: 1
#' @param bo_iters Maximum iteration for Bayesian optimazation.Default value: 10
#' @author Resul Akay
#' @examples 
#' \dontrun{
#' to do..
#' }
#' @import ParamHelpers
#' @import mlrMBO
#' @import smoof
#' @export
auto_catboost_reg <- function(data, 
                           label_col_name = NULL, 
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
                           bo_iters = 10){
  
  loadNamespace(package = "catboost")
  
  obj.fun  <- smoof::makeSingleObjectiveFunction(
    name = "catboost",
    fn =   function(x){
      
      cv <- catboost::catboost.cv(pool = learn_pool,
                        params = list(
                          iterations =             x["iterations"],
                          depth =                  x["depth"],
                          learning_rate =          x["learning_rate"],
                          rsm =                    x["rsm"],
                          l2_leaf_reg =            x["l2_leaf_reg"],
                          has_time =               has_time,
                          bagging_temperature =    x["bagging_temperature"],
                          border_count =           x["border_count"]),
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
  
 learn_pool <- as.data.frame(data)
  
 learn_pool <- catboost::catboost.load_pool(dplyr::select(learn_pool, -c(paste(label_col_name))),
            label = as.matrix(dplyr::select(learn_pool, paste(label_col_name))),
            cat_features = cat_features)
  
  control = makeMBOControl()
  control = setMBOControlTermination(control, iters = bo_iters)
  
 BO = mbo(fun = obj.fun,
           control = control,
           show.info = TRUE)

  learn_pool_final <-  catboost::catboost.load_pool(dplyr::select(data,-c(paste(label_col_name))),
                  label = as.matrix(dplyr::select(data, paste(label_col_name))),
                  cat_features = cat_features)
 
  params <- BO$x
  
  model_final <- catboost::catboost.train(learn_pool = learn_pool_final, params = params)
  
  return(list("model"= model_final, "BO" = BO))
}
