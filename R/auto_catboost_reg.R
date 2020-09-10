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
