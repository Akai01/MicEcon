#' @title Catboost variable importance
#' @description 
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#' @import tibble
#' @param object A catboost model
#' @return 
#' A data frame class of \code{varimp, data.frame)}
#' @author Resul Akay
#' 
#' @examples 
#' \dont{
#' get_var_imp(fit)
#' }
#' @export
get_var_imp <- function(object){
  
  varimp <- tibble::rownames_to_column(
    data.frame(catboost::catboost.get_feature_importance(object)),
  "variables")
  
  colnames(varimp) <- c("variables", "varimp")
  
  class(varimp) <- c("varimp", "data.frame")
  
  return(varimp)
}


#' @title Catboost variable importance plot
#' @description 
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#' @import ggplot2
#' @param varimp A dataframe class of varimp, e.g. from get_var_imp
#' @return 
#' variable importance plot
#' @author Resul Akay
#' 
#' @examples 
#' \dontrun{
#' varimp <- get_var_imp(fit)
#' 
#' plot.varimp(varimp)
#' }
#' @export

plot.varimp <- function(varimp){
  ggplot(varimp, aes(y = variables, x = varimp)) + geom_col()
}


`%notin%` <- Negate(`%in%`)

