#' Predict an auto_catboost object
#' @param object An auto_catboost object.
#' @param newdata The newdata
#' @param \dots Ignored for now
#' @author Resul Akay
#' @export
predict.auto_catboost <- function(object, newdata, ...){
  model <- object[["model"]][["model"]]
  cat_features <- object[["model"]][["cat_features"]]
  cb <- loadNamespace(package = "catboost")
  newdata_pool <- cb$catboost.load_pool(data = newdata,
                                                cat_features = cat_features)
  pred <- cb$catboost.predict(model, pool = newdata_pool)
  return(pred)
}
