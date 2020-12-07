#' Accuracy measures for a regression models
#' @param actual Actual data as numeric vector
#' @param pred Predicted values as numeric vector
#' @author Resul Akay
#' @examples
#' \dontrun{
#'
#' accuracy(acc, pred)}
#'
#'
accuracy <- function (actual, pred)
{
  if(length(actual)!= length(pred)){
    stop("The length of actual and prediction is not the same")
  }
  error <- c(actual - pred)
  pe <- error/actual * 100
  me <- mean(error, na.rm = TRUE)
  mse <- mean(error^2, na.rm = TRUE)
  mae <- mean(abs(error), na.rm = TRUE)
  mape <- mean(abs(pe), na.rm = TRUE)
  mpe <- mean(pe, na.rm = TRUE)
  out <- c(me, sqrt(mse), mae, mpe, mape)
  names(out) <- c("ME", "RMSE", "MAE", "MPE", "MAPE")
  return(out)
}
