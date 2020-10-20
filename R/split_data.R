#' Split data randomly in to train and test.
#' @param data A data.frame.
#' @param p percentage of train data.
#' @param set_seed Random seed
#' @examples 
#' \dontrun{
#' dta <- split_data(data)
#' }
#' 
#' @export 
split_data  <- function(data, p = 0.75, set_seed = 24){
  
  if("data.frame" %notin% class(data)){
    stop("data must be a data.frame")
  }
  
  smp_size <- floor(p * nrow(data))
  set.seed(set_seed)
  train_ind <- sample(seq_len(nrow(data)), size = smp_size)
  
  train <- data[train_ind, ]
  test <- data[-train_ind, ]
  return(list("train" = train, "test" = test))
  
  }