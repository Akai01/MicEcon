#' @importFrom caret createFolds
#' @importFrom plyr aaply splat
catboost_cv <- function(x, y, cat_features, params = list(), k = 10, verbose){

  catboost2 <- loadNamespace(package = "catboost")

  folds <- caret::createFolds(1:NROW(y), k = k, list = T, returnTrain = T)
  fold_list <- data.frame("i"=1:k)

  acc <- plyr::aaply(fold_list, 1, plyr::splat(function(i) {
    train_x <- data.frame(x[folds[[i]], ])
    validation_x <- data.frame(x[-folds[[i]], ])
    train_y <- y[folds[[i]]]
    validation_y <- y[-folds[[i]]]

    if(verbose!= "Silent"){
      message(paste("Training model for fold ", i))
    }

    dtrain <- catboost2$catboost.load_pool(data = train_x, label = train_y,
                                           cat_features = cat_features)

    fit <- catboost2$catboost.train(learn_pool = dtrain, params = params)

    validation_x <- catboost2$catboost.load_pool(data = validation_x,
                                                 cat_features = cat_features)

    newpred <- catboost2$catboost.predict(fit, validation_x)
    acc <- accuracy(pred =  newpred, actual =  validation_y)
    acc

  }), .expand = TRUE,
  .progress = "none",
  .inform = FALSE,
  .drop = TRUE,
  .parallel = FALSE,
  .paropts = NULL
  )
  return(acc)
}
