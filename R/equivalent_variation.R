#' Equivalent variation
#' @description
#' 
#' \Sexpr[results=rd, stage=render]{lifecycle::badge("experimental")}
#' 
#' Equivalent variation (EV) is a measure of economic welfare changes 
#' associated with changes in prices. John Hicks (1939) is attributed 
#' with introducing the concept of compensating and equivalent variation. 
#' The equivalent variation is the change in wealth, at current prices, 
#' that would have the same effect on consumer welfare as would the change in 
#' prices, with income unchanged. It is a useful tool when the present prices 
#' are the best place to make a comparison.
#' @param he_m Market rent
#' @param he_p Program rent
#' @param alpha Rent/income
#' @param Y_0 Income
#' 
#' @examples 
#' \dontrun{
#' equivalent_variation(he_m, he_p, alpha, Y_0)
#' }
#' 
#' @author Resul Akay
#' 
#' @export
equivalent_variation <- function(he_m, he_p, alpha, Y_0){
  
  ev <- ((he_m/alpha)^(alpha)*((Y_0-he_p)/(1-alpha))^(1-alpha)-Y_0)
  
  return(ev)
  
}