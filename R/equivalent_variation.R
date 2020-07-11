equivalent_variation <- function(he_m, he_p, alpha, Y_0){
  
  ev <- ((he_m/alpha)^(alpha)*((Y_0-he_p)/(1-alpha))^(1-alpha)-Y_0)
  
  return(ev)
  
}