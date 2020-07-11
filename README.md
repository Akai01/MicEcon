# MicEcon

### Installing the package:
```r

if(!require(devtools)){
install.packages("devtools")
}

if(!require(catboost)){
devtools::install_github('catboost/catboost', subdir = 'catboost/R-package')
}

if(!require(mlrMBO)){
install.packages("mlrMBO")
}

if(!require(ParamHelpers)){
install.packages("ParamHelpers")
}

devtools::install_github("Akai01/MicEcon")

```