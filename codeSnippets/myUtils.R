deMardownAR <- function(cadena) {
  imagen <- gsub("!\\[\\]\\(", "", cadena)
  imagen <- gsub("\\)", "", imagen)
  imagen <- paste0("knitr::include_graphics(\"", imagen, "\")")
  rmarkdown <- sprintf("```{r, fig.align='center', out.width='100%%', fig.cap=''}\n%s\n```", imagen)
  return(rmarkdown)
}
cadena_entrada <- "![](images/reshape.png)"
cadena_formateada <- formato_rmarkdown(cadena_entrada)
cat(cadena_formateada)
