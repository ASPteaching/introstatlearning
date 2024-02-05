
options(width=100) 
if(!require("knitr")) install.packages("knitr")
library("knitr")
#getOption("width")
knitr::opts_chunk$set(comment=NA,echo = TRUE, cache=TRUE)



knitr::include_graphics("images/AI-ML-DL-1.jpg")



knitr::include_graphics("images/ML_vs_DL-2.png")



knitr::include_graphics("images/ActivationFunction0.png")



knitr::include_graphics("images/MultiLayer1.png")



if (!require(neuralnet)) 
  install.packages("neuralnet", dep=TRUE)
if (!require(caret)) 
  install.packages("caret", dep=TRUE)



mydata <- read.csv("https://raw.githubusercontent.com/MGCodesandStats/datasets/master/dividendinfo.csv")
str(mydata)



normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}
normData <- as.data.frame(lapply(mydata, normalize))



perc2Train <- 2/3
ssize <- nrow(normData)
set.seed(12345)
data_rows <- floor(perc2Train *ssize)
train_indices <- sample(c(1:ssize), data_rows)
trainset <- normData[train_indices,]
testset <- normData[-train_indices,]



# Neural Network
library(neuralnet)
nn <- neuralnet(dividend ~ fcfps + earnings_growth + de + mcap + current_ratio, 
                data=trainset, 
                hidden=c(3,2), 
                linear.output=FALSE, 
                threshold=0.01)



plot(nn, rep = "best")



summary(nn)
nn$result.matrix



#Test the resulting output
temp_test <- subset(testset, select =
                      c("fcfps","earnings_growth", 
                        "de", "mcap", "current_ratio"))
head(temp_test)
nn.results <- compute(nn, temp_test)
results <- data.frame(actual = 
                  testset$dividend, 
                  prediction = nn.results$net.result)
head(results)




roundedresults<-sapply(results,round,digits=0)
roundedresultsdf=data.frame(roundedresults)
attach(roundedresultsdf)
confMat<- caret::confusionMatrix(table(actual, prediction))
confMat



knitr::include_graphics("images/nn.jpg")

