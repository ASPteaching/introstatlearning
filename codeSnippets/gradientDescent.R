### 
# Gradient Descent Animation
###

if(!require(animation)) install.packages('animation')
# or development version
# remotes::install_github('yihui/animation')

library(animation)
ani.options(interval = 0.3, nmax = 50)
xx = grad.desc()

cat("The minimum is at: ", xx$par)

xx$persp(col = "lightblue", phi = 30)  # perspective plot
