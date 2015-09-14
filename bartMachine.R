### BART Machine tutorial for Restaurant Revenue Prediction
setwd('./R/Kaggle/ResaurantRevenue/')

library(ggplot2)
library(corrplot)
options(java.parameters="-Xmx5000m") # must be set initially
library(bartMachine)
set_bart_machine_num_cores(4)

SEED <- 123
set.seed(SEED)

data <- read.csv('train.csv', sep=',', stringsAsFactor=F)
test <- read.csv('test.csv', sep=',', stringsAsFactor=F)
revenue <- data$revenue
train <- data[,-ncol(data)]
all <- rbind(train, test)

qplot(log(data$revenue), 
      geom="histogram",
      binwidth=0.1,
      main="Histogram of log Revenue",
      xlab="log(revenue)",
      fill=I("red"),
      col=I("black"))

correlations <- cor(data[,6:43])
corrplot(correlations, method="ellipse", order="hclust")

y <- log(revenue)
X <- all[1:nrow(train), -c(1,2,3,4,5)]
X_test <- all[(nrow(train)+1):nrow(all), -c(1,2,3,4,5)]

bart <- bartMachine(X, y, num_trees=10, seed=SEED)

rmse_by_num_trees(bart, tree_list=c(seq(5, 50, by=5)), num_replicates=5) # 20 is better

bart <- bartMachine(X, y, num_trees=20, seed=SEED)
plot_convergence_diagnostics(bart)
check_bart_error_assumptions(bart)
interaction_investigator(bart)
var_selection_by_permute(bart, num_reps_for_avg=20)

nX <- X[, c("P6", "P17", "P28")]
nX_test <- X_test[, c("P6", "P17", "P28")]
nbart <- bartMachine(nX, y, num_trees=20, seed=SEED)
nbart_cv <- bartMachineCV(nX, y, num_tree_cvs=c(10,15,20),
                          k_folds=5,
                          verbose=TRUE) # bartMachine CV win: k: 3 nu, q: 3, 0.9 m: 20
# NOTE: Prediction can take up to half and hour
log_pred <- predict(nbart_cv, nX_test)
pred <- exp(log_pred)
submit <- data.frame(Id=test$Id, Prediction = pred)
write.csv(submit, file="bartMachine_P6P17P28.csv", row.names=F) 

