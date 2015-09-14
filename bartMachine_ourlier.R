setwd('./R/Kaggle/ResaurantRevenue/')

options(java.parameters="-Xmx5000m")
library(bartMachine)
set_bart_machine_num_cores(4)

SEED <- 123
set.seed(SEED)

data <- read.csv('train.csv', sep=',', stringsAsFactor=F)
test <- read.csv('test.csv', sep=',', stringsAsFactor=F)
revenue <- data$revenue
train <- data[,-ncol(data)]
all <- rbind(train, test)
Dates <- as.POSIXlt(strptime(all$Open.Date, format='%m/%d/%Y', tz="PDT"))
all$Age <- 2015 - (Dates$year + 1900)
ntrain <- all[0:nrow(data), ]
ntest <- all[138:nrow(all), ]
ntrain$logRevenue <- log(revenue)

ntrain_noOutlier <- ntrain[which((ntrain$logRevenue <=16) & (ntrain$logRevenue >= 14.5)),]

y <- ntrain_noOutlier$logRevenue
X <- ntrain_noOutlier; X$logRevenue <- NULL
X <- X[,-c(1,2,3,4,5)]

X_varSelect1 <- X[,c("P17", "P25", "P28", "Age")]

bart <- bartMachine(X_varSelect1, y, seed=SEED, num_trees=20)

rmse_by_num_trees(bart, tree_list=c(10, 20, 30, 50), num_replicates=5) # 50 trees seems better 
bart <- bartMachine(X_varSelect1, y, seed=SEED, num_trees=50)

ntest_pred <- ntest[,c("P17", "P25", "P28", "Age")]

pred <- predict(bart, ntest_pred)
Prediction <- exp(pred)
submit <- data.frame(Id=ntest$Id, Prediction=Prediction)
write.csv(submit, "bartMachine_outlier_varSelect1.csv", row.names=F)
