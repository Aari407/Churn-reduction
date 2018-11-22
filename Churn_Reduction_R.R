rm(list=ls())
setwd("D:/Project")
getwd()
library("rpart")
#install.packages('caret', dependencies = TRUE)
library("caret")
library("randomForest")

d1 <- read.csv('Train_data.csv',
               sep = ",")
d2 <- read.csv('Test_data.csv',
               sep = ",")
d1$international.plan=as.factor(d1$international.plan)
d1$voice.mail.plan=as.factor(d1$voice.mail.plan)
d1$Churn=as.factor(d1$Churn)
d1$state=as.factor(d1$state)
d1_train=subset(d1, select=-(phone.number))

d2$international.plan=as.factor(d2$international.plan)
d2$voice.mail.plan=as.factor(d2$voice.mail.plan)
d2$Churn=as.factor(d2$Churn)
d2$state=as.factor(d2$state)
d2_test=subset(d2, select=-phone.number)

#rpart_model= rpart(formula=Churn ~ ., data=d1_train)
#summary(c50_model)
#class_prediction <- predict(object = rpart_model,  
#                            newdata = d2_test,  
#                            type = "class")       
#confusionMatrix(data = class_prediction,         
#              reference = d2_test$Churn)


#Descision Tree Scores
#Accuracy: 92.98
#Recall: 141/(141+83): 0.6294643

set.seed(1)  
random_model= randomForest(formula=Churn ~ ., data=d1_train, importance=TRUE, ntree=1000)
summary(random_model)

class_prediction <- predict(object = random_model,  
                            newdata = d2_test,  
                            type = "class")       
confusionMatrix(data = class_prediction,         
                reference = d2_test$Churn) 


nodesize <- seq(2, 10, 2)
ntree <- seq(500, 1100, 100)

# Create a data frame containing all combinations 
hyper_grid <- expand.grid(ntree = ntree, nodesize = nodesize)

# Create an empty vector to store OOB error values
oob_err <- c()

# Write a loop over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  model_rf <- randomForest(formula = Churn ~ ., 
                           data = d1_train,
                           ntree = hyper_grid$ntree[i],
                           nodesize = hyper_grid$nodesize[i],
                           importance=TRUE)
  oob_err[i] <- model_rf$err.rate[nrow(model_rf$err.rate), "OOB"]
}

class_prediction <- predict(object = model_rf,  
                            newdata = d2_test,  
                            type = "class")       
confusionMatrix(data = class_prediction,         
                reference = d2_test$Churn) 

# Identify optimal set of hyperparmeters based on OOB error
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])

#Random Forest Scores
#Accuracy: 95.62
#Recall: : 70