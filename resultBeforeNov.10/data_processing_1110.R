
#######################After check feature one by one, we edit variables####
##Code for checking each variable is not included, function prepf summarizes check result. 
prepf <- function(train1){
  
  n <- dim(train1)[1]
  train2 <- train1
  
  ##number of children
  train2$n.children.cat <- train2$n.children
  train2$n.children.cat[train2$n.children.cat %in% c(4,5)] <- 4
  train2$n.children.cat[train2$n.children.cat %in% 6:12] <- 5
  
  ##number of adult
  train2$n.adults.cat <- train1$n.adults
  train2$n.adults.cat[train2$n.adults.cat %in% 5:9] <- 5
  
  train2$tenure.con <- train1$tenure
  train2$tenure.con[train2$tenure.con > 22] <- 22
  
  train2$ni.age.con <- train1$ni.age
  train2$ni.age.con[train1$ni.age <=25] = 25
  train2$ni.age.con[train1$ni.age <=30 & train1$ni.age>25] = 30
  train2$ni.age.con[train1$ni.age <=35 & train1$ni.age>30] = 35
  train2$ni.age.con[train1$ni.age <=40 & train1$ni.age>35] = 40
  train2$ni.age.con[train1$ni.age<=45 & train1$ni.age>40] = 45
  train2$ni.age.con[train1$ni.age<=50 & train1$ni.age>45] = 50
  train2$ni.age.con[train1$ni.age<=55 & train1$ni.age>50] = 55
  train2$ni.age.con[train1$ni.age<=65 & train1$ni.age>55] = 60
  train2$ni.age.con[train1$ni.age > 65] <- 65
  
  train2$len.at.res.con <- round(train2$len.at.res)
  train2$len.at.res.con[train2$len.at.res.con < 10] <- 9
  train2$len.at.res.con[train2$len.at.res.con %in% 21:25] <- 21
  train2$len.at.res.con[train2$len.at.res.con > 25] <- 22
  
  ##Categorical
  train2$city.cat <- round(train1$zip.code/10000,1)
  train2$city.cat[train2$city.cat==9.8] <- "Washington"
  train2$city.cat[train2$city.cat==8.0] <- "colorado"
  train2$city.cat[train2$city.cat==8.5] <- "Arizona"
  train2$city.cat[train2$city.cat==5.0] <- "Iowa"
  train2$city.cat[train2$city.cat==2.0] <- "Virginia"
  train2$city.cat[train2$city.cat==1.5] <- "penni"
  
  train2$zip.code <- NULL
  
  train2$credit.con <- train2$credict
  train2$credit.con[as.character(train2$credit) == "high"] <- 3
  train2$credit.con[as.character(train2$credit) == "medium"] <- 2
  train2$credit.con[as.character(train2$credit) == "low"] <- 1
  
  train2$year.con <- train2$year - 2012
  
  return(train2)
}

train1 <- read.csv("Train.csv")
test1 <- read.csv("Test.csv")

test1$cancel <- 0
dat_full <- rbind(train1, test1)
dat_full_edit <- prepf(dat_full)

cat <- dat_full_edit[,c(3,6,7,9,10,23)]
con <- dat_full_edit[,c(8,18,19,20,21,22,24,25)]

##Factor categorical variables
cat1 <- cat
for(i in 1:dim(cat)[2]){
  cat1[,i] <- as.factor(cat[,i])
}

##Normalize continuous variables
con1 <- con
for(i in 1:dim(con)[2]){
  con1[,i] <- (con[,i] - mean(con[,i],na.rm=TRUE))/sd(con[,i],na.rm=TRUE)
}

#############
dat_full_edit1 <- cbind(cat1,con1)

##It takes time to run###
#tempData <- mice(dat_full_edit1,m=5,maxit=50,seed=500)
##
completedData <- complete(tempData,4)

library("dummies")
cat_full <- completedData[,1:6]
cat_full_dummy <- dummy.data.frame(cat_full,names=names(cat_full))

dat_complete <- cbind(cat_full_dummy, completedData[,7:14])

train_complete <- dat_complete[1:7578,]
train_complete$cancel <- train1$cancel

idx_1 <- which(train1$cancel == -1)
train_complete1 <- train_complete[-idx_1,]

test_complete <- dat_complete[-(1:7578),]

#write.csv(train_complete1, "train_complete.csv")
#write.csv(test_complete, "test_complete.csv")

##train_complete1 and test_complete are final data we use for training and prediction.
