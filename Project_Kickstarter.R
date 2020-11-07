##Kickstarter Project
library(dplyr)
library(stringr)
library(lmridge)
library(e1071)
library(caret)

##Load Data##

df <- read.csv("https://raw.githubusercontent.com/CallBark/kickstarter/main/Kickstarter_NA.csv")

df1 <- na.omit(df)

summary(df1)

##Clean data
df1 <-
df1 %>%
mutate(Category = str_replace_all(df1$Category, "[[:punct:]]", " "))

##Training & Testing##

#fraction of sample to be used for training
p<-.7

#number of observations (rows) in the dataframe
obs_count<-dim(df1)[1]

#number of observations to be selected for the training partition
#the floor() function rounds down to the nearest integer
training_size <- floor(p * obs_count)
set.seed(123)
#create a vector with the shuffled row numbers of the original dataset
train_ind <- sample(obs_count, size = training_size)

Training <- df1[train_ind, ] #pulls random rows for training
Testing <- df1[-train_ind, ] #pulls random rows for testing

dim(Training)
dim(Testing)

#####
# Create random training, validation, and test sets - another way to split the data
#####

# Set some input variables to define the splitting.
# Input 1. The data frame that you want to split into training, validation, and test.
#df <- mtcars

# Input 2. Set the fractions of the dataframe you want to split into training, 
# validation, and test.
fractionTraining   <- 0.70
fractionValidation <- 0.15
fractionTest       <- 0.15

# Compute sample sizes.
sampleSizeTraining   <- floor(fractionTraining   * nrow(df1))
sampleSizeValidation <- floor(fractionValidation * nrow(df1))
sampleSizeTest       <- floor(fractionTest       * nrow(df1))

# Create the randomly-sampled indices for the dataframe. Use setdiff() to
# avoid overlapping subsets of indices.
indicesTraining    <- sort(sample(seq_len(nrow(df1)), size=sampleSizeTraining))
indicesNotTraining <- setdiff(seq_len(nrow(df1)), indicesTraining)
indicesValidation  <- sort(sample(indicesNotTraining, size=sampleSizeValidation))
indicesTest        <- setdiff(indicesNotTraining, indicesValidation)

# Finally, output the three dataframes for training, validation and test.
dfTraining   <- df1[indicesTraining, ]
dfValidation <- df1[indicesValidation, ]
dfTest       <- df1[indicesTest, ]

##Above Method - 3 data sets
#write.csv(dfTraining,"C:\\Users\\Derek\\Documents\\Training_v2.csv")
#write.csv(dfTest,"C:\\Users\\Derek\\Documents\\Testing_v2.csv")
#write.csv(dfValidation,"C:\\Users\\Derek\\Documents\\Validation_v2.csv")

##Method 1
#write.csv(Training,"C:\\Users\\Derek\\Documents\\Training.csv")
#write.csv(Testing,"C:\\Users\\Derek\\Documents\\Testing.csv")

## Model #1 - Continuous
## What is the best goal amount to choose for a project?

##Ridge Model
M1 <- lmridge(USD.Pledged~Backers.Count+Blurb.Length, Training, K = 0.02, "sc")

M1.test <- lmridge(USD.Pledged~Backers.Count+Blurb.Length, Testing, K = 0.02, "sc")

##Another Model
#M2 <- lmridge(USD.Pledged~Backers.Count + Spotlight, Training, K = 0.02, "sc")

#Best Model
M3 <- lm(USD.Pledged~Backers.Count, Training)
summary(M3)

M3.test <- lm(USD.Pledged~Backers.Count, Testing)
summary(M3.test)

##VISUALIZING OUR RESULTS##
plot(Training$USD.Pledged~Training$Backers.Count) #scatter plot of Sales vs. AdSpend again
abline(M3$coefficients[1], M3$coefficients[2], col='blue', lwd=2) #add regression line to plot

##PLOTTING FITTED (PREDICTED) VALUES
plot(Testing$USD.Pledged~Testing$Backers.Count)
abline(M3.test$coefficients[1], M3.test$coefficients[2], col='blue', lwd=2) #add regression line to plot
abline(M3$coefficients[1], M3$coefficients[2], col='red', lwd=2)


rstats1(M2)
summary(M2)

#Ridge Residuals
mod1.residuals <- residuals.lmridge(M1)
mod1.predict <- predict(M1)

summary(M1)

summary.lmridge(M1)
rstats1(M1)
#rstats2(M1)
#plot.lmridge(M1)

#plot(mod1.residuals)

## Model #2 - classification
## What are the keys to getting a successful kickstarter?

M_LOG<-glm(State ~ Backers.Count + Goal , data = Training, family = "binomial")
summary(M_LOG)

#takes the coefficients to the base e for odds-ratio interpretation
exp(cbind(M_LOG$coefficients, confint(M_LOG)))

#generating predicted probabilities
predictions<-predict(M_LOG, Training, type="response")

#converts predictions to boolean TRUE (1) or FALSE (0) based on 1/2 threshold on output probability
binpredict <- (predictions >= .5)
View(binpredict)

#build confusion matrix based on binary prediction in-sample
confusion<-table(binpredict, Training$State == 1)
confusion

#display summary analysis of confusion matrix in-sample
confusionMatrix(confusion, positive='TRUE') #need to load the library e1071

#builds the confusion matrix to look at accuracy on testing data out-of-sample
confusionMatrix(table(predict(M_LOG, Testing, type="response") >= 0.5,
                      Testing$State == 1), positive = 'TRUE')
