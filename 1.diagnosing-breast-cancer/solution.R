## Step 1: Load libraries

library(ggplot2)
require(Amelia)
library(class)
library(gmodels)

## Data description from 
# https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.names
# Image: 
# https://www.safaribooksonline.com/library/view/data-science-for/9781449374273/httpatomoreillycomsourceoreillyimages1751040.png.jpg
# Ten real-valued features are computed for each cell nucleus:
#   
# a) radius (mean of distances from center to points on the perimeter)
# b) texture (standard deviation of gray-scale values)
# c) perimeter
# d) area
# e) smoothness (local variation in radius lengths)
# f) compactness (perimeter^2 / area - 1.0)
# g) concavity (severity of concave portions of the contour)
# h) concave points (number of concave portions of the contour)
# i) symmetry 
# j) fractal dimension ("coastline approximation" - 1)

## Step 2: Load the data 
df <- read.csv("data/breast_cancer_diagnostic.csv", stringsAsFactors = FALSE)

dim(df)

## Step 3: Explore the data

dim(df)

head(df)

# drop the id feature
df <- df[-1]

# recode diagnosis as a factor
df$diagnosis <- factor(df$diagnosis, levels = c("B", "M"),
                         labels = c("Benign", "Malignant"))

missmap(df, main="Missing Training Data Map", col=c("#FF4081", "#3F51B5"), legend=FALSE)

barplot(table(df$diagnosis), xlab = "Type of tumor", ylab="Numbers per type")

qplot(radius_mean, data=df, colour=diagnosis, geom="density",
      main="Radius mean of each tumor type")
qplot(smoothness_mean, data=df, colour=diagnosis, geom="density",
      main="Smoothness mean of each tumor type")
qplot(concavity_mean, data=df, colour=diagnosis, geom="density",
      main="Concavity mean of each tumor type")

# table or proportions with more informative labels
round(prop.table(table(df$diagnosis)) * 100, digits = 1)

# summarize three numeric features
summary(df[c("radius_mean", "area_mean", "smoothness_mean")])

## Step 4: Preprocess the data

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

# normalize the df data in range [0:1]
df_n <- as.data.frame(lapply(df[2:31], normalize))

# confirm that normalization worked
summary(df_n$area_mean)

## Step 5: Create training and test data (leave 100 data points for testing)

train_idx <- sample(nrow(df_n), nrow(df_n) - 100, replace = FALSE)

df_train <- df_n[train_idx, ]
df_test <- df_n[-train_idx, ]

# create labels for training and test data

df_train_labels <- df[train_idx, 1]
df_test_labels <- df[-train_idx, 1]

## Step 6: Create the model

df_test_pred <- knn(train = df_train, test = df_test,
                      cl = df_train_labels, k = 21)

## Step 7: Evaluate the model

evaluate.model <- function(actual.labels, predicted.labels) {
  CrossTable(x = actual.labels, y = predicted.labels, prop.chisq=FALSE)
  print(paste("Correctly predicted: ", table(actual.labels == predicted.labels)["TRUE"] / length(predicted.labels)))
}

evaluate.model(df_test_labels, df_test_pred)

## Step 8: Try to improve

# use the scale() function to z-score standardize a data frame
# (x - mean(x)) / sd(x)
df_z <- as.data.frame(scale(df[-1]))

# confirm that the transformation was applied correctly
summary(df_z$area_mean)

# create training and test datasets
df_train <- df_z[train_idx, ]
df_test <- df_z[-train_idx, ]

# re-classify test cases
df_test_pred <- knn(train = df_train, test = df_test,
                      cl = df_train_labels, k = 21)

evaluate.model(df_test_labels, df_test_pred)

# try several different values of k
df_train <- df_n[train_idx, ]
df_test <- df_n[-train_idx, ]

df_test_pred <- knn(train = df_train, test = df_test, cl = df_train_labels, k=1)
evaluate.model(df_test_labels, df_test_pred)

df_test_pred <- knn(train = df_train, test = df_test, cl = df_train_labels, k=5)
evaluate.model(df_test_labels, df_test_pred)

df_test_pred <- knn(train = df_train, test = df_test, cl = df_train_labels, k=11)
evaluate.model(df_test_labels, df_test_pred)

df_test_pred <- knn(train = df_train, test = df_test, cl = df_train_labels, k=15)
evaluate.model(df_test_labels, df_test_pred)

df_test_pred <- knn(train = df_train, test = df_test, cl = df_train_labels, k=21)
evaluate.model(df_test_labels, df_test_pred)

df_test_pred <- knn(train = df_train, test = df_test, cl = df_train_labels, k=27)
evaluate.model(df_test_labels, df_test_pred)
