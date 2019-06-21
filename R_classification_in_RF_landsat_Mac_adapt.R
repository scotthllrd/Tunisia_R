# The Purpose of this Script is to take extracted Landsat  #
# vector data and use to extract training data from a      #
# llandsat8 image and use to classify the image            #
#                                                          #
#Author: Scott Hillard                                     #
#Date:5/08/18                                              #
#Last edited: 6/21/19                                      #
#--------------------------------------------------------- #

library(rgdal)
library(raster)
library(caret)

#Bring in the raster landsat stack
setwd("~/Documents/Landsat_data/Random_For_R/")

img <- brick("LS192_35_01002018_stack.tif")
names(img) <- paste0("B", c(1:7))

#plotRGB(img * (img >= 0), r = 4, g = 5, b = 3, scale = 10000)
ndvi <- overlay(img$B4, img$B5, 
                fun = function(x, y) {
                  (y-x) / (y+x)
                })
nbr <- overlay(img$B7, img$B5, 
               fun = function(x, y) {
                 (y-x) / (y+x)
               })
ndwi <- overlay(img$B6, img$B5, 
                fun = function(x, y) {
                  (y-x) / (y+x)
                })
ndbi <- overlay(img$B5, img$B6, 
                fun = function(x, y) {
                  (y-x) / (y+x)
                })
BU   <-overlay(ndvi, ndbi, 
               fun = function(x, y) {
                 (y-x)
               })

rastlist<-list(img,ndvi,nbr,ndwi,ndbi,BU)

sat1<-stack(rastlist)
s<-brick(sat1)

##bring in the shapefile##

trainData <- shapefile("~/Documents/Landsat_data/Random_For_R/Land_Class_poly_LS2018_2.shp")
responseCol <- "Class_ID"

#This code will extract the cell values based on the shapefile, and create a matrix
dfAll = data.frame(matrix(vector(), nrow = 0, ncol = length(names(s)) + 1))
for (i in 1:length(unique(trainData[[responseCol]]))){
  category <- unique(trainData[[responseCol]])[i]
  categorymap <- trainData[trainData[[responseCol]] == category,]
  dataSet <- extract(s, categorymap)
  if(is(trainData, "SpatialPointsDataFrame")){
    dataSet <- cbind(dataSet, class = as.numeric(rep(category, nrow(dataSet))))
    dfAll <- rbind(dfAll, dataSet[complete.cases(dataSet),])
  }
  if(is(trainData, "SpatialPolygonsDataFrame")){
    dataSet <- dataSet[!unlist(lapply(dataSet, is.null))]
    dataSet <- lapply(dataSet, function(x){cbind(x, class = as.numeric(rep(category, nrow(x))))})
    df <- do.call("rbind", dataSet)
    dfAll <- rbind(dfAll, df)
  }
}
#plot(dfAll)
#plot(trainData)
set.seed(100)
train <- sample(nrow(dfAll), 0.7*nrow(dfAll), replace = FALSE)
TrainSet <- dfAll[train,]
ValidSet <- dfAll[-train,]
summary(TrainSet)
summary(ValidSet)

# There are multiple options within the caret library to run different models, beyond random forest
# Example are different linear models such as svm

#modFit_rf <- train(as.factor(class) ~ B3 + B4 + B5, method = "rf", data = sdfAll)
#modFit_svm <- train(as.factor(class) ~ B3 + B4 + B5, method = "svmLinear", data = sdfAll)
#modFit_xgb <- train(as.factor(class) ~ B3 + B4 + B5, method = "xgbLinear", data = sdfAll)

#Lets run a random forest model using the randomForest package (choice based on better back end diagnostics)
library(randomForest)
library(snow)
FIA.rf.jsh=randomForest(as.factor(class) ~ B1 + B2 + B3 +
                                  B4 + B5 + B6 + B7+layer.1+layer.2+layer.3+layer.4+layer.5 ,
                                  data = TrainSet, importance=TRUE)  # this is just an example
print(FIA.rf.jsh)
plot(FIA.rf.jsh)
FIA.rf.jsh$importance
varImpPlot(FIA.rf.jsh,main="Band")

# Predicting on train set
predTrain <- predict(FIA.rf.jsh, TrainSet, type = "class")
# Checking classification accuracy
table(predTrain, TrainSet$class)  

predValid <- predict(FIA.rf.jsh, ValidSet, type = "class")
# Checking classification accuracy
mean(predValid == ValidSet$class)                    
table(predValid,ValidSet$class)

mean(predValid == ValidSet$class)                    

table(predValid,ValidSet$class)



##SVM regression##

library(e1071)
model_svm <- svm(as.factor(class) ~ B1 + B2 + B3 +
                   B4 + B5 + B6 + B7+layer.1+layer.2+layer.3+layer.4+layer.5 ,
                 data = TrainSet)

# Predicting on train set
predTrain <- predict(model_svm, TrainSet, type = "class")
# Checking classification accuracy
table(predTrain, TrainSet$class)  

predValid <- predict(model_svm, ValidSet, type = "class")
# Checking classification accuracy
mean(predValid == ValidSet$class)                    
table(predValid,ValidSet$class)

mean(predValid == ValidSet$class)                    

table(predValid,ValidSet$class)


###--------------------------Make predictions on a raster------------------------------#
##Predict on the 2017 scene#
setwd("/Users/scotthillard/Documents/Landsat_data/outputs/")

img <- brick("landsat_time17_january2.tif")
names(img) <- paste0("B", c(1:7))
ndvi <- overlay(img$B4, img$B5, 
                fun = function(x, y) {
                  (y-x) / (y+x)
                })
nbr <- overlay(img$B7, img$B5, 
               fun = function(x, y) {
                 (y-x) / (y+x)
               })
ndwi <- overlay(img$B6, img$B5, 
                fun = function(x, y) {
                  (y-x) / (y+x)
                })
ndbi <- overlay(img$B5, img$B6, 
                fun = function(x, y) {
                  (y-x) / (y+x)
                })
BU   <-overlay(ndvi, ndbi, 
               fun = function(x, y) {
                 (y-x)
               })


rastlist<-list(img,ndvi,nbr,ndwi,ndbi,BU)

sat1<-stack(rastlist)
img_stack<-brick(sat1)
#This will parralelize the process, super useful if you have a server
beginCluster()
preds_rf <- clusterR(img_stack, raster::predict, args = list(model = FIA.rf.jsh))
endCluster()

plot(preds_rf) #plot the classification

# If clustering is desired to either expand or get rid of small areas, use a focal stats function,
# in this case a 3X3 moving window with a mean function.
preds_rf_focal <- focal(preds_rf, w=matrix(1/9, nc=3, nr=3)) #a and b are equivalent, a will run faster due to the architecture of the function
#b <- focal(x, w=matrix(1,3,3), fun=mean)

plot(preds_rf_focal) #plot the classification again
preds_rf_focal
crs<-"+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0" 
#If you like the way it looks you can then write it as a raster
writeRaster(preds_rf_focal, filename="Land_Class_RF_2017.tif","GTiff",
            options="INTERLEAVE=BAND", overwrite=TRUE)


##Predict on the 2018 scene## apply the model to another landsat raster, in this case the 2018 scene.##

##SVM regression raster prediction##

beginCluster()
preds_svm <- clusterR(img_stack, raster::predict, args = list(model = model_svm))
endCluster()

preds_SVM_focal <- focal(preds_svm, w=matrix(1/9, nc=3, nr=3)) #a and b are equivalent, a will run faster due to the architecture of the function
#b <- focal(x, w=matrix(1,3,3), fun=mean)

plot(preds_SVM_focal) #plot the classification again
preds_SVM_focal
crs<-"+proj=utm +zone=32 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0" 
#If you like the way it looks you can then write it as a raster
writeRaster(preds_SVM_focal, filename="Land_Class_SVM_2017.tif","GTiff",
            options="INTERLEAVE=BAND", overwrite=TRUE)


##Implementing an ANN model###

dfall<-dfAll
for(i in 1:12){
  dfall[,i]<-(dfall[,i]-min(dfall[,i])/(max(dfall[,i])-min(dfall[,i])))
}
set.seed(2)
set.seed(100)
train <- sample(nrow(dfall), 0.7*nrow(dfall), replace = FALSE)
TrainSet <- dfall[train,]
ValidSet <- dfall[-train,]
summary(TrainSet)
summary(ValidSet)

str(dfall)
library(neuralnet)
NN <- neuralnet(as.integer(class) ~ B1 + B2 + B3 +
                  B4 + B5 + B6 + B7+layer.1+layer.2+layer.3+layer.4+layer.5 ,
                data = TrainSet, hidden = c(5,3,2),
                linear.output = F,learningrate = 0.0001)

plot(NN)
predict_testNN = neuralnet::compute(NN, ValidSet[,c(1:13)])
predict_testNN = (predict_testNN$net.result * (max(ValidSet$class) - min(ValidSet$class))) + min(ValidSet$class)

plot(ValidSet$class, predict_testNN, col='blue', pch=16, ylab = "predicted rating NN", xlab = "real rating")
#####example earth explorer API
install.packages("epsa.tools")
library(tools)
earthexplorer_search(usgs_eros_username, usgs_eros_password, 
                     datasetName,
                     lowerLeft = NULL, 
                     upperRight = NULL, 
                     startDate = "1920-01-07",
                     endDate = as.character(Sys.Date()), 
                     months = "",
                     includeUnknownCloudCover = T, 
                     minCloudCover = 0, maxCloudCover = 100,
                     additionalCriteria = "", 
                     sp = NULL, 
                     place_order = F, 
                     products = "sr",
                     format = "gtiff", 
                     verbose = F)

earthexplorer_download(usgs_eros_username, usgs_eros_password,
                       output_folder = getwd(), 
                       earthexplorer_search_results = NULL, 
                       quicklooks_only = F, 
                       ordernum = NULL, 
                       overwrite = F, 
                       verbose = F)
