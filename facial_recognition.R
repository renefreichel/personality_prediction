# %% [markdown]
# # Facial Expression - BDA Challenge 4
# 
# ### Students:
# - Rene Freichel
# - Selina Rahman
# - Andre Nedderhoff 

# %% [code] {"_execution_state":"idle"}
## Importing packages

library(tidyverse) # metapackage with lots of helpful functions
library(png)# package that can be used to read png image files in a simple format
library(caret)
library(reshape2)

## Reading in files

# You can access files the "../input/" directory.
# You can see the files by running  

list.files(path = "../input/")

# %% [code]
# Show the availabe directories
dirs = dir("../input", pattern="[^g]$", recursive=TRUE, include.dirs = TRUE, full.names = TRUE)
dirs

# %% [code]
# Get all image files: file names ending ".png" 
anger   = dir(grep("anger",   dirs, value = TRUE), pattern = "png$", full.names = TRUE)
disgust = dir(grep("disgust", dirs, value = TRUE), pattern = "png$", full.names = TRUE)
happy   = dir(grep("happy",   dirs, value = TRUE), pattern = "png$", full.names = TRUE)
sad     = dir(grep("sad",     dirs, value = TRUE), pattern = "png$", full.names = TRUE)
test_im = dir(grep("test",    dirs, value = TRUE), pattern = "png$", full.names = TRUE)

str(anger)
str(disgust)
str(happy)
str(sad)
str(test_im)

# %% [markdown]
# The pictures (video stills) show faces that were captured while more or less spontaneously expressing an emotion. Some of the images are repeated, but then shifted, rotated, or both. Not all are easy to classify:

# %% [code]
ok = file.copy(  happy[60], "happy.png", overwrite = TRUE)
ok = file.copy(    sad[61],   "sad.png", overwrite = TRUE)
ok = file.copy(  anger[61], "anger.png", overwrite = TRUE)
ok = file.copy(disgust[61], "disgust.png", overwrite = TRUE)
IRdisplay::display_html(glue::glue('<img src="happy.png?{runif(1)}" width="200" style="float:left" /><img src="sad.png" width="200" style="float:left" /><img src="anger.png" width="200" style="float:left" /><img src="disgust.png?{runif(1)}" width="200" style="float:left" />'))

# %% [markdown]
# Clearly the first is a happy face, but is the second a sad face, an angry face, or both?

# %% [markdown]
# # Data considerations
# 
# It's not difficult to find out how the data were collected. This is an important question to answer about any data set you use in a machine learning project because it will determine to what new data your model will generalize.
# 
# Answer the 3 most important questions for any ML project:
# 
# 1. Where do the data come from? (To which population will results generalize?)
# 2. What are candidate machine learning methods? (models? features?)
# 3. What is the Bayes' error bound? (Any guestimate from scientific literature or web resources?)
# 
# 
# 
# 

# %% [markdown]
# <div style=color:darkblue;background-color:#fafaff;min-height:8em; >
# 
# 
# <br>
#     <ol>
#         <li>
#             The data stems from the CK+ database. Participants were between 18 to 50 years old, 69 % female,
#             81%,Euro-American, 13% Afro-American, and 6% other groups-(Lucey, et al., 2010; p. 1).
#             Ideally, the results will generalize to a similar population so any data that has been created
#             photographed in a similar way.
#         </li>
#         <li>
#             <ul>
#                 <li>Multinomial Regression</li>
#                 <li>Random Forest / Other Tree-Based Methods</li>
#                 <li>Support Vector Machines</li>
#                 <li>K-Nearest Neighbours</li>
#                 <li>QDA / LDA</li>
#                 <li>Boosted tree</li>
#             </ul>
#         </li>
#         <li>
#             We would expect an accuracy of at least around 85%. Previous research suggests that children are
#             able to recognize emotions in human faces around 80 %. See here:
#             Mancini, G., Agnoli, S., Baldaro, B., Ricci Bitti, P. E., & Surcinelli, P. (2013). Facial
#             expressions of emotions: recognition accuracy and affective reactions during late childhood. The
#             Journal of psychology, 147(6), 599-617.
#             <br>
#             <a href="https://psycnet.apa.org/record/2013-34018-006">View article</a>
#         </li>
#     </ol>
#     
# 

# %% [markdown]
# # Bayes bound
# 
# To have an idea of a lower bound on the Bayes bound (i.e., the minimum accuracy that should be achievable). The best 'machine' we have at hand to recognize emotion from facial expression in the human brain. How often do human judges get it correct? In a small experiment where 42 of you have classified a subset 30 of the pictures each, an estimate for human classification accuracy was obtained:
# 
# 
# | disgust | anger | happy | sad |
# |--------:|------:|------:|-----|
# |   ..%   | ..%   | ..%   | ..% |
# 
# The average human classification accuracy was ...% ± ..%.
# 
# As always, it's handy to evaluate how the algorithm does on the training set: If the training set is not classified accurately, how can you expect the test set to do any better. This obvious fact is often overlooked—surprisingly.

# %% [markdown]
# 
# <img src="https://i.imgur.com/scE5XWh.png" width=300 style="float:right" />
# 
# # Feature extraction for images
# 
# 
# 
# So let's look at feature extraction. We discuss three approaches to feature extraction from images: 
# 
# 
# 1. Direct pixel analysis
#     - Use pixel intensities are features
#         - possibly resize images to reduce number of potential features (i.e., account for pixel correlations)
#         - make sure humans can still perform the task with chosen size
#     - Akin [MNIST and fashion-MNIST practical](https://www.kaggle.com/datasniffer/example-image-data-mnist)
# 
# <br />
# 
# 
# 2. Similar approach as phone sensor signals:
# 
#     <img src="https://i.imgur.com/qc6JIST.png" width=100 style="float:right" />
# 
#     - Compute descriptive statistics of the histograms:
#         - raw bin counts
#         - mean (per color channel)
#         - standard deviation
#         - shifted correlations (or inner product)
#         - "gradients" (so called HoG features: Histograms of Gradients, popular for detecting humans)
#         - SIFT features (too complicated to discuss here)
#         - spectral measures
#     - **Key issues**: scale, shift and *rotation* invariance, and intensity invariance
#     - Especially useful for edge coordinates
#     - Break up picture into patches (analogeous sub-epochs of phone sensor signals)
#     - Akin the Frey-Slate features in the [letter recognition practical](https://www.kaggle.com/datasniffer/k-nn-and-similarity-in-letter-recognition)
# 
# <img src="https://image.slidesharecdn.com/98f7f5b1-28c5-4385-a77b-368f27f95cd8-150419012246-conversion-gate01/95/lecture-21-image-categorization-computer-vision-spring2015-43-638.jpg?cb=1429406714" width=400 style="float:right;margin-left:50px" />
# 
# 3. Bag-of-features models:
# 
#     - Use a "dictionary" of pattern patches and count how often each pattern patch 'occurs' 
#         - has a match to a high degree in the image, thresholding
#         - use the counts (i.e., histograms) of these dictionary patches as features
#     - Where to find such a dictionary? 
#         - Extract from images themselves (just like in text processing: tokens are obtained from the available texts) 
#             - Use clustering methods
#         - Use pretrained "convolutional neural nets" 
#             - trained for specific recognition tasks
#             - trained for image generation
#             
# The 3rd method is tricky and computationally heavy, because the entire image has to be searched for a match of the "dictionary" of patterns. Although the Fast Fourier Transform (FFT; in two dimensions) offers a fast way to do this, it requires careful programming to implement this effectively. Convolutional neural network libraries indeed use FFT to efficiently implement this in specialized software. We've encountered FFT when we discussed the spectrum of signals.

# %% [markdown]
# # Import data
# 
# When working with image data, you often have many more Gigabytes of raw data than you have RAM memory available. Therefore, it is often not possible to work with all data "in memory". Resizing images often helps, but may cause loss of information.
# 
# The images for this competition are
# 
# - gray scale, so we need only one *color channel* 
# - are only 48 by 48 pixels
# 
# Furthermore there are only 2538 pictures in the training set. Therefore, we are lucky enough to be able to retain all images in RAM, and don't have to do "special stuff" to handle reading in image files while fitting a model.

# %% [markdown]
# Reading in images pixelwise is easiest: We simply store each image as a long vector of pixel intensities, row by row. Also we will need a vector that contains the emotion label for each of the images.

# %% [code]
# Combine all filenames into a single vector
train_image_files = c(anger, happy, sad, disgust)

# Read in the images as pixel values (discarding color channels)
X = sapply(train_image_files, function(nm) c(readPNG(nm)[,,1])) %>% t()            
y = c(rep("anger", length(anger)), rep("happy", length(happy)), rep("sad", length(sad)), rep("disgust", length(disgust)))

X_test = sapply(test_im, function(nm) c(readPNG(nm)[,,1])) %>% t() 


# Change row and column names of X to something more managable
rownames(X)      = gsub(".+train/", "", rownames(X))
rownames(X_test) = gsub(".+test/",  "", rownames(X_test))

colnames(X) = colnames(X_test) = paste("p",1:ncol(X), sep="")

# Check result (are X, X_test, and y what we expect)
X[1:6,20:23] %>% print
table(y)

X_test[1:6,20:23] %>% print

# %% [code]
# Look at the data
head(X)
head(X_test)

# %% [code]
# Visualization utility function
as_image = function(x, nr=sqrt(length(x))) {opar=par(mar=rep(0,4)); on.exit(par(opar)); image(t(matrix(x,nr))[,nr:1], col = gray(0:255/255),axes=F)}


options(repr.plot.width=4, repr.plot.height=4)
as_image(X[13,])
as_image(X_test[13,])

# %% [markdown]
# ## Near Zero Variance Pixels

# %% [markdown]
# Below it is calculated whether there are present nonvariant pixels in the data. Surprisingly, only 2 of those pixels would remain if $p=.15$. When checking for correlation among the pixels, it seems that there is a multicollinearity issue going on. Even when filtering for about 90 pixels not having near zero variance, still 68 of those pixels are classified as being highly correlated. Due to a high loss of pixels, we decided against removing those from the data. 

# %% [code]
Sums <- numeric(100)
p <- c(seq(0,1,by=0.01))
for (i in 1:100){
  
  # Compute P(X > 0)
  props = colMeans(X > 0)
  
  # Find pixels that are not near-zero-variance
  use_pix = props > p[i] & props < 1-p[i]
  
  # What is the number of pixels retained?
  Sums[i] <- sum(use_pix)
}

plot(Sums, type = "l")

print(Sums)
use_pix = props > p[7] & props < 1-p[7]
cor_indx <- findCorrelation(cor(X[,use_pix]), .95)
length(cor_indx)


# %% [markdown]
# # Histogram features  
# We look at the histograms of the different emotions. We plot the different emotions and later compute different features based on these histograms. We plot them to explore the differences between the histograms of the different emotions. We see in the plots that the histograms are quite similar, but differ slightly in the proportions of the pixelvalues. (We saw this idea in the notebook of team 5)
# 
# We then compute the descriptive histogram features. We first define functions to compute different features. Then we create a function to compute all of the features for the data. 

# %% [code]
# First we combine the data with the corresponding emotion
X_hist <- as_tibble(X, rownames = "id") %>%
  mutate(outcome = y) %>%
  select(outcome, id, everything()) %>%
  filter(!is.na(outcome))

# Change the data into long format 
X_histogram <- X_hist[,-2] %>%
  melt(direction = "long") %>%
  rename(pix = variable,
         value = value) 

# Plot histograms
X_histogram %>% ggplot(mapping = aes(x = value, color = outcome, fill = outcome)) +
  geom_bar(mapping = aes(y = ..prop.., group = outcome)) +
  facet_wrap(~ outcome, nrow = 2) +
  coord_cartesian(xlim = c(.1, .9), ylim = c(0,0.006)) +
  scale_y_continuous(labels = function(x) { paste0(x * 100, "%")}) +
  labs(x = "Pixel_value", y = "Percentage") +
  theme(legend.position = "none")


# %% [code]
# Define histogram features (from competition 2)

# Entropy function (from lecture slides)
entropy  <- function(x, nbreaks = nclass.Sturges(x)) {
  r = range(x)
  x_binned = findInterval(x, seq(r[1], r[2], len= nbreaks))
  h = tabulate(x_binned, nbins = nbreaks) # fast histogram
  p = h/sum(h)
  -sum(p[p>0] * log(p[p>0]))
}

# Function to compute mode 
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

# Function to get frequency peak 
getpeak <- function(x) {
  spec <- spectrum(x, log = "y",plot=FALSE) 
  peak <- spec$freq[which.max(spec$spec)]
  return(peak)
}

# Function to get frequency mean
freqmean <- function(x) {
  sbar = sum(spectrum(x, plot = FALSE)$freq * 
               spectrum(x, plot = FALSE)$spec * 
               (spectrum(x, plot = FALSE)$freq[2] - 
                  spectrum(x, plot = FALSE)$freq[1]))
  return(sbar)
}

freqvar <- function(x) {
  svar = sum((spectrum(x, plot = FALSE)$freq - 
                sum(hist(x, breaks = 40, plot = FALSE)$mids * 
                      hist(x, breaks = 40, plot = FALSE)$density * 
                      hist(x, breaks = 40, plot = FALSE)$mids[2] - 
                      hist(x, breaks = 40, plot = FALSE)$mids[1])) ^ 2 * 
               spectrum(x, plot = FALSE)$spec * 
               (spectrum(x, plot = FALSE)$freq[2] - 
                  spectrum(x, plot = FALSE)$freq[1]))  
  return(svar)
}


# %% [markdown]
# ## Possible features
# There are a whole lot of possible features, but with many features there is a chance of overfitting, so we chose to include the following:
# * mean
# * standard deviation
# * skewtosis
# * kurtosis
# * minimum and maximum
# * mean spectrum
# * variance spectrum
# * peak spectrum

# %% [code]
# Define a function to extract all the histogram features
extract_histogram_features <- function(filename) {
  # change data into long format
  long <- melt(filename, direction = "long") %>%
    rename(
      id = Var1,
      pix = Var2
    )
  
  long %>%
    group_by(id) %>%
    summarise(
      # mean
      m = mean(value),
      
      # standard deviation
      sd = sd(value),
      
      # skewness
      skew = e1071::skewness(value),
      
      # kurtosis
      kurt = e1071::kurtosis(value),
      
      # minimum
      min = min(value),
      
      # maximum
      max = max(value),
      
      # mean spectrum
      freq_m = freqmean(value),
      
      # variance spectrum
      freq_var = freqvar(value),
      
      # peak spectrum
      freq_peak = getpeak(value),
    )
  
}

# %% [code]
# Compute the features for the train and test sets respectively
hist_features <- extract_histogram_features(X)
head(hist_features)

hist_features_test <- extract_histogram_features(X_test)
head(hist_features_test)

# %% [code]
# Change the features into a tibble for inner_join for the train and test sets respectively
X_df <- as_tibble(X, rownames = "id")
head(X_df)

X_test_df <- as_tibble(X_test, rownames = "id")
head(X_test_df)


# %% [code]
# Join the features and the original data for both train and test data
X_features <- X_df %>%
  inner_join(hist_features, by = "id") %>%
  select(!id) %>%
  as.matrix()

head(X_features)

X_features_test <- X_test_df %>%
  inner_join(hist_features_test, by = "id") %>%
  select(!id) %>%
  as.matrix()

head(X_features_test)

# %% [markdown]
# # Histogram features from edges
# 
# Histogram features work best on edges. How can you detect edges? To answer that question we have to consider what an edge is: An edge is a rapid change in pixel intensities, so if we compute the difference between two consecutive pixels, and check if it is larger than a certain threshold, we can find the pixels that are at the edge of an abrupt intensity change. In principle this can be done in any possible direction (north, south, west, east, north-west, south-west, etc.), but it turns out that it is enough to do it in only two directions: north and west (or south and east for that matter). These give horizontal and vertical edges respectively. By computing differences in both north and west direction consecutively we filter out pixels that are part of a diagonal edge. 

# %% [code]
options(repr.plot.width=4*4, repr.plot.height=4)

# Compute edges by differencing neighboring pixels
im = matrix(X[756,],48)
h_edge = im[-1,] - im[-48,] # horizontal
v_edge = im[,-1] - im[,-48] # vertical
d_edge = h_edge[,-1] - h_edge[,-48] # diagonal

# Specify a threshold (hand tuned here on visual result)
threshold = .0625 

layout(t(1:4))
as_image(im)
as_image(h_edge < threshold,   47); mtext("horizontal edge pixels")
as_image(v_edge < threshold,   48); mtext("vertical edge pixels")
as_image(d_edge < threshold/2, 47); mtext("diagonal edge pixels")
as_image((h_edge[,-1] < 0.1) & (v_edge[-1,] < 0.1), 47); mtext("edge pixels")

# %% [markdown]
# You can use the detected edge pixels to compute Frey and Slate type features: Histogram descriptives of the x and y locations of the 'on' pixels in the edge pixel maps.

# %% [code]
# Load FreySlateFeatures function 
source("https://bit.ly/32um24j")

FreySlateFeatures(h_edge < threshold)

# %% [markdown]
# Note that Frey & Slate features were designed for distinguishing capital characters; not for distinguishing emotions in pictures of human faces. If you want to go this handicraft way of feature extraction you may want to consider choosing different histogram descriptors.
# 
# You will notice that not all features computed this way have non-zero variance, and it is generally dificult to predict which features are highly correlated or multi-collinear.

# %% [markdown]
# Credits go to Team 5, which calculated the edge features as shown in the code chunk below. At last, those edge features are binded to the train and test data.

# %% [code]
Features = tibble()
# Create loop for histogram features seen above:
for (i in 1:nrow(X)) {
  im = matrix(X[i,],48)
  h_edge = im[-1,] - im[-48,] # horizontal
  v_edge = im[,-1] - im[,-48] # vertical
  d_edge = h_edge[,-1] - h_edge[,-48] # diagonal
  Features <- bind_rows(Features, FreySlateFeatures(h_edge < threshold))
}

head(Features)

X_features <- cbind(X_features, Features)


Features = tibble()
# Create loop for histogram features seen above:
for (i in 1:nrow(X_test)) {
  im = matrix(X_test[i,],48)
  h_edge = im[-1,] - im[-48,] # horizontal
  v_edge = im[,-1] - im[,-48] # vertical
  d_edge = h_edge[,-1] - h_edge[,-48] # diagonal
  Features <- bind_rows(Features, FreySlateFeatures(h_edge < threshold))
}

head(Features)

X_features_test <- cbind(X_features_test,Features)

# %% [markdown]
# ## Near Zero variance and multicollinearity among all the new features

# %% [code]
# Remove near zero variance and correlated features

# Change into tibble to use select
X_nzv <- as_tibble(X_features)

# Remove near zero variance
nzv <- X_features %>% caret::nearZeroVar(names = TRUE)
nzv

X_clean <- X_nzv %>% select(-all_of(nzv))

head(X_clean)

corv <- caret::findCorrelation(cor(X_clean), .98, names = TRUE)
length(corv)

X_clean <- X_clean %>% select(-all_of(corv))
head(X_clean)


# Removing only those predictors that were also removed in the train data
X_clean_test <- X_features_test %>%
  as_tibble %>%
  select(-all_of(nzv), -all_of(corv)) %>%
  as.matrix()


# %% [markdown]
# # Fit a model
# 
# To figure out which model provides the best trade off between bias and variance, between accuracy and flexibility, one strategy is to fit both a flexible and a more rigid model and determine from CV error which direction on the flexiblity axis we should go to avoid overtraining.
# 
# We'll consider classification trees, random forests, multinomial regression, LDA, QDA with PCA preprocessing, kNN, and boosted trees here. Random forests are probably the least susceptible to overtraining and is considered one of the best "off the shelf" machine learning algorithms in the sense that they require little expertise in application, and easily perform well without tuning. (This is not to say that tuning should not be considered!) It's your uncle Bob's prefered algorithm so to say.
# 
# 

# %% [markdown]
# Before running the models, access multiple cores and create k-folds for manual cross validation
# - Faster for boosted trees
# - needed to get an accuracy value for the random forest

# %% [code]
## Load libraries for recruiting multiple CPU cores
library(parallel)
suppressMessages(library(doParallel))

## Turn multiple cores usage on
cat("Nr. of cores: ", detectCores(), "\n")
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

## Specify trainControl() object for caret
trCntrl = trainControl('cv', number = 5, allowParallel = TRUE)

# %% [code]
X_final <- as.matrix(X_clean)
X_final_test <- as.matrix(X_clean_test)

# %% [code]
# Create the training set
TrainSet <- cbind(y,X_final)

# Prepare the cross validation with K=5 folds manually
K <- 5
index <- rep(1:K, floor(nrow(TrainSet)/K)+1)[1:nrow(TrainSet)]
fold.index <- sample(index)

# %% [markdown]
# Checking how the training set looks

# %% [code]
head(TrainSet)

# %% [markdown]
# ## Classification tree
# 
# As an example, here we fit a classification tree, using the pixel based approach.

# %% [code]
## Fit a CART using 5-fold cross-validation to tune the complexity parameter

set.seed(2020) # for repeatability (generally don't do this!)

tt <- Sys.time()
fit_tree = train(x=X_final, y=y, method='rpart', trControl = trCntrl, tuneGrid = data.frame(cp=.02))
fit_tree

(dur <- Sys.time() - tt)

# %% [code]
## Graphical visualization of the decision tree
options(repr.plot.width=14, repr.plot.height=8)
plot(fit_tree$final, compress=TRUE, uniform=TRUE, margin=0.05, branch=.75); 
text(fit_tree$final, cex=0.8, all=TRUE, use.n=TRUE)

## Textual visualization of the decision tree
fit_tree$finalModel

# %% [markdown]
# ## Random Forest Model 

# %% [markdown]
# #### Note that cross validation was implemented manually here. The actual accuracy might deviate some bit.

# %% [code]
fitted_rfs <- list()
AccuracyRandomForest <- numeric(K)

for (i in 1:K){
  X_Train <- TrainSet[!fold.index==(K-(i-1)), ]
  X_Test <- TrainSet[fold.index==(K-(i-1)), ]
  Y <- unname(X_Train[,1])
  X_Train <- X_Train[,-1]   
  class(X_Train) <- "numeric"
  
  Y_Test <- unname(X_Test[,1])
  X_Test <- X_Test[,-1]
  class(X_Test) <- "numeric"
  
  fitted_rfs[[i]] = train(x=X_Train, y=Y, method='ranger', 
                          trControl = trainControl("none"), 
                          tuneGrid = data.frame(mtry=9, splitrule="gini", min.node.size=10)
  )
  
  ## Check performance on training set
  predforest = predict(fitted_rfs[[i]], X_Test, type='raw') 
  Performance = confusionMatrix(predforest, factor(Y_Test))
  
  AccuracyRandomForest[i] <- Performance$overall[1]
}

rf <- mean(AccuracyRandomForest)
print(paste0("The accuracy for the random forest is: ", rf))


fit_rf <- train(x=X_final, y=y, method='ranger', 
                trControl = trainControl("none"), 
                tuneGrid = data.frame(mtry=9, splitrule="gini", min.node.size=10)
)




# %% [markdown]
# ## Multinomial Regression

# %% [code]
## Multinomial regression with lasso penalty 

# This takes a bit of time, so maybe you skip this at first
fit_glm = train(
  x=X_final, y=y, method='glmnet', trControl = trCntrl,
  tuneGrid = expand.grid(alpha=1, lambda = 0.001)
)
fit_glm

# %% [markdown]
# ## LDA

# %% [code]
fit_lda = train(x=X_final, y=y, method='lda', trControl=trCntrl)
fit_lda

# %% [markdown]
# ## QDA with PCA preprocessing

# %% [code]
fit_qda = train(x=X_final, y=y, method='qda', trControl=trCntrl, preProcess="pca")
fit_qda

# %% [markdown]
# ## k-NN

# %% [code]
fit_knn = train(x=X_final, y=y, method='knn', trControl=trCntrl)
fit_knn

# %% [markdown]
# ## Boosted Tree

# %% [code]
fitted_xgbs <- list()
AccuracyBoostedTrees <- numeric(K)


for (i in 1:K){
  X_Train <- TrainSet[!fold.index==(K-(i-1)), ]
  X_Test <- TrainSet[fold.index==(K-(i-1)), ]
  
  Y <- unname(X_Train[,1])
  X_Train <- X_Train[,-1]   
  class(X_Train) <- "numeric"
  
  Y_Test <- unname(X_Test[,1])
  X_Test <- X_Test[,-1]
  class(X_Test) <- "numeric"
  
  ## Boosting is very slow with train(), even if xgboost can be rather fast
  
  fitted_xgbs[[i]] = train(x=X_Train, y=Y, method="xgbTree", 
                           trControl = trainControl("none"), 
                           tuneGrid = data.frame(
                             nrounds=600, max_depth=4, eta=.3, 
                             gamma=0, colsample_bytree=0.95, 
                             min_child_weight=1, subsample=1)
  )
  
  
  
  predtree = predict(fitted_xgbs[[i]], X_Test, type="raw")
  Performance = confusionMatrix(predtree, factor(Y_Test))
  
  AccuracyBoostedTrees[i] <- Performance$overall[1]
}    



# see ?xgboost::xgboost for tuning parameters in tuneGrid


xgb <- mean(AccuracyBoostedTrees)
print(paste0("The accuracy for the boosted tree is: ", xgb))

fit_xgb <- train(x=X_final, y=y, method="xgbTree", 
                 trControl = trainControl("none"), 
                 tuneGrid = data.frame(
                   nrounds=600, max_depth=4, eta=.3, 
                   gamma=0, colsample_bytree=0.95, 
                   min_child_weight=1, subsample=1)             
)


# %% [markdown]
# Below the chunk would do the cross validation itself; takes 40 minutes though instead of 20

# %% [code]
#fit_xgb <- train(x=X_final, y=y, method="xgbTree", 
#    trControl = trCntrl, 
#    tuneGrid = data.frame(
#        nrounds=600, max_depth=4, eta=.3, 
#        gamma=0, colsample_bytree=0.95, 
#        min_child_weight=1, subsample=1)             
#)

# %% [markdown]
# # Comparison of the models

# %% [code]
## Store the models in a list

# taken out for this competition: ridge regularized qda
mdls = list(glm = fit_glm, 
            lda = fit_lda, qda = fit_qda,  
            knn = fit_knn, rpart = fit_tree)

Acc = sapply(mdls, function(mdl) max(mdl$results$Accuracy))
Acc[length(Acc)+1] <- rf
Acc[length(Acc)+1] <- xgb     
names(Acc) <- c("glm", "lda", "qda", "knn", "rpart", "rf", "xgb")

# make a barplot with only the best performing model in red
color = 1 + (Acc >= max(Acc)) 
barplot(Acc, horiz=T, las=1, col = color)

# the most accurate model:
bestmodel <- names(Acc %>% sort(decreasing = TRUE))[1]

# get text command for best model:
bestmodel <- paste("fit_", bestmodel, sep = "")


# test eval command for best model:
eval(parse(text = bestmodel))

# %% [markdown]
# # Formatting your submission file
# 
# To format your submission file, you can use the following code:

# %% [code]
## Make predictions
predtree = predict(eval(parse(text = bestmodel)), X_clean_test, type='raw')

## Write to file
tibble(file = rownames(X_test), category = predtree) %>% 
  write_csv(path = "submission.csv")

## Check result
file.show("submission.csv")