# %% [markdown]
# # Team Skip, Cathelijne & René

# %% [markdown]
# In this notebook we will implement various methods to find a good model. It's important to point out that during the first round submission we only included the code of the final model and not all the different things that we tried out. Eventually we ended up on rank 3 of the leaderboard with our previous LDA analysis at round 1. 
# 
# For a better overview, we removed the given "unnecessary descriptions" for some basic steps. Some code elements stem from the "quickstart", "feature extraction from signals" and the "voice gender recognition".

# %% [markdown]
# ## Importing data and organizing files
# 
# First, we import the data, bring it into the right format and plot the data as a basic check.

# %% [code]
## Loading packages
library(tidyverse) 
library(ggplot2) 


# Important pre-processing steps
list.files(path = "../input")

# Make sure the data is available
if (length(list.files("../input", pattern = "recognition")) > 0) {
  
  # Copy all files to the current directory
  system("cp -r ../input/bda-2020-physical-activity-recognition/* ./")
  
} else {
  
  # Download data for this competition
  data_url = "https://phonesensordata.netlify.app/Archive.zip"
  download.file(data_url, "Archive.zip")
  
  # Unzip all files in the current directory
  unzip("Archive.zip")
  
}

# list files in the current working directory
list.files()

# show the content of the labels file in a seperate window
file.show("activity_labels.txt")



# 1. Reading the data
act_labels = read_delim("activity_labels.txt"," ",col_names=F,trim_ws=T) 
act_labels = act_labels %>% select(X1,X2)
act_labels

labels <- read_delim("./RawData/Train/labels_train.txt", " ", col_names = F)
colnames(labels) <- c('trial', 'userid', 'activity', 'start', 'end')

labels <- labels %>% mutate(activity = act_labels$X2[activity])

print(labels)


# identify the file name and extract the 'username' (participant ID) and 'expname' (experimental run)
filename = "RawData/Train/gyro_exp01_user01.txt"
username = gsub(".+user(\\d+).+", "\\1", filename) %>% as.integer()
expname =  gsub(".+exp(\\d+).+", "\\1", filename) %>% as.integer()

# import the data from the file
user01 <- read_delim(filename, " ", col_names = F)
#head(user01)

options(repr.plot.width=12)

plot.ts(user01, xlab="Sample number")

# 1.3 Merging Signals and Labels
print(labels[1:2,])

# Add the sequence start:end to each row in a list.
# The result is a nested table:
sample_labels_nested <- 
  labels %>% 
  rowwise() %>% # do next operation(s) rowwise
  mutate(sampleid = list(start:end)) %>%
  ungroup()

# Check the resulting table:
print(sample_labels_nested, n=6)



# Unnest the nested tabel.
sample_labels <- 
  sample_labels_nested %>% 
  
  # Rows are segments, we need to keep track of different segements
  mutate(segment = row_number() ) %>% 
  
  # Expand the data frame to one sample per row
  unnest() %>% 
  
  # Remove columns we don't need anymore
  select(-start, -end) 



# Check the result (first few rows are not interesting; rows 977-990 are)
print(sample_labels[977:990, ])


user_df <- 
  # Store signals user01 in a data frame, with 'userid' and 'trial'
  data.frame(userid = username, trial = expname, user01) %>%
  
  # Add 'sampleid' for matching the sampleid's in 'sample_labels'. 
  # The first sample in user01 signals always has sampleid=0; the last
  # has sampleid is therefore nrow(user01)-1.
  mutate(sampleid = 0:(nrow(user01)-1) ) %>%
  
  # Add the labels stored in sample_labels
  left_join(sample_labels) 

# Check the result (first few rows are not interesting; the following are)
print(user_df[1227:1239, ])

spectrum(user_df$X3,plot=FALSE, span = 11, log = "n")$spec
spectrum(user_df$X3, span = 11, log = "n")


options(repr.plot.width=15) # change plot width for nicer output

user_df %>% 
  ggplot(aes(x = sampleid, y = X1, col = factor(activity), group=segment)) + 
  geom_line()

# %% [markdown]
# ## Feature Extraction
# 
# Here we use the given code for some time domain features and include code for a function that extracts spectrum features and the mode.

# %% [code]
# 2.1 Time domain features

user_df %>% 
  # change 7986 to 8586 to see shifted walk cycle
  dplyr::filter(activity == "WALKING", segment == 13, 
                7596 < sampleid & sampleid < 7986) %>% 
  
  ggplot(aes(x = sampleid %% 54, y = X1, group = sampleid %/% 54, 
             col = factor(sampleid %/% 54))) + geom_line()

user_df %>% 
  # change 7986 to 8586 to see shifted walk cycle
  dplyr::filter(activity == "WALKING", segment == 13, 
                7596 < sampleid & sampleid < 8586) %>% 
  
  ggplot(aes(x = sampleid %% 54, y = X1, group = sampleid %/% 54, 
             col = factor(sampleid %/% 54))) + geom_line()


user_df %>%
  
  ggplot(aes(X1)) + 
  geom_histogram(bins=40, fill=1, alpha=0.5) + 
  geom_histogram(aes(X2), bins=40, fill = 2, alpha=0.5) + 
  geom_histogram(aes(X3), bins=40, fill = 4, alpha=0.5) +
  facet_wrap(~activity, scales = "free_y")



##check spectrum features
spectrum(user_df$X1, plot = FALSE)

### own function to get the mode
getmode <- function(x) {
  uniqv <- unique(x)
  uniqv[which.max(tabulate(match(x, uniqv)))]
}

### own function to get entropy
getentropy <- function(x) {
  probs <- prop.table(table(x))
  -sum(probs * log2(probs))
}


# from Group 8 to get the spectral peak
getpeak <- function(x) {
  spec <- spectrum(x, log = "y",plot=FALSE) 
  peak <- spec$freq[which.max(spec$spec)]
  return(peak)
}

# %% [markdown]
# ## Including new features
# 
# Before selecting the relevant features, we here made a list with features that we aimed to include statistical features:
# 
# - Means
# - Medians
# - modes
# - minima
# - maxima
# - absolute median values
# - standard deviations
# - root means square
# - 25th and 75th quantiles
# - Interquartile range
# - skewness and curtosis
# - entropy (see function definition above)
# - autocorrelations with lag1 and lag4
# - autocorreelations with lag1 from other predictors
# - Correlations between X1, X2, X3
# - Zerocrossings
# 
# Moreover, we included spectral features:
# -   RMS frequency 
# -	Center frequency
# -	Spectral minima, maxima, SDs
# -	Energy
# -	SRA
# -   spectral peak

# %% [code]
usertimedom <- user_df %>%
  
  # add an epoch ID variable (on epoch = 2.56 sec)
  mutate(epoch = sampleid %/% 128) %>% 
  
  # extract statistical features from each epoch
  group_by(epoch) %>%
  summarise(
    
    # The epoch's activity label is the mode of 'activity'
    activity = names(which.max(table(c("-", activity)))),
    
    # keep the starting sampleid of epoch as a time marker
    sampleid = sampleid[1],
    
    ### Features
    # Mean, median, mode, min and max
    m1 = mean(X1), 
    m2 = mean(X2), 
    m3 = mean(X3),
    med1 = median(X1),
    med2 = median(X2),
    med3 = median(X3),
    mod1 = getmode(X1),
    mod2 = getmode(X2),
    mod3 = getmode(X3),
    min1 = min(X1),
    min2 = min(X2),
    min3 = min(X3),
    max1 = max(X1),
    max2 = max(X2),
    max3 = max(X3),
    
    # Median absolute value
    absmed1 = mad(X1),
    absmed2 = mad(X2),
    absmed3 = mad(X3),
    
    # Standard deviations
    sd1 = sd(X1), 
    sd2 = sd(X2),
    sd3 = sd(X3),
    
    # Root mean square
    rms1 = sqrt(mean(X1^2)),
    rms2 = sqrt(mean(X2^2)),
    rms3 = sqrt(mean(X3^2)),
    
    # Quantiles
    q1_25 = quantile(X1, .25),
    q2_25 = quantile(X2, .25),
    q3_25 = quantile(X3, .25),
    q1_75 = quantile(X1, .75),
    q2_75 = quantile(X2, .75),
    q3_75 = quantile(X3, .75),
    
    # Interquartile range
    IQR1 = IQR(X1),
    IQR2 = IQR(X2),
    IQR3 = IQR(X3),
    
    # Skewness and kurtosis
    skew1 = e1071::skewness(X1),
    skew2 = e1071::skewness(X2),
    skew3 = e1071::skewness(X3),
    kurt1 = e1071::kurtosis(X1),
    kurt2 = e1071::kurtosis(X2),
    kurt3 = e1071::kurtosis(X3),
    
    # Entropy
    ent1 = getentropy(X1),
    ent2 = getentropy(X2),
    ent3 = getentropy(X3),
    
    # Power (will be very similar to mean...)
    Pow1 = mean(X1^2),
    Pow2 = mean(X2^2),
    Pow3 = mean(X3^3),
    
    
    # peak spectrum 
    spec_peak1 = getpeak(X1),
    spec_peak2 = getpeak(X2),
    spec_peak3 = getpeak(X3),
    
    
    # Autocorrelations with lag1 up to lag4
    AR1.1 = cor(X1, lag(X1), use = "pairwise"),
    AR1.2 = cor(X1, lag(X1, n = 2), use = "pairwise"),
    AR1.3 = cor(X1, lag(X1, n = 3), use = "pairwise"),
    AR1.4 = cor(X1, lag(X1, n = 4), use = "pairwise"),
    
    AR2.1 = cor(X2, lag(X2), use = "pairwise"),
    AR2.2 = cor(X2, lag(X2, n = 2), use = "pairwise"),
    AR2.3 = cor(X2, lag(X2, n = 3), use = "pairwise"),
    AR2.4 = cor(X2, lag(X2, n = 4), use = "pairwise"),
    
    AR3.1 = cor(X3, lag(X3), use = "pairwise"),
    AR3.2 = cor(X3, lag(X3, n = 2), use = "pairwise"),
    AR3.3 = cor(X3, lag(X3, n = 3), use = "pairwise"),
    AR3.4 = cor(X3, lag(X3, n = 4), use = "pairwise"),
    
    # Autocorrelations with lag1 from other predictors 
    AR12 = cor(X1, lag(X2), use = "pairwise"),
    AR13 = cor(X1, lag(X3), use = "pairwise"),
    AR21 = cor(X2, lag(X1), use = "pairwise"),
    AR23 = cor(X2, lag(X3), use = "pairwise"),
    AR31 = cor(X3, lag(X1), use = "pairwise"),
    AR32 = cor(X3, lag(X2), use = "pairwise"),
    
    # Correlations between X1, X2, X3
    C1_2 = cor(X1,X2, use = "pairwise"),
    C1_3 = cor(X1,X3, use = "pairwise"),
    C2_3 = cor(X2,X3, use = "pairwise"),
    
    # Zerocrossings
    zcr1 = 0.5 * mean(abs(sign(X1*(sampleid +1)) - sign(X1*(sampleid)))),
    zcr2 = 0.5 * mean(abs(sign(X2*(sampleid +1)) - sign(X2*(sampleid)))),
    zcr3 = 0.5 * mean(abs(sign(X3*(sampleid +1)) - sign(X3*(sampleid)))),
    
    
    # keep track of signal lengths
    n_samples = n()
  ) 

head(usertimedom)
glimpse(usertimedom)


# obtain all files
dir("./RawData/Train/", pattern = "^acc")


extractTimeDomainFeatures <- function(filename, sample_labels) {
  
  # extract user and experimental run ID's from file name
  username = gsub(".+user(\\d+).+", "\\1", filename) %>% as.numeric()
  expname  = gsub( ".+exp(\\d+).+", "\\1", filename) %>% as.numeric()
  
  # import the sensor signals from the file
  user01 <- read_delim(filename, " ", col_names = F, progress = TRUE, 
                       col_types = "ddd")
  
  
  # merge signals with labels 
  user_df <- 
    data.frame(userid = username, trial = expname, user01) %>%
    mutate(sampleid = 0:(nrow(user01)-1) ) %>%
    left_join(sample_labels, by = c('userid','trial','sampleid')) 
  
  
  # split in epochs of 128 samples and compute features per epoch
  usertimedom <-  user_df %>%
    
    # add an epoch ID variable (on epoch = 2.56 sec)
    mutate(epoch = sampleid %/% 128) %>% 
    
    # extract statistical features from each epoch
    group_by(epoch) %>%
    summarise(
      # keep track of user and experiment information
      user_id = username, 
      exp_id = expname,   
      
      # epoch's activity labels and start sample
      activity = names(which.max(table(c("-", activity)))),
      sampleid = sampleid[1],
      
      
      ### Features
      # Mean, median, mode, min and max
      #m1 = mean(X1), 
      #m2 = mean(X2), 
      #m3 = mean(X3),
      
      #med1 = median(X1),
      #med2 = median(X2),
      #med3 = median(X3),
      
      #mod1 = getmode(X1),
      #mod2 = getmode(X2),
      #mod3 = getmode(X3),
      
      #min1 = min(X1),
      #min2 = min(X2),
      #min3 = min(X3),
      
      #max1 = max(X1),
      #max2 = max(X2),
      #max3 = max(X3),
      
      # Standard deviations
      sd1 = sd(X1), 
      sd2 = sd(X2),
      sd3 = sd(X3),
      
      # Variance 
      #var1 = var(X1),
      #var2 = var(X2),
      #var3 = var(X3),
      
      # Quantiles
      #q1_25 = quantile(X1, .25),
      #q2_25 = quantile(X2, .25),
      #q3_25 = quantile(X3, .25),
      #q1_75 = quantile(X1, .75),
      #q2_75 = quantile(X2, .75),
      #q3_75 = quantile(X3, .75),
      
      # Interquartile range
      #IQR1 = IQR(X1),
      #IQR2 = IQR(X2),
      #IQR3 = IQR(X3),
      
      # median absolute deviation (called median absolute value in article Tom found)
      #absmed1 = mad(X1),
      #absmed2 = mad(X2),
      #absmed3 = mad(X3),
      
      # mean absolute value
      absmean1 = mean(abs(X1)),
      absmean2 = mean(abs(X2)),
      absmean3 = mean(abs(X3)),
      
      #RMS  
      rms1 = sqrt(mean(X1^2)),
      rms2 = sqrt(mean(X2^2)),
      rms3 = sqrt(mean(X3^2)),
      
      # Zero crossings (the average number the sign of a time wave changes)
      #zcr1 = 0.5 * mean(abs(sign(X1*(sampleid+1)) - sign(X1*(sampleid)))),
      #zcr2 = 0.5 * mean(abs(sign(X2*(sampleid+1)) - sign(X2*(sampleid)))),
      #zcr3 = 0.5 * mean(abs(sign(X3*(sampleid+1)) - sign(X3*(sampleid)))),
      
      # Skewness and kurtosis
      skew1 = e1071::skewness(X1),
      skew2 = e1071::skewness(X2),
      skew3 = e1071::skewness(X3),
      kurt1 = e1071::kurtosis(X1),
      kurt2 = e1071::kurtosis(X2),
      kurt3 = e1071::kurtosis(X3),
      
      #crest factor
      CF1 = max(abs(X1))/sqrt(mean(X1^2)),
      CF2 = max(abs(X2))/sqrt(mean(X2^2)),
      CF3 = max(abs(X3))/sqrt(mean(X3^2)),
      
      #spectral features
      
      #RMS frequency
      spec_RMS1 = sqrt(mean(spectrum(X1,plot=FALSE)$spec^2)),
      spec_RMS2 = sqrt(mean(spectrum(X2,plot=FALSE)$spec^2)),
      spec_RMS3 = sqrt(mean(spectrum(X3,plot=FALSE)$spec^2)),
      
      #center frequency
      cenF1 = sqrt(spectrum(X1,plot=FALSE)$freq[which.max(spectrum(X1,plot=FALSE)$spec)] 
                   *spectrum(X1,plot=FALSE)$freq[which.min(spectrum(X1,plot=FALSE)$spec)]),
      cenF2 = sqrt(spectrum(X1,plot=FALSE)$freq[which.max(spectrum(X2,plot=FALSE)$spec)] 
                   *spectrum(X1,plot=FALSE)$freq[which.min(spectrum(X2,plot=FALSE)$spec)]),
      cenF3 = sqrt(spectrum(X1,plot=FALSE)$freq[which.max(spectrum(X3,plot=FALSE)$spec)] 
                   *spectrum(X3,plot=FALSE)$freq[which.min(spectrum(X1,plot=FALSE)$spec)]),
      
      #spec_mean1 = mean(spectrum(X1,plot=FALSE)$spec),
      #spec_mean2 = mean(spectrum(X2,plot=FALSE)$spec),
      #spec_mean3 = mean(spectrum(X3,plot=FALSE)$spec),
      
      #spec_median1 = median(spectrum(X1,plot=FALSE)$spec),
      #spec_median2 = median(spectrum(X2,plot=FALSE)$spec),
      #spec_median3 = median(spectrum(X3,plot=FALSE)$spec),
      
      #spec_mode1 = getmode(spectrum(X1,plot=FALSE)$spec),
      #spec_mode2 = getmode(spectrum(X2,plot=FALSE)$spec),
      #spec_mode3 = getmode(spectrum(X3,plot=FALSE)$spec),
      
      #spec_sd1 = sd(spectrum(X1,plot=FALSE)$spec),
      #spec_sd2 = sd(spectrum(X2,plot=FALSE)$spec),
      #spec_sd3 = sd(spectrum(X3,plot=FALSE)$spec),
      
      #spec_min1 = min(spectrum(X1,plot=FALSE)$spec, na.rm = TRUE),
      #spec_min2 = min(spectrum(X2,plot=FALSE)$spec, na.rm = TRUE),
      #spec_min3 = min(spectrum(X3,plot=FALSE)$spec, na.rm = TRUE),
      
      #spec_max1 = max(spectrum(X1,plot=FALSE)$spec, na.rm = TRUE),
      #spec_max2 = max(spectrum(X2,plot=FALSE)$spec, na.rm = TRUE),
      #spec_max3 = max(spectrum(X3,plot=FALSE)$spec, na.rm = TRUE),
      
      
      # Entropy
      #ent1 = getentropy(X1),
      #ent2 = getentropy(X2),
      #ent3 = getentropy(X3),
      
      # Power (will be very similar to mean...)
      #Pow1 = mean(X1^2),
      #Pow2 = mean(X2^2),
      #Pow3 = mean(X3^3),
      
      #Energy
      Ene1 = sum(X1^2),
      Ene2 = sum(X2^2),
      Ene3 = sum(X3^2),
      
      #SRA
      SRA1 = mean(sqrt(abs(X1))),
      SRA2 = mean(sqrt(abs(X2))),
      SRA3 = mean(sqrt(abs(X3))),
      
      # Autocorrelations with lag1 up to lag4
      AR1.1 = cor(X1, lag(X1), use = "pairwise"),
      #AR1.2 = cor(X1, lag(X1, n = 2), use = "pairwise"),
      #AR1.3 = cor(X1, lag(X1, n = 3), use = "pairwise"),
      #AR1.4 = cor(X1, lag(X1, n = 4), use = "pairwise"),
      
      AR2.1 = cor(X2, lag(X2), use = "pairwise"),
      #AR2.2 = cor(X2, lag(X2, n = 2), use = "pairwise"),
      #AR2.3 = cor(X2, lag(X2, n = 3), use = "pairwise"),
      #AR2.4 = cor(X2, lag(X2, n = 4), use = "pairwise"),
      
      AR3.1 = cor(X3, lag(X3), use = "pairwise"),
      #AR3.2 = cor(X3, lag(X3, n = 2), use = "pairwise"),
      #AR3.3 = cor(X3, lag(X3, n = 3), use = "pairwise"),
      #AR3.4 = cor(X3, lag(X3, n = 4), use = "pairwise"),
      
      # Autocorrelations with lag1 from other predictors 
      #AR12 = cor(X1, lag(X2), use = "pairwise"),
      #AR13 = cor(X1, lag(X3), use = "pairwise"),
      #AR21 = cor(X2, lag(X1), use = "pairwise"),
      #AR23 = cor(X2, lag(X3), use = "pairwise"),
      #AR31 = cor(X3, lag(X1), use = "pairwise"),
      #AR32 = cor(X3, lag(X2), use = "pairwise"),
      
      # Correlations between X1, X2, X3
      C1_2 = cor(X1,X2, use = "pairwise"),
      C1_3 = cor(X1,X3, use = "pairwise"),
      C2_3 = cor(X2,X3, use = "pairwise"),
      
      n_samples = n()
    ) 
  
  usertimedom 
}

# %% [code]
filename = "./RawData/Train/acc_exp01_user01.txt"
df = extractTimeDomainFeatures(filename, sample_labels) 
print(df)

# demonstrate this for only the first 5 files
filenames <- dir("./RawData/Train/", "^acc", full.names = TRUE) 

# map_dfr runs `extractTimeDomainFeatures` on all elements in 
# filenames and binds results row wise
myData = suppressMessages(map_dfr(filenames, extractTimeDomainFeatures, sample_labels)) 

# Check the result
#print(myData)

# demonstrate this for only the first 5 files
filenames <- dir("./RawData/Train/", "^gyro", full.names = TRUE) 

# map_dfr runs `extractTimeDomainFeatures` on all elements in 
# filenames and binds results row wise
myData_gyro = suppressMessages(map_dfr(filenames, extractTimeDomainFeatures, sample_labels)) 

# Check the result
#print(myData_gyro)

all_data = merge(myData, myData_gyro, by=c("epoch", "user_id", "exp_id", "activity", "sampleid", "n_samples"), suffixes = c("_acc","_gyro"))

head(all_data)
colnames(all_data)
glimpse(all_data)

# %% [markdown]
# ## Model Fitting
# 
# Here we fit various models and select relevant features.
# 
# Importantly, 
# 
# 1. we tried predictions using the entire dataset with n_sample !=128 and including the activity "-" however this led to worse performance. Thus, we decided to delete n_samples < 128, and delete all unlabelled data.
# 
# 2. we examined non-zero variance for all features however this showed no relevant predictors to remove. If there are no predictors to remove, then the nzv - object is empty and running the non-zero variance method to remove [non - existing] objects will lead to an error. Thus, we removed it from the code and added the explanation here. 
# 
# 3. we removed highly correlated predictors.
# 
# 4. used cross validation technique with five iterations. We chose the number 5 based on a short literature review and the sample size of the data.
# 
# 5. experimented with LDA, QDA, KNN, mulitnominal models

# %% [code]
#First, we will remove redundant features.

all_data = merge(myData, myData_gyro, by=c("epoch", "user_id", "exp_id", "activity", "sampleid", "n_samples"), suffixes = c("_acc","_gyro"))

##delete spec_gyro
#all_data <- all_data[,!grepl("spec.*gyro", colnames(all_data))]


## delete n_samples < 128, and we delete all unlabelled data
all_data <- all_data %>% 
  filter(n_samples == 128 & activity != '-')

# Near zero variance for all features, first 6 columns skipped
#nzv <- caret::nearZeroVar(all_data[,-c(1:6)]) + 6

all_data2 <- all_data#[,-nzv] 

# Correlated features
R <- cor(all_data2[7:ncol(all_data2)]) 
cor_ind <- caret::findCorrelation(R, cutoff = 0.8)
dplyr::glimpse(cor_ind) 

length(cor_ind)
### 96 features have very high correlations (makes sense as they are statistical proporties with a lot linear dependencies...), these will be removed from the predictor set

# account for that the first 6 rows must be left in the data
cor_ind2 <- cor_ind + 6

# removing the correlated features
all_data3 <- all_data2[,-cor_ind2]
glimpse(all_data3)

# Cross-validation
trcntr = caret::trainControl('cv', number = 5, p=0.85)

## LDA, QDA, mulitnominal

# LDA
fit_lda = caret::train(activity ~ ., data=all_data3[,-c(1:3,5,6)], method="lda", trControl = trcntr)
fit_lda

#QDA
#fit_qda = caret::train(activity ~ ., data=all_data3[,-c(1:3,5,6)], method="qda", trControl = trcntr)
#fit_qda
#this gives the error that some group is too small for a qda

# Multinomial regression
fit_multi = caret::train(activity ~ ., data=all_data3[,-c(1:3,5,6)], method="multinom", trControl = trcntr)
fit_multi

# KNNs
grid = expand.grid(k = c(6,7,8,9,10))

knns_fit = train(activity ~ ., data=all_data3[,-c(1:3,5,6)], method="knn", preProcess = c("scale"), trControl = trcntr, tuneGrid = grid)
knns_fit

#After the first round of model fitting, multinomial logistic regression (.88) and knn (.86) worked best, considering accuracy only. This was done without thinking about the potential of having way to many predictors. However, It is important to take this into account because otherwise overfitting will be a problem. So in the next block we chose predictors through stepwise analysis of the models we already fitted.

## stepwise LDA, QDA, Multinomial regression

##LDA & QDA

# LDA
fit_lda_step = caret::train(activity ~ ., data=all_data3[,-c(1:3,5,6)], method="stepLDA", trControl = trcntr)
fit_lda_step

# for completeness sake we keep the code in but since some groups are to small this will not be followed up on
# fit_qda_step = caret::train(activity ~ ., data=all_data3[,-c(1:3,5,6)], method="stepQDA", trControl = trcntr)
# fit_qda_step

# Multinomial regression
fit_multi_step = caret::train(activity ~ ., data=all_data3[,-c(1:3,5,6)], method="multinom", direction="forward", trControl = trcntr)
fit_multi_step

## KNN
# Another option that has been explained in the lectures was the unscaled k- nearest- neighbours option. 
# Earlier we only tried the scaled option so we thought to add in the unscaled option for the sake of getting the whole range of 
# options and to be sure that we didn't leave out a maybe efficient model. 

#knn not scaled
knn_fit = train(activity ~ ., data=all_data3[,-c(1:3,5,6)], method="knn", trControl = trcntr, tuneGrid = grid)
knn_fit

## Plot

## Visualize model performance differences for 

# Put all fitted models in a named list (if necessary 
# change the names to match the names you used to store 
# the models in.)
models = list(knn = knn_fit, multi_s = fit_multi_step,lda_s = fit_lda_step,knns = knns_fit, multi = fit_multi,lda = fit_lda)

# Extract the cross-validated accuracies from each model
Acc = sapply(models, function(mdl) max(mdl$results$Accuracy)) 

# make a barplot with only the best performing model in red
color = 1 + (Acc >= max(Acc)) 
barplot(Acc, horiz=T, las=1, col = color, main = "accuracy per model",xlab = "accuracy", ylab = "model name", xlim = c(0:1))

best_model = fit_multi_step
#From this we could conclude that the bar that is red is the best model (multinominal regression) 
# if you look at the accuracy so this will be the model that we use to predict the unknown instances of the data.

# %% [markdown]
# ## Visualisation of Model performance after stepwise regression

# %% [code]
barplot(Acc, horiz=T, las=1, col = color, main = "accuracy per model",xlab = "accuracy", ylab = "model name", xlim = c(0:1))

# %% [markdown]
# ## Submissions

# %% [code]
# 5. Submissions

#The test data can be imported in the same way as the training data, you only have to change `Train` to `Test` in the directory path:

filenames_test = dir("./RawData/Test/", "^acc", full.names = TRUE)

# map_dfr runs `extractTimeDomainFeatures` on all elements in 
# filenames and binds results row wise
test_acc = map_dfr(filenames_test, extractTimeDomainFeatures, sample_labels) 

# demonstrate this for only the first 5 files
filenames_test <- dir("./RawData/Test/", "^gyro", full.names = TRUE) 

# map_dfr runs `extractTimeDomainFeatures` on all elements in 
# filenames and binds results row wise
test_gyro = map_dfr(filenames_test, extractTimeDomainFeatures, sample_labels) 

test_data = merge(test_acc, test_gyro, by=c("epoch", "user_id", "exp_id", "activity", "sampleid", "n_samples"), suffixes = c("_acc","_gyro"))
glimpse(test_data)

##see whether there are NA's for features in the testdata
colSums(is.na(test_data))

sum(colSums(is.na(test_data)))

##For the AR feature there NA's, as this feature is used in the model, we will fill these NA's with the column means

for(i in 1:ncol(test_data)){
  test_data[is.na(test_data[,i]), i] <- median(test_data[,i], na.rm = TRUE)
}

colSums(is.na(test_data))

sum(colSums(is.na(test_data)))

# Predictions based on logistic regression 
pred <- predict(best_model, new = test_data)

predictions <- test_data %>%     
  mutate(activity = pred)

## Formatting the submission file

#To help you turning your predictions into the right format, the following code can help. Here it is executed on the training set data frame, but the same can be applied to the test set data frame.

predictions %>%
  
  # prepend "user" and "exp" to user_id and exp_id
  mutate(
    ##credits to group 7 for this piece of code
    user_id = case_when(user_id < 10 ~ paste("user0", user_id, sep=""),TRUE ~ paste("user", user_id, sep="")),
    exp_id = case_when(exp_id < 10 ~ paste("exp0", exp_id, sep=""), TRUE ~ paste("exp", exp_id, sep=""))
  ) %>% 
  
  # unit columnes user_id, exp_id and sample_id into a string 
  # separated by "_" and store it in the new variable `Id`
  unite(Id, user_id, exp_id, sampleid) %>%
  
  # retain only the `Id` and  predictions
  select(Id, Predicted = activity) %>%
  
  # write to file
  write_csv("test_set_predictions.csv")


# Check the result
file.show("test_set_predictions.csv")

# %% [markdown]
# ## Final comments
# 
# We spent significant amount of time on the modelling part (running both QDA, LDA, KNN, multinominal models). Regarding the features, we tried to initially use many statistical and spectral features. Our literature review on other possible features led us to more creative and sophisticated features, such as the Mel frequency cepstral coefficients (MFCC) and (ii) Perceptive Linear Prediction (PLP) features. However, the extraction of those more advanced features requires external libraries in R - thus, we could not use them in here.
# 

# %% [markdown]
# ## Other notes.
# 
# Here we will add notes from the notebook that may still be helpful.

# %% [code]
##features by previous research

##acc X1 6
#SD -> done
#correlation coefficient
#Kurtosis -> done
#RMS -> done
#SRA -> done
#Frequency Center

##acc X2 6
#SD -> done
#Skewness -> done
#Kurtosis -> done
#RMS -> done
#SRA -> done
#RMS frequency

##acc X3 7
#Mean Abolute value -> done
#SD -> done
#Correlation coefficient
#Skewness -> done
#Energy -> done
#SRA -> done
#Frequency center

##gyro X1 8
#SD -> done
#Median -> done
#Correlation coefficient
#Skewness -> done
#RMS -> done
#Energy -> done
#Frequency Center
#RMS Frequency

##gyro X2 7
#SD -> done
#Skewness -> done
#Kurtosis -> done
#Energy -> done
#SRA -> done
#Crest factor
#RMS frequency

##gyro X3 7
#SD -> done
#correlation coefficient
#Kurtosis -> done
#RMS -> done
#Energy -> done
#SRA -> done
#Frequency Center