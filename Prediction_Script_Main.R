# %% [markdown]
# 
# # Outline of Content
# 
# 1. Import and pre-process the data
# 
# -- import transcripts, personality scores, gender
# 
# -- merge the datasets
# 
# -- create a train and test dataset 
# 
# 2. Extracting features and building the model
# 
# -- frequency of words and their correlation with the Big 5
# 
# -- NRC, AFINN, BING word lists
# 
# -- total amount of words per vlogger
# 
# -- proportion of sentiment words per vlogger
# 
# 3. Feature Selection
# 
# 
# 4. Predictive Models
# 
# 
# 5. Converting data to kaggle format

# %% [code] {"_execution_state":"idle"}
# Loading required packages
library(tidyverse) 
library(tidytext)
library(devtools)

# Creating RMSE function
RMSE = function(pred, act){
  sqrt(mean((pred - act)^2))
}

# "../input/" directory
list.files(path = "../input")

# %% [markdown]
# There are three .csv files in the directory structure:

# %% [code]
directory_content = list.files("../input/bda2020big5/youtube-personality", full.names = TRUE)
print(directory_content)

# %% [markdown]
# In addition there's a "transcript" folder (see number \[2\] in the output above) in which the actual video transcripts are stored in `.txt` files. 
# 
# Store these file paths in variables for easy reference later on:

# %% [code]
# Path to the transcripts directory with transcript .txt files
path_to_transcripts = directory_content[2] 

# .csv filenames (see output above)
AudioVisual_file    = directory_content[3]
Gender_file         = directory_content[4]
Personality_file    = directory_content[5]

# %% [markdown]
# # 1. Import the data
# 
# We'll import
# 
# - transscripts
# - Personality scores
# - Gender
# 
# ## 1.1 Importing transcripts
# 
# The transcript text files are stored in the subfolder 'transcripts'. They can be listed with the following commands:

# %% [code]
transcript_files = list.files(path_to_transcripts, full.names = TRUE) 

print(head(transcript_files,2))

# %% [markdown]
# The transcript file names encode the vlogger ID that you will need for joining information from the different data frames. A clean way to extract the vlogger ID's from the names is by using the funcation `basename()` and removing the file extension ".txt".

# %% [code]
vlogId = basename(transcript_files)
vlogId = str_replace(vlogId, pattern = ".txt$", replacement = "")
head(vlogId)

# %% [markdown]
# To include features extracted from the transcript texts you will have to read the text from files and store them in a data frame. For this, you will need the full file paths as stored in `transcript_files`.
# 
# Here are some tips to do that programmatically
# 
# - use either a `for` loop, the `sapply()` function, or the `map_chr()` from the `tidyverse`
# - don't forget to also store `vlogId` extracted with the code above 
# 
# We will use the `map_chr()` function here:

# %% [code]
transcripts_df = tibble(
  
  # vlogId connects each transcripts to a vlogger
  vlogId=vlogId,
  
  # Read the transcript text from all file and store as a string
  Text = map_chr(transcript_files, ~ paste(readLines(.x), collapse = "\\n")), 
  
  # `filename` keeps track of the specific video transcript
  filename = transcript_files
)

# %% [code]
transcripts_df %>% 
  head(2)

# %% [markdown]
# ## Import personality scores
# 
# The other data files can be read in with `read_delim` (not `read_csv` because the files are not actually comma separated). For instance, the following should work:

# %% [code]
# Import the Personality scores
pers = read_delim(Personality_file, delim=" ")

# %% [code]
head(pers,2)

# %% [markdown]
# ## Import gender
# 
# Gender info is stored in a separate `.csv` which is also delimited with a space. This file doesn't have column names, so we have to add them ourselves:

# %% [code]
gender = read.delim(Gender_file, head=FALSE, sep=" ", skip = 1)

# Add column names
names(gender) = c('vlogId', 'gender')


head(gender,2)

# %% [markdown]
# ## Merging the `gender` and `pers` dataframes
# 
# Obviously, we want all the information in a single tidy data frame. While the builtin R function `merge()` can do that, the `tidyverse()` has a number of more versatile and consistent functions called `left_join`, `right_join`, `inner_join`, `outer_join`, and `anti_join`. We'll use `left_join` here to merge the gender and personality data frames:

# %% [code]
vlogger_df = left_join(gender, pers) %>% 
  mutate(gender = ifelse(gender == "Male", 1, 0)) #male = 1, female = 0
head(vlogger_df) # VLOG8 has missing personality scores: those should be predicted



transcripts_df_all <- full_join(transcripts_df, vlogger_df) %>%
  drop_na()

# %% [markdown]
# ## Creating a test and train dataset
# 
# To choose features for our model we split our dataset in a test and training set and select the number of iterations for the cross-validation

# %% [code]
##extract words per vlog
transcripts_df_tokens_tot <- unnest_tokens(transcripts_df_all, token, Text, token = 'words')

##all vloggers
ID <- unique(transcripts_df_tokens_tot$vlogId)

##amount of models
n <- 40

overall_list <- vector(mode = "list", length = n)
for(i in 1:n){
  set.seed(22*i)
  overall_list[[i]][[1]] <- sample(ID, round(length(ID)*0.85)) # train IDs
  overall_list[[i]][[2]] <- ID[!(ID %in% overall_list[[i]][[1]])] # test IDs
}

# %% [markdown]
# # 2. Extracting features & building model

# %% [markdown]
# ## Frequency per word and its correlation with the Big Five
# 
# A novel feature that we ourselves came up with is to include the frequency of words and its correlation with Big 5 profiles. Intuitively, we would hypothesize that certain words are distinctly used by vloggers with a high score on a certain Big 5 domain. Using the example of agreeableness, we illustrate this below in two tables that print the highest and lowest correlations of words and the agreeableness dimension.

# %% [code]
cor_list <- vector(mode = "list", length = n)

##list of stop words
stopwords <- get_stopwords()

for(i in 1:n){
  ##only take tokens from train set to not create bias
  transcripts_df_tokens <- transcripts_df_tokens_tot[transcripts_df_tokens_tot$vlogId %in% overall_list[[i]][[1]],]
  
  
  ##remove stop words
  transcripts_df_tokens_ns  <- 
    transcripts_df_tokens %>%
    anti_join(stopwords, by = c(token = "word"))
  
  ##get count of token per vlog
  transcripts_df_token_count_ns <- 
    transcripts_df_tokens_ns %>%
    count(vlogId, token, sort = T)
  
  ##get count of token overall
  transcripts_df_count_ns <- 
    transcripts_df_tokens_ns %>%
    count(token, sort = T) 
  
  # Merge into one file
  transcripts_df_tokens_count_ns <- full_join(transcripts_df_all, transcripts_df_token_count_ns) %>%
    select(-c(filename,Text))
  
  transcripts_df_tokens_count_top_ns <- transcripts_df_tokens_count_ns[transcripts_df_tokens_count_ns$token %in% transcripts_df_count_ns$token,]
  cor_list[[i]] <- transcripts_df_tokens_count_top_ns
}

# most used words
transcripts_df_count_ns %>% 
  arrange(desc(n)) %>%
  head()

# %% [code]
##calculate the correlation between the frequency of words and the personality dimensions
for(i in 1:n){
  transcript_count_ns <- cor_list[[i]] %>%
    spread(token, n, fill = 0) %>%
    select(-c(vlogId, gender)) %>%
    as.matrix %>%
    cor() %>%
    as.data.frame %>%
    select(c('Extr', 'Agr', 'Cons', 'Emot', 'Open')) %>%
    slice(-c(1:5))
  cor_list[[i]] <- transcript_count_ns
}

## Show highest and lowest correlations for Agreeableness as an example
transcript_count_ns %>%
  arrange(Agr) %>%
  head()

transcript_count_ns %>%
  arrange(Agr) %>%
  tail()

# %% [code]
##get all tokens (not just from training data)
transcripts_tokenized <- unnest_tokens(transcripts_df, token, Text, token = 'words')

# remove the stop words
transcripts_tokenized2  <- 
  transcripts_tokenized %>%
  anti_join(stopwords, by = c(token = "word")) 

s <- 1
feat_cor_list <- list()
for(i in 1:n){
  #Putting the words in a column instead of columnnames
  transcript_count_ns_nr <- rownames_to_column(cor_list[[i]], "word")
  
  
  ##assign correlations from training data to all tokens
  transcripts_weighed <- 
    inner_join(transcripts_tokenized2, transcript_count_ns_nr, by = c(token = 'word')) 
  
  ##seperate negative and positive correlations per personality dimension
  TR_Relab <- transcripts_weighed %>% mutate(
    Extr_neg = ifelse(Extr<0, abs(Extr), 0),
    Extr_pos = ifelse(Extr_neg!=0, 0, abs(Extr)),
    Agr_neg = ifelse(Agr<0, abs(Agr), 0),
    Agr_pos = ifelse(Agr_neg!=0, 0, abs(Agr)),
    Cons_neg = ifelse(Cons<0, abs(Cons), 0),
    Cons_pos = ifelse(Cons_neg!=0, 0, abs(Cons)),
    Emot_neg = ifelse(Emot<0, abs(Emot), 0),
    Emot_pos = ifelse(Emot_neg!=0, 0, abs(Emot)),
    Open_neg = ifelse(Open<0, abs(Open), 0),
    Open_pos = ifelse(Open_neg!=0, 0, abs(Open)),
  )
  
  
  ##calculate the sum scores and the mean of the correlations per vlog
  TR_Relab_sums <- numeric()
  TR_Relab_means <- numeric()
  for (i in unique(TR_Relab$vlogId)) {
    TR_Relab_sums <- rbind(TR_Relab_sums, colSums(TR_Relab[TR_Relab$vlogId == i,9:18]))
    TR_Relab_means <- rbind(TR_Relab_means, colMeans(TR_Relab[TR_Relab$vlogId==i,9:18]))
  }
  
  
  ##make a dataframe with sum scores
  TR_Relab_sums <- cbind(unique(TR_Relab$vlogId), TR_Relab_sums)
  TR_Relab_sums <- as_tibble(TR_Relab_sums)
  names(TR_Relab_sums) <- c("vlogId", names(TR_Relab_sums)[-1])
  TR_Relab_sums[,-1] <- apply(TR_Relab_sums[,-1],2, as.numeric)
  TR_Relab_sums[,-1] <- as.data.frame(scale(TR_Relab_sums[,-1]))
  
  ##make dataframe with mean scores
  TR_Relab_means <- cbind(unique(TR_Relab$vlogId), TR_Relab_means)
  names(TR_Relab_means) <- c("vlogId", names(TR_Relab_means)[-1])
  TR_Relab_means <- as_tibble(TR_Relab_means)
  TR_Relab_means[,-1] <- apply(TR_Relab_means[,-1],2, as.numeric)
  TR_Relab_means <- TR_Relab_means %>% 
    rename(
      vlogId = V1
    )
  TR_Relab_means[,-1] <- as.data.frame(scale(TR_Relab_means[,-1]))
  
  
  ##join all dataframes and the vlogId
  features_per_vlogger_cor <- left_join(vlogger_df, TR_Relab_means) 
  features_per_vlogger_cor <- merge(features_per_vlogger_cor, TR_Relab_sums, by = 'vlogId', suffix = c("_mean", "_sum"))
  feat_cor_list[[s]] <- features_per_vlogger_cor
  s <- s+1
}

#remove NA's from features
features <- features_per_vlogger_cor %>% 
  filter(!is.na(Extr))
head(features)

# %% [markdown]
# # Test feature
# 
# After constructing the features, we will test whether this feature performs well enough on our test data. We will compare the models with the mean scores, sum scores and a model with both scores and compare their RMSE.

# %% [code]
##split the data
training_data <- features[features$vlogId %in% overall_list[[n]][[1]],]
test_data <- features[features$vlogId %in% overall_list[[n]][[2]],]

## 1.model of the mean scores
fit_mean <- lm(cbind(Extr, Agr, Cons, Emot, Open) ~.,training_data[,-c(1,2,(ncol(training_data)-9):ncol(training_data))])
fit_mean_s = summary(fit_mean)

pred <- predict(fit_mean, new = test_data)
act <- as.matrix(test_data[,3:7])

fit_mean_s
RMSE(pred, act)

## 2.model of the sum scores
fit_sum <- lm(cbind(Extr, Agr, Cons, Emot, Open) ~.,training_data[,c(3:7,(ncol(training_data)-9):ncol(training_data))])
fit_sum_s = summary(fit_sum)

pred <- predict(fit_sum, new = test_data)
act <- as.matrix(test_data[,3:7])

fit_sum_s
RMSE(pred, act)

## 3.model with both scores
fit_both <- lm(cbind(Extr, Agr, Cons, Emot, Open) ~.,training_data[,-c(1,2)])
fit_both_s = summary(fit_both)

pred <- predict(fit_sum, new = test_data)
act <- as.matrix(test_data[,3:7])

fit_both_s
RMSE(pred, act)

# %% [markdown]
# ## NRC
# 
# Another feature database we could use is the NRC.

# %% [code]
#load NRC word list
nrc <- read.csv("https://pastebin.com/raw/EWMYszfq", stringsAsFactors = FALSE)  

#count number of words per sentiment per vlog using NRC
transcripts_tokenized_labeled1 <- 
  inner_join(transcripts_tokenized2, nrc, by = c(token = 'word')) %>%
  select(-filename) %>%
  count(vlogId, sentiment) %>%
  spread(sentiment, n, fill = 0)

head(transcripts_tokenized_labeled1)

# %% [markdown]
# ## AFINN
# 
# Another feature database we could use is the AFINN.

# %% [code]
##load AFINN package
download.file("http://www2.imm.dtu.dk/pubdb/edoc/imm6010.zip","afinn.zip")
unzip("afinn.zip")
afinn = read.delim("AFINN/AFINN-111.txt", sep="\t", col.names = c("word","score"), stringsAsFactors = FALSE)

transcripts_tokenized_labeled2 <- inner_join(transcripts_tokenized2, afinn, by = c(token = 'word')) %>%
  select(-filename)

head(transcripts_tokenized_labeled2)

# %% [code]
# function to find the mode.
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#Calculating AFINN centre measure scores and joining it to the file with features
transcripts_tokenized_labeled1$mean_AFINN <- 0
transcripts_tokenized_labeled1$median_AFINN <- 0
transcripts_tokenized_labeled1$mode_AFINN <- 0
for (i in unique(transcripts_tokenized_labeled2$vlogId)) {
  transcripts_tokenized_labeled1[transcripts_tokenized_labeled1$vlogId==i,]$mean_AFINN <- mean(transcripts_tokenized_labeled2[transcripts_tokenized_labeled2$vlogId==i,]$score)
  transcripts_tokenized_labeled1[transcripts_tokenized_labeled1$vlogId==i,]$median_AFINN <- median(transcripts_tokenized_labeled2[transcripts_tokenized_labeled2$vlogId==i,]$score)
  transcripts_tokenized_labeled1[transcripts_tokenized_labeled1$vlogId==i,]$mode_AFINN <- getmode(transcripts_tokenized_labeled2[transcripts_tokenized_labeled2$vlogId==i,]$score)
}

head(transcripts_tokenized_labeled1)

# %% [markdown]
# ## BING
# 
# Another feature database we could use is the BING.

# %% [code]
##get number of negative/ positive words per vlog
transcripts_tokenized_labeled <- inner_join(transcripts_tokenized2, get_sentiments('bing'), by = c(token = 'word')) %>%
  select(-filename) %>%
  count(vlogId, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  rename(negative_BING = negative, positive_BING = positive)

transcripts_tokenized_labeled1 <- merge(transcripts_tokenized_labeled1, transcripts_tokenized_labeled, by = 'vlogId')

head(transcripts_tokenized_labeled1)

# %% [markdown]
# ## Total amount of words per vlogger
# 
# Another feature we could use.

# %% [code]
#computing amount of words used in total per vlogger
nwords <- transcripts_tokenized2 %>% 
  count(`vlogId`, vlogId)

names(nwords) <- c("vlogId", "total_words")

transcripts_tokenized_labeled1 <- merge(transcripts_tokenized_labeled1, nwords, by = 'vlogId')

head(transcripts_tokenized_labeled1)

# %% [markdown]
# ## Proportion of sentiment words per vlog
# 
# Another feature we could use.

# %% [code]
proportions <- transcripts_tokenized_labeled1[c(1:11,15:17)] %>%
  mutate_if(is.numeric, funs(proportion = ./total_words)) %>%
  select(-c(2:14,"total_words_proportion"))

transcripts_tokenized_labeled1 <- merge(transcripts_tokenized_labeled1, proportions, by = 'vlogId')

head(transcripts_tokenized_labeled1)

# %% [markdown]
# Merging all features together.

# %% [code]
feat_list <- list()
for(i in 1:n){
  df <- feat_cor_list[[i]] 
  
  #merge all features
  all_features <- merge(df ,transcripts_tokenized_labeled1, by = 'vlogId')
  feat_list[[i]] <- all_features
  
}
head(all_features)

# %% [markdown]
# # 3. Feature selection linear model
# 
# We have extracted a bunch of features. We will now select useful features and optimize our model. We will do this per personality dimension using stepwise regression.

# %% [code]
##get all the column names of the features to have an overview
names(all_features)

##get index of anger
index = which(colnames(all_features)=="anger")

train_sets <- list()
test_sets <- list()
for(i in 1:n){
  df <- feat_list[[i]]
  ##scale all features
  df[index:ncol(df)] <- lapply(df[index:ncol(df)], function(x) c(scale(x)))
  
  ##split the data in i different sets
  train_sets[[i]] <- df[df$vlogId %in% overall_list[[i]][[1]],]
  test_sets[[i]] <- df[df$vlogId %in% overall_list[[i]][[2]],]                                                       
  
}





##take a look at the correlations of all features
cor_all_features <- all_features %>%
  select(-c(vlogId, gender, Extr, Agr, Cons, Emot, Open)) %>%
  drop_na() %>%
  as.matrix %>%
  cor()
head(cor_all_features)

# %% [markdown]
# ## Subset features
# As some features highly correlate, we will make subset of features.

# %% [code]
##Indices of the sum features
index_sum1 = which(colnames(all_features)=="Extr_neg_sum")
index_sum2 = which(colnames(all_features)=="Open_pos_sum")

##Indices of the proportion features
index_proportion1 = which(colnames(all_features)=="anger_proportion")
index_proportion2 = which(colnames(all_features)=="positive_BING_proportion")

##we want to test proportion features rather than the normal ones
##index of AFINN features as these are not in the proportion features
index_AFINN1 = which(colnames(all_features)=="mean_AFINN")
index_AFINN2 = which(colnames(all_features)=="mode_AFINN")

for(i in 1:n){
  df <- train_sets[[i]]
  train_sets[[i]] = df[,c(1:index_sum1-1, index_AFINN1:index_AFINN2, index_proportion1:index_proportion2)]
}

# %% [code]
for(i in 1:n){
  df <- train_sets[[i]]
  
  ##calculate correlations of all features
  cor_proportion_features <- df %>%
    select(-c(vlogId, Extr, Agr, Cons, Emot, Open)) %>%
    drop_na() %>%
    as.matrix %>%
    cor()
  
  
  features_correlated <- caret::findCorrelation(cor_proportion_features, cutoff = 0.69, verbose=FALSE, names=TRUE, exact = TRUE)
  
  train_sets[[i]] <- df %>%
    select(-c(all_of(features_correlated)))
  
  ##features that are dropped out
  print(features_correlated)
}

# %% [markdown]
# # 4. Predictive model

# %% [markdown]
# ## Feature selection & modelling Extraversion

# %% [code]
##make lists that will hold the best model, the predictions and a vector that holds the RMSEA of each model
best_models_extr <- list()
best_RMSEA_extr <- c()
best_pred_extr <- list()

for(i in 1:n){
  
  train_df <- train_sets[[i]]
  test_df <- test_sets[[i]]
  
  #Extraversion models with proportions
  null_mod_Extr <- lm(Extr ~ 1, data = train_df[,-c(1,4:7)])
  full_mod_Extr <- lm(Extr ~ ., data = train_df[,-c(1,4:7)])
  
  #stepwise regression
  mod_Extr_f <- step(null_mod_Extr, scope = list(lower = null_mod_Extr, upper = full_mod_Extr), direction = "forward", trace=0)
  mod_Extr_b <- step(full_mod_Extr, scope = list(lower = null_mod_Extr, upper = full_mod_Extr), direction = "backward", trace=0)
  
  #actual scores
  act <- as.matrix(test_df[,3])
  
  ##prediction forward
  pred_Extr_f <- predict(mod_Extr_f, new = test_df)
  rmsea_f <- RMSE(pred_Extr_f, act)
  
  ##prediction backward
  pred_Extr_b <- predict(mod_Extr_b, new = test_df)
  rmsea_b <- RMSE(pred_Extr_b, act)
  
  ##choose best model
  if(rmsea_f < rmsea_b){
    best_models_extr[[i]] <- mod_Extr_f
    best_RMSEA_extr <- c(best_RMSEA_extr, rmsea_f)
    best_pred_extr[[i]] <- pred_Extr_f
  }else{
    best_models_extr[[i]] <- mod_Extr_b
    best_RMSEA_extr <- c(best_RMSEA_extr, rmsea_b)
    best_pred_extr[[i]] <- pred_Extr_b
  }
  
}

best_RMSEA_extr

# %% [markdown]
# ## Feature selection & modelling Agreeableness

# %% [code]
##make lists that will hold the best model, the predictions and a vector that holds the RMSEA of each model
best_models_agr <- list()
best_RMSEA_agr <- c()
best_pred_agr <- list()

for(i in 1:n){
  
  train_df <- train_sets[[i]]
  test_df <- test_sets[[i]]
  
  #models agreeableness with proportions
  null_mod_Agr <- lm(Agr ~ 1, data = train_df[,-c(1,3,5:7)])
  full_mod_Agr <- lm(Agr ~ ., data = train_df[,-c(1,3,5:7)])
  
  #stepwise regression
  mod_Agr_f <- step(null_mod_Agr, scope = list(lower = null_mod_Agr, upper = full_mod_Agr), direction = "forward", trace = 0)
  mod_Agr_b <- step(full_mod_Agr, scope = list(lower = null_mod_Agr, upper = full_mod_Agr), direction = "backward", trace = 0)
  
  #actual scores
  act <- as.matrix(test_df[,4])
  
  #Prediction Agr forward
  pred_Agr_f <- predict(mod_Agr_f, new = test_df)
  rmsea_f <- RMSE(pred_Agr_f, act)
  
  #Prediction Agr backward
  pred_Agr_b <- predict(mod_Agr_b, new = test_df)
  rmsea_b <- RMSE(pred_Agr_b, act)
  
  ##choose best model
  if(rmsea_f < rmsea_b){
    best_models_agr[[i]] <- mod_Agr_f
    best_RMSEA_agr <- c(best_RMSEA_agr, rmsea_f)
    best_pred_agr[[i]] <- pred_Agr_f
  }else{
    best_models_agr[[i]] <- mod_Agr_b
    best_RMSEA_agr <- c(best_RMSEA_agr, rmsea_b)
    best_pred_agr[[i]] <- pred_Agr_b
  }
}

best_RMSEA_agr

# %% [markdown]
# ## Feature selection & modelling Conscientiousness

# %% [code]
##make lists that will hold the best model, the predictions and a vector that holds the RMSEA of each model
best_models_cons <- list()
best_RMSEA_cons <- c()
best_pred_cons <- list()

for(i in 1:n){
  
  train_df <- train_sets[[i]]
  test_df <- test_sets[[i]]
  
  #Conscientiousness models with proportions
  null_mod_Cons <- lm(Cons ~ 1, data = train_df[,-c(1,3:4,6:7)])
  full_mod_Cons <- lm(Cons ~ ., data = train_df[,-c(1,3:4,6:7)])
  
  #stepwise regression
  mod_Cons_f <- step(null_mod_Cons, scope = list(lower = null_mod_Cons, upper = full_mod_Cons), direction = "forward", trace = 0)
  mod_Cons_b <- step(full_mod_Cons, scope = list(lower = null_mod_Cons, upper = full_mod_Cons), direction = "backward", trace = 0)
  
  #actual score
  act <- as.matrix(test_df[,5])
  
  ##prediction forward
  pred_Cons_f <- predict(mod_Cons_f, new = test_df)
  rmsea_f <- RMSE(pred_Cons_f, act)
  
  ##prediction backward
  pred_Cons_b <- predict(mod_Cons_b, new = test_df)
  rmsea_b <- RMSE(pred_Cons_b, act)
  
  ##choose best model
  if(rmsea_f < rmsea_b){
    best_models_cons[[i]] <- mod_Cons_f
    best_RMSEA_cons <- c(best_RMSEA_cons, rmsea_f)
    best_pred_cons[[i]] <- pred_Cons_f
  }else{
    best_models_cons[[i]] <- mod_Cons_b
    best_RMSEA_cons <- c(best_RMSEA_cons, rmsea_b)
    best_pred_cons[[i]] <- pred_Cons_b
  }
}

best_RMSEA_cons

# %% [markdown]
# ## Feature selection & modelling Emotional Stability

# %% [code]
##make lists that will hold the best model, the predictions and a vector that holds the RMSEA of each model
best_models_emot <- list()
best_RMSEA_emot <- c()
best_pred_emot <- list()

for(i in 1:n){
  
  train_df <- train_sets[[i]]
  test_df <- test_sets[[i]]
  
  #Models emotional stability with proportions
  null_mod_Emot <- lm(Emot ~ 1, data = train_df[,-c(1,3:5,7)])
  full_mod_Emot <- lm(Emot ~ ., data = train_df[,-c(1,3:5,7)])
  
  mod_Emot_f <- step(null_mod_Emot, scope = list(lower = null_mod_Emot, upper = full_mod_Emot), direction = "forward", trace = 0)
  mod_Emot_b <- step(full_mod_Emot, scope = list(lower = null_mod_Emot, upper = full_mod_Emot), direction = "backward", trace = 0)
  
  #actual scores
  act <- as.matrix(test_df[,6])
  
  #Prediction Emot forward
  pred_Emot_f <- predict(mod_Emot_f, new = test_df)
  rmsea_f <- RMSE(pred_Emot_f, act)
  
  #Prediction Emot backward
  pred_Emot_b <- predict(mod_Emot_b, new = test_df)
  rmsea_b <- RMSE(pred_Emot_b, act)
  
  ##choose best model
  if(rmsea_f < rmsea_b){
    best_models_emot[[i]] <- mod_Emot_f
    best_RMSEA_emot <- c(best_RMSEA_emot, rmsea_f)
    best_pred_emot[[i]] <- pred_Emot_f
  }else{
    best_models_emot[[i]] <- mod_Emot_b
    best_RMSEA_emot <- c(best_RMSEA_emot, rmsea_b)
    best_pred_emot[[i]] <- pred_Emot_b
  }
}

best_RMSEA_emot

# %% [markdown]
# ## Feature selection & modelling Openness

# %% [code]
##make lists that will hold the best model, the predictions and a vector that holds the RMSEA of each model
best_models_open <- list()
best_RMSEA_open <- c()
best_pred_open <- list()

for(i in 1:n){
  
  train_df <- train_sets[[i]]
  test_df <- test_sets[[i]]
  
  
  #Openness models with proportions
  null_mod_Open <- lm(Open ~ 1, data = train_df[,-c(1,3:6)])
  full_mod_Open <- lm(Open ~ ., data = train_df[,-c(1,3:6)])
  
  #stepwise regression
  mod_Open_f <- step(null_mod_Open, scope = list(lower = null_mod_Open, upper = full_mod_Open), direction = "forward", trace = 0)
  mod_Open_b <- step(full_mod_Open, scope = list(lower = null_mod_Open, upper = full_mod_Open), direction = "backward", trace = 0)
  
  #actual scores
  act <- as.matrix(test_df[,7])
  
  ##prediction forward
  pred_Open_f <- predict(mod_Open_f, new = test_df)
  rmsea_f <- RMSE(pred_Open_f, act)
  
  ##prediction backward
  pred_Open_b <- predict(mod_Open_b, new = test_df)
  rmsea_b <- RMSE(pred_Open_b, act)
  
  ##choose best model
  if(rmsea_f < rmsea_b){
    best_models_open[[i]] <- mod_Open_f
    best_RMSEA_open <- c(best_RMSEA_open, rmsea_f)
    best_pred_open[[i]] <- pred_Open_f
  }else{
    best_models_open[[i]] <- mod_Open_b
    best_RMSEA_open <- c(best_RMSEA_open, rmsea_b)
    best_pred_open[[i]] <- pred_Open_b
  }
}

best_RMSEA_open

# %% [markdown]
# ## Predictions merged

# %% [code]
for(i in 1:n){
  test_df <- test_sets[[i]]
  pred2 = cbind(best_pred_extr[[i]], best_pred_agr[[i]], best_pred_cons[[i]], best_pred_emot[[i]], best_pred_open[[i]])
  act2 = test_df %>% select(Extr, Agr, Cons, Emot, Open) %>% as.matrix
  
  print(RMSE(pred2, act2))
}

# %% [markdown]
# # 5. Format for Kaggle Competition

# %% [code]
test_data <- all_features %>% filter(is.na(Extr))
head(test_data)

# %% [code]
best_models_extr_2 <- best_models_extr[which(best_RMSEA_extr < mean(best_RMSEA_extr))]
best_models_agr_2 <- best_models_agr[which(best_RMSEA_agr < mean(best_RMSEA_agr))]
best_models_cons_2 <- best_models_cons[which(best_RMSEA_cons < mean(best_RMSEA_cons))]
best_models_emot_2 <- best_models_emot[which(best_RMSEA_emot < mean(best_RMSEA_emot))]
best_models_open_2 <- best_models_open[which(best_RMSEA_open < mean(best_RMSEA_open))]

#Predictions Extr
n <- 10
pred_Extr_matrix = numeric()
for (i in 1:length(best_models_extr_2)) {
  pred_Extr_vec <- predict(best_models_extr_2[[i]], new = test_data)
  pred_Extr_matrix <- cbind(pred_Extr_matrix, pred_Extr_vec)
}
pred_Extr <- rowMeans(pred_Extr_matrix)

#Predictions Agr
pred_Agr_matrix = numeric()
for (i in 1:length(best_models_agr_2)) {
  pred_Agr_vec <- predict(best_models_agr_2[[i]], new = test_data)
  pred_Agr_matrix <- cbind(pred_Agr_matrix, pred_Agr_vec)
}
pred_Agr <- rowMeans(pred_Agr_matrix)

#Predictions Cons
pred_Cons_matrix = numeric()
for (i in 1:length(best_models_cons_2)) {
  pred_Cons_vec <- predict(best_models_cons_2[[i]], new = test_data)
  pred_Cons_matrix <- cbind(pred_Cons_matrix, pred_Cons_vec)
}
pred_Cons <- rowMeans(pred_Cons_matrix)

#Predictions Emot
pred_Emot_matrix = numeric()
for (i in 1:length(best_models_emot_2)) {
  pred_Emot_vec <- predict(best_models_emot_2[[i]], new = test_data)
  pred_Emot_matrix <- cbind(pred_Emot_matrix, pred_Emot_vec)
}
pred_Emot <- rowMeans(pred_Emot_matrix)

#Predictions Open
pred_Open_matrix = numeric()
for (i in 1:length(best_models_open_2)) {
  pred_Open_vec <- predict(best_models_open_2[[i]], new = test_data)
  pred_Open_matrix <- cbind(pred_Open_matrix, pred_Open_vec)
}
pred_Open <- rowMeans(pred_Open_matrix)

pred_Final <- tibble(vlogId = test_data$vlogId, Extr = pred_Extr, Agr = pred_Agr, Cons = pred_Cons, Emot = pred_Emot, Open = pred_Open)

# %% [code]
# compute output data frame
testset_pred <- pred_Final

head(testset_pred)

# %% [code]
# put data frame in the right format
testset_pred_long  <- 
  testset_pred %>% 
  gather(pers_axis, value, -vlogId) %>%
  arrange(vlogId, pers_axis)

head(testset_pred_long)

# %% [code]
# Obtain the right format for Kaggle
testset_pred_final <- 
  testset_pred_long %>%
  unite(Id, vlogId, pers_axis) 

# Check if we succeeded
testset_pred_final

# Write to csv
testset_pred_final %>% 
  rename(Id = Id,
         Expected = value) %>%
  write_csv(path = "predictions.csv")

# Check if the file was written successfully.
list.files()

# %% [code]
hist(pred_Final$Extr)
hist(pred_Final$Agr)
hist(pred_Final$Emot)
hist(pred_Final$Open)
hist(pred_Final$Cons)

# %% [code]
[code]