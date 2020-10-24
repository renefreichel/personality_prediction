# %% [markdown]
# # Team 1: Bea, Jasmina & René

# %% [code] {"_execution_state":"idle"}
## Importing packages

library(tidyverse) # metapackage with lots of helpful functions
library(tidytext)
library(glmnet)
library(lexicon) # used for adding features 
library(glmnet) # used for fitting models

# Here we can find the input
list.files(path = "../input")

# Using this to hopefully have more computational power
doMC::registerDoMC(cores = 4)

# %% [markdown]
# # 1. The project
# 
# In this competition you will predict customer sentiments regarding [Baby products purchased on Amazon.com](http://jmcauley.ucsd.edu/data/amazon/), on the basis of their written reviews. 
# 
# Answer the 3 most important questions for any ML project:
# 
# 1. Where do the data come from? (To which population will results generalize?)
# 2. What are candidate machine learning methods? (models? features?)
# 3. What is the Bayes' error bound?
# 

# %% [markdown]
# # Our answers
# 1. This question can be answered on different levels:
# 
# -The data stems originally from people who wrote reviews on Amazon about baby products. 
# 
# -The dataset was provided by Emily Fox, Ph.D.  and Carlos Guestrin and is summarized in a dataset that is on the UCSD server: http://jmcauley.ucsd.edu/data/amazon/
# 
# 2. The candidate machine learning algorithms are Lasso and ridge regression next to PCA.
# 3. Bayes error bound: Humans are quite good at deducing whether a review is positive or negative, so they should be able to distinguish between a satisfactory and unsatisfactory review with high %.

# %% [markdown]
# # 2. Read Data
# 
# Locate and load the data into memory.

# %% [code]
dir("../input", recursive=TRUE)

# %% [code]
# Find the right file path
csv_filepath = dir("..", pattern="amazon_baby.csv", recursive=TRUE, full.names = TRUE)

# Read in the csv file
amazon = read_csv(csv_filepath) %>%
  rownames_to_column('id') 

# %% [code]
# Checking results
head(amazon)

# %% [markdown]
# The data frame contains both the train and test data. The test data are the reviews for which the rating is missing and you need to provide a prediction. 
# 
# The following logical index variable will be of help selecting the desired rows without the need to split the data frame into seperate sets. This makes it easier to extract features.

# %% [code]
trainidx = !is.na(amazon$rating)
table(trainidx)

# %% [markdown]
# From the above, there are 153,531 training samples and 30,000 test samples.

# %% [markdown]
# # 3. Preprocessing
# 
# The table contains, the product `name`, the textual `review`, and the `rating`. Should we use only the `review`'s or also the `name`? Clearly the products differ on quality which will cause the ratings to differ, so we want to include product identiy as a predictive feature. How can we do this? Well, there are several ways, but we'll use a trick: we'll prepend the product name to the review text. That is we will paste the `name` string and `review` string into a single string. In that way, we incorporate both product names and review text without haveing to handle product names separately. 
# 
# Doing so will also handle another problem: Some of the reviews that are empty strings, and we wont be able to make a prediction for them. By pasting `name` and `review` we'll at least have the name to predict their rating. 
# 
# Here is code for pasting `name` and `review` using the `unite()` function:

# %% [code]
# Paste name and review into a single string separated by a "–".
# The new string replaces the original review.
amazon = amazon %>% 
  unite(review, name, review, sep = " — ", remove = FALSE)

print(amazon)

# %% [markdown]
# # 3.1 Tokenization
# 
# We're going to use tidytext to break up the text into separate tokens and count the number of occurences per review. To keep track of the review to which the review belongs, we have added the rownames as `id` above, which is simply the row number. As tokens you can consider single words, pairs of words called bi-grams, or n-grams. 

# %% [code]
reviews = amazon %>% 
  
  # tokinize reviews at word level
  unnest_tokens(token, review) %>%
  
  # count tokens within reviews as 'n'
  # (keep id, name, and rating in the result)
  count(id, name, rating, token)

head(reviews)

# %% [markdown]
# ## Stopwords
# 
# A common step in text analytics is to remove unimportant words. But what are unimportant words? Linguists have come up with lists of so called stop words that we've run into before. Words on these lists, such as 'me', 'what', 'a', 'the', etcetera, putatively do not carry a lot of information. This is because they occur so frequently in any text, but especially because their usage does not vary a lot across texts on different topics.
# 
# However, you should always ask yourself "information on what specifically?". In the Personality competion you may have observed that words like 'me', 'myself', 'we', do seem to carry information about personality. Hence, habitual stop word removal (that you'll find advocated in online posts all the time) is generally ill advised.
# 
# A priori we mostly have little reason to assume any potential feature is uninformative with respect to the target variable, and the only way to find out, is to test the hypothesis statistically. For instance, if we look specifically whether the stop words listed in the table returned by `get_stopwords()`, we find that their counts are highly predictive for `rating`, even if we control for stop word totals per review:

# %% [markdown]
# # Decision to not remove stopwords
# We initially removed the stop words with the code provided and evaluated our model. The model seems to perform better when the stop words are not removed. Thus, we commented out the following code chunk.

# %% [code]
# sw = reviews %>% 

# Add the total number of tokens per review as 'N'
#  add_count(id, name = "N") %>% 

# Retain only tokens that are stopwords
#  inner_join(get_stopwords(), by = c(token='word')) %>% 

# Compute the total number of stopwords per review
#  group_by(id, rating, N) %>% 
#  summarise(n_stopwords = sum(n))

# head(sw)

# %% [markdown]
# The resulting data frame allows us to relate `rating` to the number stopwords in the review, and check if this relation is statistical significant. To make sure the relation isn't caused by the total number of tokens `N`, we correct for it in the regression:

# %% [code]
# lm(rating ~ n_stopwords + N, sw) %>% summary()

# %% [markdown]
# The regression coefficients are highly significant, even thought the explained variance is rather small. Individual words like "a" and "what" are also highly significant. 
# 
# __Note__: This should not be perceived in any way as an encouragement to conduct regression analyses for all kinds of text features&mdash;there are simply too many of them. It is just to demonstrate that you shouldn't mindlessly follow common praxis or convention in removing "uninformative" features. 
# 
# The only reason to mention stopwords here is because too often it prescribed as non-optional in internet posts. It is best to rely on non-zero variance and correlations (and multicolinearity) to decide on "non-informative features").

# %% [markdown]
# 
# # 4. Features engineering
# 
# Features computed for tokens in text are based on the Bag of Words (BoW) model: Each document is considered a bag of words, in wich order plays no particular. A better name would be Bag of Tokens, because tokens can also be bi-grams etc, but we're stuck with BoW. Common features are
# 
# 
# - **document occurence**: 
#     > 0-1 encoding of the presence or absence of a token in a document (here: review)
#     
# - **token counts**: 
#     > simple counts $n_{t,d}$ of each token $t$ within documents $d$ (resulting in a document by term matrix, or DTM)
# 
# - **term frequency ($TF_{d,t}$)**: 
#     > the relative frequency of a term within a document $\displaystyle {n_{d,t} \over  \sum_t n_{d,t}}$
# 
# - **inverse document frequency ($IDF_t$)**: 
#     > inverse the relative frequency with which a term occurs among the $N$ documents, expressed on a log scale (a measure of 'surprise') as  $-\log\left({DF_t \over N}\right)$ Here $DF_t$ is the number of documents that contain the token $t$.
# 
# - **the $TFIDF_{d,t}$**: 
#     > the product of TF and IDF
# 
# - **vector space embeddings**: 
#     > advanced features like factor loadings (eigen vectors) from a PCA of the DTM, or "word2vec" representations of words, sentences, and paragraphs (not discussed here), usually obtained by training neural networks on a very large corpus
# 
# 
# The motivation for $TF_{d,t}$ is simply that the more often a token $t$ occurs in a document, the more likely it is that the topic of the document is closely related to that token. A problem of $TF_{d,t}$ is that it does not take into account that certain words simply occur more frequently because of their role in language (such as 'a', 'but', etc.). 
# 
# The motivation for the $IDF_t$ is that the more wide spread the use of a token $t$ is among all documents, the less likely it conveys information about the topic of any particular document. Hence, the more surprising a word is, the more likely it conveys information about the topic of the document in which it is found. 
# 
# The $TFIDF_{d,t}$ banks on both of these ideas and quantifies the important of a term for a given document. 
# 
# While $TFIDF_{d,t}$ is extensively applied and very successful in document retrieval systems (i.e., search engines), the $IDF_t$ part has much less use over $TF_{d,t}$ in *predictive* models because the $IDF_t$ part simply scales the $TF_{d,t}$ features accross documents. This scaling may have an effect on scale sensitive algorithms like PCA and algorithms that rely on Euclidean distances such as kNN. 
# 
# (Btw: While linear and logistic regression are scale insensitive because they can absorb scale differences in the regression coefficients, LDA and QDA are insenstive to scaling because the compute Mahalanobis distance and not Euclidean distance. The Mahalanobis distance is the Euclidean distance after standardizing in SVD space.)
# 

# %% [markdown]
# # Features we considered
# Now we move on to computing the features. We used the code from the *Huge design matrices* exercise for this. We included the following features in our model:
# * n: a count of each word within a review
# * total number of words per review: see below. Mainly used to get a proportion, but could be a useful feature as well. For example, maybe if someone is really enthousiastic about the product they write a longer review.
# * tf: the relative frequency of a word within a review. This is useful because it shows that the more a word is used in a review, the more likely it is that the nature of the review is related to the word. For example, if 'hate' is used frequently, it is more likely that the nature of the review is related to that word, which can help to distinguish unsatisfactory reviews from satisfactory ones. 
# * idf: the number of reviews that contain a certain word. Included because the more a certain word is used in all the reviews, the less information it contains about the nature of the review (thus, whether it's satisfactory or unsatisfacory).
# * tf_idf: product of tf and idf.
# * Lexicons (NRC, AFINN): see below. We included this because we expect negative reviews to have more negative sentiment scores and positive reviews more positive sentiment scores.
# 

# %% [markdown]
# First, we prepare the data for additional feature computation. We do this by first splitting the amazon data set into a train and test set. Afterwards, we make 'rating' dichotomous by setting ratings higher than 1 to Satisfacory and Not satisfactory otherwise. Lastly, we compute the tf_idf for the test en training set, where we also filter surprise lower than −log(0.01/100).

# %% [code]
# Splitting amazon data set into train and test so we can compute additional features
amazon_train <- amazon %>%
  filter(!is.na(rating)) # training set

amazon_test <- amazon %>%
  filter(is.na(rating)) # test set

# Making rating dichotomous
amazon_train <- amazon_train %>% mutate(rating = if_else(rating > 3, "Satisfactory", "Not satisfactory"))

# Computing tf_idf for test set
tf_idf_test <- amazon_test %>%
  unnest_tokens(words, review, token = 'words', drop = FALSE) %>%
  count(words, id) %>% 
  bind_tf_idf(words, id, n) %>% 
  filter(idf <= -log(0.01/100)) # filtering surprise lower than -log(0.1/100)

# Computing tf_idf for training set
tf_idf  <- amazon_train %>%
  unnest_tokens(words, review, token = 'words', drop = FALSE) %>%
  count(words, id) %>% 
  bind_tf_idf(words, id, n) %>% 
  filter(idf <= -log(0.01/100)) %>%
  semi_join(tf_idf_test, by = "words") # those that don't appear in both test and training set are not useful, so we semi-join

# Semi-joining again
tf_idf_test <- tf_idf_test %>%
  semi_join(tf_idf, by = "words")

# We use this in our function later
info <- tf_idf[,1:3]
info_test <- tf_idf_test[,1:3]

# %% [markdown]
# # Adding Bigrams

# %% [markdown]
# To improve our predictions further, we added bigrams using partly the code from group 12. We filtered out stop words, otherwise the notebook took too long to run.

# %% [code]
stop_words <- get_stopwords()

# Computing bigrams for training set
amazon_bigrams <- amazon_train %>%
  unnest_tokens(bigram, review, token = "ngrams", n = 2) %>% 
  count(bigram, id) %>% 
  bind_tf_idf(bigram, id, n) %>% 
  filter(idf <= -log(0.1/100)) # filter out surprise lower than -log(0.1/100)

# Computing bigrams for test set
amazon_bigrams_test <- amazon_test %>%
  unnest_tokens(bigram, review, token = "ngrams", n = 2) %>% 
  count(bigram, id) %>% 
  bind_tf_idf(bigram, id, n) %>% 
  filter(idf <= -log(0.1/100))

# Bigrams that don't appear in the other set are not useful in this case, so we semi-join the sets.
amazon_bigrams_test <- amazon_bigrams_test %>% 
  semi_join(amazon_bigrams, by = "bigram") %>% 
  select(feature = bigram, id, value = tf_idf)  

amazon_bigrams <- amazon_bigrams %>% 
  semi_join(amazon_bigrams_test, by = c("bigram" = "feature")) %>% 
  select(feature = bigram, id, value = tf_idf)

# %% [markdown]
# # Adding lexicons
# We added lexicons, because we thought an unsatisfactory review might have more negative sentiment scores than a satisfactory review. We devide the lexicon counts by the total amount of words in the review, otherwise we would just get counts which could potentially cause multicolinearity issues. To add all the features we've computed together, we use the function from group 12. 

# %% [code]
# Loading NRC and AFINN lexicons
nrc_content <- list.files("../input/nrcsentiments", full.names = TRUE)
#afinn_content <- list.files("../input/afinnsentiments", full.names = TRUE)
# Calculating AFINN scores
#afinn <- read_delim(afinn_content[1], delim=",")
#colnames(afinn) <- c('words', 'sentiment')
#joined_afinn <- inner_join(info, afinn) %>%
#count(`id`, `sentiment`) %>%
#spread(sentiment, n, fill = 0)
#joined_afinn <- joined_afinn %>% select(2:10) 
#afinn_div <- joined_afinn/total_words$words_total 
#afinn_score <- cbind(afinn_div, joined_afinn) 
#afinn_score <- afinn_score %>% select(1:8) 

# Using function to add all the features together
features_function <- function(info, tf_idf, bigram){
  
  # Computing the total amount of words per review
  total_words <- info %>% 
    group_by(id) %>% 
    summarize(words_total = n())
  
  # Calculating NRC scores
  nrc <- read_delim(nrc_content[1], delim=",")
  colnames(nrc) <- c('words', 'sentiment')
  joined_nrc <- inner_join(info, nrc) %>%
    count(`id`, `sentiment`) %>%
    spread(sentiment, n, fill = 0)
  nrc_div <- joined_nrc[,2:11]/total_words$words_total # dividing by total amount of words
  nrc_score <- cbind(nrc_div, joined_nrc) 
  nrc_score <- nrc_score[,1:11] # selecting relevant columns
  
  # Computing features
  features <- inner_join(nrc_score, total_words, by = 'id')
  features <- features %>% 
    select(id, everything()) %>% 
    gather(2:ncol(features), key = "feature", value = "value")
  
  data_complete <- tf_idf %>% 
    select(feature = words, id, value = tf_idf) %>% 
    bind_rows(features, bigram)
  
  data_complete
}

# %% [markdown]
# ## Correlated features
# 
# Although correlated features may exist, with thousands of features it's computationally too cumbersome to try to remove them directly. Instead we'll have to rely on the properties of the Lasso and Ridge regression to deal with them (look it up in the ISLR book; it might come up in an exam question).
# 
# 

# %% [markdown]
# 
# # 5. Models
# 
# ## Not relying on manual feature selection
# 
# In the Personality competition we computed features by utilizing word lists that in previous research were found to be predictive of sentiment. This requires substantial input from experts on the subject. If such knowledge is not (yet) available a process of trial and error can be used. But with many thousands of features automation of this process is essential. 
# 
# 
# In addition forward and/or backward selection, automated methods that try to automatically ballance flexibility and predictive performance are
# 
# 1. Lasso and Ridge regression
# 2. Principal Components and Partial Least Squares regression
# 3. Smoothing 
# 4. Regression and Classification trees (CART)
# 5. Random Forests
# 6. Support Vector Machines
# 
# Methods (1) and (2) on this list involve methods are able to take many features while automatically reducing redundant flexibility to any desired level. Multicollinearity, the epithome of reduancy, is also automatically taken care of by these methods.
# 
# Number (3) on the list, smoothing, grants more flexibility by allowing for some non-linearity in the relations between features and the target variable, without the need to manually specify a specific mathematical form (as is necessary in polynomial regression).
# 
# Methods (4), (5), and (6) are not only able to remove redundant features, but also can automatically recognize interactions between  features.
# 
# Hence, all of these methods remove the necessity of finding the best features by hand. 
# 
# All of these methods are associated with a small set of 1 to 3 (or 4 in some cases) parameters that control the flexibility of the model in a more or less continuous way&mdash;much like the $k$ parameter in k-nearest neighbers. Like the $k$ parameter in k-NN, these parameters can and need to be adjusted (*'tuned'*) for optimal predictive performance. Tuning is best done on a validation set (a subset from the training data), or using cross-validation, depending on the size of the data set.

# %% [markdown]
# # 5.1 Model fitting
# 
# Not all algorithms can deal with sparse matrices. For instance `lm()` can't. The package `glmnet`, which is extensively discussed in chapter 6 of ISLR, has a function with the same name `glmnet()` which can handle sparse matrices, and also allow you to reduce the model's flexibility by means of the Lasso penalty or ridge regression penalty. Furthermore, like the standard `glm()` function, it can also handle a variety of dependent variable families, including gaussian (for linear regression), binomial (for logistic regression), multinomial (for multinomial logistic regression), Poisson (for contingency tables and counts), and a few others. It is also quite caple of dealing computationally efficiently with the many features we have here.
# 
# > <span style=color:brown>The aim of this competition is the predict the probability that a customer is ***satisfied***. This is deemed to be the case if `rating > 3`.  Hence, you will need as a dependent variable `y` a factor that specifies whether this is the case. </span>
# 
# The performance of your submission will be evaluated using the area under the curve (AUC) of the receiver operating curve (ROC). See chapter 4 in the ISLR book. See also the help file for how `cv.glmnet` can works with this measure.
# 
# As said, `glmnet()` allows you to tune the flexibility of the model by means of _regularizing_ the regression coefficients. The type of regularization (i.e., the Lasso or ridge) that is used is controled by the `alpha` parameter. Refer to the book for an explanation. The amount of regularization is specified by means of the `lambda` parameter. Read the warning in the `help(glmnet)` documentation about changing this parameter. To tune this parameter look at the `cv.glmnet()` function.
# 

# %% [markdown]
# Before we move on to fitting the model, we prepare the data by adding the features with our function, turning the features into a sparse design matrix and lastly getting the target values from the design matrix.

# %% [code]
# Adding features to test and train set
data_complete <- features_function(info, tf_idf, amazon_bigrams)
data_complete_test <- features_function(info_test, tf_idf_test, amazon_bigrams_test)

# Turn features into a sparse design matrix 
X <- cast_sparse(data_complete, id, feature, value)

y <- data.frame(id=rownames(X)) %>%
  inner_join(amazon_train) %>%
  pull(rating) %>% as.factor()

# Checking results
table(y)

# Making sparse design matrix for test set, used for predictions later
X_test = cast_sparse(data_complete_test, id, feature, value)

# %% [markdown]
# Now we can use lasso and ridge regression. In our lasso and ridge regression fits, we optimize around the AUC, since this is the measure that determines the kaggle score. We also tried some other measures, but AUC worked best. 

# %% [code]
# Fitting lasso
cv_lasso = cv.glmnet(X, y, family = "binomial", type.measure = "auc")

# Fitting ridge regression
cv_ridge <- cv.glmnet(X, y, family = "binomial", type.measure = "auc", alpha = 0)

# %% [markdown]
# # 5.2 Model evaluation
# 
# 

# %% [markdown]
# To evaluate our models we will first look into
# 
# 1. How they compare visually
# 2. Which models we ended up with
# 3. Which model we will prefer and chose for our final submission

# %% [markdown]
# ## Visual Comparison

# %% [code]
# Plotting Lasso
plot(cv_lasso)

# Plotting Ridge
plot(cv_ridge)

# %% [markdown]
# In these plots we see that for the Lasso model, the AUC decreases a lot when log(lambda) increases as the models with high lambda will be very resstraint and contain very few variables as the penalty parameter is high. With a very low penalty parameter (lambda) we also see a small decrease in AUC as they are becoming too complex and might be overfitting the model. For the Ridge model we see somewhat similar behaviour but at a way less extreme level. As lambda increases the AUC decreases. But in this case the AUC stays high even at very high lambda and only seems to extremely drop with a lambda higher than 4 which is way above -5 when we see the big drop in the Lasso model.

# %% [markdown]
# # What are our models?

# %% [markdown]
# ### How many variables do they contain?
# 
# To investigate the models we ended up with we will look into how many variables each of the models contain after using the shrinkage functions on them

# %% [code]
# How many coefficients did either model have before shrinking? 
length(coef(cv_lasso))
length(coef(cv_ridge))

# How many coefficients did either model have after shrinking? 
# (In the best-performing model)
# Lasso 
length(which(coef(cv_lasso)!=0)) 
# Ridge (as the ridge does not shrink to 0, we counted all variables that were more than 0.1 from 0 in either direction)
length(which(coef(cv_ridge)>0.1)) + length(which(coef(cv_ridge) <(-0.1)))

# %% [markdown]
# As we see both models started out with 4162 variables. But we see a big difference in the number of variables after shrinking. As Ridge regression doesn't technically shrink any variable to 0 the ridge regression still contains 4162 variables. We then filtered out the variables which are near 0 so less then 0.1 above or below 0 which gives us 3993 variables. In the lasso model we are left with a model that contains 2038 variables.

# %% [markdown]
# ### How do they perform?
# In this section we are looking at the performance of the model using the cvm.

# %% [code]
# We are extracting the cvm from our model objects and finding the value of the best performing one
print(max(cv_lasso$cvm))
print(max(cv_ridge$cvm))

# %% [markdown]
# Lasso seems to perform better, so this is our model of choice.

# %% [markdown]
# 
# # 6. Submitting your predictions
# 
# A sample file is provided to make sure that you predict the right cases and submit your predictions in the right format:

# %% [code]
# Making prediction on the test set
pred = predict(cv_lasso, X_test, s = "lambda.min", type = 'response')
head(pred)

# Writing csv file for submission
as_tibble(pred, rownames = "Id") %>% 
  rename(Prediction = '1') %>% 
  mutate(Id = as.numeric(Id)) %>% 
  arrange(Id) %>% 
  write_csv("predictions_team1.csv")

# Checking results
file.show("predictions_team1.csv")

#sample_filepath = dir("..", pattern="sample.csv", recursive=TRUE, full.names = TRUE)
#sample_submission = read_csv(sample_filepath, col_types = cols(col_character(), col_double()))
#head(sample_submission)