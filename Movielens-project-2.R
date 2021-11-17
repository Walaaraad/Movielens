##Code provided by edx staff to download the dataset
#Load libraries
#Create edx set, validation set
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")


#Load the data
# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


#The dataset edx
str(edx)


#Number of unique movies and users in the edx dataset
edx %>% 
  summarize(
    n_movies= n_distinct(edx$movieId),
    n_users=n_distinct(edx$userId),
    n_genres=n_distinct(edx$genres))

#The dataset modified formats
edx$userId <- as.factor(edx$userId)#converts 'userId' to factor
edx$movieId <- as.factor(edx$movieId)#converts 'movieId' to factor
edx$genres <- as.factor(edx$genres)#converts 'genres' to factor
edx$timestamp <- as.POSIXct(edx$timestamp, origin = "1970-01-01")#converts 'timestamp' to POSIXct
edx <- edx %>% 
  mutate(year = as.numeric(str_sub(title,-5,-2)))#extracts the release year of the movie
edx <- edx %>%
  mutate(year_rate=year(timestamp))#extracts the year that the rate was given by the user

#Rating Distribution
edx%>%
  ggplot(aes(rating))+
  geom_histogram(binwidth=0.25 , fill="magenta")+
  labs(title = "Rating Distribution",
       x="Rating",
       y="Frequency")

#Year_rate Distribution
edx%>%
  ggplot(aes(year_rate))+
  geom_histogram(binwidth=0.35 , fill="blue")+
  labs(title = " Year_rate Distribution",
       subtitle = "The year when the rate was given by the user", 
       x="Year Rate",
       y= "Frequency")

#Release_year Distribution
edx%>%
  ggplot(aes(year))+
  geom_histogram(binwidth=0.3 , fill="green")+
  labs(title = " Release Year Distribution",
       subtitle = "The year when the movie was created", 
       x="Release Year",
       y= "Frequency")

#List of the highest five genres rating genres 
edx %>% separate_rows(genres, sep = "\\|") %>%
  group_by(genres) %>%
  summarize(count = n()) %>%
  arrange(desc(count))%>%
  head(5)

#List of the first ten movies of highest movie rating
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange(desc(count))%>%
  head(10)

### Data Modeling
#Create 'train' and 'test' set
edx <- edx %>% select(userId, movieId, rating)
test_index <- createDataPartition(edx$rating, times = 1, p = .2, list = F)
train <- edx[-test_index, ] # Create Train set
test <- edx[test_index, ] # Create Test set
test <- test %>% # The same movieId and usersId appears in both set. 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")


##Baseline Model_The mean of the movie rating effect model
mu <- mean(train$rating)
mu

#check the test result
navie_rmse<-RMSE(test$rating, mu)
navie_rmse

#Save results in table
rmse_results<- tibble(method="Mean of rating model",RMSE = navie_rmse)
rmse_results %>% knitr::kable(caption = "RMSE")

## The movie effect model
movie_mean<- train %>% 
  group_by(movieId) %>%
  summarize(b_i= mean(rating-mu))

#check the test result
movie_prediction <- test %>%
  right_join(movie_mean, by = "movieId")%>%
  mutate(prediction = mu + b_i)

movie_effect_rmse <- RMSE(test$rating, movie_prediction$prediction)
movie_effect_rmse

#Save results in table
rmse_results <- rmse_results %>%
  add_row(method="Movie effect model", RMSE = movie_effect_rmse)
rmse_results %>% knitr::kable()

##The User and Movie effect model
user_mean <- train %>%
  left_join(movie_mean, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating-mu-b_i))

#check the test result
user_prediction <- test %>%
  left_join(movie_mean, by = "movieId") %>%
  left_join(user_mean, by = "userId") %>%
  mutate(prediction = mu+b_i+b_u)

user_effect_rmse <- RMSE(test$rating, user_prediction$prediction)
user_effect_rmse  

#Save results in table
rmse_results <- rmse_results %>%
  add_row(method = "User & Movie effect model", RMSE = user_effect_rmse)
rmse_results %>% knitr::kable()

##Regularized Movie and User effect model
#check the test result
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(lambda){
  mu <- mean(train$rating)
  b_i <- train %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating-mu)/(n()+lambda))
  b_u <- train %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating-b_i-mu)/(n()+lambda))
  predictions <- test %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu+b_i+b_u) %>%
    .$pred
  RMSE(test$rating, predictions)
})

#Plot the lambdas vs RMSEs
qplot(lambdas, rmses, 
      main = "Regularized Movie & User Model",
      xlab = "lambda", ylab = "RMSE")

#According to the plot, the best lambda is:
lambda <- lambdas[which.min(rmses)]
lambda

min(rmses)

#Save results in table
rmse_results <- rmse_results %>%
  add_row(method = "Regularized Movie & User effect model", RMSE = min(rmses))
rmse_results %>% knitr::kable()

