
# Packages ----------------------------------------------------------------
library(plyr) # <- general data work
library(dplyr) # <- general data work
library(tidyr) # <- general data work
library(ggplot2) # <- data visualisation
library(caret) # <- ML feature engineering and model fitting 
library(VIM) # <- k-nn missing value imputation
library(xgboost) # <- xgboost model
library(randomForest) # <- random forest model
# Read Data ---------------------------------------------------------------

# Read in our car data.
# The car data set can be found here: https://www.kaggle.com/orgesleka/used-cars-database
# It contains a variety of information on car adverts scraped from ebay.
data <- read.csv("C:\\Users\\Roland\\Documents\\git_repositories\\UsedCarsEDA\\autos.csv",
                 stringsAsFactors = FALSE)


# Examine Data ------------------------------------------------------------
dim(data) # The data has ~19k rows and 20 columns. Thats 19k observations with
# 19 features and our response value as the 20th column.
glimpse(data)
# The columns of the data are as follows:
# dateCrawled : when this ad was first crawled, all field-values are taken from this date
# name : "name" of the car
# seller : private or dealer
# offerType
# price : the price on the ad to sell the car
# abtest
# vehicleType
# yearOfRegistration : at which year the car was first registered
# gearbox
# powerPS : power of the car in PS
# model
# kilometer : how many kilometers the car has driven
# monthOfRegistration : at which month the car was first registered
# fuelType
# brand
# notRepairedDamage : if the car has a damage which is not repaired yet
# dateCreated : the date for which the ad at ebay was created
# nrOfPictures : number of pictures in the ad (unfortunately this field contains everywhere a 0 and is thus useless (bug in crawler!) )
# postalCode
# lastSeenOnline : when the crawler saw this ad last online



# Split Data into Train Test ----------------------------------------------
set.seed(1111)

# Determine proportion of data used for training
kTrainingPortion <- 0.7
# Get row indexes for training data
training_rows <- sample(1:nrow(data))[1:(kTrainingPortion*nrow(data))]
# Filter data for training and testing data
train <- data[training_rows,]
test <- data[-training_rows,]

# Categorical Variables ---------------------------------------------------
# We'll start by examining the categorical features in the dataset:
cat_var <- c("dateCrawled","name","seller","offerType","abtest","vehicleType",
             "gearbox","model","fuelType","brand","notRepairedDamage",
             "dateCreated")
# We'll examine the number of unique values in each feature. We don't want features
# with lots of unique values:
cat_var_levels <- data[,cat_var] %>%
  apply(MARGIN = 2, FUN = function(x){length(unique(x))}) %>%
  ldply()
names(cat_var_levels) <- c("feature", "n_unique_values")
cat_var_levels %>%
  arrange(desc(n_unique_values))
# We can see that the dataCrawled, name, model, dataCreated and brand features all have
# a high number of unique values.

# Date Columns ------------------------------------------------------------

date_cols <- c("dateCrawled","dateCreated","lastSeen")

head(data[,date_cols])
# These won't be much use to us, we'll simply remove them:

data[,date_cols] <- NULL

cat_var <- cat_var[!(cat_var %in% date_cols)]

# Name Column -------------------------------------------------------------

# The name column doesn't give us any information that isn't in the other more
# valuable features. We can remove this without worry
data$name <- NULL
cat_var <- cat_var[cat_var != "name"]

# Numeric Variables -------------------------------------------------------

# Now we'll examine the numerical features in the data set
num_var <- c("powerPS", "kilometer", "nrOfPictures","postalCode")

# We'll start by examining the sample variance for each column:
variance_num_var <- data[,num_var] %>%
  apply(MARGIN = 2, FUN = function(x){var(x)}) %>%
  ldply()
names(variance_num_var) <- c("feature", "variance")
variance_num_var %>%
  arrange(desc(variance))
# Immediately we see that the nrOfPictures column have a variance of 0.
# In other words, there is only a single value in this column and gives us no information
# about the response variable, we can remove this column.

# Number of Pictures ------------------------------------------------------

# We will remove the number of pictures column as it has 0 variance
data <- data %>% select(-nrOfPictures)

num_var <- num_var[num_var != "nrOfPictures"]


# Postal Code Column ------------------------------------------------------

# A difficult column to work with, lots of factors and difficult to work with

# Lets look at the frequencies
data.frame(table(data$postalCode)) %>%
  select(Freq) %>%
  summary()

# A mean value of 23, a sparse data column, we'll just remove it and try adding later
# if we want (we could attempt a bin counting method)
data$postalCode <- NULL
num_var <- num_var[num_var != "postalCode"]
# NA Values ---------------------------------------------------------------

# Replace blank values with NA
data[data == ""] <- NA

# Count NAs
na_values <- data %>%
  apply(MARGIN = 2, FUN = function(x){sum(is.na(x))}) %>%
  ldply()

names(na_values) <- c("feature","na_counts")

na_values <- na_values %>%
  arrange(desc(na_counts))

head(na_values)

# Impute NA Values --------------------------------------------------------

Mode <- function(x) {
  uniques <- unique(x[!is.na(x)]) # <- remove na values in case na is the most common level
  uniques[which.max(tabulate(match(x, uniques)))]
}

features_to_impute <- c("gearbox","model","fuelType","vehicleType", "notRepairedDamage")

# First job is to check whether any brands have ONLY NA values
data %>%
  group_by(brand) %>%
  summarize_at(features_to_impute,
               .funs = function(x){length(unique(x))}) %>%
  gather(key = "feature", value = "n_unique", -brand) %>%
  filter(n_unique == 1)

unique(data$model[data$brand == "sonstige_autos"])
# Replace this with other
data$model[data$brand == "sonstige_autos"] <- "other"

# Now impute missing values with the mode for each brand
data <- ddply(data,
              c("brand"),
              .fun = function(x){
                modes <- apply(x[,features_to_impute], 2, Mode)
                for(feature in features_to_impute){
                  x[is.na(x[,feature]),feature] <- modes[[feature]]
                }
                x
              })

# Ensure we removed all our NA values:
any(is.na(data)) # <- SHOULD READ FALSE

# Response Value ----------------------------------------------------------

# The response value (price)
summary(data$price)

# The maximum price is obviously wrong, let's boxplot to see the outliers:
ggplot(data, aes(x = as.factor(1), y = price)) + geom_boxplot()

# Lots of significant outliers that will cause us problems in the modelling stage
(priceQuantiles <- quantile(data$price, seq(from = 0.1, to = 1, by = 0.05)))
# Let's cap our max value
data$price[data$price > priceQuantiles[['90%']]] <- priceQuantiles[["90%"]]

ggplot(data, aes(x = as.factor(1), y = price)) + geom_boxplot()

# We can call our plot again and we see we now have a few remaining outliers
# but they seem more reasonable. We can see if they're luxury cars:
ggplot(data, aes(x = as.factor(1), y = price)) + geom_boxplot()





# Dealing with the Brand Column -------------------------------------------

# How many brands do we have?
length(unique(data$brand))

# Let's tier the brands (using a bit of data and a bit of intuition)
brands <- data %>%
  group_by(brand) %>%
  summarize(brand_mean_price = mean(price)) %>%
  ungroup()

data <- data %>%
  merge(brands,
        by = "brand")

num_var <- c(num_var, "brand_mean_price")


# Tier cars to reduce levels in brand feature:

# Car Tiers
ggplot(brands,aes(x = reorder(brand, brand_mean_price), 
                  y = brand_mean_price, 
                  fill = brand_mean_price)) +
  geom_bar(stat = "identity") +
  theme(legend.position = "none") +
  geom_hline(yintercept = quantile(data$price), col = "red") +
  coord_flip()

head(brands)
# Initialize a brand_tier column
brands$brand_tier <- "mid"

brands$brand_tier[brands$brand_mean_price > quantile(data$price)[["75%"]]] <- "high"
brands$brand_tier[brands$brand_mean_price < quantile(data$price)[["25%"]]] <- "low"
# Merge with data to add brand tier list to data
data <- data %>%
  merge(brands %>% select(-brand_mean_price),
        by = "brand") %>%
  dplyr::select(-brand)

cat_var <- c(cat_var, "brand_tier")
cat_var <- cat_var[cat_var != "brand"]

# Different Type of Sellers -----------------------------------------------

unique(data$seller)

# Translate from German to English
data$seller[data$seller == "gewerblich"] <- "commercial"
data$seller[data$seller == "privat"] <- "private"

ggplot(data, aes(x = seller, y = price)) + 
  geom_boxplot()



# Year of Registration ----------------------------------------------------

# Initial analysis
unique(data$yearOfRegistration)
quantile(data$yearOfRegistration, seq(from = 0.1, to = 1, by = 0.1))

# Impute using kilometers value

ggplot(data %>% filter(yearOfRegistration > 1975,
                       yearOfRegistration < 2020),
       aes(x = as.factor(yearOfRegistration),
           y = kilometer)) +
  geom_boxplot()

# We'll impute using a simple cap:
kMinYear <- 1950
kMaxYear <- 2019

data$yearOfRegistration[data$yearOfRegistration < kMinYear] <- kMinYear
data$yearOfRegistration[data$yearOfRegistration > kMaxYear] <- kMaxYear

# Plot our data
ggplot(data ,
       aes(x = as.factor(yearOfRegistration),
           y = price)) +
  geom_boxplot()

# Types of Car ------------------------------------------------------------

unique(data$vehicleType)

# Rename our vehicles:
vehicle_type_translation <- data.frame("vehicleType" = c("andere","bus","cabrio","coupe",
                                                         "kleinwagen","kombi","limousine","suv"),
                                       "vehicleTypeEng" = c("other", "bus", "convertable",
                                                            "coupe","small car","estate",
                                                            "limo","suv"))

data <- data %>%
  merge(vehicle_type_translation,
        by = "vehicleType") %>%
  dplyr::select(-vehicleType) %>%
  rename("vehicleType" = vehicleTypeEng)

data %>%
  group_by(vehicleType) %>%
  summarise(mean_price = mean(price),
            count = n()) %>%
  ungroup() %>%
  ggplot(aes(x = reorder(vehicleType, mean_price), y = mean_price, fill = count)) +
  geom_bar(stat = 'identity') +
  coord_flip()

ggplot(data, aes(x = vehicleType, y = price)) + geom_boxplot()



# Model Prep --------------------------------------------------------------
processed_params <- preProcess(data[,num_var],
                               method = c("range"))

processed_data <- predict(processed_params,
                          data)

encoded_data <- processed_data %>%
  mutate_at(cat_var,
            .funs = as.factor)

encoded_data$model <- NULL
encoded_data$brand_mean_price <- NULL

dmy <- dummyVars(" ~ .", data = encoded_data, fullRank =  TRUE)

encoded_data <- data.frame(predict(dmy, newdata = encoded_data))


# Random Forest -----------------------------------------------------------
training_rows <- sample(1:nrow(data), nrow(data)*0.05)

training_data <- encoded_data[training_rows,]

random_forest_model <- randomForest(price ~ ., 
                                    data = training_data)

test_rows <- 1:nrow(encoded_data)
test_rows <- test_rows[!(test_rows %in% training_rows)]

test_data <- encoded_data[test_rows,]

predictions <- predict(random_forest_model, test_data)

results <- data.frame("observed" = test_data$price,
                      "predicted" = predictions)


results <- results %>%
  mutate(residuals = observed - predicted,
         residuals_squared = residuals^2)

ss_tot <- sum((results$observed - mean(results$observed))^2)
ss_res <- sum((results$observed - results$predicted)^2)

(R_squared  <- 1 - (ss_res/ss_tot))

# XGBOOST -----------------------------------------------------------------

names(processed_data)


# Fit xgboost model
x <- training_data %>% select(-price)
y <- training_data %>% select(price)

train.x.matrix <- xgb.DMatrix(data = as.matrix(x),
                              label = as.matrix(y))

xgbmodel <- xgboost(data = train.x.matrix, # the data
                    booster = "gbtree",
                    nrounds = 20# max number of boosting iterations
)  # the objective function

xgbmodel

predictions <- predict(xgbmodel, as.matrix(test_data %>% dplyr::select(-price)))

results <- data.frame("observed" = test_data$price,
                      "predicted" = predictions)

results <- results %>%
  mutate(residuals = observed - predicted,
         residuals_squared = residuals^2)

ss_tot <- sum((results$observed - mean(results$observed))^2)
ss_res <- sum((results$observed - results$predicted)^2)

(R_squared  <- 1 - (ss_res/ss_tot))







