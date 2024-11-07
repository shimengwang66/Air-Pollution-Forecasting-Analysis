###########################################################################################
# Group Project: Air Pollution Forecasting

# Group Members:
# FENG, PEILAN 006309856
# MEDHI, MRIGANGKA 606309858
# WANG, SHIMENG (SUMMER) 504882854
# ZHAO, PEILIN 706318584

###########################################################################################

setwd("/Users/mrigangkamedhi/Desktop/Forecasting & Time Series/Week 5/Project/")

# Install required packages if not already installed
packages <- c("tidyverse", "forecast", "vars", "prophet", "lubridate", 
              "caret", "tsibble", "gridExtra", "randomForest", "xgboost", 
              "e1071", "zoo")

install_if_missing <- function(packages) {
    new_packages <- packages[!(packages %in% installed.packages()[,"Package"])]
    if(length(new_packages)) install.packages(new_packages)
}

install_if_missing(packages)

# Load libraries
library(tidyverse)
library(forecast)
library(vars)
library(prophet)
library(lubridate)
library(caret)
library(tsibble)
library(gridExtra)
library(randomForest)
library(xgboost)
library(e1071)
library(zoo)

# Set up environment for reproducibility
set.seed(123)

# 1. Data Loading and Preprocessing
# Load the dataset
data <- read.csv("LSTM-Multivariate_pollution.csv")
data <- data %>% drop_na()  # Remove or handle missing values

# Convert date-time columns to datetime format
data$datetime <- as.POSIXct(data$date, format="%Y-%m-%d %H:%M:%S")

# Select relevant columns
data <- data %>% dplyr::select(datetime, pollution, dew, temp, press, wnd_dir, wnd_spd, snow, rain)

# 2. Exploratory Data Analysis (EDA)
# Summary statistics
summary(data)

# Plot pollution concentration over time
pollution_plot <- ggplot(data, aes(x = datetime, y = pollution)) +
    geom_line(color = 'blue') + labs(title = "PM2.5 Pollution Concentration Over Time") +
    theme_minimal()

# Seasonal plots: Monthly and Hourly Averages
data_monthly <- data %>% mutate(month = month(datetime)) %>% group_by(month) %>% summarize(avg_pollution = mean(pollution, na.rm = TRUE))
data_hourly <- data %>% mutate(hour = hour(datetime)) %>% group_by(hour) %>% summarize(avg_pollution = mean(pollution, na.rm = TRUE))

monthly_plot <- ggplot(data_monthly, aes(x = month, y = avg_pollution)) +
    geom_line(color = 'green') + labs(title = "Monthly Average PM2.5 Pollution Concentration")

hourly_plot <- ggplot(data_hourly, aes(x = hour, y = avg_pollution)) +
    geom_line(color = 'purple') + labs(title = "Hourly Average PM2.5 Pollution Concentration")

# Display plots
grid.arrange(pollution_plot, monthly_plot, hourly_plot, ncol = 1)

# 3. Train-Test Split
# Splitting data into training and testing sets (e.g., last 6 months for testing)
train_size <- floor(0.8 * nrow(data))
train <- data[1:train_size, ]
test <- data[(train_size + 1):nrow(data), ]

# 4. Feature Engineering
# Lagged values, rolling means
train <- train %>%
    mutate(lag_pollution = lag(pollution, 1),
           rollmean_pollution = zoo::rollmean(pollution, 24, fill = NA))

# 5. Time Series Forecasting Models
# 5.1 ARIMA Model
pollution_ts <- ts(train$pollution, frequency = 24)
fit_arima <- auto.arima(pollution_ts)
forecast_arima <- forecast(fit_arima, h = nrow(test))

# 5.2 VAR Model (Vector Autoregression)
var_data <- train %>% dplyr::select(pollution, dew, temp, press)
var_model <- VAR(var_data, p = 2, type = "const")
forecast_var <- predict(var_model, n.ahead = nrow(test))

# 5.3 Prophet Model
df_prophet <- train %>% rename(ds = datetime, y = pollution)
df_prophet$ds <- as.POSIXct(df_prophet$ds, format = "%Y-%m-%d %H:%M:%S")
df_prophet <- df_prophet %>% drop_na(ds, y)

m <- prophet(df_prophet)
future <- make_future_dataframe(m, periods = nrow(test), freq = "hour")
forecast_prophet <- predict(m, future)

# 5.4 Random Forest Model
# Train random forest model on non-time-dependent features
rf_model <- randomForest(pollution ~ dew + temp + press + wnd_spd + snow + rain, data = train)
rf_forecast <- predict(rf_model, test)

# 5.5 XGBoost Model
# Prepare data for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train %>% dplyr::select(dew, temp, press, wnd_spd, snow, rain)), label = train$pollution)
dtest <- xgb.DMatrix(data = as.matrix(test %>% dplyr::select(dew, temp, press, wnd_spd, snow, rain)))

xgb_model <- xgboost(data = dtrain, max.depth = 3, eta = 0.1, nrounds = 100, objective = "reg:squarederror")
xgb_forecast <- predict(xgb_model, dtest)

# 5.6 Support Vector Regression (SVR)
# Prepare data for SVR: only numeric features
numeric_train <- train %>% dplyr::select(pollution, dew, temp, press, wnd_spd, snow, rain)
numeric_test <- test %>% dplyr::select(dew, temp, press, wnd_spd, snow, rain)

# Train SVR model
svr_model <- svm(pollution ~ ., data = numeric_train, kernel = "radial")

# Predict on test data
svr_forecast <- predict(svr_model, newdata = numeric_test)

# 6. Model Evaluation
# Actual test values
y_test <- test$pollution

# Helper function for error metrics
calc_metrics <- function(pred, actual) {
    rmse <- sqrt(mean((pred - actual)^2))
    mae <- mean(abs(pred - actual))
    return(list(RMSE = rmse, MAE = mae))
}

# Calculate RMSE and MAE for each model
arima_metrics <- calc_metrics(forecast_arima$mean, y_test)
var_forecast <- sapply(forecast_var$fcst$pollution[, "fcst"], as.numeric)
var_metrics <- calc_metrics(var_forecast, y_test)
prophet_metrics <- calc_metrics(forecast_prophet$yhat[(nrow(forecast_prophet)-nrow(test)+1):nrow(forecast_prophet)], y_test)
rf_metrics <- calc_metrics(rf_forecast, y_test)
xgb_metrics <- calc_metrics(xgb_forecast, y_test)
svr_metrics <- calc_metrics(svr_forecast, y_test)

# 7. Generate Report Table
metrics_df <- data.frame(
    Model = c("ARIMA", "VAR", "Prophet", "Random Forest", "XGBoost", "SVR"),
    RMSE = c(arima_metrics$RMSE, var_metrics$RMSE, prophet_metrics$RMSE, rf_metrics$RMSE, xgb_metrics$RMSE, svr_metrics$RMSE),
    MAE = c(arima_metrics$MAE, var_metrics$MAE, prophet_metrics$MAE, rf_metrics$MAE, xgb_metrics$MAE, svr_metrics$MAE)
)

print(metrics_df)

# 8. Visualization of Forecasts
# Combine actual and forecast data for visualization
results <- data.frame(
    DateTime = test$datetime,
    Actual = y_test,
    ARIMA = forecast_arima$mean,
    VAR = var_forecast,
    Prophet = forecast_prophet$yhat[(nrow(forecast_prophet)-nrow(test)+1):nrow(forecast_prophet)],
    RandomForest = rf_forecast,
    XGBoost = xgb_forecast,
    SVR = svr_forecast
)

# Plot forecasts vs actuals
results_long <- gather(results, "Model", "Prediction", -DateTime, -Actual)
ggplot(results_long, aes(x = DateTime, y = Prediction, color = Model)) +
    geom_line() +
    geom_line(aes(y = Actual), color = "black", linetype = "dashed") +
    labs(title = "Comparison of Forecasts vs Actual PM2.5 Pollution Concentration") +
    theme_minimal()


# Prepare data for plotting XGBoost predictions vs actuals
xgb_results <- data.frame(
    DateTime = test$datetime,
    Actual = y_test,
    XGBoost = xgb_forecast
)

# Plot XGBoost predictions vs actuals with legend
ggplot(xgb_results, aes(x = DateTime)) +
    geom_line(aes(y = Actual, color = "Actual"), linetype = "dashed") +
    geom_line(aes(y = XGBoost, color = "XGBoost Prediction")) +
    labs(title = "XGBoost Forecast vs Actual PM2.5 Pollution Concentration",
         y = "PM2.5 Concentration", x = "DateTime") +
    scale_color_manual(values = c("Actual" = "black", "XGBoost Prediction" = "blue")) +
    theme_minimal() +
    theme(legend.title = element_blank())
