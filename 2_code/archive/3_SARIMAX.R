# Load required libraries
library(tidyverse)
library(lubridate)
library(forecast)  # For time series analysis
library(tseries)   # For KPSS test
library(gridExtra) # For arranging multiple ggplots

# Load the data
df <- read_csv("../0_data/preprocessed/df_final.csv") |> 
  filter(datetime >= ymd_hms("2021-09-01 00:00:00"))

# Preprocessing: Missing values
df <- df |>
  fill(everything(), .direction = "down")

# Feature Selection: Selected variables
df <- df |>
  select(
    datetime, kWh, contains("shortwave_radiation"), 
    contains("precipitation"), contains("soil_temperature_7")
  )

# Visualization of ACF and PACF
lags = 96
acf_result <- acf(df$kWh, plot = T, lag.max = lags)
acf_df <- data.frame(
  ACF = acf_result$acf,
  Lag = 0:lags
)
significance_thr <- 1.96 / sqrt(length(data))

# Set up full dataset for model fitting
df_full <- df |>
  arrange(datetime) |>
  complete(datetime = seq(min(datetime), max(datetime), by = "hour")) |>
  fill(everything(), .direction = "down")

# Train and Validation Splits
df_train <- df_full |> 
  filter(datetime <= ymd("2022-08-31"))

df_val <- df_full |> 
  filter(datetime > ymd("2022-08-31")) |> 
  filter(datetime <= ymd("2023-08-31"))

# Fit SARIMAX model
start_time = Sys.time()
forecaster <- Arima(
  df_train |> pull(kWh), 
  xreg = df_train |> select(-c(datetime, kWh)) |> as.matrix() |> scale(),
  order = c(5, 1, 5), 
  seasonal = list(order=c(1, 0, 1), period=24),
  optim.method = "BFGS"
)
end_time = Sys.time()
end_time - start_time

# Predict the first day after the training period
y_pred <- forecast(
  forecaster, 
  xreg = df_val |> select(-c(datetime, kWh)) |> as.matrix(), 
  h = 24
)

y_pred |> 
  as_tibble() |> 
  transmute(.pred = `Point Forecast`) |> 
  mutate(
    datetime = df_val$datetime,
    .actual = df_val$kWh
  ) |> 
  pivot_longer(-datetime) |> 
  ggplot(aes(datetime, value, colour = name)) +
  geom_line() +
  labs(y = "kWh", title = "SARIMAX Predictions vs Actual Values")

# Additional predictions
# Predicting the second day
y_pred_2 <- forecast(forecaster, xreg = df_val[25:48, -which(colnames(df_val) == "kWh")], 
                     h = 24, 
                     lambda = TRUE)

# Add your logic for further predictions as needed