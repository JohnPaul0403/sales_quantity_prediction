# Online Retail Store Forecasting

## Project Overview

This project aims to help small retail stores predict their future sales, enabling better inventory decisions and providing insights into business development. The dataset used for this project is the [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) dataset by Daqing Chen, which contains data from 2009-2011.

## Dataset Description

The dataset contains the following columns:

- **InvoiceNo**: Invoice number. A 6-digit integral number uniquely assigned to each transaction. If this code starts with the letter 'C', it indicates a cancellation.
- **StockCode**: Product (item) code. A 5-digit integral number uniquely assigned to each distinct product.
- **Description**: Product (item) name.
- **Quantity**: The quantities of each product (item) per transaction.
- **InvoiceDate**: Invoice date and time. The day and time when a transaction was generated.
- **UnitPrice**: Unit price. Product price per unit in sterling (£).
- **CustomerID**: Customer number. A 5-digit integral number uniquely assigned to each customer.
- **Country**: Country name. The name of the country where a customer resides.

## Project Steps

### 1. Data Exploration

- Load the dataset and inspect the first few rows.
- Check for missing values and anomalies.
- Identify patterns for anomalies and missing values.

### 2. Data Cleaning

- Remove rows with invalid `Invoice` and `StockCode` values.
- Drop rows with missing `CustomerID`.
- Remove rows with `Price` equal to 0.

### 3. Data Engineering

- Preprocess `InvoiceDate` to extract date features.
- Visualize daily revenue.
- Add new indicators for predicting (e.g., rolling mean, lag features, differencing).
- Find correlations with the indicators.

### 4. Feature Engineering

- Detect outliers using the Interquartile Range (IQR) method.
- Preprocess data to add date features, rolling features, lag features, and differencing features.
- Handle missing data using the `SimpleImputer` class.
- Standardize the data using `StandardScaler`.

### 5. Model Training

- Split the dataset into training and testing sets.
- Train an ARIMA model using the training data and exogenous variables.
- Evaluate the model using Mean Absolute Error (MAE) and Mean Squared Error (MSE).
- Save and load the trained model.

### 6. Model Fine-tuning

- Train additional ARIMA models using smoothed and scaled data.
- Evaluate the models to find the best-performing one.

### 7. Forecasting

- Forecast the revenue for the next 62 days using the best-performing model.
- Unscale the forecasted values and visualize the results.

## Conclusion

This project demonstrates how to preprocess and clean a retail dataset, engineer features, train and evaluate ARIMA models, and forecast future sales. The best-performing model was identified and used to forecast the revenue for the next 62 days, providing valuable insights for inventory and business planning.
