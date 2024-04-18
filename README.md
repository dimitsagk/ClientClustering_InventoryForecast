# Online Retail Store Client CLustering and Inventory Forecasting Project

## Company Overview
- **Company:** Online retail store
- **Location:** UK-based, with 90% of customers in the UK
- **Business Model:** Wholesale and retail

## Data
- **Source:** [Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) by Daqing Chen (2015).
- **Type:** Transactional data
- **Date Range:** From 01/12/2010 until 09/12/2011 (1 year)
- **Number of Instances:** 541,909
- **Columns:** Invoice No, Stock Code, Description, Quantity, Invoice Date, Unit Price, Customer ID, Country

## Project Objective
- **Objective:** Optimize inventory management to ensure timely availability and minimize stockouts.
- **Approach:** Employ machine learning models to forecast inventory needs for top-selling products, considering trends. Additionally, perform client clustering to enhance forecast accuracy.


## Challenges
- **Data Cleaning:** Duplicate removal, product filtering, cancellation handling
- **Seasonality:** Limited data span hinders clear seasonality identification
- **Outliers:** Presence of numerous outliers and noise
- **Client Behavior:** Large variability in ordering behavior among clients before clustering

## Solution
- **Model 1:** Utilized a supervised time series machine learning model with Prophet, incorporating feature engineering.
- **Model 1 Performance Metrics:** Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
- **Model 2:** Employed an unsupervised clustering model (Gaussian Mixture with 4 clusters), integrating feature engineering, data scaling, and dimensionality reduction.
- **Model 2 Performance Metrics:** Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index.


## Results
- **Comparative Analysis:** Compared forecasting models with and without client clustering
- **Error Rate:** Models with clustering showed a 70% lower error rate compared to those without
