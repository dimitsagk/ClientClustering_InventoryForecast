import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.metrics import pairwise_distances, davies_bouldin_score

from prophet import Prophet

from statsmodels.tools.eval_measures import rmse
from sklearn.metrics import mean_absolute_error



def cleaning(df):

    # Missing values
    # Droppping rows with UnitPrice 0
    df = df[df.UnitPrice != 0.0]

    # Drop duplicates
    df.drop_duplicates(inplace=True)
   
    # Formatting, removing .0 from CustomerID -> column will become Object after and NaN will be preserved
    df['CustomerID'] = df['CustomerID'].astype('Int64')
    
    # Changing country EIRE to Ireland
    df.Country = df.Country.replace('EIRE','Ireland')
   
    # Dropping invoices that are cancellations -> the Invoice codes that start with 'C'
    df = df[~df.InvoiceNo.str.startswith('C', na=False)]
    
    # Adding anything that is not Sock to StockCode 'OTHER'
    df.loc[df.StockCode.isin(['M', 'D', 'S', 'B', 'm','C2','POST','DOT','AMAZONFEE','CRUK']),'StockCode'] = 'OTHER'


    # Resetting index
    df.reset_index(drop=True, inplace=True)
    
    return df



def product_info_adjust(df):
    '''Adjusting StockCode, Description, UntiPrice, where necessary'''
    # Correcting a few product codes that are clearly referring to different products
    df.loc[df.Description == 'PICNIC BASKET WICKER 60 PIECES','StockCode'] = str(df[df.Description=='PICNIC BASKET WICKER 60 PIECES'].StockCode.mode()[0]) + 'A'

    # Firstly, for the Description with multiple prices, I want only one price, and based on my analysis, I will choose the 
    # mode price value
    descr_price = pd.DataFrame(df.groupby('Description')['UnitPrice'].agg(lambda x: x.value_counts().index[0])).reset_index()
    df.UnitPrice = df.Description.map(dict(zip(descr_price.Description, descr_price.UnitPrice)))

    # Secondly, I am having only one Description per StockCode, again based on the analysis on notebook 01_InitialExploration
    stock_descr = pd.DataFrame(df.groupby('StockCode')['Description'].agg(lambda x: x.value_counts().index[0])).reset_index()
    df.Description = df.StockCode.map(dict(zip(stock_descr.StockCode, stock_descr.Description)))

    return df



def feat_eng(df):
    ''' This function, does feature engineering and returns the constructed dataframe.
    Thsi is for the ML model for Client clustering, used mostly at the notebook 03_ML_ClientClustering-notebook.'''
    #Total quantity purchased
    cust = df.groupby('CustomerID')['Quantity'].sum().reset_index().rename(columns={'Quantity': 'TotalQuantity'})

    # Total value of orders
    cust = pd.merge(cust, df.groupby('CustomerID')['TotalPrice'].sum().reset_index().rename(columns={'TotalPrice': 'TotalValue'}), 
                                                                                       how='left', on='CustomerID')

    # Average quantity per item per order
    # Grouping by StockCode, in case in the same invoice client has added the product more than once
    item_quant = df.groupby(['CustomerID','InvoiceNo','StockCode'])['Quantity'].sum().reset_index()

    cust = pd.merge(cust, item_quant.groupby('CustomerID')['Quantity'].mean().round(2).reset_index().rename(columns={'Quantity':'AvrgQuantity'}), 
                                                                                       how='left', on='CustomerID')

    # Average value per order
    order_val = df.groupby(['CustomerID','InvoiceNo'])['TotalPrice'].sum().reset_index()
    avrg_val = order_val.groupby('CustomerID')['TotalPrice'].mean().round(2).reset_index().rename(columns={'TotalPrice':'AvrgOrderValue'})

    cust = pd.merge(cust, avrg_val, how='left', on='CustomerID')

    # Total orders number
    cust = pd.merge(cust, df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index().rename(columns={'InvoiceNo': 'TotalOrders'}), 
                                                                                       how='left', on='CustomerID')

    # Setting CustomerID as index
    cust.set_index('CustomerID', inplace=True)

    return cust    


def outlier_scaling(df):
    '''Function that performs outlier scaling.
    There are two outliers that have been spoted throught the EDA, that have huge difference in scale, and seem to be two stand-alone
    cases (they are not repeating customers).
    Through the analysis I have decided to scale the Quantity of them to the next highest value after these two."
    Receives the daframe as parameter and returns the updated dataframe with outlier clipping, 
    and updated the Total Price column accordingly.'''

    condition = df.InvoiceNo.isin(['581483','541431'])

    df.loc[condition,'Quantity'] = df.Quantity.sort_values(ascending=False).values[2]
    df.loc[condition,'TotalPrice'] = df.Quantity * df.UnitPrice  

    return df


def outliers_clipping_STD(df, col_name):
    ''' Function that does outlier clipping based on the Standard Deviation Method.
    It receives as parameters the dataframe and the column where oultier clipping will be performed.
    Returns updated dataframe and prints how many instances had outlier clipping performed to them.
    '''

    # Checking if column if integer or float type, to format accordingly
    # checking only upper limit. I don't have lower limit, cause everything start from 0
    if df[col_name].dtype == 'int64':        
        tq_UpperLimit = (df[col_name].mean() + df[col_name].std()*3).round(0).astype(int)
    else:
        tq_UpperLimit = (df[col_name].mean() + df[col_name].std()*3).round(2)         

    
    print("Instances that needed outlier clipping: ", df[df[col_name] > tq_UpperLimit].shape[0],
              ", out of total instances: ",df[col_name].shape[0])
    
    df.loc[df[col_name] > tq_UpperLimit , col_name] = tq_UpperLimit  

    return df



def scaling_data(df):
    ''' Scaling data using the Standard Scaler. Returns the scaled dataframe.'''
    scaler = StandardScaler()

    scaler.fit(df)

    # applying the transformation
    df_stndrd = scaler.transform(df)

    df_stndrd = pd.DataFrame(df_stndrd, columns = df.columns, index=df.index)

    return df_stndrd


def k_means_model(df, n_clusters):
    ''' Function for K-Means clustering. Number of clusters to be provided.
    Returns updated dataframe with a column for the labels from the clustering and performance metrics:
    Silhouette coefficient, Calinski-Harabasz Index, Davies-Bouldin Index.
    Additionally it prints (but does not return) the percentage of instances in each cluster to the total instances.
    Returns in total 4 elements.
    '''
    df_m = df.copy()
    kmeans = KMeans(n_clusters)
    kmeans.fit(df_m)
    df_m['kmeans_labels'] = kmeans.labels_

    # Performance metrics
    sc = metrics.silhouette_score(df_m, df_m['kmeans_labels'], metric='euclidean')
    ch = metrics.calinski_harabasz_score(df_m, df_m['kmeans_labels'])
    db = davies_bouldin_score(df_m, df_m['kmeans_labels'])

    # Printing results
    print(f"Kmeans silhouette coefficient: {sc: .4f}")
    print(f"Kmeans Calinski-Harabasz Index: {ch: .4f}")
    print(f"Kmeans Davies-Bouldin Index: {db: .4f}")
    
    # Count of instances per cluster
    for i in range(n_clusters):
        print(f"Label: {df_m['kmeans_labels'].unique()[i]}, Percentage total customers: {round(df_m[df_m['kmeans_labels'] == i].shape[0] *100/df_m.shape[0],2)}%")
        
    return df_m


def dbscan_model(df, n_eps, n_min_samples): 
    ''' Function for DBSCAN clustering. Dataframe, eps and min_samples values to be provided.
    Returns updated dataframe with a column for the labels from the clustering and performance metrics:
    Silhouette coefficient, Calinski-Harabasz Index, Davies-Bouldin Index.
    Additionally it prints (but does not return) the percentage of instances in each cluster to the total instances.
    Returns in total 4 elements.
    '''
    df_m = df.copy()
    
    dbscan = DBSCAN(eps=n_eps, min_samples=n_min_samples)
    df_m['dbscan_labels'] = dbscan.fit_predict(df_m)

    # Performance metrics
    sc = metrics.silhouette_score(df_m, df_m['dbscan_labels'], metric='euclidean')
    ch = metrics.calinski_harabasz_score(df_m, df_m['dbscan_labels'])
    db = davies_bouldin_score(df_m, df_m['dbscan_labels'])

    # Printing results
    print(f"DBSCAN silhouette coefficient: {sc: .4f}")
    print(f"DBSCAN Calinski-Harabasz Index: {ch: .4f}")
    print(f"DBSCAN Davies-Bouldin Index: {db: .4f}")
    
    # Count of instances per cluster
    for label in df_m['dbscan_labels'].unique():
        print(f"Label: {label}, Percentage total customers: {round(df_m[df_m['dbscan_labels'] == label].shape[0] *100/df_m.shape[0],2)}%")
        
    return df_m


def scatter_plot(df, labels, col1='TotalQuantity', col2='AvrgOrderValue'):
    ''' Scatter plot function, specifically to visualize outliers/ noise from dbscan/ hdbscan models.'''
    # Define colors for each cluster
    cluster_colors = sns.color_palette('tab10')

    plt.figure(figsize=(4, 3))
    
    # Visualizing the clusters
    for label in labels.unique():
        if label == -1:
            # Plot points labeled as noise/outliers in black
            plt.scatter(df.loc[labels == label, col1], df.loc[labels == label, col2], s=10, color='black', label='Noise/Outliers')
        else:
            # Plot points for each cluster with different color
            plt.scatter(df.loc[labels == label, col1], df.loc[labels == label, col2], s=10, color=cluster_colors[label], label=f'Cluster {label}')

    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend()
    plt.show()

    return


def hdbscan_model(df, n_min_cluster_size): 
    ''' Function for HDBSCAN clustering. Dataframe, and min_cluster size to be provided.
    Returns updated dataframe with a column for the labels from the clustering and performance metrics:
    Silhouette coefficient, Calinski-Harabasz Index, Davies-Bouldin Index.
    Additionally it prints (but does not return) the percentage of instances in each cluster to the total instances.
    Returns in total 4 elements.
    '''
    df_m = df.copy()
    
    hdb = HDBSCAN(min_cluster_size=n_min_cluster_size).fit(df_m)
    df_m['hdbscan_labels'] = hdb.labels_

    # Performance metrics
    sc = metrics.silhouette_score(df_m, df_m['hdbscan_labels'], metric='euclidean')
    ch = metrics.calinski_harabasz_score(df_m, df_m['hdbscan_labels'])
    db = davies_bouldin_score(df_m, df_m['hdbscan_labels'])

    # Printing results
    print(f"HDBSCAN silhouette coefficient: {sc: .4f}")
    print(f"HDBSCAN Calinski-Harabasz Index: {ch: .4f}")
    print(f"HDBSCAN Davies-Bouldin Index: {db: .4f}")
    
    # Count of instances per cluster
    for label in df_m['hdbscan_labels'].unique():
        print(f"Label: {label}, Percentage total customers: {round(df_m[df_m['hdbscan_labels'] == label].shape[0] *100/df_m.shape[0],2)}%")
        
    return df_m


def gm_model(df, n_clusters):
    ''' Function for Gaussian Mixture clustering. Number of clusters to be provided.
    Returns updated dataframe with a column for the labels from the clustering and performance metrics:
    Silhouette coefficient, Calinski-Harabasz Index, Davies-Bouldin Index.
    Additionally it prints (but does not return) the percentage of instances in each cluster to the total instances.
    Returns in total 4 elements.
    '''
    df_m = df.copy()
    
    gm = GaussianMixture(n_components=n_clusters).fit(df_m)
    df_m['gm_labels'] = gm.predict(df_m)

    # Performance metrics
    sc = metrics.silhouette_score(df_m, df_m['gm_labels'], metric='euclidean')
    ch = metrics.calinski_harabasz_score(df_m, df_m['gm_labels'])
    db = davies_bouldin_score(df_m, df_m['gm_labels'])

    # Printing results
    print(f"GaussianMixture silhouette coefficient: {sc: .4f}")
    print(f"GaussianMixture Calinski-Harabasz Index: {ch: .4f}")
    print(f"GaussianMixture Davies-Bouldin Index: {db: .4f}")
    
    # Count of instances per cluster
    for i in range(n_clusters):
        print(f"Label: {df_m['gm_labels'].unique()[i]}, Percentage total customers: {round(df_m[df_m['gm_labels'] == i].shape[0] *100/df_m.shape[0],2)}%")
        
    return df_m


def mshift_model(df,n_bandwidth):
    ''' Function for Mean Shift clustering. Takes also as parameter the bandwidth number.
    Returns updated dataframe with a column for the labels from the clustering and performance metrics:
    Silhouette coefficient, Calinski-Harabasz Index, Davies-Bouldin Index.
    Additionally it prints (but does not return) the percentage of instances in each cluster to the total instances.
    Returns in total 4 elements.
    '''
    df_m = df.copy()
    
    mshclust = MeanShift(bandwidth=n_bandwidth).fit(df_m)
    df_m['mshift_labels'] = mshclust.labels_

    # Performance metrics
    sc = metrics.silhouette_score(df_m, df_m['mshift_labels'], metric='euclidean')
    ch = metrics.calinski_harabasz_score(df_m, df_m['mshift_labels'])
    db = davies_bouldin_score(df_m, df_m['mshift_labels'])

    # Printing results
    print(f"MeanShift silhouette coefficient: {sc: .4f}")
    print(f"MeanShift Calinski-Harabasz Index: {ch: .4f}")
    print(f"MeanShift Davies-Bouldin Index: {db: .4f}")
    
    # Count of instances per cluster
    for i in range(df_m['mshift_labels'].nunique()):
        print(f"Label: {df_m['mshift_labels'].unique()[i]}, Percentage total customers: {round(df_m[df_m['mshift_labels'] == i].shape[0] *100/df_m.shape[0],2)}%")
        
    return df_m



def prophet_model(df2, days):
    ''' Prophet model used for inventory forecast.
    Takes the dataframe as parameter, and how many days will be assigned for the test group, and splits the dataset internally.
    Returns the forecasts and RMSE value, and prints the model evaluation metrics.'''

    df = df2.copy()
    # Splitting, train, test
    train = df.iloc[:len(df) - days]
    test = df.iloc[len(df) - days:]


    # Training model
    m = Prophet()
    m.add_country_holidays(country_name='UK')
    m.fit(train)
    future = m.make_future_dataframe(periods = days)
    forecast = m.predict(future)

    # Model evaluation
    predictions = forecast.iloc[-days:]['yhat']
    actual_values = test['y']
    rmse_n = rmse(predictions, actual_values)
    mae_n = mean_absolute_error(actual_values, predictions)
    print("Root Mean Squared Error: ",rmse_n)
    print("Mean Absolute Error: ", mae_n)


    return forecast, rmse_n, mae_n



def df_prophet_prep(df, product, clipping = False):
    '''Function to prepare the dataframe for the prophet model.
    The dataframe is filtered to refer only to 1 product (one of the top selling ones).
    Receives as parameters the dataframe that needs adjustment, the product, and whether outlier clipping will be performed.
    Returns updated dataframe and prints in how many instances there needed to be done outlier clipping.
    '''
    
    df = df[df.StockCode == product].drop(columns='StockCode').reset_index(drop=True)

    # if needed, clipping of outliers, using the Standard Deviation method
    if clipping == True:
        # I am using the function I have defined for outliers clipping
        df = outliers_clipping_STD(df,'Quantity')
    
    
    # Adding rows for all dates, even if there is no data
    # Defining the known start and end dates of the dataset and creating dataframe with all the dates in between
    start_date = '2010-12-01'
    end_date = '2011-12-09'
    all_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    all_dates_df = pd.DataFrame({'InvoiceDate': all_dates})

    # merging the date dataframe with the main dataframe
    merged_df = all_dates_df.merge(df, on='InvoiceDate', how='left').fillna(0)

    # and then grouping to make sure, I have only ont line per day
    merged_df = merged_df.groupby('InvoiceDate')['Quantity'].sum().reset_index()

    
    # Renaming columns as needed for the prophet model
    for i in range(merged_df.shape[1]):
        if merged_df.iloc[:,i].dtypes =='<M8[ns]':
            merged_df.rename(columns={merged_df.columns[i]: "ds"}, inplace=True)
        else:
            merged_df.rename(columns={merged_df.columns[i]: "y"}, inplace=True)

    # making sure the column order is as needed for the prophet model
    merged_df = merged_df[['ds', 'y']] 

    
    return merged_df


def metrics_calc_clusters(df, product, rmse_TopPr0, mae_TopPr0):
    """
    Function that calculates the collective RMSE and MAE values for predictions based on clusters,
    where the total success is proportional to the size of each cluster.

    Receives as parameters the dataframe containing the data, the product for which the forecasts are made
    list of RMSE and MAE values for the forecasts for the given product per cluster.

    Prints the the collective RMSE and MAE and returns nothing.
    """
    
    # Calculating the percentage of instances per cluster
    label_percentages = [round((df[(df.StockCode==product)&(df.Label==i)].shape[0]/
                          df[df.StockCode==product].shape[0])*100,2) for i in range(4)]

    # Calculating the propotional RMSE and MAE values per cluster
    proportional_rmse_TopPr0_values = [rmse * percentage * 0.01 for rmse, percentage in zip(rmse_TopPr0, label_percentages)]
    proportional_mae_TopPr0_values = [mae * percentage * 0.01 for mae, percentage in zip(mae_TopPr0, label_percentages)]

    # Calculating the collective RMSE and MAE values
    print('RMSE: ',sum(proportional_rmse_TopPr0_values),', MAE: ',sum(proportional_mae_TopPr0_values))

    return



def prophet_model_with_clusters(df_TopPr0_cl0, df_TopPr0_cl1, df_TopPr0_cl2, df_TopPr0_cl3, days, df, product):
    '''
    Performs Prophet forecasting for a product with cluster separation.

    This function takes dataframes for the top product with cluster separation and applies Prophet forecasting
    separately to each cluster. It calculates RMSE and MAE for each cluster and prints the results. Then, it
    calculates collective metrics considering the size of each cluster and prints the results.

    Parameters:
    - Dataframes for the selected top product with already cluster separation done.
    - days (int): Number of days for forecasting.
    - the original dataframe, needed for the metrics
    - the chosen product needed for the metrics

    Returns: None
    '''
    
    rmse_TopPr0 = [0] * 4
    mae_TopPr0 = [0] * 4

    # Creating dataframes for the top product 1, for the dataframes with cluster seperation, and no outlier clipping
    # Cluster 0
    print('Cluster 0:')
    _, rmse_TopPr0[0], mae_TopPr0[0] = prophet_model(df_TopPr0_cl0, days)
    # Cluster 1
    print('Cluster 1:')
    _, rmse_TopPr0[1], mae_TopPr0[1] = prophet_model(df_TopPr0_cl1, days)
    # Cluster 2
    print('Cluster 2:')
    _, rmse_TopPr0[2], mae_TopPr0[2] = prophet_model(df_TopPr0_cl2, days)
    # Cluster 3
    print('Cluster 3:')
    _, rmse_TopPr0[3], mae_TopPr0[3] = prophet_model(df_TopPr0_cl3, days)

    print('\nCollective metrics for this model:')
    metrics_calc_clusters(df, product, rmse_TopPr0, mae_TopPr0)

    return


def prophet_model_per_product(df, product, product_n):
    ''' Temporary function for testing Prophet forecasting on multiple top-selling products.

    Due to time constraints, this function duplicates code from the main file to test the Prophet
    forecasting model on products other than the top 1 product. It separates the dataframe per selected 
    product and applies the Prophet forecasting model to each product separately. This function is a 
    temporary solution until individual functions are built to handle forecasting for multiple products.

    Parameters:
    - df (DataFrame): The dataframe containing sales data.
    - product (str): The selected product for forecasting.
    - product number, to print it, for clarity

    Returns: None
    '''
    
    df_TopPr0 = df_prophet_prep(df, product, False) 
    df_TopPr0_clip = df_prophet_prep(df, product, True)

    # Splitting dataset per cluster
    df_cl0 = df[df.Label == 0].reset_index(drop=True)
    df_cl1 = df[df.Label == 1].reset_index(drop=True)
    df_cl2 = df[df.Label == 2].reset_index(drop=True)
    df_cl3 = df[df.Label == 3].reset_index(drop=True)

    # Creating dataframes for the top product 1, for the dataframes with cluster seperation, and no outlier clipping
    df_TopPr0_cl0 = df_prophet_prep(df_cl0.drop(columns='Label'), product, False)
    df_TopPr0_cl1 = df_prophet_prep(df_cl1.drop(columns='Label'), product, False)
    df_TopPr0_cl2 = df_prophet_prep(df_cl2.drop(columns='Label'), product, False)
    df_TopPr0_cl3 = df_prophet_prep(df_cl3.drop(columns='Label'), product, False)

    # Creating dataframes for the top product 1, for the dataframes with cluster seperation, and with outlier clipping
    df_TopPr0_cl0_clip = df_prophet_prep(df_cl0.drop(columns='Label'), product, True)
    df_TopPr0_cl1_clip = df_prophet_prep(df_cl1.drop(columns='Label'), product, True)
    df_TopPr0_cl2_clip = df_prophet_prep(df_cl2.drop(columns='Label'), product, True)
    df_TopPr0_cl3_clip = df_prophet_prep(df_cl3.drop(columns='Label'), product, True)

    # Model
    print(f"\033[1m\n\nTop {product_n+1} product, no clusters, no outlier clipping:\033[0m")
    _,_,_ = prophet_model(df_TopPr0, 60)
    print(f"\033[1m\n\nTop {product_n+1} product, no clusters, with outlier clipping:\033[0m")
    _,_,_ = prophet_model(df_TopPr0_clip, 60)
    print(f"\033[1m\n\nTop {product_n+1} product, with clusters, no outlier clipping:\033[0m")
    prophet_model_with_clusters(df_TopPr0_cl0, df_TopPr0_cl1, df_TopPr0_cl2, df_TopPr0_cl3, 60, df, product)
    print(f"\033[1m\n\nTop {product_n+1} product, with clusters, with outlier clipping:\033[0m")
    prophet_model_with_clusters(df_TopPr0_cl0_clip, df_TopPr0_cl1_clip, df_TopPr0_cl2_clip, df_TopPr0_cl3_clip, 60, df, product)

    return

    