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



def prophet_model(df, days):
    ''' Prophet model used for inventory forecast.
    Takes the dataframe as parameter, and how many days will be assigned for the test group, and splits the dataset internally.
    Returns the forecasts, and prints the model evaluation metrics.'''

    # Splitting, train, test
    train = df.iloc[:len(df) - days]
    test = df.iloc[len(df) - days:]

    # Training model
    m = Prophet()
    m.fit(train)
    future = m.make_future_dataframe(periods = days)
    forecast = m.predict(future)

    # Model evaluation
    predictions = forecast.iloc[-73:]['yhat']
    print("Root Mean Squared Error: ",rmse(predictions, test['y']))
    print("Mean Absolute Error: ", mean_absolute_error(test['y'], predictions))


    return forecast



    