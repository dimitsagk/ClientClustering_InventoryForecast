import pandas as pd
import numpy as np

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

    



    