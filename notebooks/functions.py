import pandas as pd
import numpy as np

def cleaning(df):

    # Missing values
    # Droppping rows with UnitPrice 0
    df = df[df.UnitPrice != 0.0]

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    
    # Formatting, removing .0 from CustomerID -> column will become Object after and NaN will be preserved
    df.CustomerID = df.CustomerID.apply(lambda x: format(x, '.0f') if not pd.isna(x) else x)
    # Changing country EIRE to Ireland
    df.Country = df.Country.replace('EIRE','Ireland')

    
    # Dropping invoices that are cancellations -> the Invoice codes that start with 'C'
    df = df[~df.InvoiceNo.str.startswith('C', na=False)]
    
    # Dropping orders that were samples or adjusted bad debt (StockCode 'S' or 'B')
    df = df[~df.StockCode.isin(['S','B'])]
    
    # for Manual orders (StockCode 'M' or 'm') I will add to StockCode and Description NaN, to treat them as such
    df.loc[df.StockCode.isin(['M','m']),['Description','StockCode']] = np.nan


    # I see that there are also the shipping costs with SrockCode 'POST' and 'DOT'
    # For clarity I will replace the code and description with shipping
    df.loc[df.StockCode.isin(['POST','DOT']),['Description','StockCode']] = 'SHIPPING'

    # There are two entries with amazon fee, and these invoices contain only that item, will drop them
    df = df[~df.Description.str.contains('AMAZON', na=False)]

    
    # it looks that the top Unit prices are either Shipping or NaN
    # I will drop the NaN from StockCode (they are from the MANUAL code) because they don't offer any clear insights
    df.dropna(subset='StockCode', inplace=True)

    # Resetting index
    df.reset_index(drop=True, inplace=True)
    
    return df
