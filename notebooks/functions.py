import pandas as pd
import numpy as np

def cleaning(df):

    # Missing values
    # Droppping rows with UnitPrice 0
    df = df[df.UnitPrice != 0.0]

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    
    # Formatting, removing .0 from CustomerID -> column will become Object after and NaN will be preserved
    #df.CustomerID = df.CustomerID.apply(lambda x: format(x, '.0f') if not pd.isna(x) else x)
    df['CustomerID'] = df['CustomerID'].astype('Int64')
    
    # Changing country EIRE to Ireland
    df.Country = df.Country.replace('EIRE','Ireland')

    
    # Dropping invoices that are cancellations -> the Invoice codes that start with 'C'
    df = df[~df.InvoiceNo.str.startswith('C', na=False)]

    
    # Adding anything that is not Sock to StockCode 'OTHER'
    df.loc[df.StockCode.isin(['M', 'D', 'S', 'B', 'm','C2','POST','DOT','AMAZONFEE','CRUK']),'StockCode'] = 'OTHER'

    
    # Correcting multiple descriptions for StockCodes
    #stock_descr = pd.DataFrame(df.groupby('StockCode')['Description'].agg(lambda x: x.value_counts().index[0]))
    #stock_descr.reset_index(inplace=True)
    #df.Description = df.StockCode.map(dict(zip(stock_descr.StockCode, stock_descr.Description)))

    
    # Resetting index
    df.reset_index(drop=True, inplace=True)
    
    return df
