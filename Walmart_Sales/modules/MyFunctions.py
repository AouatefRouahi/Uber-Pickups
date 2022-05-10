#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff


# ******************************************************  Extract Data **********************************


def get_year(date_str):
    if date_str.lower() != 'nan':
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.year
    else:
        return np.nan

def get_month(date_str):
    if date_str.lower() != 'nan':
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.month
    else:
        return np.nan


def get_weekday(date_str):
    if date_str.lower() != 'nan':
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.weekday()
    else:
        return np.nan


def get_day(date_str):
    if date_str.lower() != 'nan':
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.day
    else:
        return np.nan


def get_week(date_str):
    if date_str.lower() != 'nan':
        dt = datetime.strptime(date_str, "%d-%m-%Y")
        return dt.isocalendar()[1]
    else:
        return np.nan


def get_date(dt):
    dic = {'date': dt, 'hour': dt.hour ,'day': dt.day, 'year': dt.year, 'month': dt.month,
           'dayofweek':dt.weekday(), 'quarter': dt.month //4 + 1, 'dayofyear': dt.timetuple().tm_yday, 
           'dayofmonth': dt.timetuple().tm_mday, 'weekofyear': dt.isocalendar()[1] }
    return dic

# ******************************************************  Statistics  **********************************

def explore(dataset):
    print("Shape : {}".format(dataset.shape))
    print()

    print("data types : \n{}".format(dataset.dtypes))
    print()

    print("Display of dataset: ")
    display(dataset.head())
    print()

    print("Basics statistics: ")
    display(dataset.describe(include='all'))
    print()

    print("Distinct values: ")
    display(pd.Series(dataset.nunique(dropna = False)))


def unique_count(dataset, Cols):
    for col in Cols:
        print(f"unique values of {col}:")
        display(dataset[col].value_counts(dropna=False, ascending=False))


def missing(dataset):
    missing = pd.DataFrame(columns=['Variable', 'n_missing', 'p_missing'])

    miss = dataset.isnull().sum() # series

    missing['Variable'] = miss.index
    missing['n_missing'] = miss.values
    missing['p_missing'] = round(100*miss/dataset.shape[0],2).values

    return missing.sort_values(by='n_missing')  


def duplicates_count(dataset):
    count_dup = len(dataset)-len(dataset.drop_duplicates())
    if count_dup == 0:
        print('No duplicated rows found')
    else:
        df = dataset.groupby(dataset.columns.tolist())\
              .size().reset_index()\
              .rename(columns={0:'records'})
        mask = df['records']> 1
        n = df[mask]['records'].sum()
        print(f'{n} duplicated rows found')
        display(df[mask])


def outliers_count(dataset, columns):
    index = ['count', 'mean', 'std', 'low_fence', 'high_fence', 'outliers', 'outliers_p', 'count_after_drop']
    df_outliers = pd.DataFrame(columns= columns, index = index)
    for col in columns:
        count = int(dataset[col].count())
        mean = dataset[col].mean()
        std = dataset[col].std()
        low_fence = mean - 3 * std
        high_fence =  mean + 3 * std

        mask = (dataset[col] < low_fence) | (dataset[col] > high_fence)
        outliers = int(dataset[col][mask].count())
        outliers_p = round(outliers/count*100,2)
        count_after_drop = int(count - outliers)

        df_outliers[col] = [count, mean, std, low_fence, high_fence, outliers, outliers_p, count_after_drop]
        
    df_outliers= df_outliers.T   
    display(df_outliers)


def remove_outlier(dataset, col):
    mean = dataset[col].mean()
    std = dataset[col].std()
    low_fence  = mean - 3 * std
    high_fence = mean + 3 * std
    mask = (dataset[col] < low_fence)| (dataset[col] > high_fence)
    df_out = dataset.loc[~mask]
    return df_out


def remove_missing(dataset, col):      
    mask = dataset[col].isna()
    df_out = dataset.loc[~mask]

    return df_out

# ******************************************************  Metrics **********************************

def sum_squares(y_true, y_pred): 
    # SST = sum(y_true - y_true.mean())^2
    SST = np.sum((y_true - np.mean(y_true))**2)
    # SSR = sum(y_pred - y_true.mean())^2
    SSR = np.sum((y_pred - np.mean(y_true))**2)
    # SSE = sum(y_true - y_pred)^2
    SSE = np.sum((y_pred - y_true)**2)

    return SST, SSR, SSE


def adjusted_r2(r2, n, p): 
    # r2: Sample R squared score of a given model;  n : total Sample size; p: number of features (independent variables)
    adj_r = 1 - ((1-r2)*(n-1)/(n-p-1))

    return adj_r


def f_value(mode, **kargs): 
    '''
    mode = SLR, MLR, MvM ; n : sample size; p: number of independent variables
    SLR (Simple Linear Model): compare a model with one independent variable to a dummy mean regressor F(1, n-2)
    MLR (Multiple Linear Model) : compare a model with multiple independent variables to a dummy mean regressor F(p, n-p-1)
    MvM (Model vs Model): compare two modeles with different number of independent variables F(p2 - p1, n-p1)
    ''' 
    f_value = -1
    if mode == 'SLR' :
        # args = {n, SSE, SSR} --> F(1, n-2) = [SSR/1]*[(n-2)/SSE]
        f_value = (kargs['SSR'])*(n-2)/(kargs['SSE'])
    elif mode == 'MLR' :
        # args = {n, p, SSE, SSR} --> F(p, n-p-1) = [SSR/p]*[(n-p-1)/SSE]
        print(kargs)
        f_value = (kargs['SSR']/kargs['p'])*((kargs['n']-kargs['p']-1)/kargs['SSE'])
    elif mode == 'MvM' :
        # args = {n, p1, p2, SSR1, SSR2} --> F(p2- p1, n-p2-1) = [(SSR1 - SSR2)/(p2-p1)]*[(n-p2-1)/SSR2]
        f_value = ((kargs['SSR1'] - kargs['SSR2'])/(kargs['p2'] - kargs['p1']))*((kargs['n']-kargs['p2'] -1)/kargs['SSR2'])
    else:
        print('Invalid Mode (SLR, MLR, MvM)')

    return f_value
# ******************************************************  Graphics **********************************

def my_box_plotter(data):
    """
    1) étudier la symétrie, la dispersion ou la centralité de la distribution des valeurs associées à une variable.
    3) détecter les valeurs aberrantes pour  
    2) comparer des variables basées sur des échelles similaires et pour comparer les valeurs 
       des observations de groupes d'individus sur la même variable
       all / outliers / suspectedoutliers
    """
    out = go.Box(y=data, boxpoints='all', name = data.name, pointpos=-1.8, boxmean=True) # add params
    return out


def my_scatter_plotter(dx, dy):
    out = go.Scatter(x = dx, y = dy, mode ='markers')
    return out


def my_scatter_plotter_l(dx, dy, name, color):
    out = go.Scatter(x=dx, y=dy, name=name, mode='lines+markers', marker_color=color, text =color)
    return out


def my_hist_plotter(dx):
    out = go.Histogram(x=dx)
    return out

def my_hist_plotter2(dx, label):
    out = ff.create_distplot([dx], [label], show_rug=False)
    return out

def my_bar_plotter(dx, dy):
    out = go.Bar( x=dx, y=dy)
    return out


def my_heatmap(dataset, title):
    corr = round(abs(dataset.corr()),2)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    df_mask = corr.mask(mask)

    fig = ff.create_annotated_heatmap(z=df_mask.to_numpy(), 
                              x=df_mask.columns.tolist(),
                              y=df_mask.columns.tolist(),
                              colorscale='Viridis',
                              hoverinfo="none", #Shows hoverinfo for null values
                              showscale=True, ygap=1, xgap=1
                             )

    fig.update_xaxes(side="bottom")

    fig.update_layout(
        title_text= title, 
        title_x=0.5, 
        width=1000, 
        height=1000,
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        yaxis_autorange='reversed',
        template='plotly_white'
    )

    # NaN values are not handled automatically and are displayed in the figure
    # So we need to get rid of the text manually
    for i in range(len(fig.layout.annotations)):
        if fig.layout.annotations[i].text == 'nan':
            fig.layout.annotations[i].text = ""

    return fig  

# ********************************************************Enumeration ************************************************
from enum import Enum

class Status(Enum):
    NOT_SELECTED = 'not selected'
    SELECTED = 'selected'
    LATER = 'later'
    MAYBE = 'maybe'

