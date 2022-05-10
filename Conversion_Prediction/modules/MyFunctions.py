#!/usr/bin/env python
# coding: utf-8

import os
from time import time  
from datetime import datetime
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

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


def get_date(date_str):
    dt = datetime.strptime(date_str, "%d-%m-%Y")
    dic = {'date': dt, 'day': dt.day, 'year': dt.year, 'month': dt.month, 'weekday':dt.weekday()}
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
    if dataset.isnull().sum().sum() == 0:
        print('there is no missing values in this dataset')
    else:
        miss = dataset.isnull().sum() # series
        missing = pd.DataFrame(columns=['Variable', 'n_missing', 'p_missing'])
        missing['Variable'] = miss.index
        missing['n_missing'] = miss.values
        missing['p_missing'] = round(100*miss/dataset.shape[0],2).values

        display(missing.sort_values(by='n_missing'))

    
def duplicates_count(dataset):
    count_dup = len(dataset)-len(dataset.drop_duplicates())
    if count_dup == 0:
        print('No duplicated rows found')
    else: 
        display(
            dataset.groupby(dataset.columns.tolist())\
              .size().reset_index()\
              .rename(columns={0:'records'}))

    
def outliers_count(dataset, columns):
    index = ['count', 'mean', 'std', 'lower_fence', 'upper_fence', 'outliers', 'count_after_drop']
    df_outliers = pd.DataFrame(columns= columns, index = index)

    for col in columns:
        count = dataset[col].count()
        mean = dataset[col].mean()
        std = dataset[col].std()
        lower_fence = mean - 3 * std
        upper_fence =  mean + 3 * std

        mask = (dataset[col] < lower_fence) | (dataset[col] > upper_fence)
        outliers = dataset[col][mask].count()
        count_after_drop = count - outliers

        df_outliers[col] = [count, mean, std, lower_fence, upper_fence, outliers, count_after_drop]

    display(df_outliers)

    
def remove_outlier(dataset, col):
    mean = dataset[col].mean()
    std = dataset[col].std()
    lower_fence  = mean - 3 * std
    upper_fence = mean + 3 * std
    mask = (dataset[col] < lower_fence)| (dataset[col] > upper_fence)
    df_out = dataset.loc[~mask]
    return df_out

    
def remove_missing(dataset, col):      
    mask = dataset[col].isna()
    df_out = dataset.loc[~mask]

    return df_out

    
def group_count(dataset, cols):      
    df_out = dataset.groupby(by =cols).size().reset_index(name='count')
    return df_out

    
def numeric_categorical(dataset):      
    # Automatically detect positions of numeric/categorical features
    # not useful in cases where categorical features are coded as integers or floats
    idx = 0
    numeric_features = []
    numeric_indices = []
    categorical_features = []
    categorical_indices = []
    for i,t in dataset.dtypes.iteritems():
        if ('float' in str(t)) or ('int' in str(t)) :
            numeric_features.append(i)
            numeric_indices.append(idx)
        else :
            categorical_features.append(i)
            categorical_indices.append(idx)

        idx = idx + 1

    print('Found numeric features ', numeric_features,' at positions ', numeric_indices)
    print('Found categorical features ', categorical_features,' at positions ', categorical_indices)

    return numeric_features, numeric_indices, categorical_features, categorical_indices

    
def correlated(dataset):   
    mask = (dataset.corr().abs() < 1.0) & (dataset.corr().abs() > 0.9)
    high_corr = dataset.corr()[mask]
    high_corr.dropna(axis='index', how='all', inplace=True)
    high_corr.dropna(axis='columns', how='all', inplace=True)
    return high_corr

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

   
def my_hist_plotter(dx, size):
    out = go.Histogram(x=dx, xbins = dict(size =size))
    return out

   
def my_bar_plotter(dx, dy, kargs):
    out = go.Bar( x=dx, y=dy, **kargs)
    return out

   
def my_heatmap(dataset, title, folder):
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
        width=500, 
        height=500,
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

    # Export to a png image
    fig.to_image(format="png", engine="kaleido")
    if os.path.exists(folder+title+".png"):
        os.remove(folder+title+".png")

    fig.write_image(folder+title+".png")
    
    return fig    

#********************************************************************************************************
# ******************************************************  ML functions **********************************
#********************************************************************************************************

def train_val(X, y, train_ratio, val_ratio, seed):
    assert sum([train_ratio, val_ratio])==1.0, "wrong given ratio, all ratios have to sum to 1.0"
    assert X.shape[0]==len(y), "X and y shape mismatch"
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                      test_size = val_ratio, 
                                                      random_state=seed,
                                                      stratify =y)
    return X_train, X_val, y_train, y_val

def roc_auc(clf, X, y):
    prob_y = clf.predict_proba(X)
    prob_y_1 = [p[1] for p in prob_y]
    return roc_auc_score(y, prob_y_1)

def model_validation(estimator, X, y, cv, scoring):
    t0 = time()
    scores = cross_validate(estimator, 
                            X, y, 
                            cv = cv,
                            scoring=(scoring),
                            n_jobs = -1
                           )
    
    name, mean_acc, std_acc  = str(estimator).split('(')[0], float(np.mean(scores['test_score'])), float(np.std(scores['test_score']))
    print(f'fitting {name} is done in {time() - t0}s')
    
    return name, mean_acc, std_acc 

def model_selection(estimator, X_train, y_train, X_val, y_val, params, scoring):
    X = np.concatenate((X_train, X_val), axis=0)
    y = np.concatenate((y_train, y_val), axis=0)
    validation_indexes = [-1]*len(X_train) + [0]*len(X_val)
    ps = PredefinedSplit(test_fold=validation_indexes)
    print('cv = ', ps)
    
    t0 = time()
    grid = GridSearchCV(estimator = estimator, 
                        param_grid = params,
                        cv = ps,
                        scoring=(scoring),
                        n_jobs = -1,
                        verbose = 1
                       )
    grid.fit(X, y)
    name = str(estimator).split('(')[0]
    print(f'Tuning {name} hyperparameters is done in {time() - t0}s')
    
    print('\nBest Estimator \n') 
    best_estimator = grid.best_estimator_
    print('Best Params \n') 
    print(grid.best_params_)
    print('Best score \n') 
    print(grid.best_score_)
 
    return best_estimator

def learn(estimator, X_train, y_train, X_val, y_val):
    estimator.fit(X_train, y_train)
    y_train_pred = estimator.predict(X_train)
    y_val_pred  = estimator.predict(X_val)
    acc_train, acc_test = accuracy_score(y_train, y_train_pred), accuracy_score(y_val, y_val_pred)
    
    return acc_train, acc_test

def learn(estimator, X_train, y_train, X_test, y_test, name):
    # fitting
    print(f"fitting {name} is launched")
    t0=time()
    estimator.fit(X_train, y_train)
    print(f"fitting {name} is done in {(time() - t0)} s")
    
    # Predictions
    y_train_pred = estimator.predict(X_train)
    y_test_pred = estimator.predict(X_test)

    # evaluation
    acc_train, acc_test = accuracy_score(y_train, y_train_pred), accuracy_score(y_test, y_test_pred)
    f_train, f_test = f1_score(y_train, y_train_pred), f1_score(y_test, y_test_pred)
    roc_train, roc_test = roc_auc(estimator, X_train, y_train), roc_auc(estimator, X_test, y_test)
    
    return acc_train, acc_test, f_train, f_test, roc_train, roc_test