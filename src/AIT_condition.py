import numpy as np
import pandas as pd
import os
os.environ['R_HOME'] = 'D:\ApplicationSpace\Anaconda3\envs\py38\Lib\R'

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import src.indTest.HSIC2 as fasthsic
from rpy2.robjects import r
import rpy2.robjects.packages as rpackages
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
def AIT_test(data, Z, **params):
    alpha = params.get('alpha', 10 / len(data['Treatment'][0]))
    verbose = params.get('verbose', False)
    relation = params.get('relation', 'nonlinear')

    indexs = list(data.columns)
    if 'Treatment' not in indexs or 'Outcome' not in indexs:
        print('Please ensure the input is the variable of data!')
        exit(-1)

    if any(col.startswith('W') for col in data.columns):    # Determine whether covariates exist.
        if verbose: print("There are covariates.")
        if relation == 'linear':
            A, Z_data = linear_get_A_with_W(data, Z)
        else:
            A, Z_data = cf_with_W(data, Z)
    else:
        if verbose: print("There are no covariates.")
        if relation == 'linear':
            A, Z_data = linear_get_A(data,Z)
            # A = linear_get_A(data, Z)
            # Z_data = data[Z].values.reshape(-1,1) ##
        else:
            A, Z_data = cf_no_W(data, Z)
    # print(type(A),type(Z_data))
    # print(A.shape,Z_data.shape)
    pValue_Z = fasthsic.test(A, Z_data, alpha=alpha, verbose=verbose)
    # pValue_Z = test_independence(A, Z_data, alpha=alpha)
    # print(pValue_Z)

    if pValue_Z < alpha :
        valid_IV = False
    else:
        valid_IV = True
    return {'IV_validity':valid_IV,'pValue_Z': pValue_Z}



def linear_get_A(df, Z):
    X_data = np.array([np.array(val) for val in df['Treatment']])[0]
    Y_data = np.array([np.array(val) for val in df['Outcome']])[0]
    Z_data = np.array([np.array(val) for val in df[Z]])[0]

    X_data_flat = X_data.reshape(-1, X_data.shape[1]) 
    Y_data_flat = Y_data.reshape(-1, Y_data.shape[1]) 
    Z_data_flat = Z_data.reshape(-1, Z_data.shape[1]) 
    beta_list = []


    for i in range(Y_data.shape[1]):  
        Y = Y_data_flat[:, i]  

        ZTX = np.dot(Z_data_flat.T, X_data_flat)  # Z^T X
        ZTY = np.dot(Z_data_flat.T, Y)       # Z^T Y
        try:
            beta = np.linalg.inv(ZTX).dot(ZTY)  # β = (Z^T X)^(-1) Z^T Y
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(ZTX).dot(ZTY)

        beta_list.append(beta) 

    A = Y_data_flat - np.dot(X_data_flat, np.array(beta_list).T)
    return A, Z_data_flat


def linear_get_A_with_W(data, Z):

    X_data = data['Treatment'].values.reshape(-1, 1)
    Y_data = data['Outcome'].values.reshape(-1, 1)
    Z_data = data[Z].values.reshape(-1, 1)
    W_data = data.filter(like='W').values

    # Linear regression
    model_YW = LinearRegression().fit(W_data, Y_data)
    residual_Y = Y_data - model_YW.predict(W_data)
    model_XW = LinearRegression().fit(W_data, X_data)
    residual_X = X_data - model_XW.predict(W_data)
    model_ZW = LinearRegression().fit(W_data, Z_data)
    residual_Z = Z_data - model_ZW.predict(W_data)

    if len(residual_Y.shape) == 2: residual_Y = residual_Y.reshape(-1)
    if len(residual_X.shape) == 2: residual_X = residual_X.reshape(-1)
    if len(residual_Z.shape) == 2: residual_Z = residual_Z.reshape(-1)

    cov_YZ_given_W = np.cov(residual_Y, residual_Z)[0, 1]
    cov_XZ_given_W = np.cov(residual_X, residual_Z)[0, 1]

    if cov_XZ_given_W == 0:
        raise ValueError("Covariance of X and Z is zero, cannot divide by zero")
    f_hat = cov_YZ_given_W / cov_XZ_given_W
    A = residual_Y - f_hat * residual_X
    if len(A.shape) == 1: A = A.reshape(-1, 1)


    return A, residual_Z.reshape(-1, 1)


def cf_no_W(data, Z):

    if not rpackages.isinstalled('readxl'):
        rpackages.importr('readxl')
    if not rpackages.isinstalled('Formula'):
        rpackages.importr('Formula')
    # path = os.path.join('control_IV/controlfunctionIV-main/R/using_cf.R')
    robjects.r.source('pretest.R')
    robjects.r.source('cf.R')
    path = os.path.join('using_cf.R')
    robjects.r.source(path)
    pandas2ri.activate() 

    r_dataframe = pandas2ri.py2rpy(data)
    result = robjects.r.using_R_cf_no_W(r_dataframe, Z)

    A = np.array(result).reshape(-1, 1)
    Z_data = data[Z].values.reshape(-1, 1)
    return A, Z_data


def cf_with_W(data, Z):

    if not rpackages.isinstalled('readxl'):
        rpackages.importr('readxl')
    if not rpackages.isinstalled('Formula'):
        rpackages.importr('Formula')
    # path = os.path.join('control_IV/controlfunctionIV-main/R/using_cf.R')
    robjects.r.source('pretest.R')
    robjects.r.source('cf.R')
    path = os.path.join('using_cf.R')
    robjects.r.source(path)
    pandas2ri.activate()

    r_dataframe = pandas2ri.py2rpy(data)
    result = robjects.r.using_R_cf_with_W(r_dataframe, Z)

    A = np.array(result).reshape(-1, 1)
    W_data = data.filter(like='W')
    residual_Z = random_forest_residuals(data[Z], W_data)
    return A, residual_Z.reshape(-1, 1)


def random_forest_residuals(Z_data, Ws_data):

    combined_data = pd.concat([Z_data, Ws_data], axis=1).dropna()
    dependent_var_clean = combined_data.iloc[:, 0]
    independent_vars_clean = combined_data.iloc[:, 1:]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(independent_vars_clean, dependent_var_clean.values.ravel())
    residuals = dependent_var_clean.values.ravel() - model.predict(independent_vars_clean)
    return residuals


