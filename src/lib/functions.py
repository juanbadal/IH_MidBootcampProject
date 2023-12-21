
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from unidecode import unidecode
import scipy.stats as st
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant



def snakecase_columns(df: pd.DataFrame) -> pd.DataFrame:
    
    '''
    Converts the name of a columns of a dataframe to snakecase (lowercase with '_' instead of white spaces), and replaces any instance of neighborhood      with neighbourhood (for consistency)
    Input: pd.DataFrame
    Output: pd.DataFrame with snakecase column names
    '''

    snakecase_cols = []

    for col in df.columns:
        col = col.lower().replace(' ', '_')
        col = col.replace('neighborhood', 'neighbourhood')
        snakecase_cols.append(col)

    df.columns = snakecase_cols

    return df



def remove_accents(input: str) -> str:
    '''
    Removes accents and other letter punctuation symbols for letters and returns the converted string.
    '''
    return unidecode(input)



def remove_percent_div_100(input_str: str) -> float:
    '''
    Takes a string, removes the % symbol, converts it to a float and divides it by 100.
    '''
    if isinstance(input_str, str):
        converted_value = input_str.replace('%', '')
        divided_value = float(converted_value) / 100
    else:
        return input_str
    return divided_value



def remove_dollar_symbol_commas(input_str: str) -> float:
    '''
    Takes a string, removes the $ symbol and converts it to a float.
    '''
    if isinstance(input_str, str):
        no_dollar = input_str.replace('$', '')
        no_comma = no_dollar.replace(',', '')
        float_val = float(no_comma)
    else:
        return input_str
    return float_val



def make_histograms_categories(df, col_filter, col_plot, categories, bins=75, figsize=(12, 15), sharex=False, sharey=False):
    
    """
    Plot Seaborn histograms in a grid for different categories.
    Inputs:
    - df: DataFrame containing the data
    - col_filter: Column in the DataFrame for which the category filter will be done
    - col_plot: Column in the DataFrame for which histograms will be plotted
    - categories: List of unique values in the column for which histograms will be plotted
    - bins: Number of bins for histograms (default is 75)
    - figsize: size of the figure (default is 12, 25)
    """
    
    num_categories = len(categories)

    if num_categories == 2:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=sharex, sharey=sharey)
    else:
        num_rows = num_categories // 2
        num_cols = 2 if num_categories % 2 == 0 else 1

        fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize, sharex=sharex, sharey=sharey)
        fig.subplots_adjust(hspace=0.5)

    for i, category in enumerate(categories):
        if num_categories == 2:
            sns.histplot(df[df[col_filter] == category][col_plot], bins=bins, ax=axes[i])
            axes[i].set_title(category)
        else:
            sns.histplot(df[df[col_filter] == category][col_plot], bins=bins, ax=axes[i // 2, i % 2])
            axes[i // 2, i % 2].set_title(category)

    plt.show()



def make_histograms_cols(df: pd.DataFrame, figsize=(12, 15)):
    
    """
    Takes a dataframe and creates histograms for all the columns.
    Parameters:
    - df: DataFrame
    - figsize: Modifies the size of the plotting figure (default (12, 15))
    """
    
    num_cols = 2
    total_cols = len(df.columns)
    num_rows = (total_cols + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)

    for i, col in enumerate(df.columns):
        row_idx = i // num_cols
        col_idx = i % num_cols
        sns.histplot(x=df[col], data=df, ax=axes[row_idx, col_idx]) 
        axes[row_idx, col_idx].set_title(col)

    plt.show()



def compute_skewness(df: pd.DataFrame, threshold: int=-1):
    
    '''
    Computes and prints the skewness of the columns in a dataframe.
    Inputs: pandas DataFrame
    '''
    
    print('Skewness of columns in the dataframe:\n')
    
    for col in df.columns:
        if st.skew(df[col]) > abs(threshold) or st.skew(df[col]) < threshold:
            print(f'{col}: {round(st.skew(df[col]), 2)} -> Out of threshold')
        else:
            print(f'{col}: {round(st.skew(df[col]), 2)}')



def compute_vif(df: pd.DataFrame, columns: list):

    X = df.loc[:, columns]
    # the calculation of variance inflation requires a constant
    X.loc[:,'intercept'] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.loc[vif['Variable']!='intercept'].sort_values('VIF', ascending=False).reset_index(drop=True)
    return vif



def make_countplots(df: pd.DataFrame, figsize=(12, 15)):
    
    """
    Takes a dataframe and creates countplots for all the columns.
    If the column has more than 5 categories, the data goes in the y axis.
    Bars are arranged in descending order based on count.
    Parameters:
    - df: DataFrame
    - figsize: Modifies the size of the plotting figure (default (12, 15))
    """
    
    num_cols = 2
    total_cols = len(df.columns)
    num_rows = (total_cols + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.5)

    for i, col in enumerate(df.columns):
        row_idx = i // num_cols
        col_idx = i % num_cols

        if df[col].nunique() > 5:
            order = df[col].value_counts().index
            sns.countplot(y=df[col], data=df, ax=axes[row_idx, col_idx], hue=df[col], palette='Set2', order=order)
        else:
            order = df[col].value_counts().index
            sns.countplot(x=df[col], data=df, ax=axes[row_idx, col_idx], hue=df[col], palette='Set2', order=order)

        axes[row_idx, col_idx].set_title(col)
        axes[row_idx, col_idx].set_xlabel('Count' if df[col].nunique() <= 5 else 'Frequency')
        axes[row_idx, col_idx].set_ylabel('Categories' if df[col].nunique() > 5 else 'Count')

    plt.show()



def ordinal_cat_conversion_dict(df: pd.DataFrame, col_convert: str, col_value: str) -> dict:
    '''
    Creates a mapping dictionary to convert an ordinal category based on the values of a numerical feature.
    Inputs:
    - df: pandas DataFrame
    - col_convert: categorical value to convert
    - col_value: numerical column which serves as a basis for the conversion
    Outputs:
    - mapping dictionary
    '''
    grouped = df.groupby(col_convert)[col_value].median().sort_values(ascending=False)
    grouped_df = pd.DataFrame(grouped)
    grouped_df['proportion'] = grouped_df[col_value] / grouped_df[col_value].min()
    conv_list = list(grouped_df.index)
    value_list = list(grouped_df['proportion'])
    mapping_dict = dict(zip(conv_list, value_list))
    
    return mapping_dict



def regression_model_trainer(models: list, X_train, y_train) -> list:
    '''
    - Takes a list of models, X_train and y_train sets
    - Fits the models according to the data
    - Outputs a list with the trained models
    '''
    
    trained_models = []
    
    for model in models:
        if model == MLPRegressor():
            model.fit(X_train, y_train, random_state=13)
            trained_models.append(model)
        else:
            model.fit(X_train, y_train)
            trained_models.append(model)
    return trained_models



def error_metrics_df(y_test, y_test_pred, y_train, y_train_pred) -> pd.DataFrame:
    '''
    - Takes y_test values y_test predicted values, y_train values, y_train predicted values
    - Calculates error metrics (MAE, MSE, RMSE, MAPE, R2)
    - Outputs a dataframe with values for each metric and for each set
    '''
    
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    test_r2score = r2_score(y_test, y_test_pred)
    test_metrics = [test_mae, test_mse, test_rmse, test_mape, test_r2score]
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    train_r2score = r2_score(y_train, y_train_pred)
    train_metrics = [train_mae, train_mse, train_rmse, train_mape, train_r2score]
    
    df = pd.DataFrame(index=['MAE', 'MSE', 'RMSE', 'MAPE', 'R2'])
    df['test_value'] = test_metrics
    df['train_value'] = train_metrics
    
    return df
