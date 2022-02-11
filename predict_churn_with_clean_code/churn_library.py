"""
This library contains all the functions needed to:
* Perform an EDA on a given dataset
* Pre-process and feature engineer a given dataset
* Apply ML models to a given dataset predict a target variable

Author: Eugenio
Date: Feb, 2022
 """

import os
import io
import logging
from typing import List, Tuple
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
from pandas import DataFrame, Series
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth: str) -> DataFrame:
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        assert pth.lower().endswith(('.csv'))
        df = pd.read_csv(os.path.abspath(pth))

    except AssertionError as err:
        logging.error('Only .csv files are currently supported')
        raise err

    return df


def df_report(df: DataFrame, pth: str = "./logging/eda_report.txt") -> None:
    """
    Log most important statistics and information of a DataFrame
    input
           df: pandas dataframe
           pth: a path where report is saved
    """
    # convert df.info into a string using io.StringIO
    # see
    # https://stackoverflow.com/questions/39440253/how-to-return-a-string-from-pandas-dataframe-info
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    nans_str = df.isnull().sum().to_string()
    head_str = df.head().to_string()
    describe_str = df.describe().to_string()

    _dict = {"Dataset Info": info_str,
             "Number of NaNs per feature": nans_str,
             "First 10 rows": head_str,
             "Dataset stats": describe_str}

    encoded_str = "".join(
        f"\n{key}:\n{value}\n" for key,
        value in _dict.items())

    with open(pth, "w+",
              encoding="utf-8") as file:

        file.write(encoded_str)


def eda_plot(
        df: DataFrame,
        feature: str,
        kind: str = "hist",
        **kwargs) -> plt.Figure:
    """
    helper function to plot a specific feature of a dataframe
    input:
            df: pandas dataframe
            feature: feature of the dataframe to be plotted
            kind: type of plot. Choose between histogram ("hist"), bar plot ("bar"). \
                distribution plot ("dist") or heatmap ("heatmap")
            **kwargs: additional key arguments of specific plots according \
                to matplotlib or sns libraries
    output:
            figure: matplot figure
    """
    fig = plt.figure(figsize=(20, 10), **kwargs)

    if kind == "hist":
        df[feature].hist(**kwargs)
    elif kind == "bar":
        df[feature].value_counts('normalise').plot(kind='bar', **kwargs)
    elif kind == "dist":
        try:
            sns.distplot(df[feature], **kwargs)
        except ValueError:
            logging.error(
                f'Feature {feature} is categorical. A distribution plot cannot be generated')
            return

    elif kind == "heatmap":
        sns.heatmap(
            df.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2,
            **kwargs)
    else:
        raise ValueError(
            "Please choose plot kind amongst the following options \
                 ['hist', 'bar', 'dist', 'heatmap']")

    return fig


def perform_eda(df: DataFrame,
                features_to_plot: List[str] = None,
                generate_report: bool = True,
                output_pth: str = './images') -> None:
    '''
    perform eda on df, create an automatic report and save figures to images folder
    input:
            df: pandas dataframe
            generate_report: generate an automatic report of dataframe. Default True
            features_to_plot:
            output_pth
    output:
            None
    '''
    if generate_report:
        df_report(df)

    if not features_to_plot:
        features_to_plot = df.columns

    for feature in features_to_plot:
        for kind in ['hist', 'bar', 'dist']:
            plot = eda_plot(df, feature, kind=kind)
            try:
                plot.savefig(os.path.join(output_pth, f'{feature.lower()}_{kind}.png'))
            except AttributeError:
                continue
            finally:
                plt.close()  # save memory

    heatmap_plot = eda_plot(df, feature, kind='heatmap')
    heatmap_plot.savefig(os.path.join(output_pth, f'df_corr_heatmap.png'))
    plt.close()  # save memory


def encoder_helper(
        df: DataFrame,
        category_lst: List[str],
        origin: str,
        response: str = '') -> DataFrame:
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            origin: original column (pandas series) to be encoded
            response: string of response name [optional argument that could be used \
                for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_lst:
        grouped_df = df.groupby(col).mean()[origin].rename(f'{col}_{response}')
        df = df.join(grouped_df, on=col)

    return df


def perform_feature_engineering(
        df_in: DataFrame,
        target: DataFrame or Series,
        **kwargs) -> Tuple[DataFrame]:
    '''
    input:
              input: pandas dataframe. Input variables to ingest into model
              target: pandas dataframe or series. Target variable

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    return train_test_split(df_in, target, **kwargs)


def classification_report_image(y_train: DataFrame,
                                y_test: DataFrame,
                                y_train_preds: DataFrame,
                                y_test_preds: DataFrame,
                                model_name: str = 'unnamed',
                                output_pth: str = './images') -> None:
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds: training predictions from model
            y_test_preds: test predictions from model
            model_name: model name to print in report and filename

    output:
             None
    '''

    plt.rc('figure', figsize=(8, 8))
    plt.text(0.01, 1.25, str(f'{model_name} Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str(f'{model_name} Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')

    # Save figure
    model_name = '_'.join(model_name.split(
        ' ')).lower()  # appropriate filename
    plt.savefig(
        os.path.join(
            output_pth,
            f'classification_report_{model_name}.png'))
    plt.close()  # save memory


def feature_importance_plot(
        model: object,
        X_data: DataFrame,
        output_pth: str = './images',
        model_name: str = None) -> None:
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
            model_name: model name to print in report and filename

    output:
             None
    '''

    if not model_name:
        model_name = type(model).__name__

    # Calculate feature importances
    try:
        importances = model.feature_importances_
    except AttributeError as err:
        logging.error('Model has no feature_importance_ attributed')
        raise err

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save Figure
    model_name = '_'.join(model_name.split(
        ' ')).lower()  # appropriate filename
    plt.savefig(
        os.path.join(
            output_pth,
            f'feature_importance_plot_{model_name}.png'))
    plt.close()  # save memory


def save_model(model: object, model_name: str = None,
               output_pth: str = './models') -> None:
    """
    Save sklearn model into path as .pkl file
    input:
              model: model to save
              output_pth: folder where model is save
    output:
              None
    """
    if not model_name:
        model_name = type(model).__name__

    joblib.dump(model, os.path.join(output_pth, f'{model_name}.pkl'))


def load_model(pth: str) -> object:
    """
    Save sklearn model into path as .pkl file
    input:
            pth: file path where model is stored
    output:
            model: model to load
    """

    return joblib.load(pth)


def roc_curve_plot(model_lst: List[object],
                   X_data: DataFrame,
                   y_data: DataFrame,
                   output_pth: str = './images') -> None:
    """
    Save individual and combined roc curve models into path images
    input:
            model_lst: list of models
            X_data: input data
            y_data: target data
            output_pth: folder where images are stored
    output:
            model: model to load
    """
    # Single plots
    fig_comb = plt.figure(figsize=(15, 8))
    ax_comb = plt.gca()

    for model in model_lst:
        model_name = type(model).__name__

        # Create Figure
        fig = plt.figure(figsize=(15, 8))
        ax = plt.gca()
        model_plot = plot_roc_curve(model, X_data, y_data)
        model_plot.plot(ax=ax)
        model_plot.plot(ax=ax_comb, alpha=0.8)

        # Save
        fig.savefig(os.path.join(output_pth, f'roc_curve_{model_name}.png'))

    # Combined plots
    fig_comb.savefig(os.path.join(output_pth, f'roc_curves_all_models.png'))
    plt.close()


def train_models(X_train: DataFrame,
                 X_test: DataFrame,
                 y_train: DataFrame,
                 y_test: DataFrame) -> None:
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # grid search
    print('Model definition started...')
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
        'n_estimators': [200]  # , 500],
        # 'max_features': ['auto', 'sqrt'],
        # 'max_depth' : [4,5,100],
        # 'criterion' :['gini', 'entropy']
    }

    print('Random Forest fit started...')
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=2,
        verbose=3)
    cv_rfc.fit(X_train, y_train)

    print('Linear Regression fit started...')
    lrc.fit(X_train, y_train)

    # Save Model
    print('Linear Regression fit started...')
    save_model(lrc, 'lrc')
    save_model(cv_rfc.best_estimator_, 'cv_rfc_best',)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Report
    print('Classification Report is being printed.')
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        'Random Forest')
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        'Linear Regression')

    # ROC curve plot
    print('ROC curves are being generated.')
    roc_curve_plot([lrc, cv_rfc.best_estimator_], X_test, y_test)

    # Treen Explainer
    print('Tree Explainer is being generated and saved.')
    #ax = plt.gca()
    #explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    #shap_values = explainer.shap_values(X_test)
    #shap.summary_plot(shap_values, X_test, plot_type="bar", ax=ax)
    # plt.savefig('./images/rf_shap_explainer.png')
    # plt.close()

    # Feature Importance
    print('Feature importance plot is being generated and saved.')
    feature_importance_plot(cv_rfc.best_estimator_, X_test)


if __name__ == '__main__':
    df = import_data('./data/bank_data.csv')
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    perform_eda(
        df,
        features_to_plot=[
            'Churn',
            'Customer_Age',
            'Marital_Status',
            'Total_Trans_Ct'],
        output_pth='./images/eda')
    y = df['Churn']
    X = pd.DataFrame()
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    df = encoder_helper(df, cat_columns, 'Churn', 'Churn')
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        X, y, test_size=0.3, random_state=42)
    train_models(X_train, X_test, y_train, y_test)
