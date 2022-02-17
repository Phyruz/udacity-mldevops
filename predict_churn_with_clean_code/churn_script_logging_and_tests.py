"""
Test and logging script for the churn_library.py script

Author: Eugenio
Date: Feb, 2022
"""
import os
import logging
import pandas as pd
import numpy as np
import churn_library as cl

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w+',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''

    perform_eda(df, features_to_plot=['Customer_Age'])

    try:
        assert os.path.isfile('./logs/eda_report.txt')
        logging.info("Testing perform_eda: SUCCESS - EDA report was generated")
        os.remove('./logs/eda_report.txt')  # remote after testing
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Report was not generated as expected. Something went wrong")
        raise err

    try:
        assert os.path.isfile('./images/df_corr_heatmap.png')
        logging.info(
            "Testing perform_eda: SUCCESS - Correlation heatmap was generated")
        os.remove('./images/df_corr_heatmap.png')  # remote after testing
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Correlation heatmap was not generated as expected. Something went wrong")
        raise err

    try:
        assert os.path.isfile('./images/customer_age_bar.png')
        logging.info("Testing perform_eda: SUCCESS - Bar plot was generated")
        os.remove('./images/customer_age_bar.png')  # remote after testing
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Bar plot was not generated as expected. Something went wrong")
        raise err

    try:
        assert os.path.isfile('./images/customer_age_hist.png')
        logging.info("Testing perform_eda: SUCCESS - Hist plot was generated")
        os.remove('./images/customer_age_hist.png')  # remote after testing
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Hist plot was not generated as expected. Something went wrong")
        raise err

    try:
        assert os.path.isfile('./images/customer_age_dist.png')
        logging.info("Testing perform_eda: SUCCESS - Dist plot was generated")
        os.remove('./images/customer_age_dist.png')  # remote after testing
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: Dist plot was not generated as expected. Something went wrong")
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    df = pd.DataFrame({'categorical': ['one', 'one', 'two', 'two', 'three', 'three'],
                       'numerical': [10, 0, 5, 5, 2, 4]})

    df_post = encoder_helper(
        df,
        category_lst=['categorical'],
        origin='numerical',
        response='response')
    try:
        # check that values are equal to wanted logic
        assert np.array_equal(df_post['categorical_response'].values.T, np.array(
            [5.0, 5.0, 5.0, 5.0, 3.0, 3.0]))
        logging.info(
            "Testing encoder_helper: SUCCESS - dataframe was encoded as expected")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: test failed, dataframe was not encoded as expected")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    df_in = pd.DataFrame({'x_1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'x_2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                          'y': [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})

    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df_in[['x_1', 'x_2']], df_in['y'], test_size=0.2, shuffle=False)

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - All train and test datasets have at least one row")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: At least one of the testing or train datasets has no rows!")

    try:
        assert X_train.shape[0] + X_test.shape[0] == df_in['x_1'].shape[0]
        assert y_train.shape[0] + y_test.shape[0] == df_in['y'].shape[0]
        logging.info(
            "Testing perform_feature_engineering: SUCCESS - No rows were lost in splitting between train and test datasets")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: some rows were lost during split")

    try:
        # check that values are split according to 80/20
        # comparing values using transpose matrices here for better readibility
        assert np.array_equal(X_train.values.T, np.array(
            [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 4, 5, 6, 7, 8]]))
        assert np.array_equal(X_test.values.T, np.array([[9, 10],
                                                         [9, 10]]))
        assert np.array_equal(y_train.values.T, np.array(
            [11, 12, 13, 14, 15, 16, 17, 18]))
        assert np.array_equal(y_test.values.T, np.array([19, 20]))

        logging.info(
            "Testing perform_feature_engineering: SUCCESS - Dataset was correctly split in training and testing datasets")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Dataset was not correctly split into training and testing datasets")
        raise err


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''

    train_models(X_train, X_test, y_train, y_test)

    try:
        assert os.path.isfile('./models/lrc.pkl')
        logging.info(
            "Testing train_models: SUCCESS - Linear Regression Classifier was saved as .pkl")
        os.remove('./models/lrc.pkl')  # remove after testing
    except AssertionError as err:
        logging.error(
            "Testing train_models: Linear Regression Classifier was not saved. Something went wrong")
        raise err

    try:
        assert os.path.isfile('./models/cv_rfc_best.pkl')
        logging.info(
            "Testing train_models: SUCCESS - Random Forest Classifier was saved as .pkl")
        os.remove('./models/cv_rfc_best.pkl')  # remove after testing
    except AssertionError as err:
        logging.error(
            "Testing train_models: Random Forest Classifier was not saved. Something went wrong")
        raise err

    try:
        assert os.path.isfile(
            './images/classification_report_linear_regression.png')
        logging.info(
            "Testing train_models: SUCCESS - Linear Regression Classifier report was generated")
        # remove after testing
        os.remove('./images/classification_report_linear_regression.png')

    except AssertionError as err:
        logging.error(
            "Testing train_models: Linear Regression Classifier report was not generated as expected. Something went wrong")
        raise err

    try:
        assert os.path.isfile(
            './images/classification_report_random_forest.png')
        logging.info(
            "Testing train_models: SUCCESS - Random Forest Classifier report was generated")
        # remove after testing
        os.remove('./images/classification_report_random_forest.png')
    except AssertionError as err:
        logging.error(
            "Testing train_models: Random Forest Classifier report was not generated as expected. Something went wrong")
        raise err


if __name__ == "__main__":
    test_import(cl.import_data)

    df = cl.import_data("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    test_eda(cl.perform_eda, df)
    test_encoder_helper(cl.encoder_helper)
    test_perform_feature_engineering(cl.perform_feature_engineering)

    y = df['Churn']
    X = pd.DataFrame()

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    df = cl.encoder_helper(df, cat_columns, 'Churn', 'Churn')

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

    X = df[keep_cols]

    X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
        X, y, test_size=0.3, random_state=42)

    test_train_models(cl.train_models, X_train, X_test, y_train, y_test)
