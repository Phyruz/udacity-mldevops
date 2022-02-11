"""
Test and logging script for the churn_library.py script

Author: Eugenio
Date: Feb, 2022
"""
import os
import logging

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
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
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(import_data, perform_eda):
	'''
	test perform eda function
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_data: The file wasn't found")
		raise err
	
	perform_eda(df, features_to_plot = ['Customer_Age'])
	
	try:
		assert os.path.filexists('./logging/eda_report.txt')
	except AssertionError as err:
		logging.error("Testing perform_eda: Report was not generated as expected. Something went wrong")
		raise err

	try:
		assert os.path.filexists('./images/df_corr_heatmap.png')
	except AssertionError as err:
		logging.error("Testing perform_eda: Correlation heatmap was not generated as expected. Something went wrong")
		raise err

	try: 
		assert os.path.filexists('./images/customer_age_bar.png')
	except AssertionError as err:
		logging.error("Testing perform_eda: Bar plot was not generated as expected. Something went wrong")
		raise err
	
	try: 
		assert os.path.filexists('./images/customer_age_hist.png')
	except AssertionError as err:
		logging.error("Testing perform_eda: Hist plot was not generated as expected. Something went wrong")
		raise err

	try: 
		assert os.path.filexists('./images/customer_age_dist.png')
	except AssertionError as err:
		logging.error("Testing perform_eda: Dist plot was not generated as expected. Something went wrong")
		raise err


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass








