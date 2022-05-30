import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from train import *
import h2o
from h2o.automl import *


def preprocess_test_df(test_data):
    """
    Tranforms the test dataframe into suitable format for the model file to read

    :param test_data: pd.DataFrame
    :return: pd.DataFrame
    """
    encoded_column_names = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges',
                            'day', 'month', 'weekday', 'week_of_year',
                            'gender_Female', 'gender_Male', 'Partner_No',
                            'Partner_Yes', 'Dependents_No', 'Dependents_Yes',
                            'PhoneService_No', 'PhoneService_Yes', 'MultipleLines_No',
                            'MultipleLines_Yes', 'InternetService_DSL',
                            'InternetService_Fiber optic', 'InternetService_No',
                            'OnlineSecurity_No', 'OnlineSecurity_Yes', 'OnlineBackup_No',
                            'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_Yes',
                            'TechSupport_No', 'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_Yes',
                            'StreamingMovies_No', 'StreamingMovies_Yes', 'Contract_Month-to-month',
                            'Contract_One year', 'Contract_Two year', 'PaperlessBilling_No',
                            'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)',
                            'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
                            'PaymentMethod_Mailed check', 'year_2021']

    test_data = test_data.drop(['customerID', 'Churn'], axis=1)
    date_df = pd.read_csv('start_date_from_tenure.csv')

    test_df = pd.merge(test_data, date_df, on=['tenure'], how='left')

    test_df = test_df.replace('No internet service', 'No')
    test_df.MultipleLines = test_df.MultipleLines.replace('No phone service', 'No')
    test_df['TotalCharges'] = test_df['TotalCharges'].astype('float')
    test_df.start_date = pd.to_datetime(test_df.start_date)
    test_df['day'] = test_df.start_date.dt.day
    test_df['month'] = test_df.start_date.dt.month
    test_df['weekday'] = test_df.start_date.dt.weekday
    test_df['week_of_year'] = test_df.start_date.dt.week
    test_df['year'] = test_df.start_date.dt.year

    test_df.year = test_df.year.astype(str)
    test_df = test_df.drop('start_date', axis=1)
    categorical_features_list = list(test_df.select_dtypes('object').columns)
    test_df = encode_categorical_columns(categorical_features_list, test_df)

    for i in encoded_column_names:
        if i not in test_df.columns:
            test_df[i] = np.uint8(0)
    test_df = test_df[encoded_column_names]
    return test_df


def predict(df_input):
    """
    Main function for retrieving predictions
    :param df_input: pd.DataFrame
    :return: dictionary with predictions
    """
    df_input.columns =  df_input.columns.str[:-2]
    test_df = preprocess_test_df(df_input)
    h2o.init()
    model = h2o.upload_model('StackedEnsemble_AllModels_1_AutoML_2_20220530_115137')
    test_h2o = h2o.H2OFrame(test_df)
    output_df = model.predict(test_h2o)
    result = output_df.as_data_frame().predict.values[0]
    d = {"1": "Yes", "0": "No"}
    return {"prediction": d[str(result)]}

