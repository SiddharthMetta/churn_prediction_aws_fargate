import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline
import h2o
from h2o.automl import *
from sklearn.preprocessing import LabelEncoder


def encode_categorical_columns(categorical_columns, df):
    """
    Encodes categorical columns in the dataframe

    :param categorical_columns: list of categorical columns
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    for column in categorical_columns:
        tempdf = pd.get_dummies(df[column], prefix=column)
        df = pd.merge(
            left=df,
            right=tempdf,
            left_index=True,
            right_index=True,
        )
        df = df.drop(columns=column)
    return df


def add_time_factor(data):
    """
    Adds start_data column from the tenure field
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """
    min_tenure = min(data.tenure)
    max_tenure = max(data.tenure)
    print(min_tenure, max_tenure)
    tenure_list = list(range(min_tenure, max_tenure + 1))
    date_df = pd.DataFrame(tenure_list[::-1], columns=['tenure'])

    date_df['start_date'] = pd.to_datetime(tenure_list, unit='D', origin=pd.Timestamp('2021-06-01'))

    date_df.to_csv('start_date_from_tenure.csv', index=False)
    data = pd.merge(data, date_df, on=['tenure'], how='left')
    return data


def preprocess_training_data(csv_file):
    """
    Function preprocesses the data by removing outliers and adds other features

    :param csv_file: .csv file
    :return: pd.DataFrame: processed dataframe
    """
    orig_data = pd.read_csv(csv_file)
    data = orig_data.copy()
    data = add_time_factor(data)

    # EDA
    data['TotalCharges']= data['TotalCharges'].apply(lambda x: x if x!= ' ' else np.nan).astype(float)

    y = data['Churn']
    print(f'Percentage of Churn:  {round(y.value_counts(normalize=True)[1]*100,2)} %  --> ({y.value_counts()[1]} customers)\nNo churn: {round(y.value_counts(normalize=True)[0]*100,2)}  %  --> ({y.value_counts()[0]} customers)')

    le = LabelEncoder()
    data['Churn'] = le.fit_transform(data['Churn'])

    numerical_features_list = data.select_dtypes('number').columns
    data = data.drop(['customerID'], axis=1).copy()
    categorical_features_list = list(data.select_dtypes('object').columns)

    data = data.replace('No internet service', 'No')
    data.MultipleLines = data.MultipleLines.replace('No phone service','No')

    data.start_date = pd.to_datetime(data.start_date)
    data['day'] = data.start_date.dt.day
    data['month'] = data.start_date.dt.month
    data['weekday'] = data.start_date.dt.weekday
    data['week_of_year'] = data.start_date.dt.week
    data['year'] = data.start_date.dt.year

    data.year = data.year.astype(str)
    categorical_features_list.append('year')

    data = data[~data.TotalCharges.isna()]

    data = encode_categorical_columns(categorical_features_list, data)

    data = data.drop(['start_date'],axis=1)

    return data


def train_automl_model(data):
    """
    Main function for auto ml model training
    Saves model in the source directory

    :param data: pd.DataFrame
    :return: None

    """
    h2o.init()
    dataset = h2o.H2OFrame(data)
    train, test = dataset.split_frame([0.8], seed=42)
    print("train:%d test:%d" % (train.nrows, test.nrows))
    x = train.columns
    y = "Churn"
    x.remove(y)

    # For binary classification, response should be a factor
    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    aml = H2OAutoML(max_runtime_secs=90,
                    max_models=None,
                    seed=42,
                    project_name='churn_prediction_2',
                    sort_metric="AUC")

    aml.train(x=x, y=y, training_frame=train)

    model = aml.leader
    model_file = h2o.download_model(model)
    print(model.auc()) #0.89


if __name__ == '__main__':
    data = preprocess_training_data('churn_data.csv')
    train_automl_model(data)