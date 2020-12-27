from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core.dataset import Dataset

run = Run.get_context()


def clean_data(data):
    embarked = {"C": 1, "S": 2, "Q": 3}
    # Clean and one hot encode data
    x_df = data.dropna()
    x_df.drop("name", inplace=True, axis=1)
    x_df.drop("boat", inplace=True, axis=1)
    x_df.drop("home.dest", inplace=True, axis=1)
    x_df.drop("body", inplace=True, axis=1)
    x_df.drop("ticket", inplace=True, axis=1)
    x_df.drop("fare", inplace=True, axis=1)
    x_df.drop("cabin", inplace=True, axis=1)
    x_df["sex"] = x_df.sex.apply(lambda s: 1 if s == "male" else 0)
    x_df["embarked"] = x_df.embarked.map(embarked)
    x_df["age"] = x_df.age.apply(lambda s: np.NaN if s == "?" else s)
    x_df = x_df.dropna()
    y_df = x_df.pop("survived")

    return x_df, y_df


pd.set_option('mode.chained_assignment', None)
dataset = TabularDatasetFactory.from_delimited_files("https://www.openml.org/data/get_csv/16826755/phpMYEkMl")
ds = dataset.to_pandas_dataframe()

x, y = clean_data(ds)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, shuffle=True)


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0,
                        help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))


if __name__ == '__main__':
    main()


