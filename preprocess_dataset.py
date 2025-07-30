import os
from typing import List

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from utils_and_constants import (
    DROP_COLNAMES,
    PROCESSED_DATASET,
    RAW_DATASET,
    TARGET_COLUMN,
)


def read_dataset(filename: str) -> pd.DataFrame:
    """
    Reads the raw data file and returns a pandas dataframe.

    Parameters:
    filename (str): raw data filename
    """
    return pd.read_csv(filename)


def target_encode_categorical_features(
    df: pd.DataFrame, categorical_columns: List[str], target_column: str
) -> pd.DataFrame:
    """
    Target encodes the categorical features of the dataframe
    (http://www.saedsayad.com/encoding.htm)


    Parameters:
    df (pd.Dataframe): Pandas dataframe containing features and targets
    categorical_columns (List[str]): categorical column names that will be target encoded
    target_column (str): name of target column

    Returns:
    pd.Dataframe: Target encoded dataframe
    """
    encoded_data = df.copy()

    # Iterate through categorical columns
    for col in categorical_columns:
        # Calculate mean target value for each category
        encoding_map = df.groupby(col)[target_column].mean().to_dict()

        # Apply target encoding
        encoded_data[col] = encoded_data[col].map(encoding_map)

    return encoded_data


def impute_and_scale_data(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes numerical data to its mean value
    and then scales the data to a normal distribution
    Parameters:
    df_features (pd.DataFrame): DataFrame containing the features to be imputed and scaled.
    Returns:
    pd.Dataframe: Imputed and Scaled dataframe
    """
    # Create a copy to avoid modifying the original DataFrame
    df_processed = df_features.copy()
    # Impute data with mean strategy
    imputer = SimpleImputer(strategy="mean")
    df_processed[df_processed.columns] = imputer.fit_transform(df_processed)
    # Scale and fit with zero mean and unit variance
    scaler = StandardScaler()
    df_processed[df_processed.columns] = scaler.fit_transform(df_processed)
    return df_processed

def main():
    # Read data
    weather = read_dataset(filename=RAW_DATASET)

    # Preprocess data
    weather = weather.drop(columns=DROP_COLNAMES)
    if weather[TARGET_COLUMN].dtype == "object":
        weather[TARGET_COLUMN] = weather[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    # Target encode categorical columns
    # results in all columns becoming numerical
    categorical_columns = weather.select_dtypes(include=[object]).columns.to_list()
    weather = target_encode_categorical_features(
        df=weather, categorical_columns=categorical_columns, target_column=TARGET_COLUMN
    )

    # Impute and scale features
    weather_features_processed = impute_and_scale_data(
        weather.drop(columns=TARGET_COLUMN, axis=1)
    )

    # Write processed dataset
    weather_labels = weather[TARGET_COLUMN]
    weather = pd.concat([weather_features_processed, weather_labels], axis=1)
    os.makedirs(os.path.dirname(PROCESSED_DATASET), exist_ok=True)
    weather.to_csv(PROCESSED_DATASET, index=False)


if __name__ == "__main__":
    main()
