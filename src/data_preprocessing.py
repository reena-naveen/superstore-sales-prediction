import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def load_data(file_path):
    """ Load data from a CSV file. """
    return pd.read_csv(file_path)


def handle_missing_values(data, numerical_features, categorical_features):
    """ Handle missing values in the dataset. """
    # Impute numerical features
    numerical_transformer = SimpleImputer(strategy='mean')
    # Impute categorical features
    categorical_transformer = SimpleImputer(strategy='most_frequent')
    
    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Fit and transform the data
    return pd.DataFrame(preprocessor.fit_transform(data), columns=numerical_features + categorical_features)


def encode_categorical_variables(data, categorical_features):
    """ Encode categorical variables using one-hot encoding. """
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(data[categorical_features])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    return pd.concat([data.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1).drop(categorical_features, axis=1)


def feature_scaling(data, numerical_features):
    """ Scale numerical features using StandardScaler. """
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data