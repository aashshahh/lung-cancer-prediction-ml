import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def encode_columns(df):
    encoder = LabelEncoder()
    df['LUNG_CANCER'] = encoder.fit_transform(df['LUNG_CANCER'])
    df['GENDER'] = encoder.fit_transform(df['GENDER'])
    return df

def split_features(df):
    X = df.drop('LUNG_CANCER', axis=1)
    y = df['LUNG_CANCER']
    return X, y
