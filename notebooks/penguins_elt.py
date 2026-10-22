# -*- coding: utf-8 -*-
"""
Section 3 - ELT with WebHDFS InsecureClient (script version of the notebook)
Run on Windows (Jupyter / Python 3) against the HDP sandbox.
pip install hdfs pandas
"""
import io
import pandas as pd
from hdfs import InsecureClient

# HDP 2.6.x NameNode web port = 50070 ; HDP 3.x = 9870
WEBHDFS_URL = "http://localhost:9870"   # HDP 2.6 sandbox: http://sandbox-hdp.hortonworks.com:50070
HDFS_USER = "maria_dev"
RAW_PATH = "/user/maria_dev/datasets/penguins.csv"
CLEAN_DIR = "/user/maria_dev/datasets/clean"
CLEAN_PATH = CLEAN_DIR + "/penguins_clean.csv"

NUMERIC_COLS = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]


def extract(client):
    with client.read(RAW_PATH, encoding="utf-8") as reader:
        return pd.read_csv(reader, na_values=["NA", ""])


def analyze(df):
    print("Shape:", df.shape)
    print(df.dtypes)
    df.info()
    print(df.describe(include="all").T)
    missing = df.isna().sum().to_frame("missing")
    missing["missing_%"] = (missing["missing"] / len(df) * 100).round(2)
    print(missing)
    print("Duplicate rows:", df.duplicated().sum())


def clean(df):
    df = df.copy()
    df = df.drop_duplicates()
    # 1) Drop records where ALL body measurements are missing (no useful info)
    df = df.dropna(subset=NUMERIC_COLS, how="all")
    # 2) Impute remaining numeric gaps with the species median
    for col in NUMERIC_COLS:
        df[col] = df.groupby("species")[col].transform(lambda s: s.fillna(s.median()))
    # 3) Standardise and impute sex with the most frequent value per species
    df["sex"] = df["sex"].str.strip().str.lower()
    df["sex"] = df.groupby("species")["sex"].transform(
        lambda s: s.fillna(s.mode().iloc[0]))
    # 4) Convert body mass grams -> kilograms
    df["body_mass_kg"] = (df["body_mass_g"] / 1000.0).round(3)
    df = df.drop(columns=["body_mass_g"])
    df["year"] = df["year"].astype(int)
    cols = ["species", "island", "bill_length_mm", "bill_depth_mm",
            "flipper_length_mm", "body_mass_kg", "sex", "year"]
    return df[cols].reset_index(drop=True)


def load(client, df):
    client.makedirs(CLEAN_DIR)
    client.write(CLEAN_PATH, data=df.to_csv(index=False), encoding="utf-8", overwrite=True)
    print("Uploaded:", CLEAN_PATH)
    print(client.list(CLEAN_DIR, status=True))


if __name__ == "__main__":
    client = InsecureClient(WEBHDFS_URL, user=HDFS_USER)
    print(client.list("/user/maria_dev/datasets"))
    raw = extract(client)
    analyze(raw)
    clean_df = clean(raw)
    print("After cleaning:", clean_df.shape)
    print(clean_df.isna().sum())
    load(client, clean_df)
