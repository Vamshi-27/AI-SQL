import pandas as pd
from sqlalchemy import create_engine

def load_csv(file):
    return pd.read_csv(file)

def clean_column_names(df):
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
    return df

def save_to_sqlite(df, db_path="db/data.db", table_name="data"):
    engine = create_engine(f"sqlite:///{db_path}")
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    return engine