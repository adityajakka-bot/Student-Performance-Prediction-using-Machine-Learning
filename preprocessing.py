import pandas as pd
def clean_data(path):
    df=pd.read_csv(path).drop_duplicates()
    cols=df.select_dtypes(include="number").columns
    df[cols]=df[cols].fillna(df[cols].median())
    return df
if __name__=="__main__":
    print(clean_data("data/student_data.csv").head())
