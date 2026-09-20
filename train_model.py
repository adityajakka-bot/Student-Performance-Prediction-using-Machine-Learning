import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

DATA_PATH="data/student_data.csv"
MODEL_PATH="models/best_model.joblib"

def main():
    os.makedirs("models",exist_ok=True); os.makedirs("outputs",exist_ok=True)
    df=pd.read_csv(DATA_PATH)
    X=df.drop(columns=["final_score"]); y=df["final_score"]
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
    models={
      "Linear Regression":Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler()),("model",LinearRegression())]),
      "Random Forest":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestRegressor(n_estimators=250,max_depth=10,random_state=42))]),
      "Gradient Boosting":Pipeline([("imputer",SimpleImputer(strategy="median")),("model",GradientBoostingRegressor(n_estimators=200,learning_rate=.05,max_depth=3,random_state=42))])
    }
    results={}; best_name=None; best_model=None; best_r2=float("-inf")
    for name,model in models.items():
        model.fit(Xtr,ytr); p=model.predict(Xte)
        m={"MAE":mean_absolute_error(yte,p),"RMSE":mean_squared_error(yte,p)**.5,"R2":r2_score(yte,p)}
        results[name]=m; print(name,m)
        if m["R2"]>best_r2: best_name,best_model,best_r2=name,model,m["R2"]
    joblib.dump(best_model,MODEL_PATH)
    pd.DataFrame(results).T.to_csv("outputs/model_comparison.csv")
    p=best_model.predict(Xte)
    plt.figure(figsize=(7,5)); plt.scatter(yte,p,alpha=.7)
    plt.xlabel("Actual Final Score"); plt.ylabel("Predicted Final Score")
    plt.title("Actual vs Predicted - "+best_name); plt.tight_layout()
    plt.savefig("outputs/actual_vs_predicted.png",dpi=150); plt.close()
    print("Best model:",best_name)
if __name__=="__main__": main()
