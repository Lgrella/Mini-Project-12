"""runs a simple machine learning expirement"""
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlflow.models import infer_signature
import numpy as np

def main():
    """runs a basic multiple linear regression model predicting bike share count from weather situation variables
    and logs it with mlflow"""
    df = pd.read_csv("day.csv", delimiter=",")
    df['demand'] = np.where(df['cnt']> 4000, 1, 0)
    df2 = pd.get_dummies(df, columns=['weathersit'], drop_first=True)

    #model2 = smf.ols('cnt ~  C(weathersit) + temp + hum + windspeed',data = df).fit()

    features = df2[['temp', 'hum', 'windspeed', 'weathersit_2', 'weathersit_3']]
    target = df2['demand']

    #75 train/25 test split
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=1
    )

    params = {"solver": "lbfgs", "max_iter": 1000, "multi_class": "auto", "random_state": 8888}

    model = LogisticRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    with mlflow.start_run():
        mlflow.log_param("model", "MultLinRegression")
        mlflow.log_metric("accuracy", accuracy)
        mlflow.set_tag("Training Info", "Basic Mult Linear Reg model on day.csv")
        #mlflow.sklearn.log_model(model, "mlruns/0")
        signature = infer_signature(X_train, model.predict(X_train))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="DAY_model",
            signature=signature,
            input_example=X_train,
            registered_model_name="MLflow-example",
        )


if __name__ == "__main__":
    main()
