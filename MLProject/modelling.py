import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_dataset(path):
    df = pd.read_csv(path)

    def combine_risk_category(row):
        if row["Risk_Category_Low"] == 1:
            return "Low"
        elif row["Risk_Category_Medium"] == 1:
            return "Medium"
        else:
            return "High"

    df["Risk_Category"] = df.apply(combine_risk_category, axis=1)
    df = df.drop(columns=["Risk_Category_Low", "Risk_Category_Medium"])

    return df


def train_model(data_path):
    df = load_dataset(data_path)

    X = df.drop(columns=["Risk_Category"])
    y = df["Risk_Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=None,
            random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
        mlflow.log_metric("recall_macro", report["macro avg"]["recall"])
        mlflow.log_metric("f1_macro", report["macro avg"]["f1-score"])

        mlflow.sklearn.log_model(model, "model")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        os.makedirs("artifacts", exist_ok=True)
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)

        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        fi_path = "artifacts/feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        mlflow.log_artifact(fi_path)

        print("Training selesai. Artefak tersimpan di MLflow.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_path)
