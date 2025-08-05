import pandas as pd
import yaml
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

def load_config(path='config/config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def train_and_validate(model, X_train, y_train, X_val, y_val, model_name, save_path):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"✅ {model_name} Validation Accuracy: {acc:.4f}")
    print(f"📊 {model_name} Classification Report:\n{classification_report(y_val, y_pred)}")

    joblib.dump(model, save_path)
    print(f"💾 {model_name} saved to: {save_path}\n")

def main():
    config = load_config()

    features_dir = config['data']['features_dir']
    processed_dir = config['data']['processed_dir']

    # Load training and validation sets
    X_train = pd.read_csv(f"{features_dir}/X_train_scaled.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").values.ravel()

    # SMOTE Oversampling try
    smote = SMOTE(random_state=0)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    X_val = pd.read_csv(f"{features_dir}/X_val_scaled.csv")
    y_val = pd.read_csv(f"{processed_dir}/y_val.csv").values.ravel()

    # Load hyperparameters from config
    dt_params = config['models']['decision_tree']
    rf_params = config['models']['random_forest']

    # Train Decision Tree
    dt_model = DecisionTreeClassifier(class_weight='balanced', **dt_params)
    train_and_validate(
        dt_model,
        X_train_res, y_train_res,
        X_val, y_val,
        "Decision Tree",
        config['models']['dt_save_path']
    )

    # Train Random Forest
    rf_model = RandomForestClassifier(class_weight='balanced_subsample',**rf_params)
    train_and_validate(
        rf_model,
        X_train_res, y_train_res,
        X_val, y_val,
        "Random Forest",
        config['models']['rf_save_path']
    )

if __name__ == "__main__":
    main()
