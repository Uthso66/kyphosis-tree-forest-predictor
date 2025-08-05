import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_split_data(processed_dir):
    X_train = pd.read_csv(os.path.join(processed_dir, "X_train.csv"))
    X_val = pd.read_csv(os.path.join(processed_dir, "X_val.csv"))
    X_test = pd.read_csv(os.path.join(processed_dir, "X_test.csv"))
    return X_train, X_val, X_test

def scale_features(X_train, X_val, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)

    X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train_scaled, X_val_scaled, X_test_scaled

def save_features(X_train, X_val, X_test, features_dir):
    os.makedirs(features_dir, exist_ok=True)
    X_train.to_csv(os.path.join(features_dir, "X_train_scaled.csv"), index=False)
    X_val.to_csv(os.path.join(features_dir, "X_val_scaled.csv"), index=False)
    X_test.to_csv(os.path.join(features_dir, "X_test_scaled.csv"), index=False)

def main():
    config = load_config()
    data_cfg = config['data']
    
    print("🚀 [1/4] Loading split data from preprocessing... 📂")
    X_train, X_val, X_test = load_split_data(data_cfg['processed_dir'])

    print("⚙️ [2/4] Scaling numeric features (StandardScaler)... 📊")
    X_train_scaled, X_val_scaled, X_test_scaled = scale_features(X_train, X_val, X_test)

    print("💾 [3/4] Saving scaled features... 💽")
    save_features(X_train_scaled, X_val_scaled, X_test_scaled, data_cfg['features_dir'])

    print("\n🌟 [4/4] Feature building complete! ✅")
    print("🎯 All sets are ready for training 🎓")

if __name__ == "__main__":
    main()
