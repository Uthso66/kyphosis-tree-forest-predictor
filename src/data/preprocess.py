import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_raw_data(path):
    return pd.read_csv(path)

def encode_target(df):
    df['Kyphosis'] = df['Kyphosis'].map({'present': 1, 'absent': 0})
    return df

def split_data(df, test_size, val_size, random_state):
    X = df.drop('Kyphosis', axis=1)
    y = df['Kyphosis']
    
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    val_fraction = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=random_state, stratify=y_temp
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def save_split_data(X_train, X_val, X_test, y_train, y_val, y_test, processed_dir):
    os.makedirs(processed_dir, exist_ok=True)
    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_val.to_csv(os.path.join(processed_dir, "X_val.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_val.to_csv(os.path.join(processed_dir, "y_val.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)

def main():
    config = load_config()
    data_cfg = config['data']
    
    print("🚀 [1/4] Loading raw data... 📂")
    df = load_raw_data(data_cfg['raw_path'])

    print("🧠 [2/4] Encoding target labels... 🔄")
    df = encode_target(df)

    print("✂️ [3/4] Splitting data into train/val/test sets...")
    splits = split_data(
        df,
        test_size=data_cfg['test_size'],
        val_size=data_cfg['val_size'],
        random_state=data_cfg['random_state']
    )

    print("💾 [4/4] Saving processed data splits...")
    save_split_data(*splits, processed_dir=data_cfg['processed_dir'])

    print("\n🌈✨ [SUCCESS] Preprocessing complete! ✅")
    print("🎯 Ready for model training! 🚀")

if __name__ == "__main__":
    main()