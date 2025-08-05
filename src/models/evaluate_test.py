# 🎯 Model Evaluation Script with Enhanced Error Handling & Visualizations
import pandas as pd
import yaml
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split

def load_config():
    """📂 Load configuration file with error handling"""
    try:
        with open("config/config.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ Error: config.yaml file not found!")
        exit(1)
    except yaml.YAMLError:
        print("❌ Error: Invalid YAML format in config file!")
        exit(1)

def main():
    print("🚀 Starting model evaluation pipeline...")
    
    # 1️⃣ Load configuration
    print("⚙️ Loading configuration...")
    config = load_config()
    
    # 2️⃣ Load and preprocess data
    print("📊 Loading and preprocessing data...")
    try:
        df = pd.read_csv(config["data"]["raw_path"])
        df["Kyphosis"] = df["Kyphosis"].map({"absent": 0, "present": 1})
        X = df.drop("Kyphosis", axis=1)
        y = df["Kyphosis"]
    except Exception as e:
        print(f"❌ Data loading error: {str(e)}")
        exit(1)

    # 3️⃣ Split data
    print("✂️ Splitting data into train/val/test sets...")
    try:
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"],
            stratify=y
        )
        val_fraction = config["data"]["val_size"] / (1 - config["data"]["test_size"])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_fraction,
            random_state=config["data"]["random_state"],
            stratify=y_temp
        )
    except Exception as e:
        print(f"❌ Data splitting error: {str(e)}")
        exit(1)

    # 4️⃣ Load models
    print("🤖 Loading trained models...")
    models = {}
    try:
        models["Decision Tree"] = joblib.load(config["models"]["dt_save_path"])
        models["Random Forest"] = joblib.load(config["models"]["rf_save_path"])
    except Exception as e:
        print(f"❌ Model loading error: {str(e)}")
        exit(1)

    # 5️⃣ Evaluate models
    print("🧪 Evaluating models...")
    metrics_dict = {}
    os.makedirs("outputs", exist_ok=True)

    for name, model in models.items():
        print(f"\n🔍 Evaluating {name}...")
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            acc = accuracy_score(y_test, y_pred)
            clf_report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Store metrics
            metrics_dict[name] = {
                "accuracy": acc,
                "classification_report": clf_report,
                "confusion_matrix": conf_matrix.tolist()
            }
            
            # Print results
            print(f"✅ {name} Test Accuracy: {acc:.4f}")
            print(f"📊 Classification Report:\n{classification_report(y_test, y_pred)}")
            
            # Plot confusion matrix
            plt.figure(figsize=(5, 4))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name} - Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            conf_plot_path = f"outputs/{name.lower().replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(conf_plot_path)
            plt.close()
            print(f"🖼️ Saved confusion matrix to: {conf_plot_path}")
            
            # Plot ROC curve if available
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                plt.figure()
                plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
                plt.plot([0, 1], [0, 1], linestyle="--")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title(f"{name} - ROC Curve")
                plt.legend(loc="lower right")
                roc_plot_path = f"outputs/{name.lower().replace(' ', '_')}_roc_curve.png"
                plt.savefig(roc_plot_path)
                plt.close()
                print(f"📈 Saved ROC curve to: {roc_plot_path}")
                
        except Exception as e:
            print(f"⚠️ Error evaluating {name}: {str(e)}")
            continue

    # 6️⃣ Save metrics
    try:
        with open(config["evaluation"]["metrics_output"], "w") as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"\n💾 Saved evaluation metrics to: {config['evaluation']['metrics_output']}")
    except Exception as e:
        print(f"❌ Error saving metrics: {str(e)}")

    print("\n🎉 Evaluation pipeline completed successfully! ✅")

if __name__ == "__main__":
    main()