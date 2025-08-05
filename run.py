import os

print("🚀 Running full Kyphosis Tree-Forest pipeline...\n")

print("📦 Step 1: Data Preprocessing")
os.system("python src/data/preprocess_data.py")

print("\n🧠 Step 2: Feature Engineering")
os.system("python src/features/build_features.py")

print("\n🤖 Step 3: Model Training")
os.system("python src/models/train_model.py")

print("\n📊 Step 4: Model Evaluation")
os.system("python src/models/evaluate_test.py")

print("\n✅ All done. Pipeline finished successfully!")
