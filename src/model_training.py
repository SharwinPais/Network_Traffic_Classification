import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from data_preprocessing import find_dataset, load_and_clean_data

def prepare_train_and_visualize():
    dataset_path = find_dataset()
    df, target_col = load_and_clean_data(dataset_path)

    print("\n--- Feature Engineering ---")
    X = df.drop(columns=[target_col])
    # Drop any non-numeric columns like IPs or timestamps if they snuck in
    X = X.select_dtypes(include=[np.number]) 
    y = df[target_col]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Model Training ---")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    print("\n--- Evaluation & Visualization ---")
    predictions = rf_model.predict(X_test_scaled)
    print(f"Overall Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%\n")
    
    # 1. Plot Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Network Traffic Classification - Confusion Matrix')
    plt.ylabel('Actual Traffic Type')
    plt.xlabel('Predicted Traffic Type')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('confusion_matrix.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    # 2. Plot Top 10 Feature Importances
    print("Generating Feature Importance Chart...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:10] # Grab top 10
    top_features = X.columns[indices]
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=importances[indices], y=top_features, palette='viridis', hue=top_features, legend=False)
    plt.title("Top 10 Most Important Network Flow Features")
    plt.xlabel("Relative Importance")
    plt.ylabel("Feature Name")
    plt.tight_layout()
    plt.savefig('feature_importance.pdf', format='pdf', bbox_inches='tight')
    plt.close()

    print("✅ Success! 'confusion_matrix.pdf' and 'feature_importance.pdf' have been saved to your folder.")

if __name__ == "__main__":
    prepare_train_and_visualize()