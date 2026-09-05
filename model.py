import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay, roc_curve, confusion_matrix
)

# 1. Load data
df = pd.read_csv('dataset_08_email_priority_classification.csv')

print("--- DATASET SUMMARY ---")
print("Shape:", df.shape)
print("\nTarget Balance:\n", df['target'].value_counts())

# 2. Features and target split
X = df.drop(columns=['target'])
y = df['target']

# 3. Stratified train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 7. Print Metrics
print("\n=== METRIC RESULTS ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1-score:  {f1_score(y_test, y_pred) * 100:.2f}%")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob) * 100:.2f}%")

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# 8. Feature Coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)
print("\n=== FEATURE COEFFICIENTS ===")
print(coef_df.to_string(index=False))

# 9. Visualizations
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap='Blues')
ax.set_title('Confusion Matrix')
plt.tight_layout()
plt.savefig('Figure_1.png', dpi=300)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc_score(y_test, y_prob):.2f})', color='darkorange')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_title('ROC Curve')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
plt.tight_layout()
plt.savefig('Figure_2.png', dpi=300)
plt.close()

print("\nSuccess: Figure_1.png and Figure_2.png generated.")