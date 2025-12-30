import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("data/LoanApprovalPrediction.csv")

# Rename target
df.rename(columns={"Loan_Status": "Credit_Risk"}, inplace=True)

# Drop ID
df.drop("Loan_ID", axis=1, inplace=True)

# Handle missing values
df["LoanAmount"].fillna(df["LoanAmount"].median(), inplace=True)
df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].median(), inplace=True)
df["Credit_History"].fillna(df["Credit_History"].mode()[0], inplace=True)

# Split X, y
X = df.drop("Credit_Risk", axis=1)
y = df["Credit_Risk"]

# Identify column types
categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# Model
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    class_weight="balanced",
    random_state=42
)

# Pipeline
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X, y)

# Save pipeline
joblib.dump(pipeline, "model/model.pkl")

print("✅ Model trained & saved correctly")
