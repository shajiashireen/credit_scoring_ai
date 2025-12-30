import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

st.set_page_config(page_title="Explainable Credit Scoring", layout="wide")
st.title("💳 Explainable Credit Scoring System")

model = joblib.load("model/model.pkl")

preprocessor = model.named_steps["preprocess"]
clf = model.named_steps["model"]

# FILE UPLOAD

uploaded_file = st.file_uploader("Upload Credit Scoring Dataset (CSV)", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV file to continue.")
    st.stop()

data = pd.read_csv(uploaded_file)

data.drop("Loan_ID", axis=1, inplace=True, errors="ignore")
data.rename(columns={"Loan_Status": "Credit_Risk"}, inplace=True)

data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].median())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].median())
data["Credit_History"] = data["Credit_History"].fillna(data["Credit_History"].mode()[0])

X = data.drop("Credit_Risk", axis=1)
y = data["Credit_Risk"]

X_transformed = preprocessor.transform(X)
feature_names = preprocessor.get_feature_names_out()

# SELECT APPLICANT

st.subheader("👤 Select Applicant")

row_id = st.slider(
    "Choose applicant index",
    min_value=0,
    max_value=len(X) - 1,
    value=0
)

# CREDIT DECISION 

instance = X.iloc[[row_id]]

prediction = model.predict(instance)[0]
proba = model.predict_proba(instance)[0]

classes = model.named_steps["model"].classes_

st.subheader("📊 Credit Decision")

st.write("Predicted class:", prediction)


label_map = {
    0: "Bad Credit (Rejected)",
    1: "Good Credit (Approved)",
    "N": "Bad Credit (Rejected)",
    "Y": "Good Credit (Approved)"
}


if prediction in [1, "Y"]:
    st.success(label_map[prediction])
else:
    st.error(label_map[prediction])


probability_output = {}
for i, cls in enumerate(classes):
    if cls in [1, "Y"]:
        probability_output["Good Credit"] = round(float(proba[i]), 3)
    else:
        probability_output["Bad Credit"] = round(float(proba[i]), 3)

st.write("Prediction Probabilities:")
st.json(probability_output)

st.subheader("🔍 SHAP & LIME Explanation")
preprocessor = model.named_steps["preprocess"]
clf = model.named_steps["model"]

X_transformed = preprocessor.transform(X)
raw_feature_names = preprocessor.get_feature_names_out()

pretty_feature_names = []

for f in raw_feature_names:
    f = f.replace("num__", "")
    f = f.replace("cat__", "")
    f = f.replace("_", " ")
    f = f.title()
    pretty_feature_names.append(f)
feature_names = pretty_feature_names

# SHAP EXPLANATION
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_transformed)

class_index = list(clf.classes_).index(1) if 1 in clf.classes_ else 1

# 🌍 GLOBAL FEATURE IMPORTANCE
st.markdown("### 🌍 Global Feature Importance")

global_exp = shap.Explanation(
    values=shap_values[:, :, class_index],
    base_values=np.repeat(
        explainer.expected_value[class_index],
        X_transformed.shape[0]
    ),
    data=X_transformed,
    feature_names=feature_names
)

fig_global = plt.figure(figsize=(10, 6))
shap.plots.bar(global_exp, max_display=15, show=False)
st.pyplot(fig_global)

# 👤 LOCAL FEATURE EXPLANATION

st.markdown("👤 Local Explanation (Selected Applicant)")

row = row_id 

local_exp = shap.Explanation(
    values=shap_values[row, :, class_index],
    base_values=explainer.expected_value[class_index],
    data=X_transformed[row],
    feature_names=feature_names
)

fig_local = plt.figure(figsize=(10, 6))
shap.plots.waterfall(local_exp, show=False)
st.pyplot(fig_local)

# LIME EXPLANATION (STABLE VERSION)

st.subheader(" LIME Explanation")

def predict_fn_lime(x):
    return clf.predict_proba(x)

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_transformed,
    feature_names=list(feature_names),
    class_names=["Bad Credit", "Good Credit"],
    mode="classification",
    categorical_features=[],
    discretize_continuous=False
)

lime_exp = lime_explainer.explain_instance(
    X_transformed[row],
    predict_fn_lime,
    num_features=10
)

fig_lime = lime_exp.as_pyplot_figure()
st.pyplot(fig_lime)
