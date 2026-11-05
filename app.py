
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.set_page_config(
    page_title="AI Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based Network Traffic Anomaly Detection")

st.markdown("""
An intelligent cybersecurity application using
**Machine Learning and Isolation Forest** to identify
unusual network traffic patterns.
""")

# Sidebar configuration
st.sidebar.header("Model Configuration")

contamination = st.sidebar.slider(
    "Expected anomaly proportion",
    0.01, 0.20, 0.05, 0.01
)

st.sidebar.info(
    "Upload a network traffic CSV dataset or use "
    "the built-in synthetic demonstration data."
)

# Generate demonstration dataset
def generate_data(n=1000):
    np.random.seed(42)

    normal_count = int(n * 0.95)
    anomaly_count = n - normal_count

    normal = pd.DataFrame({
        "duration": np.random.exponential(2, normal_count),
        "bytes_sent": np.random.normal(500, 100, normal_count),
        "packets": np.random.normal(50, 10, normal_count),
        "failed_logins": np.random.poisson(0.2, normal_count),
        "destination_port": np.random.choice(
            [80, 443, 53, 8080], normal_count
        )
    })

    anomaly = pd.DataFrame({
        "duration": np.random.exponential(15, anomaly_count),
        "bytes_sent": np.random.normal(3000, 800, anomaly_count),
        "packets": np.random.normal(300, 50, anomaly_count),
        "failed_logins": np.random.poisson(8, anomaly_count),
        "destination_port": np.random.choice(
            [22, 23, 4444, 9999], anomaly_count
        )
    })

    data = pd.concat(
        [normal, anomaly],
        ignore_index=True
    )

    data["Label"] = (
        ["BENIGN"] * normal_count
        + ["ATTACK"] * anomaly_count
    )

    return data.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)


# Dataset selection
st.subheader("1. Network Dataset")

uploaded_file = st.file_uploader(
    "Upload network traffic CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file, low_memory=False)
        st.success("Dataset uploaded successfully!")
        source = "Uploaded CSV dataset"
    except Exception as e:
        st.error(f"Unable to read CSV: {e}")
        st.stop()
else:
    data = generate_data()
    source = "Synthetic demonstration dataset"
    st.info("Using built-in synthetic demonstration data.")

st.write(f"Dataset source: **{source}**")

if len(data) < 2:
    st.error("Dataset must contain at least two records.")
    st.stop()

# Dataset preview
st.write("Dataset dimensions:", data.shape)

with st.expander("Preview uploaded dataset"):
    st.dataframe(data.head(20), width="stretch")

# Identify possible label column
label_candidates = [
    col for col in data.columns
    if col.strip().lower() in [
        "label", "class", "attack", "target"
    ]
]

default_label = label_candidates[0] if label_candidates else "None"

label_options = ["None"] + list(data.columns)

label_column = st.selectbox(
    "Select actual traffic label column (optional)",
    label_options,
    index=(
        label_options.index(default_label)
        if default_label in label_options else 0
    )
)

if label_column == "None":
    label_column = None

# Prepare numerical features
excluded = [label_column] if label_column else []

numeric_data = data.drop(
    columns=excluded
).select_dtypes(include=[np.number])

numeric_data = numeric_data.replace(
    [np.inf, -np.inf], np.nan
)

numeric_data = numeric_data.dropna(
    axis=1, how="all"
)

# Remove constant columns
numeric_data = numeric_data.loc[
    :, numeric_data.nunique(dropna=True) > 1
]

if numeric_data.shape[1] == 0:
    st.error(
        "No suitable numerical features found. "
        "Please upload a dataset with numerical traffic columns."
    )
    st.stop()

# Limit processing size for performance
if len(numeric_data) > 50000:
    st.warning(
        "Dataset contains more than 50,000 records. "
        "A random sample of 50,000 records will be analysed."
    )

    selected_indices = numeric_data.sample(
        n=50000, random_state=42
    ).index

    numeric_data = numeric_data.loc[selected_indices]
    data = data.loc[selected_indices]

# Model training
st.subheader("2. Machine Learning Analysis")

if st.button("Run Anomaly Detection", type="primary"):

    with st.spinner("Training Isolation Forest model..."):

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
        )

        predictions = model.fit_predict(numeric_data)

        results = data.copy()

        results["Prediction"] = np.where(
            predictions == -1,
            "Anomaly",
            "Normal"
        )

        results["Anomaly Score"] = (
            -model.decision_function(numeric_data)
        )

        st.session_state["results"] = results
        st.session_state["label_column"] = label_column

    st.success("Anomaly detection completed!")

# Display results
if "results" in st.session_state:

    results = st.session_state["results"]
    label_column = st.session_state["label_column"]

    normal_count = (
        results["Prediction"] == "Normal"
    ).sum()

    anomaly_count = (
        results["Prediction"] == "Anomaly"
    ).sum()

    st.subheader("3. Detection Results")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", len(results))
    c2.metric("Normal Traffic", normal_count)
    c3.metric("Detected Anomalies", anomaly_count)

    st.divider()

    left, right = st.columns(2)

    with left:
        st.write("### Traffic Classification")
        st.bar_chart(
            results["Prediction"].value_counts()
        )

    with right:
        st.write("### Network Traffic Overview")

        numeric_columns = results.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_columns) >= 2:
            chart_data = results.copy()
            chart_data["Traffic Type"] = (
                results["Prediction"]
            )

            st.scatter_chart(
                chart_data,
                x=numeric_columns[0],
                y=numeric_columns[1],
                color="Traffic Type"
            )

    # Optional evaluation with actual labels
    if label_column and label_column in results.columns:

        st.subheader("4. Model Evaluation")

        actual = results[label_column].astype(str).str.strip()

        actual_binary = (
            ~actual.str.lower().str.contains(
                "benign|normal", regex=True
            )
        ).astype(int)

        predicted_binary = (
            results["Prediction"] == "Anomaly"
        ).astype(int)

        if actual_binary.nunique() == 2:

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Accuracy",
                f"{accuracy_score(actual_binary, predicted_binary):.2%}"
            )

            m2.metric(
                "Precision",
                f"{precision_score(actual_binary, predicted_binary, zero_division=0):.2%}"
            )

            m3.metric(
                "Recall",
                f"{recall_score(actual_binary, predicted_binary, zero_division=0):.2%}"
            )

            m4.metric(
                "F1 Score",
                f"{f1_score(actual_binary, predicted_binary, zero_division=0):.2%}"
            )

            st.write("Confusion Matrix")
            st.write(
                pd.DataFrame(
                    confusion_matrix(
                        actual_binary, predicted_binary
                    ),
                    index=["Actual Normal", "Actual Attack"],
                    columns=["Predicted Normal", "Predicted Attack"]
                )
            )

            st.caption(
                "These metrics compare unsupervised anomaly "
                "predictions with the dataset's actual labels. "
                "They are evaluation results, not proof of "
                "production-level detection accuracy."
            )

        else:
            st.warning(
                "Evaluation requires both benign and attack records."
            )

    else:
        st.info(
            "Select a label column to calculate evaluation metrics."
        )

    st.divider()

    st.subheader("5. Detailed Detection Report")

    filter_type = st.selectbox(
        "Filter traffic",
        ["All", "Normal", "Anomaly"]
    )

    if filter_type == "All":
        filtered = results
    else:
        filtered = results[
            results["Prediction"] == filter_type
        ]

    st.dataframe(filtered, width="stretch")

    csv = results.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download Detection Report",
        data=csv,
        file_name="network_anomaly_report.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "Educational cybersecurity project | "
    "Isolation Forest | Python and Streamlit"
)
