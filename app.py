
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

st.set_page_config(
    page_title="Network Anomaly Detection",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI-Based Network Traffic Anomaly Detection")
st.write(
    "A Machine Learning application for identifying "
    "unusual patterns in network traffic."
)

st.sidebar.header("Model Configuration")

samples = st.sidebar.slider(
    "Number of network records",
    min_value=200,
    max_value=2000,
    value=1000,
    step=100
)

contamination = st.sidebar.slider(
    "Expected anomaly proportion",
    min_value=0.01,
    max_value=0.20,
    value=0.05,
    step=0.01
)


@st.cache_data
def generate_data(n):
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
        ),
        "protocol_code": np.random.choice(
            [0, 1, 2], normal_count
        )
    })

    anomaly = pd.DataFrame({
        "duration": np.random.exponential(15, anomaly_count),
        "bytes_sent": np.random.normal(3000, 800, anomaly_count),
        "packets": np.random.normal(300, 50, anomaly_count),
        "failed_logins": np.random.poisson(8, anomaly_count),
        "destination_port": np.random.choice(
            [22, 23, 4444, 9999], anomaly_count
        ),
        "protocol_code": np.random.choice(
            [0, 1, 2], anomaly_count
        )
    })

    data = pd.concat(
        [normal, anomaly],
        ignore_index=True
    )

    return data.sample(frac=1, random_state=42).reset_index(drop=True)


data = generate_data(samples)

features = [
    "duration",
    "bytes_sent",
    "packets",
    "failed_logins",
    "destination_port",
    "protocol_code"
]

model = IsolationForest(
    contamination=contamination,
    random_state=42,
    n_estimators=100
)

predictions = model.fit_predict(data[features])

data["Prediction"] = np.where(
    predictions == -1,
    "Anomaly",
    "Normal"
)

data["Anomaly Score"] = -model.decision_function(
    data[features]
)

total = len(data)
anomalies = (data["Prediction"] == "Anomaly").sum()
normal = total - anomalies

st.subheader("Network Traffic Analysis")

col1, col2, col3 = st.columns(3)

col1.metric("Total Records", total)
col2.metric("Normal Traffic", normal)
col3.metric("Detected Anomalies", anomalies)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Traffic Classification")
    chart_data = data["Prediction"].value_counts()
    st.bar_chart(chart_data)

with right:
    st.subheader("Network Traffic Overview")
    st.scatter_chart(
        data,
        x="bytes_sent",
        y="packets",
        color="Prediction"
    )

st.divider()

st.subheader("Detected Network Records")

filter_option = st.selectbox(
    "Filter traffic",
    ["All", "Normal", "Anomaly"]
)

if filter_option == "All":
    filtered_data = data
else:
    filtered_data = data[
        data["Prediction"] == filter_option
    ]

st.dataframe(filtered_data, use_container_width=True)

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Analysis Report",
    data=csv,
    file_name="network_anomaly_report.csv",
    mime="text/csv"
)

st.info(
    "Educational demonstration using synthetic network traffic. "
    "This application does not analyse live network packets."
)
