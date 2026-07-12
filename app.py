import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Customer Segmentation — Live Demo", page_icon="🧩", layout="wide")

st.title("🧩 Customer Segmentation — Live Demo")
st.caption(
    "Synthetic customer data only — no real business or customer information. "
    "Demonstrates EDA, feature scaling, and unsupervised clustering, end to end and interactive."
)

rng = np.random.default_rng(21)

@st.cache_data
def generate_customers(n=600):
    # three latent segments with overlapping but distinct behavior
    segment = rng.choice(["budget", "regular", "premium"], n, p=[0.40, 0.40, 0.20])
    base = {"budget": (25, 3), "regular": (65, 8), "premium": (150, 25)}
    annual_spend = np.array([rng.normal(*base[s]) for s in segment]).clip(5)

    freq_base = {"budget": (3, 1.5), "regular": (10, 3), "premium": (18, 4)}
    purchase_frequency = np.array([rng.normal(*freq_base[s]) for s in segment]).clip(1)

    recency_base = {"budget": (60, 20), "regular": (25, 10), "premium": (8, 4)}
    days_since_last_purchase = np.array([rng.normal(*recency_base[s]) for s in segment]).clip(1)

    tenure_days = rng.integers(30, 1800, n)

    df = pd.DataFrame({
        "annual_spend": annual_spend.round(2),
        "purchase_frequency": purchase_frequency.round(1),
        "days_since_last_purchase": days_since_last_purchase.round(0),
        "tenure_days": tenure_days,
        "true_segment": segment,  # kept for reference, not used in clustering
    })
    return df

df = generate_customers()

tab1, tab2, tab3 = st.tabs(["1. EDA", "2. Scaling", "3. Clustering"])

with tab1:
    st.subheader("Descriptive statistics")
    st.dataframe(df.describe().round(2), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Annual spend distribution")
        hist_data = pd.cut(df["annual_spend"], bins=20).value_counts().sort_index()
        hist_df = pd.DataFrame({"count": hist_data.values}, index=[str(i) for i in hist_data.index])
        st.bar_chart(hist_df)
    with right:
        st.subheader("Purchase frequency vs spend")
        st.scatter_chart(df, x="purchase_frequency", y="annual_spend")

with tab2:
    st.subheader("Feature scaling")
    st.markdown(
        "Clustering algorithms are distance-based, so features on very different scales "
        "(spend in dollars vs. frequency in purchase counts) would otherwise let the largest-magnitude "
        "feature dominate the distance calculation. All numeric features are standardized before clustering."
    )
    feature_cols = ["annual_spend", "purchase_frequency", "days_since_last_purchase", "tenure_days"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)
    st.dataframe(scaled_df.head(10), use_container_width=True)

with tab3:
    st.subheader("Cluster the customers")
    algo = st.selectbox("Choose algorithm", ["K-Means", "DBSCAN"])

    if algo == "K-Means":
        k = st.slider("Number of clusters (k)", 2, 8, 3)
    else:
        eps = st.slider("DBSCAN eps (neighborhood radius)", 0.2, 2.0, 0.6, 0.1)
        min_samples = st.slider("DBSCAN min_samples", 3, 20, 8)

    if st.button("🚀 Run clustering"):
        if algo == "K-Means":
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X_scaled)
        else:
            model = DBSCAN(eps=eps, min_samples=min_samples)
            labels = model.fit_predict(X_scaled)

        df_result = df.copy()
        df_result["cluster"] = labels.astype(str)
        st.session_state["cluster_result"] = df_result
        st.session_state["cluster_labels"] = labels

        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int(np.sum(labels == -1)) if -1 in labels else 0
        st.success(f"Found {n_clusters_found} clusters" + (f" ({n_noise} points marked as noise)" if n_noise else ""))

    if "cluster_result" in st.session_state:
        df_result = st.session_state["cluster_result"]
        labels = st.session_state["cluster_labels"]

        valid_mask = labels != -1
        if valid_mask.sum() > 1 and len(set(labels[valid_mask])) > 1:
            score = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
            st.metric("Silhouette score", f"{score:.3f}", help="Ranges -1 to 1. Higher means better-separated, more cohesive clusters.")

        st.subheader("Clusters visualized (PCA-reduced to 2D)")
        pca = PCA(n_components=2, random_state=42)
        coords = pca.fit_transform(X_scaled)
        plot_df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1], "Cluster": df_result["cluster"]})
        st.scatter_chart(plot_df, x="PC1", y="PC2", color="Cluster")

        st.subheader("Cluster profiles (mean feature values)")
        profile = df_result.groupby("cluster")[["annual_spend", "purchase_frequency", "days_since_last_purchase", "tenure_days"]].mean().round(1)
        st.dataframe(profile, use_container_width=True)

with st.expander("How this maps to production work"):
    st.markdown("""
This demo clusters on four behavioral features for clarity. In a real client engagement, this
typically extends to a broader RFM (Recency, Frequency, Monetary) feature set, elbow-method or
silhouette-based selection of the optimal cluster count, and a final deliverable pairing each
cluster with a named business segment and a recommended marketing or retention action.
""")

st.caption("Built by Ghulam Nasir — Machine Learning Engineer · Python, scikit-learn, Unsupervised Learning")
