import os
import glob
import shutil
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# ---------------- SETTINGS ----------------
KAGGLE_SLUG = "sudhanshu2198/dating-app-behavior-dataset"
FILE_NAME = "dating_app_behavior_dataset_extended1.csv"
DATA_DIR = "data"
LOCAL_PATH = os.path.join(DATA_DIR, FILE_NAME)

COLOR_A = "#E8416F"   # rose
COLOR_B = "#1FA8A0"   # teal
PALETTE = [COLOR_A, COLOR_B, "#F08AA6", "#7AD3CD", "#A52A4D", "#0F6F6A", "#F6B8C9", "#B5E8E4"]

st.set_page_config(page_title="Dating App Behavior Dashboard", layout="wide")

st.markdown(f"""
<style>
h1, h2, h3 {{ color: {COLOR_A}; }}
[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {COLOR_A}22, {COLOR_B}22);
    border-left: 5px solid {COLOR_A};
    border-right: 5px solid {COLOR_B};
    padding: 12px; border-radius: 10px;
}}
.stTabs [aria-selected="true"] {{ color: {COLOR_B}; border-bottom-color: {COLOR_B}; }}
</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADING ----------------
def download_from_kaggle():
    """Download dataset with kagglehub and copy the CSV into ./data. Not cached."""
    try:
        import kagglehub
        folder = kagglehub.dataset_download(KAGGLE_SLUG)
        csvs = glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True)
        if not csvs:
            return None, "No CSV found in the Kaggle download."
        pick = next((c for c in csvs if os.path.basename(c).lower() == FILE_NAME.lower()), csvs[0])
        os.makedirs(DATA_DIR, exist_ok=True)
        shutil.copy(pick, LOCAL_PATH)
        return LOCAL_PATH, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner=False)
def read_csv_path(path):
    return pd.read_csv(path)


# Uploader lives OUTSIDE any cached function (this fixes CachedWidgetWarning)
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload dating app CSV (optional)", type=["csv"])

df = None
if uploaded is not None:
    df = pd.read_csv(uploaded)
elif os.path.exists(LOCAL_PATH):
    df = read_csv_path(LOCAL_PATH)
else:
    with st.spinner("Downloading dataset from Kaggle..."):
        path, err = download_from_kaggle()
    if path:
        df = read_csv_path(path)
    else:
        st.warning("Could not auto-download from Kaggle. Upload the CSV in the sidebar.")
        st.caption(f"Reason: {err}")
        st.stop()

# ---------------- CLEANING / COLUMN DETECTION ----------------
df = df.drop_duplicates().reset_index(drop=True)
id_like = [c for c in df.columns if df[c].nunique() == len(df)]
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in id_like]
cat_cols = [c for c in df.columns
            if c not in id_like and c not in num_cols and df[c].nunique() <= 30]
low_card_num = [c for c in num_cols if df[c].nunique() <= 10]
color_options = ["None"] + cat_cols + low_card_num

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.header("Filters")
fdf = df.copy()
for col in cat_cols[:4]:
    opts = sorted(fdf[col].dropna().astype(str).unique())
    chosen = st.sidebar.multiselect(col, opts, default=opts)
    fdf = fdf[fdf[col].astype(str).isin(chosen)]

if fdf.empty:
    st.error("No rows match the filters.")
    st.stop()

# ---------------- HEADER ----------------
st.title("Dating App User Behavior Analytics")
st.caption("Swipe patterns, engagement, and behavioral segmentation")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Users (rows)", f"{len(fdf):,}")
k2.metric("Columns", f"{df.shape[1]}")
if num_cols:
    k3.metric(f"Avg {num_cols[0]}", f"{fdf[num_cols[0]].mean():,.2f}")
if len(num_cols) > 1:
    k4.metric(f"Avg {num_cols[1]}", f"{fdf[num_cols[1]].mean():,.2f}")

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Distributions", "Relationships", "Segmentation"])

# ---------------- TAB 1: OVERVIEW ----------------
with tab1:
    st.subheader("Data preview")
    st.dataframe(fdf.head(100), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Missing values")
        miss = fdf.isna().sum()
        miss = miss[miss > 0]
        if miss.empty:
            st.success("No missing values.")
        else:
            fig = px.bar(miss, color_discrete_sequence=[COLOR_A], labels={"value": "Missing", "index": "Column"})
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Summary statistics")
        if num_cols:
            st.dataframe(fdf[num_cols].describe().T.round(2), use_container_width=True)

# ---------------- TAB 2: DISTRIBUTIONS ----------------
with tab2:
    col = st.selectbox("Choose a column", [c for c in df.columns if c not in id_like])
    if col in num_cols and df[col].nunique() > 10:
        fig = px.histogram(fdf, x=col, nbins=40, marginal="box", color_discrete_sequence=[COLOR_A])
        fig.update_traces(marker_line_color=COLOR_B, marker_line_width=1)
    else:
        counts = fdf[col].astype(str).value_counts().reset_index()
        counts.columns = [col, "count"]
        fig = px.bar(counts, x=col, y="count", color=col, color_discrete_sequence=PALETTE)
    st.plotly_chart(fig, use_container_width=True)

    if cat_cols:
        st.subheader("Category share")
        pie_col = st.selectbox("Pie chart column", cat_cols)
        pc = fdf[pie_col].astype(str).value_counts().reset_index()
        pc.columns = [pie_col, "count"]
        st.plotly_chart(px.pie(pc, names=pie_col, values="count", hole=0.45,
                               color_discrete_sequence=PALETTE), use_container_width=True)

# ---------------- TAB 3: RELATIONSHIPS ----------------
with tab3:
    if len(num_cols) >= 2:
        st.subheader("Correlation heatmap")
        corr = fdf[num_cols].corr()
        st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                                  color_continuous_scale=[COLOR_B, "#FFFFFF", COLOR_A]),
                        use_container_width=True)

        st.subheader("Scatter explorer")
        a, b, c = st.columns(3)
        x = a.selectbox("X axis", num_cols, index=0)
        y = b.selectbox("Y axis", num_cols, index=1)
        hue = c.selectbox("Colour by", color_options)
        sample = fdf.sample(min(len(fdf), 5000), random_state=1)
        fig = px.scatter(sample, x=x, y=y,
                         color=None if hue == "None" else sample[hue].astype(str),
                         color_discrete_sequence=PALETTE, opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least two numeric columns.")

    if cat_cols and num_cols:
        st.subheader("Average by group")
        g1, g2 = st.columns(2)
        grp = g1.selectbox("Group by", cat_cols)
        val = g2.selectbox("Average of", num_cols)
        agg = fdf.groupby(grp)[val].mean().reset_index()
        st.plotly_chart(px.bar(agg, x=grp, y=val, color=grp, color_discrete_sequence=PALETTE),
                        use_container_width=True)

# ---------------- TAB 4: SEGMENTATION ----------------
with tab4:
    if len(num_cols) >= 2:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        feats = st.multiselect("Features for segmentation", num_cols, default=num_cols[:min(6, len(num_cols))])
        k = st.slider("Number of segments", 2, 8, 4)
        if len(feats) >= 2:
            X = fdf[feats].dropna()
            if len(X) > 20000:
                X = X.sample(20000, random_state=1)
            Xs = StandardScaler().fit_transform(X)
            labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(Xs)
            pcs = PCA(n_components=2).fit_transform(Xs)
            plot_df = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1],
                                    "Segment": [f"Segment {i+1}" for i in labels]})
            if len(plot_df) > 5000:
                plot_df = plot_df.sample(5000, random_state=1)
            st.plotly_chart(px.scatter(plot_df, x="PC1", y="PC2", color="Segment",
                                       color_discrete_sequence=PALETTE, opacity=0.65),
                            use_container_width=True)

            prof = X.copy()
            prof["Segment"] = [f"Segment {i+1}" for i in labels]
            left, right = st.columns([2, 1])
            with left:
                st.subheader("Segment profile (averages)")
                st.dataframe(prof.groupby("Segment").mean().round(2), use_container_width=True)
            with right:
                st.subheader("Segment sizes")
                sizes = prof["Segment"].value_counts().reset_index()
                sizes.columns = ["Segment", "Users"]
                st.plotly_chart(px.pie(sizes, names="Segment", values="Users", hole=0.4,
                                       color_discrete_sequence=PALETTE), use_container_width=True)
        else:
            st.info("Pick at least two features.")
    else:
        st.info("Need at least two numeric columns.")