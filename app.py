
import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import kaggle_sync
from train_model import build_features, feature_columns

DATA = Path("data")
MODELS = Path("models")

st.set_page_config(page_title="Manufacturing Predictive Maintenance", page_icon="🏭", layout="wide")


@st.cache_data
def load_tables():
    machines = pd.read_csv(DATA / "machines.csv")
    failures = pd.read_csv(DATA / "failures.csv", parse_dates=["failure_time"])
    maint = pd.read_csv(DATA / "maintenance_logs.csv", parse_dates=["start_time", "end_time"])
    shifts = pd.read_csv(DATA / "production_shifts.csv", parse_dates=["shift_date"])
    return machines, failures, maint, shifts


@st.cache_data
def load_sensor_count():
    import pyarrow.parquet as pq
    return pq.ParquetFile(DATA / "sensor_readings.parquet").metadata.num_rows


@st.cache_data
def load_machine_sensors(machine_id: str):
    return pd.read_parquet(DATA / "sensor_readings.parquet", filters=[("machine_id", "==", machine_id)])


@st.cache_data
def latest_risk_scores(_machines):
    """Score every machine on its most recent readings with the trained model."""
    model = joblib.load(MODELS / "failure_24h_model.joblib")
    df = pd.read_parquet(DATA / "sensor_readings.parquet")
    last_ts = df["timestamp"].max()
    window = df[df["timestamp"] > last_ts - pd.Timedelta(days=2)]  
    feats_df = build_features(window, _machines)
    latest = feats_df.sort_values("timestamp").groupby("machine_id").tail(1)
    latest["failure_risk"] = model.predict_proba(latest[feature_columns(latest)])[:, 1]
    return latest[["machine_id", "machine_type", "machine_status", "temperature_c", "vibration_mm_s",
                   "oil_quality_index", "failure_risk"]].sort_values("failure_risk", ascending=False), last_ts


def pull_from_kaggle(dataset_override: str):
    """Download the Kaggle dataset into data/ and reload. Returns an error message, or None on success."""
    try:
        api, dataset = kaggle_sync.connect()
        dataset = dataset_override.strip() or dataset
        with st.spinner(f"Downloading {dataset} from Kaggle..."):
            kaggle_sync.pull(api, dataset, DATA, quiet=True)
    except Exception as e:
        return f"Kaggle download failed: {e}"
    st.cache_data.clear()
    return None


with st.sidebar:
    st.header("Data source")
    dataset_override = st.text_input("Kaggle dataset", placeholder=f"<your-username>/{kaggle_sync.SLUG}",
                                     help="Leave blank to use your own dataset, or enter owner/dataset-name.")
    refresh = st.button("🔄 Refresh from Kaggle", width="stretch")

if refresh or not (DATA / "sensor_readings.parquet").exists():
    err = pull_from_kaggle(dataset_override)
    if err:
        st.error(err)
        if not (DATA / "sensor_readings.parquet").exists():
            st.info("Log in with `kaggle auth login` and publish the dataset with "
                    "`python kaggle_sync.py push -m \"first upload\"`, then refresh this page.")
            st.stop()
    else:
        st.rerun()

source = kaggle_sync.data_source(DATA)
with st.sidebar:
    if source:
        st.success(f"**Kaggle:** [{source['dataset']}](https://www.kaggle.com/datasets/{source['dataset']})  \n"
                   f"Downloaded {source['pulled_at'].replace('T', ' ')}")
    else:
        st.warning("**Local data** from `generate_dataset.py`, not downloaded from Kaggle. "
                   "Push it, then click Refresh from Kaggle.")
    if not (MODELS / "failure_24h_model.joblib").exists():
        st.caption("No trained model yet. Run `python train_model.py`.")

machines, failures, maint, shifts = load_tables()

st.title("🏭 Manufacturing Predictive Maintenance")
st.caption("Synthetic, open-source big-data dataset · 4 plants · 120 machines · 10-minute sensor telemetry")

tab_overview, tab_machine, tab_risk, tab_data = st.tabs(
    ["📊 Overview", "🔧 Machine explorer", "⚠️ Failure risk (ML)", "📁 Data & schema"])


with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sensor readings", f"{load_sensor_count():,}")
    c2.metric("Machines", len(machines))
    c3.metric("Failures", len(failures))
    c4.metric("Maintenance jobs", len(maint))
    c5.metric("Avg availability", f"{shifts['availability_pct'].mean():.1f}%")

    left, right = st.columns(2)
    fm = failures.merge(machines[["machine_id", "machine_type", "city"]], on="machine_id")
    left.plotly_chart(px.bar(fm.groupby(["machine_type", "failure_mode"]).size().reset_index(name="count"),
                             x="count", y="machine_type", color="failure_mode", orientation="h",
                             title="Failures by machine type and mode"), width="stretch")
    right.plotly_chart(px.histogram(fm, x="failure_time", color="city", nbins=40,
                                    title="Failures over time by plant"), width="stretch")

    left, right = st.columns(2)
    cost = maint.groupby("maintenance_type")["cost_inr"].agg(["sum", "count", "mean"]).reset_index()
    left.plotly_chart(px.bar(cost, x="maintenance_type", y="sum", color="maintenance_type", text_auto=".3s",
                             title="Total maintenance cost (INR): preventive vs corrective"),
                      width="stretch")
    daily = shifts.groupby("shift_date")[["units_produced", "units_rejected"]].sum().reset_index()
    daily["reject_rate_pct"] = 100 * daily.units_rejected / daily.units_produced.clip(lower=1)
    right.plotly_chart(px.line(daily, x="shift_date", y="reject_rate_pct", title="Daily reject rate (%)"),
                       width="stretch")


with tab_machine:
    c1, c2 = st.columns([1, 3])
    mtype = c1.selectbox("Machine type", sorted(machines.machine_type.unique()))
    ids = machines[machines.machine_type == mtype].machine_id.tolist()
    failed_ids = [m for m in ids if m in set(failures.machine_id)]
    mid = c1.selectbox("Machine", ids, index=ids.index(failed_ids[0]) if failed_ids else 0)
    info = machines[machines.machine_id == mid].iloc[0]
    c1.write(f"**Plant:** {info.city}, {info.state}")
    c1.write(f"**Manufacturer:** {info.manufacturer}")
    c1.write(f"**Age:** {info.age_years} years · **Criticality:** {info.criticality}")
    c1.write(f"**Failures:** {(failures.machine_id == mid).sum()}")

    s = load_machine_sensors(mid)
    days = c1.slider("Days to show", 7, int((s.timestamp.max() - s.timestamp.min()).days) + 1, 30)
    s = s[s.timestamp >= s.timestamp.max() - pd.Timedelta(days=days)]
    plot_cols = c1.multiselect("Sensors", ["temperature_c", "vibration_mm_s", "power_kw", "acoustic_db",
                                           "oil_quality_index", "pressure_bar", "rpm", "health_index"],
                               default=["temperature_c", "vibration_mm_s", "oil_quality_index", "health_index"])
    if plot_cols:
        fig = make_subplots(rows=len(plot_cols), cols=1, shared_xaxes=True, subplot_titles=plot_cols,
                            vertical_spacing=0.04)
        for i, col in enumerate(plot_cols, 1):
            fig.add_trace(go.Scattergl(x=s.timestamp, y=s[col], mode="lines", name=col,
                                       line=dict(width=1)), row=i, col=1)
        for ft in failures[(failures.machine_id == mid) & (failures.failure_time >= s.timestamp.min())].itertuples():
            fig.add_vline(x=ft.failure_time, line_color="red", line_dash="dash")
        fig.update_layout(height=230 * len(plot_cols), showlegend=False,
                          title="Red dashed lines = failures")
        c2.plotly_chart(fig, width="stretch")
    c2.dataframe(failures[failures.machine_id == mid], width="stretch", hide_index=True)


with tab_risk:
    if not (MODELS / "failure_24h_model.joblib").exists():
        st.warning("No trained model yet. Run `python train_model.py` and refresh.")
    else:
        m = json.loads((MODELS / "metrics.json").read_text())
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC (test)", m["roc_auc"])
        c2.metric("PR-AUC (test)", m["pr_auc"], help=f"Random guessing = {m['baseline_pr_auc']}")
        c3.metric("Features used", len(m["features"]))
        with st.spinner("Scoring all machines on their latest readings..."):
            risk, last_ts = latest_risk_scores(machines)
        st.subheader(f"Probability of failure in the next 24 hours (as of {last_ts:%d %b %Y %H:%M})")
        threshold = st.slider("Alert threshold", 0.1, 0.9, 0.5, 0.05)
        risk["alert"] = risk.failure_risk >= threshold
        st.metric("Machines needing attention", int(risk.alert.sum()))
        st.plotly_chart(px.bar(risk.head(20), x="machine_id", y="failure_risk", color="machine_type",
                               title="Top 20 highest-risk machines"), width="stretch")
        st.dataframe(risk.style.format({"failure_risk": "{:.1%}"}).background_gradient(
            subset=["failure_risk"], cmap="Reds"), width="stretch", hide_index=True)

with tab_data:
    st.markdown("All tables join on **machine_id**. Sensor data is in Parquet for speed; other tables are CSV.")
    table = st.selectbox("Preview table", ["machines", "failures", "maintenance_logs", "production_shifts",
                                           "sensor_readings (sample)"])
    if table == "sensor_readings (sample)":
        df = load_machine_sensors(machines.machine_id.iloc[0]).head(500)
    else:
        df = {"machines": machines, "failures": failures, "maintenance_logs": maint,
              "production_shifts": shifts}[table]
    st.write(f"{len(df):,} rows × {df.shape[1]} columns")
    st.dataframe(df.head(500), width="stretch", hide_index=True)
    st.subheader("Column types")
    st.dataframe(pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str).values}),
                 width="stretch", hide_index=True)
