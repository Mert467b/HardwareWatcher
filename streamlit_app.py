import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import sys
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Throttle Predictor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
h1, h2, h3 { font-family: 'JetBrains Mono', monospace; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib theme ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3,
})
ACCENT = "#1f77b4"; GREEN = "#2ca02c"; ORANGE = "#ff7f0e"; RED = "#d62728"

# ── Stdout capture ────────────────────────────────────────────────────────────
class _Capture(list):
    def __enter__(self):
        self._old = sys.stdout
        sys.stdout = self._buf = io.StringIO()
        return self
    def __exit__(self, *a):
        self.extend(self._buf.getvalue().splitlines())
        sys.stdout = self._old

# ── Import project modules ────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from src.models import RandomForestClassifier as RFClassifier
    from src.utils import (
        preprocess_classification_data,
        calculate_metrics,
        print_confusion_matrix,
        quick_fit,
    )
    from src.visualization import plot_confusion_matrix
    MODULES_OK = True
except ImportError as e:
    MODULES_OK = False
    IMPORT_ERROR = str(e)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _collect_importance(node, arr):
    if not isinstance(node, dict):
        return
    feat = node.get("feature")
    if feat is not None and feat < len(arr):
        arr[feat] += 1
    _collect_importance(node.get("left"),  arr)
    _collect_importance(node.get("right"), arr)

def _cm_stats(y_true, y_pred):
    with _Capture():
        return print_confusion_matrix(y_true, y_pred)

def _plot_cm(y_true, y_pred):
    """Plot confusion matrix without relying on src.visualization internals."""
    import seaborn as sns
    unique = np.unique(np.concatenate([y_true, y_pred]))
    cm = np.zeros((len(unique), len(unique)), dtype=int)
    for i, tv in enumerate(unique):
        for j, pv in enumerate(unique):
            cm[i, j] = np.sum((y_true == tv) & (y_pred == pv))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=unique, yticklabels=unique)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌡️ Throttle Predictor")
    st.caption("CPU/GPU throttle classifier")
    st.divider()

    # ── Data upload ───────────────────────────────────────────────────────────
    st.markdown("### 1 · Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                label_visibility="collapsed")

    # ── Mode ─────────────────────────────────────────────────────────────────
    st.markdown("### 2 · Mode")
    mode = st.radio(
        "mode",
        ["Auto (CPU throttle dataset)", "Custom (any dataset)"],
        label_visibility="collapsed",
        help=(
            "**Auto** uses the built-in preprocessing pipeline designed for "
            "the CPU throttle dataset collected with get_data.py.\n\n"
            "**Custom** lets you define features, target and throttle "
            "conditions for any tabular CSV."
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    st.markdown("### 3 · Model")
    task = st.selectbox("Algorithm", ["RandomForest", "XGBoost"],
                        label_visibility="collapsed")

    st.markdown("### 4 · Hyperparameters")
    if task == "RandomForest":
        n_trees           = st.slider("Trees",               10, 200,  50, 10)
        max_depth         = st.slider("Max depth",            2,  10,   4,  1)
        min_samples_leaf  = st.slider("Min samples / leaf",  10, 200,  50, 10)
        min_samples_split = st.slider("Min samples / split", 20, 300, 100, 10)
        threshold         = st.slider("Decision threshold",  0.50, 0.95, 0.75, 0.05)
    else:
        n_estimators  = st.slider("Estimators",          50, 300, 150, 10)
        xgb_depth     = st.slider("Max depth",            2,   8,   3,  1)
        lr            = st.slider("Learning rate",      0.01, 0.30, 0.10, 0.01)
        xgb_threshold = st.slider("Decision threshold", 0.50, 0.95, 0.85, 0.05)
        # Always define RF defaults (used in validation tests)
        n_trees, max_depth, min_samples_leaf, min_samples_split = 50, 4, 50, 100

    st.divider()
    run_btn = st.button("▶  Run Pipeline", use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
st.markdown("# Throttle Prediction Dashboard")

if not MODULES_OK:
    st.error(
        f"**Could not import project modules.**\n\n"
        f"Make sure `src/` is in the same folder as `streamlit_app.py`.\n\n"
        f"`{IMPORT_ERROR}`"
    )
    st.stop()

if uploaded is None:
    st.info("👈  Upload your CSV in the sidebar to get started.")
    st.markdown("### Expected columns (Auto mode)")
    sample_cols = [
        "timestamp", "session_id", "activity", "surface_type", "cpu_boost_mode",
        "cpu_temp_C", "cpu_power_W", "cpu_util_pct", "cpu_freq_MHz",
        "gpu_temp_C", "gpu_util_pct", "gpu_clock_MHz", "gpu_power_W",
    ]
    st.dataframe(pd.DataFrame(columns=sample_cols), use_container_width=True)
    st.stop()

df_raw = pd.read_csv(uploaded)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_config, tab_results, tab_tests = st.tabs([
    "📊  Data Explorer",
    "⚙️  Dataset Config",
    "🎯  Model Results",
    "🔬  Validation Tests",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — DATA EXPLORER
# ═════════════════════════════════════════════════════════════════════════════
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows",      f"{len(df_raw):,}")
    c2.metric("Columns",   len(df_raw.columns))
    c3.metric("Sessions",  df_raw["session_id"].nunique()
              if "session_id" in df_raw.columns else "—")
    c4.metric("Missing %", f"{df_raw.isnull().mean().mean()*100:.1f}%")

    st.markdown("## Raw data preview")
    st.dataframe(df_raw.head(300), use_container_width=True, height=260)

    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()

    st.markdown("## Sensor time-series")
    default_ts = [c for c in ["cpu_temp_C", "cpu_power_W", "gpu_temp_C"] if c in num_cols]
    chosen_ts  = st.multiselect("Select channels", num_cols, default=default_ts,
                                key="ts_select")
    if chosen_ts:
        palette = [ACCENT, GREEN, ORANGE, RED, "#9467bd", "#8c564b"]
        fig, axes = plt.subplots(len(chosen_ts), 1,
                                 figsize=(12, 2.5 * len(chosen_ts)), sharex=True)
        if len(chosen_ts) == 1:
            axes = [axes]
        for ax, col, color in zip(axes, chosen_ts, palette * 10):
            ax.plot(df_raw.index, df_raw[col], color=color, lw=0.8, alpha=0.9)
            ax.set_ylabel(col, fontsize=9)
        axes[-1].set_xlabel("Sample index")
        fig.tight_layout(h_pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("## Distributions")
    default_dist = [c for c in ["cpu_temp_C", "cpu_power_W", "gpu_temp_C"] if c in num_cols]
    dist_cols = st.multiselect("Columns for histogram", num_cols,
                               default=default_dist[:3], key="dist_select")
    if dist_cols:
        cols_per_row = 3
        rows = (len(dist_cols) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row,
                                 figsize=(12, 3.5 * rows))
        axes = np.array(axes).flatten()
        for i, col in enumerate(dist_cols):
            axes[i].hist(df_raw[col].dropna(), bins=50,
                         color=ACCENT, edgecolor="white", alpha=0.85)
            axes[i].set_title(col, fontsize=9)
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — DATASET CONFIG  (only shown / used in Custom mode)
# ═════════════════════════════════════════════════════════════════════════════
with tab_config:
    if mode == "Auto (CPU throttle dataset)":
        st.info(
            "You are in **Auto mode**. The built-in preprocessing pipeline is used — "
            "no configuration needed.\n\n"
            "Switch to **Custom** mode in the sidebar to configure features and "
            "target for your own dataset."
        )
    else:
        st.markdown("## Custom dataset configuration")
        st.caption(
            "Define which columns to use as features, how to build the target label, "
            "and optional feature engineering. These settings are used when you run the pipeline."
        )

        num_cols_all = df_raw.select_dtypes(include=np.number).columns.tolist()
        all_cols     = df_raw.columns.tolist()

        # ── Feature selection ─────────────────────────────────────────────────
        st.markdown("### Features")
        st.caption("Select the input columns the model will train on. Only numeric columns are supported.")
        default_feats = [c for c in num_cols_all
                         if c not in ["timestamp", "session_id", "sec_since_start", "is_clogged"]]
        selected_features = st.multiselect(
            "Input features", num_cols_all,
            default=default_feats[:15],
            key="custom_features",
        )

        st.divider()

        # ── Target definition ─────────────────────────────────────────────────
        st.markdown("### Target column")
        target_mode = st.radio(
            "How should the target be defined?",
            ["Use an existing column", "Build from threshold conditions"],
            key="target_mode",
        )

        if target_mode == "Use an existing column":
            st.caption(
                "Pick a binary column (0/1) that already exists in your CSV. "
                "Make sure it contains only 0s and 1s."
            )
            target_col = st.selectbox("Target column", all_cols, key="target_col_existing")
            prediction_window_custom = st.slider(
                "Prediction window (rows to shift target forward)",
                0, 300, 0, 5,
                help="0 = no shift (predict current state). "
                     "30 = predict 30 rows into the future.",
                key="pred_window_custom",
            )
            use_conditions = False

        else:
            st.caption(
                "Define up to two threshold conditions. "
                "A row is labelled **1 (positive)** when the condition(s) are met. "
                "The label is then shifted forward so the model predicts future events."
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Condition 1**")
                cond1_col = st.selectbox("Column",    num_cols_all,
                                         index=num_cols_all.index("cpu_temp_C")
                                         if "cpu_temp_C" in num_cols_all else 0,
                                         key="cond1_col")
                cond1_op  = st.selectbox("Operator", [">", ">=", "<", "<="],
                                         key="cond1_op")
                cond1_val = st.number_input("Threshold", value=85.0, key="cond1_val")

            with col_b:
                st.markdown("**Condition 2** (optional)")
                use_cond2  = st.checkbox("Enable condition 2", value=True, key="use_cond2")
                cond2_col  = st.selectbox("Column",   num_cols_all,
                                          index=num_cols_all.index("cpu_freq_MHz")
                                          if "cpu_freq_MHz" in num_cols_all else 0,
                                          key="cond2_col",
                                          disabled=not use_cond2)
                cond2_op   = st.selectbox("Operator", [">", ">=", "<", "<="],
                                          index=3,
                                          key="cond2_op",
                                          disabled=not use_cond2)
                cond2_val  = st.number_input("Threshold", value=2000.0,
                                             key="cond2_val",
                                             disabled=not use_cond2)

            logic_op = "AND"
            if use_cond2:
                logic_op = st.radio("Combine conditions with", ["AND", "OR"],
                                    horizontal=True, key="logic_op")

            prediction_window_custom = st.slider(
                "Prediction window (rows)",
                0, 300, 30, 5,
                help="How many rows into the future to predict. "
                     "At 1 row/sec, 30 = predict 30 seconds ahead.",
                key="pred_window_custom2",
            )
            use_conditions = True

        st.divider()

        # ── Optional feature engineering ──────────────────────────────────────
        st.markdown("### Feature engineering (optional)")
        st.caption("Automatically add derived columns to improve the model.")

        col_fe1, col_fe2 = st.columns(2)
        with col_fe1:
            add_rolling = st.checkbox("Rolling mean & std", value=True,
                                      help="Adds 30-row and 60-row rolling mean/std for selected columns.")
            if add_rolling:
                rolling_cols = st.multiselect(
                    "Columns for rolling features",
                    num_cols_all,
                    default=[c for c in ["cpu_temp_C", "cpu_power_W"] if c in num_cols_all],
                    key="rolling_cols",
                )
        with col_fe2:
            add_lag = st.checkbox("Lag features", value=True,
                                  help="Adds t-5, t-10, t-30 lag values for selected columns.")
            if add_lag:
                lag_cols = st.multiselect(
                    "Columns for lag features",
                    num_cols_all,
                    default=[c for c in ["cpu_temp_C", "cpu_power_W", "cpu_util_pct"]
                             if c in num_cols_all],
                    key="lag_cols",
                )

        st.divider()

        # ── Preview ───────────────────────────────────────────────────────────
        if st.button("👁  Preview processed data", key="preview_btn"):
            with st.spinner("Building preview…"):
                df_prev = df_raw.copy()

                # Apply conditions
                if use_conditions:
                    ops = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
                           "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}
                    c1 = ops[cond1_op](df_prev[cond1_col], cond1_val)
                    if use_cond2:
                        c2 = ops[cond2_op](df_prev[cond2_col], cond2_val)
                        throttle_now = (c1 & c2) if logic_op == "AND" else (c1 | c2)
                    else:
                        throttle_now = c1
                    df_prev["_target_now"] = throttle_now.astype(int)
                    if prediction_window_custom > 0:
                        df_prev["target"] = (
                            df_prev["_target_now"]
                            .shift(-prediction_window_custom)
                            .rolling(window=prediction_window_custom, min_periods=1).max()
                        )
                    else:
                        df_prev["target"] = df_prev["_target_now"]
                else:
                    df_prev["target"] = df_prev[target_col]
                    if prediction_window_custom > 0:
                        df_prev["target"] = df_prev["target"].shift(-prediction_window_custom)

                # Rolling
                if add_rolling and rolling_cols:
                    for col in rolling_cols:
                        df_prev[f"{col}_roll_mean_30"] = df_prev[col].rolling(30).mean()
                        df_prev[f"{col}_roll_std_30"]  = df_prev[col].rolling(30).std()
                        df_prev[f"{col}_roll_mean_60"] = df_prev[col].rolling(60).mean()

                # Lag
                if add_lag and lag_cols:
                    for col in lag_cols:
                        for lag in [5, 10, 30]:
                            df_prev[f"{col}_lag_{lag}"] = df_prev[col].shift(lag)

                df_prev = df_prev.dropna().reset_index(drop=True)

                class_counts = df_prev["target"].value_counts().sort_index()
                p1, p2 = st.columns(2)
                p1.metric("Rows after processing", f"{len(df_prev):,}")
                p2.metric("Class balance  0 : 1",
                          f"{class_counts.get(0,0):,} : {class_counts.get(1,0):,}")
                st.dataframe(df_prev.head(100), use_container_width=True, height=220)


# ═════════════════════════════════════════════════════════════════════════════
# PREPROCESSING HELPER
# ═════════════════════════════════════════════════════════════════════════════
def run_custom_preprocessing(df):
    """Build features + target from sidebar/config widget state."""
    df = df.copy()
    ops = {">": lambda a, b: a > b, ">=": lambda a, b: a >= b,
           "<": lambda a, b: a < b, "<=": lambda a, b: a <= b}

    pw = (st.session_state.get("pred_window_custom2")
          or st.session_state.get("pred_window_custom") or 0)

    if st.session_state.get("target_mode") == "Use an existing column":
        df["target"] = df[st.session_state["target_col_existing"]]
        if pw > 0:
            df["target"] = df["target"].shift(-pw)
    else:
        c1 = ops[st.session_state["cond1_op"]](
            df[st.session_state["cond1_col"]], st.session_state["cond1_val"])
        if st.session_state.get("use_cond2", False):
            c2 = ops[st.session_state["cond2_op"]](
                df[st.session_state["cond2_col"]], st.session_state["cond2_val"])
            throttle_now = (c1 & c2) if st.session_state.get("logic_op","AND") == "AND" \
                           else (c1 | c2)
        else:
            throttle_now = c1
        df["_target_now"] = throttle_now.astype(int)
        if pw > 0:
            df["target"] = (
                df["_target_now"].shift(-pw)
                .rolling(window=pw, min_periods=1).max()
            )
        else:
            df["target"] = df["_target_now"]

    # Rolling features
    if st.session_state.get("add_rolling_custom", True):
        rc = st.session_state.get("rolling_cols", [])
        for col in rc:
            if col in df.columns:
                df[f"{col}_roll_mean_30"] = df[col].rolling(30).mean()
                df[f"{col}_roll_std_30"]  = df[col].rolling(30).std()
                df[f"{col}_roll_mean_60"] = df[col].rolling(60).mean()

    # Lag features
    if st.session_state.get("add_lag_custom", True):
        lc = st.session_state.get("lag_cols", [])
        for col in lc:
            if col in df.columns:
                for lag in [5, 10, 30]:
                    df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    df = df.dropna().reset_index(drop=True)

    # Build final feature list: base selected + any newly engineered columns
    base_feats = [f for f in st.session_state.get("custom_features", [])
                  if f in df.columns]
    extra_feats = [c for c in df.columns
                   if (c.endswith(("_roll_mean_30","_roll_std_30","_roll_mean_60",
                                   "_lag_5","_lag_10","_lag_30"))
                       and c not in base_feats)]
    features = base_feats + extra_feats
    features = [f for f in features if f in df.columns]

    X = df[features].values
    y = df["target"].values.astype(int)
    return X, y, features, pw, df


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not run_btn:
        st.info("Configure your model in the sidebar and click **▶ Run Pipeline**.")
        st.stop()

    # ── Preprocess ────────────────────────────────────────────────────────────
    with st.spinner("Preprocessing data…"):
        if mode == "Auto (CPU throttle dataset)":
            with _Capture() as logs:
                try:
                    X, y, features, prediction_window, df_proc = \
                        preprocess_classification_data(df_raw.copy())
                    df_proc = df_proc.rename(
                        columns={"target_future_throttle": "target"})
                    if "target" not in df_proc.columns and "target_future_throttle" in df_proc.columns:
                        df_proc["target"] = df_proc["target_future_throttle"]
                except Exception as e:
                    st.error(f"Auto preprocessing failed: {e}")
                    st.stop()
            with st.expander("Preprocessing log", expanded=False):
                st.code("\n".join(logs), language="text")
        else:
            # Check minimum config
            if not st.session_state.get("custom_features"):
                st.warning("Go to the **⚙️ Dataset Config** tab and select your features first.")
                st.stop()
            try:
                X, y, features, prediction_window, df_proc = \
                    run_custom_preprocessing(df_raw)
            except Exception as e:
                st.error(f"Custom preprocessing failed: {e}")
                st.stop()

    target_col_name = "target"

    st.metric("Features used", len(features))
    with st.expander("Feature list", expanded=False):
        st.write(features)

    # ── Train / test split ────────────────────────────────────────────────────
    split_idx = int(len(df_proc) * 0.8)
    pw        = max(prediction_window, 1)
    train_df  = df_proc.iloc[:split_idx - pw]
    test_df   = df_proc.iloc[split_idx:]

    if len(train_df) < 50 or len(test_df) < 50:
        st.error("Not enough rows after preprocessing. Check your configuration.")
        st.stop()

    X_train = train_df[features].values; y_train = train_df[target_col_name].values.astype(int)
    X_test  = test_df[features].values;  y_test  = test_df[target_col_name].values.astype(int)

    # ── Train ─────────────────────────────────────────────────────────────────
    with st.spinner(f"Training {task}…"):
        np.random.seed(42)
        if task == "RandomForest":
            model = RFClassifier(
                n_trees=n_trees, max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
            )
            model.fit(X_train, y_train)
            _pt = np.array([t.predict_batch(X_test,  t.tree) for t in model.trees]).mean(0)
            _pr = np.array([t.predict_batch(X_train, t.tree) for t in model.trees]).mean(0)
            y_pred       = (_pt >= threshold).astype(int)
            y_pred_train = (_pr >= threshold).astype(int)
        else:
            try:
                from xgboost import XGBClassifier
            except ImportError:
                st.error("XGBoost is not installed — run `pip install xgboost`.")
                st.stop()
            model = XGBClassifier(
                n_estimators=n_estimators, max_depth=xgb_depth, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train)
            y_pred       = (model.predict_proba(X_test)[:,1]  >= xgb_threshold).astype(int)
            y_pred_train = (model.predict_proba(X_train)[:,1] >= xgb_threshold).astype(int)

    # ── Metrics ───────────────────────────────────────────────────────────────
    precision, recall, f1 = calculate_metrics(y_test, y_pred)
    train_acc = float(np.mean(y_pred_train == y_train))
    test_acc  = float(np.mean(y_pred == y_test))
    gap       = train_acc - test_acc

    st.markdown("## Performance metrics")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Test Accuracy",  f"{test_acc*100:.2f}%")
    m2.metric("Train Accuracy", f"{train_acc*100:.2f}%")
    m3.metric("Precision",      f"{precision:.3f}")
    m4.metric("Recall",         f"{recall:.3f}")
    m5.metric("F1-Score",       f"{f1:.3f}")

    gap_color = GREEN if gap < 0.05 else ORANGE if gap < 0.10 else RED
    gap_msg   = ("✓ good generalisation" if gap < 0.05
                 else "⚠ possible overfit" if gap < 0.10
                 else "✗ overfit — reduce depth / add regularisation")
    st.markdown(
        f"<span style='font-family:monospace;font-size:0.85rem;color:{gap_color}'>"
        f"Train/Test gap: {gap:.4f}  {gap_msg}</span>",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Confusion matrix + class report ───────────────────────────────────────
    col_cm, col_rep = st.columns(2)

    with col_cm:
        st.markdown("## Confusion matrix")
        fig_cm = _plot_cm(y_test, y_pred)
        st.pyplot(fig_cm, use_container_width=True)
        plt.close(fig_cm)

    with col_rep:
        st.markdown("## Classification report")
        p0, r0, f1_0, s0, p1, r1, f1_1, s1, _ = _cm_stats(y_test, y_pred)
        report_df = pd.DataFrame({
            "Class":     ["Negative (0)", "Positive (1)"],
            "Precision": [p0, p1],
            "Recall":    [r0, r1],
            "F1-Score":  [f1_0, f1_1],
            "Support":   [int(s0), int(s1)],
        })
        def _color(v):
            if isinstance(v, float):
                if v >= 0.80: return f"color:{GREEN}"
                if v >= 0.60: return f"color:{ORANGE}"
                return f"color:{RED}"
            return ""
        st.dataframe(
            report_df.style
                     .applymap(_color, subset=["Precision","Recall","F1-Score"])
                     .format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-Score":"{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("#### Class balance (test set)")
        class_counts = pd.Series(y_test).value_counts().sort_index()
        fig_b, ax_b = plt.subplots(figsize=(5, 1.8))
        bars = ax_b.barh(["Negative", "Positive"],
                         [class_counts.get(0,0), class_counts.get(1,0)],
                         color=[ACCENT, ORANGE], height=0.5)
        ax_b.set_xlabel("Test samples")
        for bar in bars:
            ax_b.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                      f"{int(bar.get_width()):,}", va="center", fontsize=8)
        fig_b.tight_layout()
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

    # ── Feature importance ────────────────────────────────────────────────────
    if task == "RandomForest":
        st.markdown("## Feature importance")
        try:
            importances = np.zeros(len(features))
            for tree in model.trees:
                _collect_importance(tree.tree, importances)
            importances /= (importances.sum() + 1e-9)
            imp_df = (pd.DataFrame({"Feature": features, "Importance": importances})
                        .sort_values("Importance").tail(20))
            bar_colors = [ACCENT if v >= imp_df["Importance"].quantile(0.75)
                          else "#aec7e8" for v in imp_df["Importance"]]
            fig_i, ax_i = plt.subplots(figsize=(10, max(4, len(imp_df)*0.38)))
            ax_i.barh(imp_df["Feature"], imp_df["Importance"],
                      color=bar_colors, height=0.65)
            ax_i.set_xlabel("Relative importance")
            ax_i.set_title("Top features (gini-based)", fontsize=10)
            fig_i.tight_layout()
            st.pyplot(fig_i, use_container_width=True)
            plt.close(fig_i)
        except Exception:
            st.info("Feature importance could not be computed.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALIDATION TESTS
# ═════════════════════════════════════════════════════════════════════════════
with tab_tests:
    if not run_btn:
        st.info("Run the pipeline first (Model Results tab).")
        st.stop()

    def _make_rf():
        return RFClassifier(n_trees=n_trees, max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            min_samples_split=min_samples_split)

    # ── Test 1: Temporal degradation ─────────────────────────────────────────
    st.markdown("## Test 1 — Temporal degradation")
    st.caption("Score vs training fraction. A rising curve = more data helps the model.")

    fractions   = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    fixed_split = int(len(df_proc) * 0.8)
    Xte_f = df_proc.iloc[fixed_split:][features].values
    yte_f = df_proc.iloc[fixed_split:][target_col_name].values.astype(int)

    td_results = []
    prog = st.progress(0, text="Running temporal degradation…")
    for i, frac in enumerate(fractions):
        cutoff = int(len(df_proc) * frac) - max(prediction_window, 1)
        if cutoff < 50:
            prog.progress((i+1)/len(fractions))
            continue
        tr = df_proc.iloc[:cutoff]
        Xtr = tr[features].values
        ytr = tr[target_col_name].values.astype(int)
        with _Capture():
            _, acc, auc = quick_fit(_make_rf(), Xtr, ytr, Xte_f, yte_f)
        td_results.append((frac, acc, auc))
        prog.progress((i+1)/len(fractions), text=f"frac={frac:.1f} → acc={acc:.4f}")
    prog.empty()

    if td_results:
        fracs, accs, aucs = zip(*td_results)
        fig_td, ax_td = plt.subplots(figsize=(10, 4))
        ax_td.plot(fracs, accs, marker="o", color=ACCENT, lw=2, label="Accuracy", ms=7)
        ax_td.plot(fracs, aucs, marker="s", color=GREEN,  lw=2, label="AUC",      ms=7)
        ax_td.set_xlabel("Training fraction"); ax_td.set_ylabel("Score")
        ax_td.set_title("Temporal Degradation", fontsize=11); ax_td.legend()
        fig_td.tight_layout()
        st.pyplot(fig_td, use_container_width=True); plt.close(fig_td)
        st.dataframe(pd.DataFrame({
            "Fraction": fracs,
            "Accuracy": [f"{a:.4f}" for a in accs],
            "AUC":      [f"{a:.4f}" for a in aucs],
        }), use_container_width=True, hide_index=True)

    st.divider()

    # ── Test 4: Walk-forward ──────────────────────────────────────────────────
    st.markdown("## Test 4 — Walk-forward validation")
    st.caption("Consistent AUC across folds = the model generalises well over time.")

    cutoffs    = [0.5, 0.6, 0.7, 0.8]
    wf_results = []
    prog2 = st.progress(0, text="Running walk-forward…")
    for i, cf in enumerate(cutoffs):
        split  = int(len(df_proc) * cf)
        window = 15_000
        tstart = max(0, split - max(prediction_window,1) - window)
        tr     = df_proc.iloc[tstart:split - max(prediction_window,1)]
        te     = df_proc.iloc[split:]
        if len(tr) < 50 or len(te) < 50:
            prog2.progress((i+1)/len(cutoffs))
            continue
        Xtr = tr[features].values; ytr = tr[target_col_name].values.astype(int)
        Xte = te[features].values; yte = te[target_col_name].values.astype(int)
        with _Capture():
            _, acc, auc = quick_fit(_make_rf(), Xtr, ytr, Xte, yte)
        wf_results.append((cf, len(tr), len(te), acc, auc))
        prog2.progress((i+1)/len(cutoffs), text=f"cutoff={cf:.1f} → acc={acc:.4f}")
    prog2.empty()

    if wf_results:
        cuts, trsz, tesz, accs2, aucs2 = zip(*wf_results)
        fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
        ax_wf.plot(cuts, accs2, marker="o", color=ACCENT, lw=2, label="Accuracy", ms=7)
        ax_wf.plot(cuts, aucs2, marker="s", color=GREEN,  lw=2, label="AUC",      ms=7)
        ax_wf.fill_between(cuts, [a-0.02 for a in accs2],
                           [a+0.02 for a in accs2], color=ACCENT, alpha=0.08)
        ax_wf.set_xlabel("Train cutoff fraction"); ax_wf.set_ylabel("Score")
        ax_wf.set_title("Walk-Forward Validation", fontsize=11); ax_wf.legend()
        fig_wf.tight_layout()
        st.pyplot(fig_wf, use_container_width=True); plt.close(fig_wf)
        st.dataframe(pd.DataFrame({
            "Cutoff":     cuts, "Train rows": trsz, "Test rows": tesz,
            "Accuracy":   [f"{a:.4f}" for a in accs2],
            "AUC":        [f"{a:.4f}" for a in aucs2],
        }), use_container_width=True, hide_index=True)

        acc_std = float(np.std(accs2))
        color_v = GREEN if acc_std < 0.02 else ORANGE if acc_std < 0.05 else RED
        verdict = ("consistent ✓" if acc_std < 0.02
                   else "moderate variance ⚠" if acc_std < 0.05 else "high variance ✗")
        st.markdown(
            f"<span style='font-family:monospace;font-size:0.85rem;color:{color_v}'>"
            f"Accuracy std: {acc_std:.4f} — {verdict}</span>",
            unsafe_allow_html=True,
        )