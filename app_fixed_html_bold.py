%%writefile app.py

import streamlit as st
import pandas as pd
import statsmodels.api as sm
import plotly.express as px
from io import BytesIO
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Research Regression Lab",
    page_icon="📊",
    layout="wide"
)

HERO_IMAGE = Path("/Users/andresrotstein/Downloads/ChatGPT Image May 26, 2026, 10_57_37 PM.png")

# =========================================================
# STYLE
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main {
        background-color: #f7f8fb;
    }

    h1 {
        font-size: 2.6rem !important;
        font-weight: 850 !important;
        color: #111827;
        letter-spacing: -0.03em;
    }

    h2, h3 {
        color: #111827;
        font-weight: 750 !important;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        color: #6b7280;
        font-size: 1.05rem;
        margin-top: -10px;
        margin-bottom: 24px;
    }

    .section-card {
        background-color: white;
        border-radius: 18px;
        padding: 22px 24px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
        margin-bottom: 18px;
    }

    .section-card h3 {
        margin-top: 0;
        margin-bottom: 10px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px 18px;
        border-radius: 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stMetricLabel"] {
        color: #6b7280;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: #111827;
        font-weight: 800;
    }

    div[data-testid="stDataFrame"] {
        background-color: white;
        border-radius: 18px;
        padding: 8px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 4px 18px rgba(15, 23, 42, 0.04);
    }

    .stAlert {
        border-radius: 16px;
    }

    .info-card {
        background-color: #e8f1ff;
        border-left: 5px solid #2563eb;
        color: #111827;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        border-top: 1px solid #bfdbfe;
        border-right: 1px solid #bfdbfe;
        border-bottom: 1px solid #bfdbfe;
    }

    .warning-card {
        background-color: #fff7ed;
        border-left: 5px solid #f97316;
        color: #111827;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        border-top: 1px solid #fed7aa;
        border-right: 1px solid #fed7aa;
        border-bottom: 1px solid #fed7aa;
    }

    .success-card {
        background-color: #ecfdf5;
        border-left: 5px solid #10b981;
        color: #111827;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
        border-top: 1px solid #a7f3d0;
        border-right: 1px solid #a7f3d0;
        border-bottom: 1px solid #a7f3d0;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: white !important;
    }

    section[data-testid="stSidebar"] label {
        font-weight: 650;
    }

    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] select {
        color: #111827 !important;
        background-color: white !important;
    }

    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] {
        background-color: white !important;
        border-radius: 14px;
    }

    section[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] * {
        color: #111827 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 999px;
        padding: 10px 18px;
        border: 1px solid #e5e7eb;
        font-weight: 650;
    }

    .stTabs [aria-selected="true"] {
        background-color: #111827 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================================================
# HEADER
# =========================================================

st.title("Research Regression Lab")
st.markdown(
    """
    <div class="app-subtitle">
    Quick driver analysis for surveys, CX, NPS, satisfaction and business outcomes.
    Upload a file, select your model, and get coefficients, standardized betas, driver importance and an automated executive read.
    </div>
    """,
    unsafe_allow_html=True
)

if HERO_IMAGE.exists():
    st.image(
        str(HERO_IMAGE),
        use_column_width=True
    )
else:
    st.info(f"Hero image not found at: {HERO_IMAGE}")

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def generate_headline(model, coef_df, dependent):
    predictors_df = coef_df[coef_df["variable"] != "const"].copy()

    if predictors_df.empty:
        return "No predictors were available to generate a driver headline."

    sig_df = predictors_df[
        (predictors_df["p_value"] < 0.05) &
        (predictors_df["sig_imp_%"].notna())
    ].copy()

    if not sig_df.empty:
        top = sig_df.sort_values("sig_imp_%", ascending=False).iloc[0]
        return f"{top['variable']} emerges as the strongest significant driver of {dependent}."

    valid_df = predictors_df[predictors_df["imp_%"].notna()].copy()

    if not valid_df.empty:
        top = valid_df.sort_values("imp_%", ascending=False).iloc[0]
        return f"{top['variable']} shows the highest relative weight, but significant drivers are limited or absent."

    return "No driver importance could be calculated for this model."


def generate_model_read(model, coef_df, dependent, corr_df=None):
    r2 = model.rsquared
    adj_r2 = model.rsquared_adj
    n = int(model.nobs)

    predictors_df = coef_df[coef_df["variable"] != "const"].copy()
    sig_df = predictors_df[predictors_df["p_value"] < 0.05].copy()
    non_sig_df = predictors_df[predictors_df["p_value"] >= 0.05].copy()

    total_predictors = len(predictors_df)
    sig_count = len(sig_df)
    sig_share = sig_count / total_predictors if total_predictors > 0 else 0

    if adj_r2 < 0.20:
        strength = "Low"
        strength_text = (
            "The model has low explanatory power. It may be useful for exploration, "
            "but it should not be used as a strong driver model."
        )
    elif adj_r2 < 0.40:
        strength = "Moderate-low"
        strength_text = (
            "The model explains a modest share of the outcome. It can help identify signals, "
            "but conclusions should remain cautious."
        )
    elif adj_r2 < 0.60:
        strength = "Moderate"
        strength_text = (
            "The model has a usable level of explanatory power. It is appropriate for directional "
            "driver prioritization, especially if the main predictors are statistically significant."
        )
    elif adj_r2 < 0.75:
        strength = "Strong"
        strength_text = (
            "The model has strong explanatory power. It provides a solid basis for identifying "
            "which predictors are most associated with the outcome."
        )
    else:
        strength = "Very strong"
        strength_text = (
            "The model has very strong explanatory power. This is promising, but it is worth checking "
            "for overlapping predictors, repeated measures, overfitting, or variables that are too close "
            "conceptually to the dependent variable."
        )

    overview = (
        f"The model was estimated with <b>N = {n}</b> valid cases and <b>{total_predictors} predictors</b>. "
        f"It has an <b>R² of {r2:.3f}</b> and an <b>Adjusted R² of {adj_r2:.3f}</b>. "
        f"{strength_text}"
    )

    if total_predictors == 0:
        significance = "No predictors were included in the model."
    elif sig_count == 0:
        significance = (
            "None of the predictors are statistically significant at p < 0.05. "
            "The model may still show directional patterns, but it should not be used for firm driver conclusions."
        )
    elif sig_share < 0.35:
        significance = (
            f"Only <b>{sig_count} out of {total_predictors} predictors</b> are statistically significant "
            f"at p < 0.05. This means the model has a limited set of reliable drivers; non-significant "
            f"variables should be treated as weak or unstable signals."
        )
    elif sig_share < 0.70:
        significance = (
            f"<b>{sig_count} out of {total_predictors} predictors</b> are statistically significant "
            f"at p < 0.05. This gives the model a reasonably interpretable driver structure."
        )
    else:
        significance = (
            f"<b>{sig_count} out of {total_predictors} predictors</b> are statistically significant "
            f"at p < 0.05. The model shows a strong driver structure, although overlap between predictors "
            f"should still be checked."
        )

    if sig_count > 0 and "sig_imp_%" in sig_df.columns:
        ranked_df = sig_df[sig_df["sig_imp_%"].notna()].sort_values("sig_imp_%", ascending=False).copy()
        importance_col = "sig_imp_%"
        basis_text = "among statistically significant predictors"
    else:
        ranked_df = predictors_df[predictors_df["imp_%"].notna()].sort_values("imp_%", ascending=False).copy()
        importance_col = "imp_%"
        basis_text = "among all predictors, but not necessarily statistically significant"

    top_driver_lines = []

    for _, row in ranked_df.head(3).iterrows():
        value = row.get(importance_col, None)

        if pd.notna(value):
            top_driver_lines.append(
                f"- <b>{row['variable']}</b>: {value:.1f}% relative importance, "
                f"{row['direction'].lower()} association, p = {row['p_value']:.3f}."
            )

    if top_driver_lines:
        top_drivers = (
            f"The strongest drivers {basis_text} are:\n\n" +
            "\n".join(top_driver_lines)
        )
    else:
        top_drivers = "No driver ranking could be generated."

    if len(ranked_df) >= 2:
        top1 = ranked_df.iloc[0]
        top2 = ranked_df.iloc[1]
        top1_imp = top1[importance_col]
        top2_imp = top2[importance_col]
        top_gap = top1_imp - top2_imp

        if top1_imp >= 45:
            concentration = (
                f"The model is highly concentrated: <b>{top1['variable']}</b> alone accounts for "
                f"{top1_imp:.1f}% of the relative driver weight. This suggests a dominant driver pattern."
            )
        elif top_gap >= 15:
            concentration = (
                f"There is a clear leading driver: <b>{top1['variable']}</b> is {top_gap:.1f} percentage points "
                f"ahead of <b>{top2['variable']}</b> in relative importance."
            )
        else:
            concentration = (
                f"The top drivers are relatively close. <b>{top1['variable']}</b> leads, but the gap versus "
                f"<b>{top2['variable']}</b> is only {top_gap:.1f} percentage points. This suggests a multi-driver pattern "
                f"rather than a single dominant explanation."
            )
    elif len(ranked_df) == 1:
        top1 = ranked_df.iloc[0]
        concentration = (
            f"Only one driver is available for importance interpretation: <b>{top1['variable']}</b>. "
            f"Additional predictors or a broader model may be needed for a fuller driver read."
        )
    else:
        concentration = "Driver concentration could not be assessed."

    if len(non_sig_df) > 0:
        non_sig_list = ", ".join(non_sig_df["variable"].tolist())
        non_significant = (
            f"The following predictors are not statistically significant at p < 0.05 and should be treated with caution: "
            f"{non_sig_list}."
        )
    else:
        non_significant = "All predictors included in the model are statistically significant at p < 0.05."

    neg_df = predictors_df[predictors_df["coef"] < 0].copy()

    if len(neg_df) > 0:
        neg_list = ", ".join(neg_df["variable"].tolist())
        direction = (
            f"Negative coefficients were detected for: {neg_list}. "
            "This may be meaningful, but it should be checked carefully: negative signs can reflect reverse-coded scales, "
            "multicollinearity, suppression effects, or an actual inverse relationship."
        )
    else:
        direction = (
            f"All predictor coefficients are positive. Higher values in the selected predictors are associated with higher values in <b>{dependent}</b>."
        )

    if total_predictors > 0:
        cases_per_predictor = n / total_predictors
    else:
        cases_per_predictor = None

    if cases_per_predictor is not None:
        if cases_per_predictor < 10:
            reliability = (
                f"The model has only <b>{cases_per_predictor:.1f} cases per predictor</b>, which is low. "
                "Results may be unstable and should be validated with fewer predictors or a larger sample."
            )
        elif cases_per_predictor < 20:
            reliability = (
                f"The model has <b>{cases_per_predictor:.1f} cases per predictor</b>. This is acceptable for exploration, "
                "but a simpler model may be more stable."
            )
        else:
            reliability = (
                f"The model has <b>{cases_per_predictor:.1f} cases per predictor</b>, which is a comfortable ratio for this type of exploratory driver analysis."
            )
    else:
        reliability = "Cases-per-predictor could not be calculated."

    corr_warning = None

    if corr_df is not None and not corr_df.empty:
        high_pairs = []
        predictor_cols = [c for c in predictors_df["variable"].tolist() if c in corr_df.columns]

        for i, col_a in enumerate(predictor_cols):
            for col_b in predictor_cols[i + 1:]:
                corr_value = corr_df.loc[col_a, col_b]

                if pd.notna(corr_value) and abs(corr_value) >= 0.70:
                    high_pairs.append(f"{col_a} / {col_b} ({corr_value:.3f})")

        if high_pairs:
            corr_warning = (
                "Some predictors are highly correlated, which may affect coefficient stability: "
                + "; ".join(high_pairs[:8])
                + ("." if len(high_pairs) <= 8 else "; among others.")
            )
        else:
            corr_warning = "No very high predictor-to-predictor correlations were detected at |r| >= 0.70."

    if adj_r2 >= 0.40 and sig_count >= 2:
        bottom_line = (
            "Bottom line: this model is suitable for directional driver prioritization. "
            "Use the significant drivers and their relative importance to guide interpretation."
        )
    elif adj_r2 >= 0.40 and sig_count < 2:
        bottom_line = (
            "Bottom line: the model fit is acceptable, but the driver structure is thin. "
            "Prioritize only statistically significant predictors and avoid over-interpreting the rest."
        )
    elif adj_r2 < 0.40 and sig_count >= 2:
        bottom_line = (
            "Bottom line: there are some significant drivers, but the model explains a limited share of the outcome. "
            "Use it as exploratory evidence, not as a strong explanatory model."
        )
    else:
        bottom_line = (
            "Bottom line: the model is weak for driver prioritization in its current form. "
            "Consider revisiting predictors, checking coding, adding relevant variables, or simplifying the model."
        )

    return {
        "strength": strength,
        "overview": overview,
        "significance": significance,
        "top_drivers": top_drivers,
        "concentration": concentration,
        "non_significant": non_significant,
        "direction": direction,
        "reliability": reliability,
        "correlation_warning": corr_warning,
        "bottom_line": bottom_line
    }


# =========================================================
# SIDEBAR — UPLOAD
# =========================================================

with st.sidebar:
    st.header("1. Upload data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel file from the sidebar to start.")
    st.stop()

file_name = uploaded_file.name.lower()

# =========================================================
# READ DATA
# =========================================================

try:
    if file_name.endswith(".csv"):
        sep = st.sidebar.selectbox(
            "CSV separator",
            [",", ";", "\t"],
            index=1
        )
        df = pd.read_csv(uploaded_file, sep=sep)

    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    else:
        st.error("Unsupported file type. Please upload a CSV or XLSX file.")
        st.stop()

except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

if df.empty:
    st.error("The uploaded file is empty.")
    st.stop()

# =========================================================
# SIDEBAR — MODEL SETUP
# =========================================================

columns = df.columns.tolist()

with st.sidebar:
    st.header("2. Model setup")

dependent = st.sidebar.selectbox(
    "Dependent variable",
    columns
)

independent_options = [c for c in columns if c != dependent]

independents = st.sidebar.multiselect(
    "Independent variables",
    independent_options
)

with st.sidebar:
    st.header("3. Options")

use_dummies = st.sidebar.checkbox(
    "Convert categorical variables to dummies",
    value=True
)

hide_non_significant = st.sidebar.checkbox(
    "Hide non-significant predictors",
    value=False
)

if not independents:
    st.warning("Select at least one independent variable from the sidebar.")
    st.stop()

# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander("Data preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.markdown(
        f"Rows: <b>{len(df)}</b> | Columns: <b>{len(df.columns)}</b>",
        unsafe_allow_html=True
    )

# =========================================================
# PREPARE MODEL DATA
# =========================================================

model_df = df[[dependent] + independents].copy()

y = pd.to_numeric(model_df[dependent], errors="coerce")
X = model_df[independents].copy()

if use_dummies:
    X = pd.get_dummies(X, drop_first=True)
else:
    X = X.apply(pd.to_numeric, errors="coerce")

data = pd.concat([y, X], axis=1).dropna()

if data.empty:
    st.error("No valid data left after cleaning missing/non-numeric values.")
    st.stop()

y_clean = data[dependent]
X_clean = data.drop(columns=[dependent])

X_clean = X_clean.apply(pd.to_numeric, errors="coerce").dropna()
y_clean = y_clean.loc[X_clean.index]

if len(y_clean) < 10:
    st.error("Not enough valid rows after cleaning. Need at least 10 rows.")
    st.stop()

if X_clean.shape[1] == 0:
    st.error("No valid independent variables left after cleaning.")
    st.stop()

X_const = sm.add_constant(X_clean)

# =========================================================
# RUN REGRESSION
# =========================================================

try:
    model = sm.OLS(y_clean, X_const).fit()
except Exception as e:
    st.error(f"Model failed: {e}")
    st.stop()

# =========================================================
# COEFFICIENTS
# =========================================================

coef_df = pd.DataFrame({
    "variable": model.params.index,
    "coef": model.params.values,
    "std_error": model.bse.values,
    "t": model.tvalues.values,
    "p_value": model.pvalues.values
})

coef_df["significant_0.05"] = coef_df["p_value"] < 0.05

# =========================================================
# STANDARDIZED BETAS
# =========================================================

try:
    X_std = X_clean.copy()
    X_std = X_std.loc[:, X_std.std(ddof=0) != 0]

    y_std = y_clean.copy()

    X_std = (X_std - X_std.mean()) / X_std.std(ddof=0)
    y_std = (y_std - y_std.mean()) / y_std.std(ddof=0)

    X_std_const = sm.add_constant(X_std)

    std_model = sm.OLS(y_std, X_std_const).fit()

    std_beta_df = pd.DataFrame({
        "variable": std_model.params.index,
        "std_beta": std_model.params.values
    })

    coef_df = coef_df.merge(
        std_beta_df,
        on="variable",
        how="left"
    )

except Exception as e:
    st.warning(f"Could not calculate standardized betas: {e}")
    coef_df["std_beta"] = None

# =========================================================
# IMPORTANCE
# =========================================================

coef_df["abs_std_beta"] = coef_df["std_beta"].abs()

total_abs_beta = coef_df.loc[
    coef_df["variable"] != "const",
    "abs_std_beta"
].sum()

if pd.notna(total_abs_beta) and total_abs_beta > 0:
    coef_df["imp_%"] = coef_df.apply(
        lambda r: round(r["abs_std_beta"] / total_abs_beta * 100, 1)
        if r["variable"] != "const" else None,
        axis=1
    )
else:
    coef_df["imp_%"] = None

coef_df["sig_abs_std_beta"] = coef_df.apply(
    lambda r: r["abs_std_beta"]
    if (r["variable"] != "const" and r["p_value"] < 0.05)
    else None,
    axis=1
)

total_sig_abs_beta = coef_df["sig_abs_std_beta"].sum()

if pd.notna(total_sig_abs_beta) and total_sig_abs_beta > 0:
    coef_df["sig_imp_%"] = coef_df.apply(
        lambda r: round(r["sig_abs_std_beta"] / total_sig_abs_beta * 100, 1)
        if pd.notna(r["sig_abs_std_beta"]) else None,
        axis=1
    )
else:
    coef_df["sig_imp_%"] = None

coef_df["direction"] = coef_df["coef"].apply(
    lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral")
)

ordered_cols = [
    "variable",
    "coef",
    "std_beta",
    "abs_std_beta",
    "imp_%",
    "sig_abs_std_beta",
    "sig_imp_%",
    "direction",
    "std_error",
    "t",
    "p_value",
    "significant_0.05"
]

coef_df = coef_df[[c for c in ordered_cols if c in coef_df.columns]]

# =========================================================
# DISPLAY TABLE
# =========================================================

display_coef_df = coef_df.copy()

if hide_non_significant:
    display_coef_df = display_coef_df[
        (display_coef_df["variable"] == "const") |
        (display_coef_df["p_value"] < 0.05)
    ].copy()

    display_coef_df["imp_%"] = display_coef_df["sig_imp_%"]

display_coef_pretty = display_coef_df.rename(columns={
    "variable": "Variable",
    "coef": "Coef.",
    "std_beta": "Std. Beta",
    "abs_std_beta": "Abs. Std. Beta",
    "imp_%": "Imp. %",
    "sig_abs_std_beta": "Sig. Abs. Std. Beta",
    "sig_imp_%": "Sig. Imp. %",
    "direction": "Direction",
    "std_error": "Std. Error",
    "t": "t",
    "p_value": "p-value",
    "significant_0.05": "Sig. 0.05"
})

# =========================================================
# PREDICTIONS / RESIDUALS
# =========================================================

predicted = model.predict(X_const)
residuals = y_clean - predicted

plot_df = pd.DataFrame({
    "actual": y_clean,
    "predicted": predicted,
    "residuals": residuals
})

# =========================================================
# CORRELATION MATRIX
# =========================================================

try:
    corr_df = pd.concat([y_clean, X_clean], axis=1).corr(numeric_only=True).round(3)
except Exception:
    corr_df = pd.DataFrame()

# =========================================================
# EXECUTIVE READ DATA
# =========================================================

model_read = generate_model_read(
    model=model,
    coef_df=coef_df,
    dependent=dependent,
    corr_df=corr_df
)

headline = generate_headline(
    model=model,
    coef_df=coef_df,
    dependent=dependent
)

# =========================================================
# QUICK CHECKS
# =========================================================

checks = []

if model.rsquared < 0.2:
    checks.append(
        "Low R²: the selected variables explain a limited share of the variation in the dependent variable."
    )

if len(coef_df[(coef_df["variable"] != "const") & (coef_df["p_value"] < 0.05)]) == 0:
    checks.append("No statistically significant predictors at p < 0.05.")

# =========================================================
# TABS
# =========================================================

tab_overview, tab_read, tab_drivers, tab_diagnostics, tab_corr, tab_export = st.tabs(
    ["Overview", "Executive Read", "Drivers", "Diagnostics", "Correlations", "Export"]
)

# =========================================================
# TAB 1 — OVERVIEW
# =========================================================

with tab_overview:
    st.subheader("Model overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("N", int(model.nobs))

    with c2:
        st.metric("R²", round(model.rsquared, 3))

    with c3:
        st.metric("Adjusted R²", round(model.rsquared_adj, 3))

    with c4:
        try:
            st.metric("F p-value", round(model.f_pvalue, 5))
        except Exception:
            st.metric("F p-value", "-")

    st.markdown("---")

    st.subheader("Quick model checks")

    if not checks:
        st.success("No major quick-check warnings detected.")
    else:
        for check in checks:
            st.warning(check)

    st.markdown(
        f"""
        <div class="section-card">
        <h3>Model setup</h3>
        <p><b>Dependent variable:</b> {dependent}</p>
        <p><b>Predictors selected:</b> {len(independents)}</p>
        <p><b>Rows in original file:</b> {len(df)}</p>
        <p><b>Rows used in model:</b> {int(model.nobs)}</p>
        <p><b>Non-significant predictors hidden:</b> {hide_non_significant}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# TAB 2 — EXECUTIVE READ
# =========================================================

with tab_read:
    st.subheader("Executive Read")

    st.markdown(f"## {headline}")

    st.markdown(
        f"""
        <div class="section-card">
        <h3>Model quality</h3>
        <p><b>Strength:</b> {model_read["strength"]}</p>
        <p>{model_read["overview"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="section-card">
        <h3>Driver structure</h3>
        <p>{model_read["significance"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Top drivers")
    st.markdown(model_read["top_drivers"], unsafe_allow_html=True)

    st.markdown("### Driver concentration")
    st.markdown(
        f"""<div class="info-card">{model_read["concentration"]}</div>""",
        unsafe_allow_html=True
    )

    st.markdown("### What to treat with caution")
    st.markdown(
        f"""<div class="warning-card">{model_read["non_significant"]}</div>""",
        unsafe_allow_html=True
    )

    st.markdown("### Directional interpretation")
    st.markdown(
        f"""<div class="info-card">{model_read["direction"]}</div>""",
        unsafe_allow_html=True
    )

    st.markdown("### Model reliability")
    st.markdown(
        f"""<div class="info-card">{model_read["reliability"]}</div>""",
        unsafe_allow_html=True
    )

    if model_read["correlation_warning"]:
        st.markdown("### Correlation / overlap check")
        st.markdown(
            f"""<div class="warning-card">{model_read["correlation_warning"]}</div>""",
            unsafe_allow_html=True
        )

    st.markdown("### Bottom line")
    st.markdown(
        f"""<div class="success-card">{model_read["bottom_line"]}</div>""",
        unsafe_allow_html=True
    )

    st.caption(
        "This is a rule-based automated interpretation. Use it as a first analytical read, not as a replacement for expert judgment."
    )

# =========================================================
# TAB 3 — DRIVERS
# =========================================================

with tab_drivers:
    st.subheader("Coefficients / Model Summary Table")

    st.dataframe(display_coef_pretty, use_container_width=True)

    if hide_non_significant:
        st.caption(
            "Non-significant predictors are hidden. Imp. % is recalculated using only significant predictors, "
            "so visible driver importance sums to 100%."
        )
    else:
        st.caption(
            "Std. Beta = standardized beta. Imp. % = relative importance using all predictors. "
            "Sig. Imp. % recalculates importance using only statistically significant predictors at p < 0.05."
        )

    st.markdown("---")
    st.subheader("Driver Importance")

    if hide_non_significant:
        importance_col = "sig_imp_%"
        chart_title = "Relative Importance Based on Significant Predictors"
    else:
        importance_col = "imp_%"
        chart_title = "Relative Importance Based on All Predictors"

    importance_df = coef_df[
        (coef_df["variable"] != "const") &
        (coef_df[importance_col].notna())
    ].copy()

    if hide_non_significant:
        importance_df = importance_df[
            importance_df["p_value"] < 0.05
        ].copy()

    importance_df = importance_df.sort_values(
        importance_col,
        ascending=True
    )

    if not importance_df.empty:
        importance_df["label"] = importance_df[importance_col].apply(
            lambda x: f"{x:.1f}%"
        )

        fig_importance = px.bar(
            importance_df,
            x=importance_col,
            y="variable",
            orientation="h",
            text="label",
            title=chart_title
        )

        fig_importance.update_traces(
            textposition="inside",
            insidetextanchor="end",
            textfont=dict(
                color="white",
                size=13,
                family="Arial Black"
            ),
            cliponaxis=False
        )

        fig_importance.update_layout(
            xaxis_title="Importance (%)",
            yaxis_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font=dict(size=20),
            margin=dict(l=20, r=20, t=60, b=20),
            height=500,
            uniformtext_minsize=10,
            uniformtext_mode="show"
        )

        fig_importance.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
        fig_importance.update_yaxes(showgrid=False)

        st.plotly_chart(fig_importance, use_container_width=True)

        if hide_non_significant:
            st.info(
                "Non-significant predictors are hidden. The chart uses Sig. Imp. %, recalculated only among significant predictors."
            )
        else:
            st.info(
                "Imp. % is calculated from absolute standardized betas across all predictors in the model."
            )

    else:
        st.warning("No importance values could be calculated for this view.")

# =========================================================
# TAB 4 — DIAGNOSTICS
# =========================================================

with tab_diagnostics:
    st.subheader("Diagnostics")

    c1, c2 = st.columns(2)

    with c1:
        fig1 = px.scatter(
            plot_df,
            x="actual",
            y="predicted",
            trendline="ols",
            title="Actual vs Predicted"
        )

        fig1.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        fig1.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
        fig1.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.scatter(
            plot_df,
            x="predicted",
            y="residuals",
            title="Residuals vs Predicted"
        )

        fig2.add_hline(y=0)

        fig2.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=20, r=20, t=60, b=20)
        )

        fig2.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
        fig2.update_yaxes(showgrid=True, gridcolor="#e5e7eb")

        st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Diagnostics are quick checks. For driver analysis, prioritize coefficients, standardized betas, importance, and model fit."
    )

# =========================================================
# TAB 5 — CORRELATIONS
# =========================================================

with tab_corr:
    st.subheader("Correlation matrix")

    if corr_df.empty:
        st.warning("Could not create correlation matrix.")
    else:
        fig_corr = px.imshow(
            corr_df,
            text_auto=".3f",
            aspect="auto",
            title="Correlation Matrix"
        )

        fig_corr.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            title_font=dict(size=18),
            margin=dict(l=20, r=20, t=50, b=20),
            height=500
        )

        fig_corr.update_traces(
            textfont=dict(size=10)
        )

        st.plotly_chart(fig_corr, use_container_width=True)

# =========================================================
# TAB 6 — EXPORT
# =========================================================

with tab_export:
    st.subheader("Export results")

    output = BytesIO()

    try:
        executive_read_export = pd.DataFrame({
            "section": [
                "headline",
                "model_quality",
                "driver_structure",
                "top_drivers",
                "driver_concentration",
                "non_significant",
                "direction",
                "model_reliability",
                "correlation_warning",
                "bottom_line"
            ],
            "text": [
                headline,
                model_read["overview"],
                model_read["significance"],
                model_read["top_drivers"],
                model_read["concentration"],
                model_read["non_significant"],
                model_read["direction"],
                model_read["reliability"],
                model_read["correlation_warning"],
                model_read["bottom_line"]
            ]
        })

        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            coef_df.to_excel(writer, sheet_name="coefficients_full", index=False)
            display_coef_df.to_excel(writer, sheet_name="coefficients_visible", index=False)
            plot_df.to_excel(writer, sheet_name="predictions_residuals", index=False)
            executive_read_export.to_excel(writer, sheet_name="executive_read", index=False)

            if not corr_df.empty:
                corr_df.to_excel(writer, sheet_name="correlations")

            model_info = pd.DataFrame({
                "metric": ["N", "R2", "Adjusted_R2", "F_p_value", "hide_non_significant"],
                "value": [
                    int(model.nobs),
                    model.rsquared,
                    model.rsquared_adj,
                    model.f_pvalue,
                    hide_non_significant
                ]
            })

            model_info.to_excel(writer, sheet_name="model_info", index=False)

        excel_data = output.getvalue()

        st.download_button(
            label="Download regression results as Excel",
            data=excel_data,
            file_name="regression_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.success("Export ready.")

    except Exception as e:
        st.warning(f"Could not create Excel export: {e}")