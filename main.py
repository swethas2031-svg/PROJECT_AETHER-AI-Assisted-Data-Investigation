import os
import sys
import json
import zipfile
import warnings
import subprocess
import importlib.util
from pathlib import Path

warnings.filterwarnings("ignore")

# ================================================================
# 0. CHECK DEPENDENCIES
# ================================================================

REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "plotly": "plotly",
    "streamlit": "streamlit"
}


def check_dependencies():
    missing = [
        package
        for module, package in REQUIRED_MODULES.items()
        if importlib.util.find_spec(module) is None
    ]

    if missing:
        print("Installing missing packages:", missing)

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            *missing
        ])


check_dependencies()

import numpy as np
import pandas as pd

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

import plotly.express as px

np.random.seed(42)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "PROJECT_AETHER_OUTPUT"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# PHASE 1: DATA LOADING AND PREPROCESSING
# ================================================================

def generate_sample_data(n=2000, seed=42):
    """Generate reproducible synthetic business data."""

    rng = np.random.default_rng(seed)

    demand = rng.normal(100, 20, n)
    marketing = rng.normal(1500, 250, n)
    discount = rng.uniform(0, 25, n)
    cost = rng.normal(5000, 800, n)
    customers = rng.poisson(200, n)

    time_factor = (
        1 + 0.10 * np.sin(
            np.arange(n) * 2 * np.pi / 365
        )
    )

    sales = (
        2000
        + 18 * demand
        + 2.5 * marketing
        + 12 * customers
        + 25 * discount
        - 0.15 * cost
    ) * time_factor

    sales += rng.normal(0, 800, n)

    df = pd.DataFrame({
        "product_demand": demand,
        "marketing_spend": marketing,
        "discount_percent": discount,
        "product_cost": cost,
        "customer_count": customers,
        "sales": sales,
        "day_index": np.arange(n)
    })

    # Inject a small number of unusual sales observations.
    indices = rng.choice(
        n,
        size=max(1, int(n * 0.015)),
        replace=False
    )

    df.loc[indices, "sales"] *= 3

    return df


def load_data():
    """Load a local CSV or generate sample data."""

    print("\n" + "=" * 60)
    print("PROJECT AETHER: DATA LOADING")
    print("=" * 60)

    filename = input(
        "Enter CSV path, or press Enter for sample data: "
    ).strip().strip('"').strip("'")

    if filename:

        path = Path(filename).expanduser()

        if not path.is_absolute():
            path = Path.cwd() / path

        if not path.is_file():
            print(
                "File not found. Generating sample data instead."
            )
            df = generate_sample_data()

        elif path.suffix.lower() != ".csv":
            raise ValueError(
                "Please provide a CSV file."
            )

        else:
            df = pd.read_csv(path)
            print("Loaded:", path)

    else:
        print("Generating synthetic business data.")
        df = generate_sample_data()

    if df.empty:
        raise ValueError("The dataset is empty.")

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    df = df.loc[
        :,
        ~pd.Index(df.columns).duplicated()
    ].copy()

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    if len(numeric_columns) < 2:
        raise ValueError(
            "At least two numeric columns are required."
        )

    target_candidates = [
        column
        for column in numeric_columns
        if column.lower() in {
            "sales",
            "revenue",
            "profit",
            "target",
            "amount"
        }
    ]

    print("\nAvailable numeric columns:")

    for index, column in enumerate(
        numeric_columns,
        start=1
    ):
        print(f"{index}. {column}")

    default_target = (
        target_candidates[0]
        if target_candidates
        else numeric_columns[0]
    )

    choice = input(
        f"\nEnter target column name "
        f"[default: {default_target}]: "
    ).strip()

    target = choice if choice else default_target

    if target not in numeric_columns:
        print(
            "Invalid target selection. "
            f"Using {default_target}."
        )
        target = default_target

    df = df.dropna(
        subset=[target]
    ).reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            "At least 20 usable rows are recommended."
        )

    print("\nDataset shape:", df.shape)
    print("Target variable:", target)

    return df, target


# ================================================================
# PHASE 2: ANOMALY DETECTION
# ================================================================

def detect_anomalies(df, target):
    print("\n" + "=" * 60)
    print("PHASE 2: ANOMALY DETECTION")
    print("=" * 60)

    features = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    X = df[features].copy()

    X = X.dropna(
        axis=1,
        how="all"
    )

    X = X.loc[
        :,
        X.nunique(dropna=True) > 1
    ]

    if X.empty:
        raise ValueError(
            "No varying numeric columns are available."
        )

    imputer = SimpleImputer(
        strategy="median"
    )

    X_imputed = imputer.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=42
    )

    model.fit(X_imputed)

    result = df.copy()

    result["anomaly_label"] = model.predict(
        X_imputed
    )

    result["anomaly_score"] = (
        model.decision_function(X_imputed)
    )

    anomalies = result.loc[
        result["anomaly_label"] == -1
    ].copy()

    print("Rows analyzed:", len(result))
    print("Potential anomalies:", len(anomalies))

    return result, anomalies


# ================================================================
# PHASE 3: PATTERN DISCOVERY
# ================================================================

def discover_patterns(df, target):
    print("\n" + "=" * 60)
    print("PHASE 3: PATTERN DISCOVERY")
    print("=" * 60)

    features = [
        column
        for column in df.select_dtypes(
            include=np.number
        ).columns
        if column != target
    ]

    results = []

    for feature in features:

        valid = df[
            [feature, target]
        ].dropna()

        if (
            len(valid) < 3
            or valid[feature].nunique() < 2
            or valid[target].nunique() < 2
        ):
            continue

        results.append({
            "feature": feature,
            "correlation": float(
                valid[feature].corr(
                    valid[target]
                )
            )
        })

    report = pd.DataFrame(results)

    if not report.empty:

        report["absolute_correlation"] = (
            report["correlation"].abs()
        )

        report = report.sort_values(
            "absolute_correlation",
            ascending=False
        )

    print(report.to_string(index=False) if not report.empty
          else "No eligible correlations found.")

    return report


# ================================================================
# PHASE 4: STATISTICAL INVESTIGATION
# ================================================================

def investigate_statistics(df, target):
    print("\n" + "=" * 60)
    print("PHASE 4: STATISTICAL INVESTIGATION")
    print("=" * 60)

    features = [
        column
        for column in df.select_dtypes(
            include=np.number
        ).columns
        if column != target
    ]

    threshold = df[target].quantile(0.10)

    low = df.loc[
        df[target] < threshold
    ]

    baseline = df.loc[
        df[target] >= threshold
    ]

    results = []

    for feature in features:

        a = low[feature].dropna()
        b = baseline[feature].dropna()

        if len(a) < 2 or len(b) < 2:
            continue

        p_value = float(
            mannwhitneyu(
                a,
                b,
                alternative="two-sided"
            ).pvalue
        )

        pooled_std = np.sqrt(
            (
                a.var(ddof=1)
                + b.var(ddof=1)
            ) / 2
        )

        effect_size = (
            abs(a.mean() - b.mean()) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        results.append({
            "feature": feature,
            "low_group_median": float(a.median()),
            "baseline_median": float(b.median()),
            "median_difference": float(
                a.median() - b.median()
            ),
            "effect_size": float(effect_size),
            "p_value": p_value,
            "sample_size": len(a) + len(b)
        })

    report = pd.DataFrame(results)

    if not report.empty:

        _, adjusted, _, _ = multipletests(
            report["p_value"],
            method="fdr_bh"
        )

        report["adjusted_p_value"] = adjusted

        report = report.sort_values(
            ["adjusted_p_value", "effect_size"],
            ascending=[True, False]
        )

    print(report.to_string(index=False) if not report.empty
          else "No eligible statistical comparisons.")

    return report


# ================================================================
# PHASE 5: INVESTIGATION PRIORITIZATION
# ================================================================

def prioritize_investigations(report):
    print("\n" + "=" * 60)
    print("PHASE 5: INVESTIGATION PRIORITIZATION")
    print("=" * 60)

    if report.empty:
        return report.copy()

    result = report.copy()

    max_effect = result["effect_size"].max()
    max_sample = result["sample_size"].max()

    result["effect_score"] = (
        result["effect_size"] / max_effect
        if max_effect > 0
        else 0.0
    )

    result["sample_score"] = (
        result["sample_size"] / max_sample
        if max_sample > 0
        else 0.0
    )

    result["priority_score"] = (
        0.7 * result["effect_score"]
        + 0.3 * result["sample_score"]
    )

    result = result.sort_values(
        "priority_score",
        ascending=False
    )

    print(result.to_string(index=False))

    return result


# ================================================================
# PHASE 6: CONTROLLED CAUSAL ANALYSIS
# ================================================================

def run_causal_experiment():
    print("\n" + "=" * 60)
    print("PHASE 6: CONTROLLED CAUSAL EXPERIMENT")
    print("=" * 60)

    rng = np.random.default_rng(2026)

    n = 5000

    demand = rng.normal(50, 10, n)

    discount = (
        30
        - 0.4 * demand
        + rng.normal(0, 3, n)
    )

    sales = (
        100
        + 2.0 * demand
        + 1.5 * discount
        + rng.normal(0, 5, n)
    )

    df = pd.DataFrame({
        "demand": demand,
        "discount": discount,
        "sales": sales
    })

    naive = sm.OLS(
        df["sales"],
        sm.add_constant(
            df[["discount"]]
        )
    ).fit()

    adjusted = sm.OLS(
        df["sales"],
        sm.add_constant(
            df[["discount", "demand"]]
        )
    ).fit()

    true_effect = 1.5

    report = pd.DataFrame([
        {
            "model": "Unadjusted",
            "estimated_effect": float(
                naive.params["discount"]
            ),
            "standard_error": float(
                naive.bse["discount"]
            ),
            "p_value": float(
                naive.pvalues["discount"]
            )
        },
        {
            "model": "Adjusted for demand",
            "estimated_effect": float(
                adjusted.params["discount"]
            ),
            "standard_error": float(
                adjusted.bse["discount"]
            ),
            "p_value": float(
                adjusted.pvalues["discount"]
            )
        }
    ])

    report["absolute_error"] = (
        report["estimated_effect"] - true_effect
    ).abs()

    print(report.to_string(index=False))
    print("True simulated effect:", true_effect)

    return report, adjusted, true_effect


# ================================================================
# PHASE 7: OPTIONAL GENERATIVE AI
# ================================================================

def generate_ai_explanation(summary):
    print("\n" + "=" * 60)
    print("PHASE 7: GENERATIVE AI")
    print("=" * 60)

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print(
            "No API key configured. "
            "Using the statistical report without an LLM."
        )

        return (
            "Generative AI is not configured. "
            "Review the statistical evidence in the report. "
            "Associations must not be interpreted as proven causes."
        )

    try:

        if importlib.util.find_spec("openai") is None:
            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                "openai"
            ])

        from openai import OpenAI

        client = OpenAI(
            api_key=api_key
        )

        response = client.responses.create(
            model=os.environ.get(
                "AETHER_MODEL",
                "gpt-4.1-mini"
            ),
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are AETHER, a careful data science assistant. "
                        "Explain only evidence provided in the report. "
                        "Do not invent metrics. Distinguish association "
                        "from causation. Explain uncertainty and propose "
                        "useful follow-up tests."
                    )
                },
                {
                    "role": "user",
                    "content": summary
                }
            ]
        )

        explanation = response.output_text

        print(explanation)

        return explanation

    except Exception as exc:

        print(
            "LLM request failed:",
            type(exc).__name__
        )

        return (
            "LLM explanation unavailable. "
            "Use the statistical report instead."
        )


# ================================================================
# PHASE 8: FOLLOW-UP HYPOTHESIS GENERATION
# ================================================================

def generate_hypotheses(report, df):
    print("\n" + "=" * 60)
    print("PHASE 8: FOLLOW-UP HYPOTHESES")
    print("=" * 60)

    hypotheses = []

    if report.empty:

        hypotheses.append(
            "Check data quality and collect more observations."
        )

    else:

        for _, row in report.iterrows():

            feature = row["feature"]

            if row["adjusted_p_value"] < 0.05:

                hypotheses.append(
                    f"Investigate whether the observed difference "
                    f"in {feature} persists across time periods "
                    "and relevant subgroups."
                )

            else:

                hypotheses.append(
                    f"Review statistical power and measurement "
                    f"quality for {feature} before concluding "
                    "that no meaningful difference exists."
                )

    for column in df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns:

        hypotheses.append(
            f"Investigate target variation across categories "
            f"of {column}."
        )

    hypotheses.extend([
        "Check whether the findings persist in an independent dataset.",
        "Investigate whether missing data could bias the findings.",
        "Define a causal diagram before selecting adjustment variables.",
        "Compare findings against business and domain-specific baselines."
    ])

    for index, hypothesis in enumerate(
        hypotheses,
        start=1
    ):
        print(f"{index}. {hypothesis}")

    return hypotheses


# ================================================================
# PHASE 9: VISUALIZATION AND DASHBOARD GENERATION
# ================================================================

def generate_dashboard():
    print("\n" + "=" * 60)
    print("PHASE 9: DASHBOARD GENERATION")
    print("=" * 60)

    dashboard_code = r'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

st.set_page_config(
    page_title="PROJECT AETHER",
    layout="wide"
)

st.title("PROJECT AETHER")
st.subheader("AI-Assisted Data Investigation")

uploaded = st.file_uploader(
    "Upload CSV dataset",
    type=["csv"]
)

if uploaded is not None:

    df = pd.read_csv(uploaded)

    numeric = list(
        df.select_dtypes(include=np.number).columns
    )

    if len(numeric) < 2:
        st.error("At least two numeric columns are required.")
        st.stop()

    target = st.selectbox(
        "Select target variable",
        numeric
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna(
        subset=[target]
    ).reset_index(drop=True)

    if len(df) < 20:
        st.error("At least 20 rows are recommended.")
        st.stop()

    threshold_percent = st.slider(
        "Low-value percentile",
        1,
        30,
        10
    )

    threshold = df[target].quantile(
        threshold_percent / 100
    )

    low = df[df[target] < threshold]
    baseline = df[df[target] >= threshold]

    st.metric("Rows analyzed", len(df))
    st.metric("Low-value observations", len(low))

    features = [
        c for c in numeric if c != target
    ]

    results = []

    for feature in features:

        a = low[feature].dropna()
        b = baseline[feature].dropna()

        if len(a) < 2 or len(b) < 2:
            continue

        p = float(
            mannwhitneyu(
                a,
                b,
                alternative="two-sided"
            ).pvalue
        )

        pooled_std = np.sqrt(
            (a.var(ddof=1) + b.var(ddof=1)) / 2
        )

        effect = (
            abs(a.mean() - b.mean()) / pooled_std
            if pooled_std > 0
            else 0.0
        )

        results.append({
            "feature": feature,
            "low_group_median": float(a.median()),
            "baseline_median": float(b.median()),
            "effect_size": float(effect),
            "p_value": p
        })

    report = pd.DataFrame(results)

    if not report.empty:

        _, adjusted, _, _ = multipletests(
            report["p_value"],
            method="fdr_bh"
        )

        report["adjusted_p_value"] = adjusted

        report = report.sort_values(
            "adjusted_p_value"
        )

        st.subheader("Statistical investigation")

        st.dataframe(
            report,
            use_container_width=True
        )

        feature = st.selectbox(
            "Feature to visualize",
            report["feature"].tolist()
        )

        chart_data = df.copy()

        chart_data["group"] = np.where(
            chart_data[target] < threshold,
            "Low-value group",
            "Baseline"
        )

        fig = px.box(
            chart_data,
            x="group",
            y=feature,
            color="group",
            title=f"{feature} by target group"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.warning(
            "Statistical associations do not prove causation."
        )

    else:
        st.info("No eligible feature comparisons were found.")

else:
    st.info("Upload a CSV to start.")
'''

    dashboard_path = OUTPUT_DIR / "aether_app.py"

    dashboard_path.write_text(
        dashboard_code.strip() + "\n",
        encoding="utf-8"
    )

    print("Dashboard saved:", dashboard_path)

    return dashboard_path


# ================================================================
# PHASE 10: AUTOMATED TESTING
# ================================================================

def run_tests(df, target, causal_report, true_effect):
    print("\n" + "=" * 60)
    print("PHASE 10: AUTOMATED TESTING")
    print("=" * 60)

    tests = []

    def add_test(name, condition, details=""):
        tests.append({
            "test": name,
            "status": "PASS" if condition else "FAIL",
            "details": details
        })

    add_test(
        "Dataset has sufficient rows",
        len(df) >= 20
    )

    add_test(
        "Target is numeric",
        pd.api.types.is_numeric_dtype(df[target])
    )

    add_test(
        "Causal report contains estimates",
        not causal_report.empty
    )

    adjusted = causal_report.loc[
        causal_report["model"] == "Adjusted for demand",
        "estimated_effect"
    ]

    estimate_ok = (
        len(adjusted) == 1
        and abs(float(adjusted.iloc[0]) - true_effect) < 0.15
    )

    add_test(
        "Adjusted simulated causal estimate is accurate",
        estimate_ok
    )

    result = pd.DataFrame(tests)

    print(result.to_string(index=False))

    return result


# ================================================================
# PHASE 11: REPORTS AND DOCUMENTATION
# ================================================================

def save_outputs(
    data,
    anomalies,
    patterns,
    statistics,
    planner,
    causal,
    tests,
    summary,
    explanation,
    hypotheses,
    target
):
    print("\n" + "=" * 60)
    print("PHASE 11: SAVING PROJECT OUTPUTS")
    print("=" * 60)

    files = {
        "analyzed_data.csv": data,
        "anomalies.csv": anomalies,
        "correlations.csv": patterns,
        "investigation_results.csv": statistics,
        "planner_results.csv": planner,
        "causal_results.csv": causal,
        "test_results.csv": tests
    }

    for filename, frame in files.items():
        frame.to_csv(
            OUTPUT_DIR / filename,
            index=False
        )

    (OUTPUT_DIR / "investigation_summary.txt").write_text(
        summary + "\n\nAI EXPLANATION\n\n" + explanation,
        encoding="utf-8"
    )

    (OUTPUT_DIR / "hypotheses.json").write_text(
        json.dumps(
            hypotheses,
            indent=2
        ),
        encoding="utf-8"
    )

    readme = f"""
# PROJECT AETHER

AI-Assisted Data Investigation System.

## Current configuration

Target variable: {target}

## Features

- Data preparation and validation
- Isolation Forest anomaly detection
- Correlation analysis
- Mann–Whitney statistical comparisons
- False discovery rate adjustment
- Heuristic investigation prioritization
- Controlled causal regression experiment
- Optional Generative AI explanations
- Follow-up hypothesis generation
- Streamlit dashboard
- Automated smoke tests
- Exportable reports

## Running the dashboard

Install requirements:

    pip install -r requirements.txt

Launch:

    python -m streamlit run aether_app.py

## Limitations

Anomaly scores identify unusual observations, not necessarily errors.

Correlations and group differences do not prove causation.

The investigation planner uses a heuristic rather than a
formally optimized information-gain strategy.

The causal experiment uses synthetic data. It does not establish
causal effects in real observational datasets.

LLM integration requires a valid API key and model access.

Automated tests are basic checks, not comprehensive validation.
"""

    (OUTPUT_DIR / "README.md").write_text(
        readme.strip() + "\n",
        encoding="utf-8"
    )

    requirements = """
numpy
pandas
scipy
statsmodels
scikit-learn
plotly
streamlit
openai
"""

    (OUTPUT_DIR / "requirements.txt").write_text(
        requirements.strip() + "\n",
        encoding="utf-8"
    )

    deployment = """
# Deployment instructions

1. Review the README and generated reports.
2. Place the dashboard and requirements in a Git repository.
3. Install dependencies.
4. Run `python -m streamlit run aether_app.py`.
5. Test with independent CSV datasets.
6. Configure any API credentials through a secure secret store.

Never commit API keys or private datasets to GitHub.
"""

    (OUTPUT_DIR / "DEPLOYMENT.md").write_text(
        deployment.strip() + "\n",
        encoding="utf-8"
    )

    print("Saved project outputs to:", OUTPUT_DIR)


# ================================================================
# PHASE 12: PACKAGE FOR DISTRIBUTION
# ================================================================                                                                                                                                                                                       

def package_project():
    print("\n" + "=" * 60)
    print("PHASE 12: PROJECT PACKAGING")
    print("=" * 60)

    import zipfile
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "PROJECT_AETHER_OUTPUT"
    archive_path = base_dir / "PROJECT_AETHER_OUTPUT.zip"

    if not output_dir.exists():
        print("ERROR: PROJECT_AETHER_OUTPUT folder not found.")
        return None

    try:
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED
        ) as archive:

            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    archive_name = (
                        Path("PROJECT_AETHER_OUTPUT")
                        / file_path.relative_to(output_dir)
                    )

                    archive.write(
                        file_path,
                        arcname=str(archive_name)
                    )

        print("Project packaged successfully!")
        print(f"ZIP archive created: {archive_path}")

        return archive_path

    except OSError as error:
        print(f"Packaging failed: {error}")
        return None


# ================================================================
# MAIN EXECUTION PIPELINE
# ================================================================

def main():
    data, target = load_data()
    data, anomalies = detect_anomalies(data, target)
    patterns = discover_patterns(data, target)
    statistics = investigate_statistics(data, target)
    planner = prioritize_investigations(statistics)

    causal, adjusted_model, true_effect = run_causal_experiment()

    summary_lines = [
        "PROJECT AETHER - INVESTIGATION SUMMARY",
        "=" * 50,
        f"Rows analyzed: {len(data)}",
        f"Target variable: {target}",
        f"Potential anomalies: {len(anomalies)}",
        "",
        "Statistical findings:"
    ]

    if statistics.empty:
        summary_lines.append("No eligible statistical comparisons found.")
    else:
        for _, row in statistics.iterrows():
            summary_lines.append(
                f"{row['feature']}: "
                f"adjusted p-value = "
                f"{row['adjusted_p_value']:.5f}"
            )

    summary_lines.extend([
        "",
        "Associations do not prove causation.",
        "The causal experiment uses synthetic data."
    ])

    summary = "\n".join(summary_lines)
    explanation = generate_ai_explanation(summary)
    hypotheses = generate_hypotheses(statistics, data)
    dashboard_path = generate_dashboard()

    tests = run_tests(
        data,
        target,
        causal,
        true_effect
    )

    save_outputs(
        data,
        anomalies,
        patterns,
        statistics,
        planner,
        causal,
        tests,
        summary,
        explanation,
        hypotheses,
        target
    )

    archive = package_project()

    print("\n" + "=" * 60)
    print("PROJECT AETHER EXECUTION COMPLETED")
    print("=" * 60)
    print("Dataset rows:", len(data))
    print("Target variable:", target)
    print("Detected anomalies:", len(anomalies))
    print("Statistical comparisons:", len(statistics))
    print("Generated hypotheses:", len(hypotheses))
    print("Dashboard:", dashboard_path)
    print("Distribution ZIP:", archive)

    passed = (tests["status"] == "PASS").sum()
    failed = (tests["status"] == "FAIL").sum()

    print("Tests passed:", passed)
    print("Tests failed:", failed)

    print("\nTo start the dashboard, run:")
    print(
        f'python -m streamlit run "{dashboard_path}"'
    )


# ================================================================
# PROGRAM ENTRY POINT
# ================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as exc:
        print("\nPROJECT AETHER ERROR:", type(exc).__name__, str(exc))
        raise
