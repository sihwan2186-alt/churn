import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_FILE = Path("Baza customer Telecom v2.csv")
OUTPUT_ROOT = Path("processed")
PHASE_OUTPUT = OUTPUT_ROOT / "phase_5b_business_impact"

COMPARISON_FILE = OUTPUT_ROOT / "model_comparison_billing_zip.csv"
THRESHOLD_BEST_FILE = OUTPUT_ROOT / "threshold_tuning_best.csv"
COST_BEST_FILE = (
    OUTPUT_ROOT / "phase_3b_differentiation" / "experiment_d_cost_threshold_best.csv"
)

PAPER_ANNUAL_ARPU = 5400.0
PAPER_RETENTION_RATE = 0.60
PAPER_CAMPAIGN_COST = 120.0
PAPER_TP_BENEFIT = PAPER_ANNUAL_ARPU * PAPER_RETENTION_RATE
PAPER_REPORTED_NET_BENEFIT = 74200.0


def ensure_output_dir() -> None:
    PHASE_OUTPUT.mkdir(parents=True, exist_ok=True)


def normalize_churn(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype(int)


def load_financial_profile() -> pd.DataFrame:
    raw = pd.read_csv(INPUT_FILE)
    raw.columns = raw.columns.str.strip()

    for col in ["ARPU", "TotalRevenue", "AvgMobileRevenue", "AvgFIXRevenue"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw["CHURN_bin"] = normalize_churn(raw["CHURN"])

    dedup = raw.drop_duplicates(subset=["PID"]).copy()

    rows = []
    for dataset_name, df in [("raw", raw), ("pid_deduplicated", dedup)]:
        for group_name, group_df in [
            ("all", df),
            ("churned", df[df["CHURN_bin"].eq(1)]),
            ("non_churned", df[df["CHURN_bin"].eq(0)]),
        ]:
            rows.append(
                {
                    "dataset": dataset_name,
                    "group": group_name,
                    "n_accounts": int(len(group_df)),
                    "churn_rate": float(group_df["CHURN_bin"].mean())
                    if len(group_df)
                    else np.nan,
                    "arpu_mean": float(group_df["ARPU"].mean()),
                    "arpu_median": float(group_df["ARPU"].median()),
                    "annual_arpu_mean": float(group_df["ARPU"].mean() * 12.0),
                    "total_revenue_mean": float(group_df["TotalRevenue"].mean()),
                    "total_revenue_median": float(group_df["TotalRevenue"].median()),
                    "avg_mobile_revenue_mean": float(group_df["AvgMobileRevenue"].mean()),
                    "avg_fixed_revenue_mean": float(group_df["AvgFIXRevenue"].mean()),
                }
            )

    profile = pd.DataFrame(rows)
    profile.to_csv(
        PHASE_OUTPUT / "arpu_financial_profile.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return profile


def get_single_row(df: pd.DataFrame, **filters: object) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, value in filters.items():
        mask &= df[col].eq(value)
    subset = df.loc[mask]
    if subset.empty:
        raise ValueError(f"No row found for filters: {filters}")
    return subset.iloc[0]


def metric_from_comparison(row: pd.Series) -> dict:
    return {
        "threshold": 0.50,
        "f1": float(row["f1"]),
        "recall": float(row["recall"]),
        "precision": float(row["precision"]),
        "pr_auc": float(row["pr_auc"]),
        "tp": int(row["tp"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tn": int(row["tn"]),
    }


def metric_from_threshold(row: pd.Series) -> dict:
    return {
        "threshold": float(row["selected_threshold"]),
        "f1": float(row["test_f1"]),
        "recall": float(row["test_recall"]),
        "precision": float(row["test_precision"]),
        "pr_auc": float(row["test_pr_auc"]),
        "tp": int(row["test_tp"]),
        "fp": int(row["test_fp"]),
        "fn": int(row["test_fn"]),
        "tn": int(row["test_tn"]),
    }


def metric_from_cost_best(row: pd.Series) -> dict:
    return {
        "threshold": float(row["threshold"]),
        "f1": float(row["f1"]),
        "recall": float(row["recall"]),
        "precision": float(row["precision"]),
        "pr_auc": np.nan,
        "tp": int(row["tp"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tn": int(row["tn"]),
    }


def add_business_metrics(row: dict) -> dict:
    tp = int(row["tp"])
    fp = int(row["fp"])
    fn = int(row["fn"])
    tn = None if pd.isna(row.get("tn", np.nan)) else int(row["tn"])

    contacts = tp + fp
    churners = tp + fn
    population = contacts + fn + (0 if tn is None else tn)
    tp_revenue = tp * PAPER_TP_BENEFIT
    fp_cost = fp * PAPER_CAMPAIGN_COST
    net_benefit_formula = tp_revenue - fp_cost
    campaign_spend_all_contacts = contacts * PAPER_CAMPAIGN_COST
    break_even_fp_max = tp * (PAPER_TP_BENEFIT / PAPER_CAMPAIGN_COST)

    row.update(
        {
            "annual_arpu_assumption": PAPER_ANNUAL_ARPU,
            "retention_rate": PAPER_RETENTION_RATE,
            "tp_benefit": PAPER_TP_BENEFIT,
            "fp_campaign_cost": PAPER_CAMPAIGN_COST,
            "contacts": contacts,
            "test_churners": churners,
            "test_population": population,
            "tp_revenue": tp_revenue,
            "fp_cost": fp_cost,
            "net_benefit_formula": net_benefit_formula,
            "paper_reported_net_benefit": row.get("paper_reported_net_benefit", np.nan),
            "net_vs_paper_reported": net_benefit_formula - PAPER_REPORTED_NET_BENEFIT,
            "net_multiple_vs_paper": net_benefit_formula / PAPER_REPORTED_NET_BENEFIT,
            "campaign_spend_all_contacts": campaign_spend_all_contacts,
            "gross_roi_vs_fp_cost": tp_revenue / fp_cost if fp_cost else np.nan,
            "net_roi_vs_fp_cost": net_benefit_formula / fp_cost if fp_cost else np.nan,
            "gross_roi_vs_all_contact_spend": tp_revenue / campaign_spend_all_contacts
            if campaign_spend_all_contacts
            else np.nan,
            "net_roi_vs_all_contact_spend": net_benefit_formula
            / campaign_spend_all_contacts
            if campaign_spend_all_contacts
            else np.nan,
            "wasted_fp_cost_per_contact": fp_cost / contacts if contacts else np.nan,
            "wasted_fp_cost_per_tp": fp_cost / tp if tp else np.nan,
            "break_even_fp_max": break_even_fp_max,
            "break_even_fp_margin": break_even_fp_max - fp,
        }
    )
    return row


def build_operating_points() -> pd.DataFrame:
    comparison = pd.read_csv(COMPARISON_FILE)
    threshold_best = pd.read_csv(THRESHOLD_BEST_FILE)
    cost_best = pd.read_csv(COST_BEST_FILE)

    operating_points = [
        add_business_metrics(
            {
                "operating_point": "no_model_current_test_baseline",
                "model_family": "No Model",
                "source": "baseline",
                "variant": "current_test_distribution",
                "model": "No Model",
                "threshold": np.nan,
                "f1": 0.0,
                "recall": 0.0,
                "precision": 0.0,
                "pr_auc": np.nan,
                "tp": 0,
                "fp": 0,
                "fn": 109,
                "tn": 1579,
            }
        ),
        add_business_metrics(
            {
                "operating_point": "paper_easyensemble_reported",
                "model_family": "Paper",
                "source": "paper_reverse_engineered",
                "variant": "paper",
                "model": "EasyEnsembleClassifier",
                "threshold": 0.35,
                "f1": 0.129,
                "recall": 0.382,
                "precision": 0.077,
                "pr_auc": 0.079,
                "tp": 42,
                "fp": 503,
                "fn": 68,
                "tn": np.nan,
                "paper_reported_net_benefit": PAPER_REPORTED_NET_BENEFIT,
            }
        ),
    ]

    fixed_specs = [
        (
            "ours_lr_f1_best_fixed",
            "Efficiency",
            "fixed_model_comparison",
            {"variant": "without_billing_zip", "model": "LogisticRegression_SMOTE"},
        ),
        (
            "ours_balancedbagging_balanced_fixed",
            "Balanced",
            "fixed_model_comparison",
            {"variant": "with_billing_zip", "model": "BalancedBagging_original"},
        ),
    ]
    for operating_point, family, source, filters in fixed_specs:
        row = get_single_row(comparison, **filters)
        operating_points.append(
            add_business_metrics(
                {
                    "operating_point": operating_point,
                    "model_family": family,
                    "source": source,
                    "variant": str(row["variant"]),
                    "model": str(row["model"]),
                    **metric_from_comparison(row),
                }
            )
        )

    threshold_specs = [
        (
            "ours_catboost_recall_heavy_threshold",
            "Recall Heavy",
            "validation_threshold_tuned",
            {"variant": "with_billing_zip", "model": "CatBoost_native_categorical"},
        ),
        (
            "ours_xgboost_recall_extreme_threshold",
            "Recall Heavy",
            "validation_threshold_tuned",
            {"variant": "with_billing_zip", "model": "XGBoost_SMOTE"},
        ),
    ]
    for operating_point, family, source, filters in threshold_specs:
        row = get_single_row(threshold_best, **filters)
        operating_points.append(
            add_business_metrics(
                {
                    "operating_point": operating_point,
                    "model_family": family,
                    "source": source,
                    "variant": str(row["variant"]),
                    "model": str(row["model"]),
                    **metric_from_threshold(row),
                }
            )
        )

    paper_cost_best = get_single_row(
        cost_best,
        variant="with_billing_zip",
        model="BalancedBagging_original",
        scenario="paper_baseline",
    )
    operating_points.append(
        add_business_metrics(
            {
                "operating_point": "ours_balancedbagging_cost_optimized",
                "model_family": "Cost Optimized",
                "source": "phase_3b_cost_threshold_sweep",
                "variant": str(paper_cost_best["variant"]),
                "model": str(paper_cost_best["model"]),
                **metric_from_cost_best(paper_cost_best),
            }
        )
    )

    operating = pd.DataFrame(operating_points)
    operating.to_csv(
        PHASE_OUTPUT / "business_impact_operating_points.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return operating


def build_scenarios(operating: pd.DataFrame) -> pd.DataFrame:
    candidates = operating[
        ~operating["operating_point"].isin(
            ["no_model_current_test_baseline", "paper_easyensemble_reported"]
        )
    ].copy()
    core_points = candidates[
        candidates["operating_point"].isin(
            [
                "ours_lr_f1_best_fixed",
                "ours_balancedbagging_balanced_fixed",
                "ours_catboost_recall_heavy_threshold",
            ]
        )
    ].copy()
    fixed_candidates = candidates[candidates["source"].ne("phase_3b_cost_threshold_sweep")]

    scenarios = []
    scenario_specs = [
        (
            "budget_limit_500_contacts",
            "Campaign budget limited to 500 contacts",
            candidates[candidates["contacts"].le(500)],
        ),
        (
            "team_capacity_800_contacts",
            "Campaign team capacity limited to 800 contacts",
            candidates[candidates["contacts"].le(800)],
        ),
        (
            "unlimited_core_three_operating_points",
            "Revenue protection among LR, BalancedBagging, and CatBoost operating points",
            core_points,
        ),
        (
            "unlimited_fixed_operating_points",
            "Revenue protection including validation-tuned XGBoost",
            fixed_candidates,
        ),
        (
            "unlimited_threshold_candidates",
            "Revenue protection including validation-tuned XGBoost",
            candidates[candidates["source"].ne("phase_3b_cost_threshold_sweep")],
        ),
        (
            "unlimited_cost_optimized",
            "Revenue protection with cost-sensitive threshold optimization",
            candidates,
        ),
    ]

    for name, description, pool in scenario_specs:
        if pool.empty:
            continue
        best = pool.sort_values("net_benefit_formula", ascending=False).iloc[0]
        scenarios.append(
            {
                "scenario": name,
                "description": description,
                "recommended_operating_point": best["operating_point"],
                "recommended_model": best["model"],
                "threshold": best["threshold"],
                "contacts": int(best["contacts"]),
                "tp": int(best["tp"]),
                "fp": int(best["fp"]),
                "fn": int(best["fn"]),
                "recall": float(best["recall"]),
                "precision": float(best["precision"]),
                "net_benefit_formula": float(best["net_benefit_formula"]),
                "net_vs_paper_reported": float(best["net_vs_paper_reported"]),
                "net_multiple_vs_paper": float(best["net_multiple_vs_paper"]),
                "gross_roi_vs_fp_cost": float(best["gross_roi_vs_fp_cost"]),
                "net_roi_vs_fp_cost": float(best["net_roi_vs_fp_cost"]),
            }
        )

    scenario_df = pd.DataFrame(scenarios)
    scenario_df.to_csv(
        PHASE_OUTPUT / "business_impact_scenarios.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return scenario_df


def save_break_even_table(operating: pd.DataFrame) -> pd.DataFrame:
    table = operating[
        [
            "operating_point",
            "model",
            "tp",
            "fp",
            "break_even_fp_max",
            "break_even_fp_margin",
            "net_benefit_formula",
        ]
    ].copy()
    table = table[~table["operating_point"].eq("no_model_current_test_baseline")]
    table.to_csv(
        PHASE_OUTPUT / "business_impact_break_even.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return table


def plot_bubble(operating: pd.DataFrame) -> None:
    rows = operating[
        operating["operating_point"].isin(
            [
                "paper_easyensemble_reported",
                "ours_lr_f1_best_fixed",
                "ours_balancedbagging_balanced_fixed",
                "ours_catboost_recall_heavy_threshold",
                "ours_balancedbagging_cost_optimized",
            ]
        )
    ].copy()

    label_map = {
        "paper_easyensemble_reported": "Paper EasyEnsemble",
        "ours_lr_f1_best_fixed": "LR fixed",
        "ours_balancedbagging_balanced_fixed": "BalancedBagging fixed",
        "ours_catboost_recall_heavy_threshold": "CatBoost recall-heavy",
        "ours_balancedbagging_cost_optimized": "BalancedBagging cost-opt",
    }
    colors = {
        "paper_easyensemble_reported": "#7f8c8d",
        "ours_lr_f1_best_fixed": "#3498db",
        "ours_balancedbagging_balanced_fixed": "#2ecc71",
        "ours_catboost_recall_heavy_threshold": "#e74c3c",
        "ours_balancedbagging_cost_optimized": "#9b59b6",
    }
    markers = {
        "paper_easyensemble_reported": "D",
        "ours_lr_f1_best_fixed": "o",
        "ours_balancedbagging_balanced_fixed": "s",
        "ours_catboost_recall_heavy_threshold": "^",
        "ours_balancedbagging_cost_optimized": "P",
    }

    fig, ax = plt.subplots(figsize=(11, 7))
    for _, row in rows.iterrows():
        point = row["operating_point"]
        size = max(row["net_benefit_formula"], 0) / 240.0
        ax.scatter(
            row["fp"],
            row["tp"],
            s=size,
            color=colors[point],
            marker=markers[point],
            alpha=0.78,
            edgecolors="black",
            linewidth=1.4,
            label=f"{label_map[point]} ({row['net_benefit_formula']:,.0f})",
            zorder=5,
        )
        ax.annotate(
            f"{label_map[point]}\n{row['net_benefit_formula']/1000:.0f}K",
            (row["fp"], row["tp"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )

    fp_range = np.linspace(0, max(rows["fp"].max() * 1.08, 100), 200)
    tp_break = fp_range * PAPER_CAMPAIGN_COST / PAPER_TP_BENEFIT
    ax.plot(fp_range, tp_break, "--", color="#2c3e50", linewidth=1.2, alpha=0.65)
    ax.fill_between(
        fp_range,
        tp_break,
        np.full_like(fp_range, rows["tp"].max() * 1.15),
        color="#27ae60",
        alpha=0.06,
    )
    ax.set_xlabel("False positives (unnecessary campaign contacts)")
    ax.set_ylabel("True positives (churners captured)")
    ax.set_title("Business Impact by Operating Point\nBubble size = net benefit")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(PHASE_OUTPUT / "business_impact_bubble.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scenarios(scenarios: pd.DataFrame) -> None:
    display = scenarios[
        scenarios["scenario"].isin(
            [
                "budget_limit_500_contacts",
                "team_capacity_800_contacts",
                "unlimited_core_three_operating_points",
                "unlimited_cost_optimized",
            ]
        )
    ].copy()
    labels = [
        "Budget <=500",
        "Capacity <=800",
        "Unlimited core",
        "Unlimited cost-opt",
    ]
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]

    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(labels, display["net_benefit_formula"], color=colors, alpha=0.82)
    ax.axhline(
        PAPER_REPORTED_NET_BENEFIT,
        color="#7f8c8d",
        linestyle="--",
        linewidth=1.6,
        label="Paper reported net benefit",
    )
    for bar, (_, row) in zip(bars, display.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3000,
            f"{row['net_benefit_formula']:,.0f}\n{row['recommended_model']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Net benefit")
    ax.set_title("Recommended Model by Operating Scenario")
    ax.grid(axis="y", alpha=0.28)
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        PHASE_OUTPUT / "business_impact_scenarios.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)


def plot_dashboard(operating: pd.DataFrame) -> None:
    rows = operating[
        operating["operating_point"].isin(
            [
                "paper_easyensemble_reported",
                "ours_lr_f1_best_fixed",
                "ours_balancedbagging_balanced_fixed",
                "ours_catboost_recall_heavy_threshold",
                "ours_balancedbagging_cost_optimized",
            ]
        )
    ].copy()
    labels = [
        "Paper\nEasyEns.",
        "LR\nfixed",
        "Balanced\nfixed",
        "CatBoost\nrecall",
        "Balanced\ncost-opt",
    ]

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 4, hspace=0.42, wspace=0.28)

    kpis = [
        (
            "Max Net Benefit",
            f"{rows['net_benefit_formula'].max():,.0f}",
            "BalancedBagging cost-opt",
            "#9b59b6",
        ),
        (
            "Best Fixed Net",
            f"{rows[rows['source'].ne('phase_3b_cost_threshold_sweep')]['net_benefit_formula'].max():,.0f}",
            "CatBoost recall-heavy",
            "#e74c3c",
        ),
        (
            "Highest Gross ROI",
            f"{rows.loc[rows['operating_point'].eq('ours_lr_f1_best_fixed'), 'gross_roi_vs_fp_cost'].iloc[0]:.2f}x",
            "LR fixed",
            "#3498db",
        ),
        (
            "Paper Multiple",
            f"{rows['net_multiple_vs_paper'].max():.2f}x",
            "vs 74,200 reported",
            "#f39c12",
        ),
    ]
    for idx, (title, value, subtitle, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor(color)
        ax.text(
            0.5,
            0.63,
            value,
            ha="center",
            va="center",
            fontsize=22,
            fontweight="bold",
            color="white",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.35,
            title,
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.15,
            subtitle,
            ha="center",
            va="center",
            fontsize=8,
            color="#fff8dc",
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(rows))
    width = 0.34
    ax.bar(
        x - width / 2,
        rows["tp_revenue"],
        width,
        label="TP benefit",
        color="#27ae60",
        alpha=0.82,
    )
    ax.bar(
        x + width / 2,
        rows["fp_cost"],
        width,
        label="FP wasted cost",
        color="#c0392b",
        alpha=0.82,
    )
    ax.plot(
        x,
        rows["net_benefit_formula"],
        "ko-",
        linewidth=2,
        markersize=7,
        label="Net benefit",
    )
    for xi, value in zip(x, rows["net_benefit_formula"]):
        ax.annotate(
            f"{value:,.0f}",
            (xi, value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            fontweight="bold",
        )
    ax.axhline(
        PAPER_REPORTED_NET_BENEFIT,
        color="#7f8c8d",
        linestyle="--",
        linewidth=1.4,
        label="Paper reported 74,200",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Amount")
    ax.set_title("Revenue-Cost Structure under Paper Cost Parameters")
    ax.grid(axis="y", alpha=0.28)
    ax.legend()
    plt.tight_layout()
    fig.savefig(
        PHASE_OUTPUT / "business_impact_dashboard.png", dpi=160, bbox_inches="tight"
    )
    plt.close(fig)


def save_summary(
    profile: pd.DataFrame,
    operating: pd.DataFrame,
    scenarios: pd.DataFrame,
    break_even: pd.DataFrame,
) -> None:
    dedup_churned = profile[
        profile["dataset"].eq("pid_deduplicated") & profile["group"].eq("churned")
    ].iloc[0]
    best_overall = operating[
        ~operating["operating_point"].eq("no_model_current_test_baseline")
    ].sort_values("net_benefit_formula", ascending=False).iloc[0]
    best_fixed = operating[
        operating["source"].isin(["fixed_model_comparison", "validation_threshold_tuned"])
    ].sort_values("net_benefit_formula", ascending=False).iloc[0]
    lr_row = operating[operating["operating_point"].eq("ours_lr_f1_best_fixed")].iloc[0]

    summary = {
        "cost_parameters": {
            "annual_arpu": PAPER_ANNUAL_ARPU,
            "retention_rate": PAPER_RETENTION_RATE,
            "tp_benefit": PAPER_TP_BENEFIT,
            "fp_campaign_cost": PAPER_CAMPAIGN_COST,
            "paper_reported_net_benefit": PAPER_REPORTED_NET_BENEFIT,
        },
        "deduplicated_churned_financial_profile": dedup_churned.to_dict(),
        "best_overall_operating_point": best_overall.to_dict(),
        "best_fixed_or_validation_operating_point": best_fixed.to_dict(),
        "lr_efficiency_operating_point": lr_row.to_dict(),
        "scenario_recommendations": scenarios.to_dict(orient="records"),
        "break_even_margin_min": float(break_even["break_even_fp_margin"].min()),
    }
    with (PHASE_OUTPUT / "phase_5b_business_impact_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    ensure_output_dir()
    profile = load_financial_profile()
    operating = build_operating_points()
    scenarios = build_scenarios(operating)
    break_even = save_break_even_table(operating)

    plot_bubble(operating)
    plot_scenarios(scenarios)
    plot_dashboard(operating)
    save_summary(profile, operating, scenarios, break_even)

    print("=== Phase 5-B Business Impact ===")
    print(f"Outputs: {PHASE_OUTPUT}")
    print("\nOperating points:")
    cols = [
        "operating_point",
        "model",
        "threshold",
        "tp",
        "fp",
        "fn",
        "contacts",
        "net_benefit_formula",
        "gross_roi_vs_fp_cost",
        "net_multiple_vs_paper",
    ]
    print(operating[cols].to_string(index=False))
    print("\nScenario recommendations:")
    print(
        scenarios[
            [
                "scenario",
                "recommended_model",
                "threshold",
                "contacts",
                "net_benefit_formula",
                "net_multiple_vs_paper",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
