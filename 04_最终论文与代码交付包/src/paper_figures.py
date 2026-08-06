#!/usr/bin/env python3
"""Generate the complete publication figure set for the cement ESP paper.

All empirical plots are regenerated from the supplied processed dataset and
model result tables. Scenario-only plots are explicitly labelled in captions.
Both PNG previews and vector PDF files are exported.
"""

from __future__ import annotations

import sys
import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plot_utils import (GRAY, LIGHT_GRAY, NEUTRAL_FILL, PALETTE, add_subpanel_label,
                        figure_legend, finish_axes, headroom, note, save_figure, set_paper_style)
from scenario_model import RidgePowerModel, T_COLS, U_COLS


ROOT = SRC_DIR.parent
DATA_FILE = ROOT / "data" / "full_timeseries_with_flags.csv"
RESULTS = ROOT / "results"
OUT = ROOT / "figures_paper"

COND_SHORT = {
    1: "工况1\n低温低浓高流",
    2: "工况2\n高温中浓低流",
    3: "工况3\n高温高浓高流",
    4: "工况4\n低温高浓低流",
}

CAPTIONS = {
    "01_outlet_data_diagnosis": "出口浓度记录存在显著截顶且不覆盖目标区间。有效记录全部位于48.74～50.00 mg/Nm³，其中大量观测等于50 mg/Nm³，因而不能从该列直接辨识5或10 mg/Nm³附近的控制—排放关系。",
    "02_process_timeseries": "原始过程量具有连续时序变化和多工况切换特征。图中展示首24小时入口温度、入口浓度、烟气流量及总功率的分钟级变化。",
    "03_cluster_selection": "四类工况在轮廓系数与解释复杂度之间取得较好平衡。k=4的轮廓系数达到0.408，且避免继续增加聚类数带来的解释负担。",
    "04_condition_map": "四类工况在入口浓度—温度平面上形成清晰分区，并由烟气流量大小补充刻画负荷差异。",
    "05_condition_profile_heatmap": "标准化工况画像揭示温度、浓度、流量、粉尘负荷与历史总功率之间的结构差异，为分工况控制提供依据。",
    "06_method_overview": "本文采用数据驱动功率模型与工程灰箱排放模型协同的分层优化框架。实测数据负责功率与工况辨识，排放约束仅在显式先验情景下推演。",
    "07_power_prediction": "滚动验证选择的二次岭回归优于均值与线性基线；回顾性测试R²为0.957、RMSE为14.76 kW。左图虚线表示理想预测，右图比较两时段RMSE。",
    "08_power_residuals": "回顾性测试残差均值为−10.78 kW，且在高功率区负偏差有所扩大；验证集绝对误差90%分位数11.78 kW作为验证期误差附加量。",
    "08b_feature_importance": "二次岭功率模型的标准化系数显示，电压及其二次交互项是主要预测信息来源。系数反映相关性和预测贡献，不解释为单变量因果效应。",
    "09_voltage_power_response": "在各工况典型状态下，模型给出的总功率随四电场电压同步提高而上升。曲线仅在各工况历史电压支持区间附近绘制。",
    "10_optimal_power": "更严格的5 mg/Nm³情景约束在四类工况下均要求更高预测功率；中心情景工况加权平均功率由1609.42 kW升至1829.79 kW。",
    "11_optimal_voltage": "5 mg/Nm³情景主要通过提高前级电场电压并适度调整后级电压实现更高捕集强度。各柱为随机候选搜索得到的支持域内最优解。",
    "12_optimal_rapping_period": "最优振打周期在不同工况和排放限值间发生协同调整，前两电场周期较短、后两电场周期较长的工程顺序约束始终得到保持。",
    "12b_radar_controls": "代表工况的电压和振打周期剖面表明，限值收紧时前级电压增量更突出，但后级补偿与周期协同仍不可忽略。",
    "13_emission_decomposition": "最优方案的约束值由连续排放基值和振打峰值增量共同构成。柱顶虚线分别表示10和5 mg/Nm³情景限值。",
    "14_rapping_peak_mechanism": "灰箱情景表明，振打周期相对参考最优周期延长会放大单次再飞扬峰值；高粉尘负荷工况的峰值增幅更显著。",
    "15_search_convergence": "五个独立随机种子与局部细化得到的最低预测功率高度稳定；八个工况—限值组合的最大变异系数为0.181%，局部细化相对初始随机库最多改善0.986%。",
    "16_sensitivity_distribution": "405组结构情景全部在当前支持域内可行；5 mg/Nm³相对10 mg/Nm³的加权功率增幅范围为11.37%～14.31%，中位数为12.99%。箱线图展示出口缩放与振打相位的联合影响。",
    "17_sensitivity_heatmap": "场权重消融后，前两电场仍承担70.8%～73.7%的正向电压增量，说明前级优先并非完全由中心权重先验写入，但后级电压仍有不可忽略的补偿作用。",
    "18_condition_energy_penalty": "四类工况从10收紧至5 mg/Nm³的中心情景功率增幅为13.34%～14.11%，加权平均为13.69%；在相同运行时长下，该比例也对应电耗增幅。",
    "05b_boundary_proximity": "八组推荐解均未超过各工况经验97.5%支持域边界，但工况2的10 mg解和工况4的5 mg解已超过90%贴近阈值，必须经现场阶跃试验复核。",
}

TRACE_META = {
    "01_outlet_data_diagnosis": (str(DATA_FILE), "fig01_outlet_diagnosis", "§2.2；图2", "出口记录可能为量程截顶；不代表真实排放分布"),
    "02_process_timeseries": (str(DATA_FILE), "fig02_timeseries", "§2.1—2.2；图3", "只展示首24 h，不代表全周期分布"),
    "03_cluster_selection": (str(RESULTS / "cluster_selection_metrics.csv"), "fig03_cluster_selection", "§5.1；图6", "聚类指标只评价统计分离度，不等于物理工况唯一性"),
    "04_condition_map": (str(DATA_FILE), "fig04_condition_map", "§5.1；图7", "为可视化抽样点；聚类实际使用全部训练样本"),
    "05_condition_profile_heatmap": (str(RESULTS / "condition_profiles.csv"), "fig05_profile_heatmap", "§5.1；图8", "列内标准化仅用于相对比较"),
    "06_method_overview": (str(ROOT / "10_修订后完整论文_终稿.md"), "fig06_method_overview", "§3；图4", "概念流程图，不含新的经验数据"),
    "07_power_prediction": (str(RESULTS / "power_model_baselines.csv"), "fig07_power_prediction + RidgePowerModel", "§5.2；图9", "回顾性测试并非真正未触碰盲测"),
    "08_power_residuals": (str(DATA_FILE), "fig08_residuals + RidgePowerModel", "§5.2；图11", "存在−10.78 kW平均偏差和高功率区低估"),
    "08b_feature_importance": (str(DATA_FILE), "fig08b_feature_importance + RidgePowerModel", "§5.2；图10", "标准化系数用于预测解释，不代表单变量因果效应"),
    "09_voltage_power_response": (str(DATA_FILE), "fig09_voltage_response + RidgePowerModel", "§5.2；图12", "局部情景曲线，仅在历史电压附近解释"),
    "10_optimal_power": (str(RESULTS / "optimal_controls_central_scenario.csv"), "fig10_power", "§5.4；图14", "功率来自数据模型；可行性来自未辨识灰箱情景"),
    "11_optimal_voltage": (str(RESULTS / "optimal_controls_central_scenario.csv"), "grouped_controls(U_COLS)", "§6；图15", "最优电压是情景试验起点，不是设备安全设定"),
    "12_optimal_rapping_period": (str(RESULTS / "optimal_controls_central_scenario.csv"), "grouped_controls(T_COLS)", "§6；图17", "无实际振打事件，周期结果依赖峰值先验"),
    "12b_radar_controls": (str(RESULTS / "optimal_controls_central_scenario.csv"), "fig12b_control_profiles", "§6；图16", "控制剖面来自中心灰箱情景，不是设备安全设定"),
    "13_emission_decomposition": (str(RESULTS / "optimal_controls_central_scenario.csv"), "fig13_emission", "§7.2；图19", "全部排放值为情景推演，不是实测"),
    "14_rapping_peak_mechanism": (str(RESULTS / "condition_profiles.csv"), "fig14_peak", "§4.2；图5", "峰值指数1.35与尺度1.2属于工程先验"),
    "15_search_convergence": (str(RESULTS / "optimization_seed_stability.csv"), "fig15_seed_stability", "§8.1；图20", "随机种子稳定不等价于全局最优证明"),
    "16_sensitivity_distribution": (str(RESULTS / "structural_sensitivity.csv"), "fig16_structural_sensitivity", "§8.2；图21", "扫描范围仍由显式情景网格决定"),
    "17_sensitivity_heatmap": (str(RESULTS / "field_priority_ablation_summary.csv"), "fig17_field_ablation", "§8.3；图22", "消融仍依赖同一灰箱结构与支持域"),
    "18_condition_energy_penalty": (str(RESULTS / "question4_by_condition.csv"), "fig18_penalty", "§7.1；图18", "13.69%为单一种子中心情景点估计"),
    "05b_boundary_proximity": (str(RESULTS / "optimal_controls_central_scenario.csv"), "fig05b_boundary_proximity", "§5.4；图13", "经验支持域不是设备安全边界，贴边解仍需现场复核"),
}

MANUSCRIPT_FIGURE = {
    "01_outlet_data_diagnosis": 2,
    "02_process_timeseries": 3,
    "06_method_overview": 4,
    "14_rapping_peak_mechanism": 5,
    "03_cluster_selection": 6,
    "04_condition_map": 7,
    "05_condition_profile_heatmap": 8,
    "07_power_prediction": 9,
    "08b_feature_importance": 10,
    "08_power_residuals": 11,
    "09_voltage_power_response": 12,
    "05b_boundary_proximity": 13,
    "10_optimal_power": 14,
    "11_optimal_voltage": 15,
    "12b_radar_controls": 16,
    "12_optimal_rapping_period": 17,
    "18_condition_energy_penalty": 18,
    "13_emission_decomposition": 19,
    "15_search_convergence": 20,
    "16_sensitivity_distribution": 21,
    "17_sensitivity_heatmap": 22,
}


def _label_bars(ax, bars, fmt="{:.0f}", dy=3.0):
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + dy, fmt.format(h), ha="center", va="bottom", fontsize=7)


def fig01_outlet_diagnosis(df: pd.DataFrame) -> None:
    valid = df["C_out_mgNm3"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    add_subpanel_label(ax, "a")
    bins = np.linspace(valid.min() - 0.03, valid.max() + 0.03, 28)
    ax.hist(valid, bins=bins, color=PALETTE[0], edgecolor="white")
    ax.axvline(50, color=PALETTE[4], linestyle="--", label="记录上限 50")
    ax.set_xlim(valid.min() - 0.05, valid.max() + 0.05)
    ax.text(
        0.03,
        0.78,
        "5与10 mg/Nm³目标均在图示范围外",
        transform=ax.transAxes,
        color=PALETTE[2],
        fontsize=8,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": LIGHT_GRAY},
    )
    ax.set_xlabel("出口粉尘浓度记录 (mg/Nm³)")
    ax.set_ylabel("样本数")
    ax.legend(frameon=False, loc="upper left")
    finish_axes(ax)

    ax = axes[1]
    add_subpanel_label(ax, "b")
    total = len(df)
    categories = ["精确值", "50截顶", "缺失"]
    values = [int(df["C_out_exact_flag"].sum()), int(df["C_out_cap50_flag"].sum()), int(df["C_out_missing_flag"].sum())]
    bars = ax.bar(categories, values, color=[PALETTE[0], PALETTE[4], GRAY])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + total*0.012, f"{value}\n({100*value/total:.1f}%)", ha="center", fontsize=8)
    ax.set_ylabel("样本数")
    ax.set_ylim(0, max(values) * 1.18)
    note(ax, "百分比以全部10080条样本为分母", loc="upper right")
    finish_axes(ax)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT, "01_outlet_data_diagnosis")


def fig02_timeseries(df: pd.DataFrame) -> None:
    d = df.iloc[:1440].copy()
    hours = np.arange(len(d)) / 60
    fields = [
        ("Temp_C", "入口温度 (°C)", PALETTE[4]),
        ("C_in_gNm3", "入口浓度 (g/Nm³)", PALETTE[0]),
        ("Q_Nm3h", "烟气流量 (Nm³/h)", PALETTE[2]),
        ("P_total_kW", "总功率 (kW)", PALETTE[3]),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(8.2, 5.5), sharex=True)
    for ax, (col, label, color), tag in zip(axes, fields, ["a", "b", "c", "d"]):
        add_subpanel_label(ax, tag, x=-0.08, y=1.02)
        ax.plot(hours, d[col], color=color, linewidth=0.8)
        ax.set_ylabel(label)
        finish_axes(ax)
    axes[-1].set_xlabel("首日运行时间 (h)")
    axes[-1].set_xlim(0, 24)
    fig.tight_layout(h_pad=0.5)
    save_figure(fig, OUT, "02_process_timeseries")


def fig03_cluster_selection(cluster_eval: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.0))
    ax = axes[0]
    add_subpanel_label(ax, "a")
    ax.plot(cluster_eval["k"], cluster_eval["silhouette"], "o-", color=PALETTE[0])
    chosen = cluster_eval[cluster_eval["selected"]].iloc[0]
    ax.scatter([chosen["k"]], [chosen["silhouette"]], s=65, color=PALETTE[4], zorder=3, label="选定 k=4")
    ax.set_xlabel("聚类数 k")
    ax.set_ylabel("轮廓系数")
    ax.legend(frameon=False)
    finish_axes(ax)
    ax = axes[1]
    add_subpanel_label(ax, "b")
    ax.plot(cluster_eval["k"], cluster_eval["inertia"] / 1000, "s--", color=PALETTE[2], label="惯性/1000")
    ax.set_xlabel("聚类数 k")
    ax.set_ylabel("类内惯性 (×10³)")
    finish_axes(ax)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT, "03_cluster_selection")


def fig04_condition_map(df: pd.DataFrame) -> None:
    train = df[df["split"] == "train"]
    fig, ax = plt.subplots(figsize=(6.7, 4.4))
    for cluster, group in train.groupby("condition_cluster"):
        g = group.iloc[::5]
        sizes = 8 + 28 * (g["Q_Nm3h"] - train["Q_Nm3h"].min()) / (train["Q_Nm3h"].max() - train["Q_Nm3h"].min())
        ax.scatter(g["C_in_gNm3"], g["Temp_C"], s=sizes, alpha=0.38, color=PALETTE[int(cluster)-1], label=COND_SHORT[int(cluster)].replace("\n", " "))
    ax.set_xlabel("入口粉尘浓度 (g/Nm³)")
    ax.set_ylabel("入口温度 (°C)")
    ax.legend(frameon=False, ncol=2)
    finish_axes(ax, grid_axis="both")
    fig.tight_layout()
    save_figure(fig, OUT, "04_condition_map")


def fig05_profile_heatmap(profiles: pd.DataFrame) -> None:
    cols = ["Temp_C", "C_in_gNm3", "Q_Nm3h", "dust_load_kg_h", "historical_power_kW"]
    labels = ["温度", "入口浓度", "烟气流量", "粉尘负荷", "历史总功率"]
    values = profiles[cols].to_numpy(float)
    z = (values - values.mean(axis=0)) / values.std(axis=0)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    im = ax.imshow(z, cmap="RdBu_r", vmin=-1.7, vmax=1.7, aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(4), [COND_SHORT[i] for i in range(1, 5)])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i,j]:+.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(z[i,j]) > 0.85 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.03)
    cbar.set_label("列内标准分数")
    fig.tight_layout()
    save_figure(fig, OUT, "05_condition_profile_heatmap")


def fig05b_boundary_proximity(optimum: pd.DataFrame) -> None:
    """Audit how close each recommended point lies to its empirical support limit."""
    ordered = optimum.sort_values(["condition_cluster", "limit_mgNm3"], ascending=[True, False]).copy()
    ordered["ratio_pct"] = 100.0 * ordered["mahalanobis_d2"] / ordered["support_threshold_d2"]
    labels = [f"工况{int(c)}\n{int(lim)} mg" for c, lim in zip(ordered["condition_cluster"], ordered["limit_mgNm3"])]
    colors = ["#D55E00" if ratio > 90 else PALETTE[int(c)-1]
              for c, ratio in zip(ordered["condition_cluster"], ordered["ratio_pct"])]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.05, 1.2]})
    add_subpanel_label(axes[0], "a")
    x = np.arange(len(ordered))
    axes[0].vlines(x, 0, ordered["ratio_pct"], color=colors, linewidth=2.2, alpha=0.8)
    axes[0].scatter(x, ordered["ratio_pct"], color=colors, s=45, edgecolor="black", linewidth=0.5, zorder=3)
    axes[0].axhline(90, color=PALETTE[1], linestyle=":", linewidth=1.2, label="90%贴近阈值")
    axes[0].axhline(100, color=PALETTE[4], linestyle="--", linewidth=1.2, label="经验边界")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("边界相对距离 $d_M^2/q_{0.975}$ (%)")
    axes[0].set_ylim(0, 108)
    axes[0].legend(frameon=False, loc="lower left", fontsize=8)
    finish_axes(axes[0], grid_axis="y")

    add_subpanel_label(axes[1], "b")
    y = np.arange(len(ordered))[::-1]
    axes[1].barh(y, ordered["support_threshold_d2"], color=LIGHT_GRAY, height=0.58, label="经验97.5%边界")
    axes[1].barh(y, ordered["mahalanobis_d2"], color=PALETTE[0], height=0.38, label="推荐解距离")
    for yy, ratio, d2 in zip(y, ordered["ratio_pct"], ordered["mahalanobis_d2"]):
        if ratio > 90:
            axes[1].barh(yy, d2, color=PALETTE[4], height=0.38)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlabel("马氏距离平方")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    finish_axes(axes[1], grid_axis="x")

    fig.tight_layout(w_pad=1.8)
    save_figure(fig, OUT, "05b_boundary_proximity")


def fig06_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(9.0, 3.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.5)
    ax.axis("off")
    stages = [
        (0.25, 1.2, 1.55, 2.1, "原始数据", "过程量、总功率\n出口浓度记录"),
        (2.1, 1.2, 1.65, 2.1, "数据审计", "截顶诊断\n时间切分与支持域"),
        (4.05, 2.35, 1.75, 1.35, "数据驱动层", "工况聚类\n岭回归功率模型"),
        (4.05, 0.35, 1.75, 1.35, "灰箱情景层", "捕集指数\n振打峰值代理"),
        (6.15, 1.2, 1.65, 2.1, "约束优化", "5/10 mg情景约束\n历史支持域搜索"),
        (8.15, 1.2, 1.55, 2.1, "输出", "分工况控制策略\n功率代价与敏感性"),
    ]
    fills = ["#E8F1F8", "#F2F4F7", "#E8F5F0", "#FFF2E5", "#F3ECF7", "#E8F1F8"]
    for (x, y, w, h, title, body), fill in zip(stages, fills):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08", facecolor=fill, edgecolor="#4B5563", linewidth=1.0)
        ax.add_patch(box)
        ax.text(x+w/2, y+h*0.68, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(x+w/2, y+h*0.34, body, ha="center", va="center", fontsize=8.5, linespacing=1.45)
    arrows = [
        ((1.82, 2.25), (2.08, 2.25), "清洗数据"),
        ((3.77, 2.52), (4.03, 2.85), "可辨识部分"),
        ((3.77, 1.98), (4.03, 1.05), "不可辨识部分"),
        ((5.82, 3.0), (6.13, 2.65), "功率预测"),
        ((5.82, 1.05), (6.13, 1.75), "排放约束"),
        ((7.82, 2.25), (8.13, 2.25), "最优解"),
    ]
    for start, end, label in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, color=GRAY, linewidth=1.1))
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2+0.18, label, ha="center", fontsize=7, color=GRAY)
    ax.text(4.93, 4.12, "实测证据", ha="center", color=PALETTE[2], fontsize=8, fontweight="bold")
    ax.text(4.93, 0.04, "显式先验（不冒充实测辨识）", ha="center", color=PALETTE[4], fontsize=8, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, OUT, "06_method_overview")


def fit_power_model(df: pd.DataFrame) -> RidgePowerModel:
    train = df[df["split"] == "train"]
    metrics = pd.read_json(RESULTS / "power_model_metrics.json", typ="series")
    model = RidgePowerModel(alpha=float(metrics["alpha_selected_by_blocked_rolling_validation"]))
    model.fit(train, train["P_total_kW"].to_numpy(float))
    return model


def fig07_power_prediction(df: pd.DataFrame, model: RidgePowerModel, baselines: pd.DataFrame) -> None:
    test = df[df["split"] == "test"]
    actual = test["P_total_kW"].to_numpy(float)
    pred = model.predict(test)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    ax = axes[0]
    add_subpanel_label(ax, "a")
    ax.scatter(actual, pred, s=10, alpha=0.42, color=PALETTE[0], edgecolors="none")
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "--", color=PALETTE[4], label="理想预测")
    ax.text(0.04, 0.94, "$R^2=0.957$\nRMSE=14.76 kW", transform=ax.transAxes, va="top",
            bbox={"boxstyle":"round", "facecolor":"white", "edgecolor":LIGHT_GRAY})
    ax.set_xlabel("实际总功率 (kW)")
    ax.set_ylabel("预测总功率 (kW)")
    ax.legend(frameon=False, loc="lower right")
    finish_axes(ax, grid_axis="both")

    ax = axes[1]
    add_subpanel_label(ax, "b")
    labels = ["均值基线", "线性岭", "二次岭"]
    model_order = ["mean_predictor", "linear_ridge", "quadratic_ridge"]
    x = np.arange(3)
    val = baselines[baselines["split"] == "validation"].set_index("model").loc[model_order, "rmse_kW"]
    tst = baselines[baselines["split"] == "retrospective_test"].set_index("model").loc[model_order, "rmse_kW"]
    ax.bar(x-0.18, val, 0.36, color=PALETTE[5], label="验证集")
    ax.bar(x+0.18, tst, 0.36, color=PALETTE[0], label="回顾性测试")
    ax.set_xticks(x, labels)
    ax.set_ylabel("RMSE (kW，对数刻度)")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper right")
    finish_axes(ax)
    fig.tight_layout()
    save_figure(fig, OUT, "07_power_prediction")


def fig08_residuals(df: pd.DataFrame, model: RidgePowerModel) -> None:
    test = df[df["split"] == "test"]
    actual = test["P_total_kW"].to_numpy(float)
    pred = model.predict(test)
    res = pred - actual
    fig, axes = plt.subplots(1, 2, figsize=(7.7, 3.1))
    add_subpanel_label(axes[0], "a")
    axes[0].scatter(pred, res, s=9, alpha=0.38, color=PALETTE[0], edgecolors="none")
    axes[0].axhline(0, color=PALETTE[4], linestyle="--")
    axes[0].set_xlabel("预测总功率 (kW)")
    axes[0].set_ylabel("残差：预测−实际 (kW)")
    finish_axes(axes[0], grid_axis="both")
    add_subpanel_label(axes[1], "b")
    axes[1].hist(res, bins=28, color=PALETTE[2], edgecolor="white")
    metrics = pd.read_json(RESULTS / "power_model_metrics.json", typ="series")
    guard = float(metrics["validation_abs_error_q90_kW"])
    axes[1].axvline(res.mean(), color=PALETTE[4], linestyle="--", label=f"测试均值 {res.mean():.2f} kW")
    axes[1].axvline(guard, color=PALETTE[0], linestyle=":", label=f"验证|误差|P90 {guard:.2f} kW")
    axes[1].set_xlabel("残差 (kW)")
    axes[1].set_ylabel("样本数")
    axes[1].legend(frameon=False)
    finish_axes(axes[1])
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT, "08_power_residuals")


def fig08b_feature_importance(model: RidgePowerModel) -> None:
    """Plot standardized ridge coefficients without giving them a causal meaning."""
    names = [
        "$U_1$", "$U_2$", "$U_3$", "$U_4$",
        "$U_1^2$", "$U_1U_2$", "$U_1U_3$", "$U_1U_4$",
        "$U_2^2$", "$U_2U_3$", "$U_2U_4$", "$U_3^2$", "$U_3U_4$", "$U_4^2$",
        "$T_1$", "$T_2$", "$T_3$", "$T_4$", "入口温度 $H$", "入口浓度 $C_{in}$", "烟气流量 $Q$",
    ]
    coef = np.asarray(model.beta[1:], dtype=float)
    order = np.argsort(np.abs(coef))
    values = coef[order]
    labels = [names[i] for i in order]
    colors = [PALETTE[0] if value >= 0 else PALETTE[4] for value in values]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    bars = ax.barh(np.arange(len(values)), values, color=colors, alpha=0.86, height=0.62)
    for bar, value in zip(bars, values):
        ax.text(value + (1.0 if value >= 0 else -1.0), bar.get_y() + bar.get_height()/2,
                f"{value:+.1f}", va="center", ha="left" if value >= 0 else "right", fontsize=7)
    ax.set_yticks(np.arange(len(values)), labels)
    ax.set_xlabel("标准化岭回归系数 (kW/标准差)")
    ax.axvline(0, color=GRAY, linestyle="--", linewidth=0.8)
    finish_axes(ax, grid_axis="x")
    fig.tight_layout()
    save_figure(fig, OUT, "08b_feature_importance")


def fig09_voltage_response(df: pd.DataFrame, profiles: pd.DataFrame, model: RidgePowerModel) -> None:
    train = df[df["split"] == "train"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for _, p in profiles.iterrows():
        c = int(p["condition_cluster"])
        group = train[train["condition_cluster"] == c]
        scale = np.linspace(0.92, 1.08, 80)
        rows = []
        u0 = group[U_COLS].median().to_numpy(float)
        t0 = group[T_COLS].median().to_numpy(float)
        for s in scale:
            row = {col: val for col, val in zip(U_COLS, u0*s)}
            row.update({col: val for col, val in zip(T_COLS, t0)})
            row.update({"Temp_C":p["Temp_C"], "C_in_gNm3":p["C_in_gNm3"], "Q_Nm3h":p["Q_Nm3h"]})
            rows.append(row)
        y = model.predict(pd.DataFrame(rows))
        ax.plot(100*(scale-1), y, color=PALETTE[c-1], marker=["o","s","^","D"][c-1], markevery=16,
                label=COND_SHORT[c].replace("\n", " "))
    ax.set_xlabel("四电场电压同步变化 (%)")
    ax.set_ylabel("模型预测总功率 (kW)")
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    finish_axes(ax, grid_axis="both")
    fig.tight_layout()
    save_figure(fig, OUT, "09_voltage_power_response")


def fig10_power(optimum: pd.DataFrame, q4: pd.DataFrame) -> None:
    p = optimum.pivot(index="condition_cluster", columns="limit_mgNm3", values="predicted_power_kW")
    x = np.arange(4)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    b1 = ax.bar(x-0.18, p[10.0], 0.36, color=PALETTE[5], label="10 mg/Nm³")
    b2 = ax.bar(x+0.18, p[5.0], 0.36, color=PALETTE[4], label="5 mg/Nm³")
    _label_bars(ax, b1); _label_bars(ax, b2)
    ax.set_xticks(x, [COND_SHORT[i] for i in range(1,5)])
    ax.set_ylabel("情景最优功率 (kW)")
    ax.set_ylim(0, max(p.max())*1.15)
    ax.legend(frameon=False, ncol=2)
    finish_axes(ax)
    fig.tight_layout()
    save_figure(fig, OUT, "10_optimal_power")


def grouped_controls(optimum: pd.DataFrame, cols: list[str], stem: str, ylabel: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.8), sharey=True)
    for ax, cluster, tag in zip(axes.flat, range(1,5), ["a", "b", "c", "d"]):
        add_subpanel_label(ax, tag)
        sub = optimum[optimum["condition_cluster"] == cluster].set_index("limit_mgNm3")
        x = np.arange(4)
        ax.bar(x-0.18, sub.loc[10.0, cols], 0.36, color=PALETTE[5], label="10 mg/Nm³")
        ax.bar(x+0.18, sub.loc[5.0, cols], 0.36, color=PALETTE[4], label="5 mg/Nm³")
        ax.set_xticks(x, [c.split("_")[0] for c in cols])
        ax.set_title(COND_SHORT[cluster].replace("\n", " "), fontsize=9)
        finish_axes(ax)
    axes[0,0].set_ylabel(ylabel); axes[1,0].set_ylabel(ylabel)
    axes[0,1].legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 1.18))
    fig.tight_layout(h_pad=1.8)
    save_figure(fig, OUT, stem)


def fig12b_control_profiles(optimum: pd.DataFrame) -> None:
    """Compare voltage and rapping profiles for representative low/high-load conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    x = np.arange(4)
    fields = ["电场1", "电场2", "电场3", "电场4"]
    styles = {
        (1, 10.0): ("o--", PALETTE[0], "工况1，10 mg"),
        (1, 5.0): ("s-", PALETTE[0], "工况1，5 mg"),
        (3, 10.0): ("^--", PALETTE[2], "工况3，10 mg"),
        (3, 5.0): ("D-", PALETTE[2], "工况3，5 mg"),
    }
    for ax, cols, ylabel, tag in [
        (axes[0], U_COLS, "中心情景电压 (kV)", "a"),
        (axes[1], T_COLS, "中心情景振打周期 (s)", "b"),
    ]:
        add_subpanel_label(ax, tag)
        for (cluster, limit), (style, color, label) in styles.items():
            row = optimum[(optimum["condition_cluster"] == cluster) & (optimum["limit_mgNm3"] == limit)].iloc[0]
            ax.plot(x, row[cols].to_numpy(float), style, color=color, label=label, linewidth=1.6, markersize=5)
        ax.set_xticks(x, fields)
        ax.set_ylabel(ylabel)
        headroom(ax, 0.12)
        finish_axes(ax, grid_axis="both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=1.8, rect=(0, 0.09, 1, 1))
    figure_legend(fig, handles, labels, ncol=4, bottom=0.0)
    save_figure(fig, OUT, "12b_radar_controls")


def fig13_emission(optimum: pd.DataFrame) -> None:
    x = np.arange(8)
    ordered = optimum.sort_values(["limit_mgNm3", "condition_cluster"], ascending=[False, True])
    base = ordered["scenario_base_emission_mgNm3"].to_numpy()
    peak = ordered["scenario_peak_excess_mgNm3"].to_numpy()
    colors = [PALETTE[0]]*4 + [PALETTE[4]]*4
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x, base, color=colors, alpha=0.82, label="连续排放基值")
    ax.bar(x, peak, bottom=base, color=colors, alpha=0.35, edgecolor=colors, label="振打峰值增量")
    ax.hlines(10, -0.5, 3.5, colors=GRAY, linestyles="--", linewidth=1.0)
    ax.hlines(5, 3.5, 7.5, colors=GRAY, linestyles="--", linewidth=1.0)
    ax.text(3.45, 10.15, "10 mg/Nm³", ha="right", fontsize=8, color=GRAY)
    ax.text(7.45, 5.15, "5 mg/Nm³", ha="right", fontsize=8, color=GRAY)
    ax.set_xticks(x, [f"工况{i}" for i in range(1,5)]*2)
    ax.set_ylabel("峰值总排放情景值 (mg/Nm³)")
    ax.legend(frameon=False, ncol=2)
    finish_axes(ax)
    fig.tight_layout()
    save_figure(fig, OUT, "13_emission_decomposition")


def fig14_peak(profiles: pd.DataFrame) -> None:
    ratio = np.linspace(0.75, 1.25, 200)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for cluster in [1,3,4]:
        p = profiles[profiles["condition_cluster"] == cluster].iloc[0]
        peak = 1.2 * float(p["load_ratio_to_train_median"])**0.8 * ratio**1.35
        ax.plot(100*(ratio-1), peak, color=PALETTE[cluster-1], marker=["o","s","^","D"][cluster-1], markevery=35,
                label=COND_SHORT[cluster].replace("\n", " "))
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.set_xlabel("振打周期相对参考最优周期变化 (%)")
    ax.set_ylabel("单次再飞扬峰值增量代理 (mg/Nm³)")
    ax.legend(frameon=False)
    finish_axes(ax, grid_axis="both")
    fig.tight_layout()
    save_figure(fig, OUT, "14_rapping_peak_mechanism")


def fig15_seed_stability(seed_data: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    ax = axes[0]
    add_subpanel_label(ax, "a")
    ordered = seed_data.assign(key=seed_data["condition_cluster"].astype(str)+"-"+seed_data["limit_mgNm3"].astype(int).astype(str))
    keys = [f"{c}-{int(limit)}" for c in range(1,5) for limit in (10.0,5.0)]
    for j, key in enumerate(keys):
        g = ordered[ordered["key"] == key]
        median = g["predicted_power_kW"].median()
        rel = 100*(g["predicted_power_kW"]/median-1)
        jitter = np.linspace(-0.12,0.12,len(g))
        ax.scatter(np.full(len(g),j)+jitter, rel, color=PALETTE[j%4], marker="o" if key.endswith("-10") else "s", s=24)
        ax.vlines(j, rel.min(), rel.max(), color=LIGHT_GRAY, linewidth=2, zorder=0)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xticks(range(8), [f"工况{c}\n{limit} mg" for c in range(1,5) for limit in (10,5)])
    ax.set_ylabel("相对五种子中位数偏差 (%)")
    ax.set_xlabel("工况—限值组合")
    finish_axes(ax, grid_axis="both")

    ax = axes[1]
    add_subpanel_label(ax, "b")
    groups = [seed_data[seed_data["limit_mgNm3"]==limit]["local_refinement_improvement_pct"] for limit in (10.0,5.0)]
    boxes = ax.boxplot(groups, tick_labels=["10 mg/Nm³", "5 mg/Nm³"], patch_artist=True, widths=0.55)
    for box, color in zip(boxes["boxes"], [PALETTE[5], PALETTE[4]]):
        box.set_facecolor(color); box.set_alpha(0.75)
    ax.scatter(np.repeat([1,2], [len(groups[0]),len(groups[1])]), np.concatenate(groups), s=10, color=GRAY, alpha=0.45)
    ax.set_ylabel("局部细化相对初始库改善 (%)")
    ax.set_xlabel("排放限值")
    finish_axes(ax)
    fig.tight_layout(w_pad=2.1)
    save_figure(fig, OUT, "15_search_convergence")


def fig16_structural_sensitivity(structural: pd.DataFrame) -> None:
    valid = structural[structural["all_conditions_feasible"] == True]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    ax = axes[0]
    add_subpanel_label(ax, "a")
    scales = sorted(valid["outlet_scale"].unique())
    data = [valid[valid["outlet_scale"]==x]["weighted_power_increase_pct"] for x in scales]
    boxes = ax.boxplot(data, tick_labels=[f"{x:.2f}" for x in scales], patch_artist=True, showfliers=False)
    for i, box in enumerate(boxes["boxes"]):
        box.set_facecolor(PALETTE[i%len(PALETTE)]); box.set_alpha(0.72)
    ax.set_xlabel("出口记录缩放系数")
    ax.set_ylabel("5相对10 mg/Nm³加权功率增幅 (%)")
    finish_axes(ax)

    ax = axes[1]
    add_subpanel_label(ax, "b")
    phases = sorted(valid["phase_scale"].unique())
    profiles = ["equal", "weak_front", "central_front"]
    styles = ["o-", "s--", "^:"]
    labels = ["等权", "弱前级", "中心前级"]
    for profile, style, label, color in zip(profiles, styles, labels, [PALETTE[0],PALETTE[2],PALETTE[4]]):
        medians = [valid[(valid["phase_scale"]==p)&(valid["alpha_profile"]==profile)]["weighted_power_increase_pct"].median() for p in phases]
        ax.plot(phases, medians, style, color=color, label=label)
    ax.set_xlabel("振打相位叠加尺度")
    ax.legend(frameon=False)
    finish_axes(ax, grid_axis="both")
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT, "16_sensitivity_distribution")


def fig17_field_ablation(ablation: pd.DataFrame) -> None:
    profiles = ["equal", "weak_front", "central_front"]
    labels = ["等权", "弱前级", "中心前级"]
    cols = [f"weighted_delta_U{i}_kV" for i in range(1,5)]
    d = ablation.set_index("alpha_profile").loc[profiles, cols]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.3), gridspec_kw={"width_ratios":[1.45,1]})
    ax = axes[0]
    add_subpanel_label(ax, "a")
    im = ax.imshow(d.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_xticks(range(4), [f"U{i}" for i in range(1,5)])
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("电场")
    ax.set_ylabel("去除贡献权重设定")
    for i in range(3):
        for j in range(4):
            ax.text(j, i, f"{d.iloc[i,j]:.2f}", ha="center", va="center", color="white" if d.iloc[i,j]>5 else "black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cbar.set_label("10→5 mg时加权电压增量 (kV)")

    ax = axes[1]
    add_subpanel_label(ax, "b")
    shares = 100*ablation.set_index("alpha_profile").loc[profiles,"front_two_share_of_positive_adjustment"]
    bars = ax.bar(labels, shares, color=[PALETTE[0],PALETTE[2],PALETTE[4]])
    _label_bars(ax, bars, fmt="{:.1f}%", dy=0.7)
    ax.set_ylabel("前两电场占正向电压增量 (%)")
    ax.set_ylim(0,100)
    finish_axes(ax)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT, "17_sensitivity_heatmap")


def fig18_penalty(q4: pd.DataFrame) -> None:
    x = np.arange(4)
    weighted_p10 = float(np.sum(q4["share"] * q4["power_10_kW"]) / q4["share"].sum())
    weighted_p5 = float(np.sum(q4["share"] * q4["power_5_kW"]) / q4["share"].sum())
    weighted = 100.0 * (weighted_p5 / weighted_p10 - 1.0)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bars = ax.bar(x, q4.sort_values("condition_cluster")["increase_pct"], color=PALETTE[:4])
    _label_bars(ax, bars, fmt="{:.2f}%", dy=0.12)
    ax.axhline(weighted, color=PALETTE[4], linestyle="--", label=f"工况加权平均 {weighted:.2f}%")
    ax.set_xticks(x, [COND_SHORT[i] for i in range(1,5)])
    ax.set_ylabel("情景预测功率增幅 (%)")
    ax.set_ylim(0, max(q4["increase_pct"])*1.22)
    ax.legend(frameon=False)
    finish_axes(ax)
    fig.tight_layout()
    save_figure(fig, OUT, "18_condition_energy_penalty")


def _relpath(path: str) -> str:
    """交付物中一律使用相对路径，避免写入本机绝对路径。"""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


SCHEMATIC_ROW = {
    "figure_id": "01", "manuscript_figure": 1, "stem": "00_esp_schematic",
    "caption": "四电场电除尘器结构与变量定义",
    "png": "figures_paper/00_esp_schematic.pdf", "pdf": "figures_paper/00_esp_schematic.pdf",
    "source_data": "题目附图（赛题给定的设备结构示意）",
    "transformation": "00_esp_schematic.tex（TikZ矢量重绘，xelatex编译）",
    "supported_manuscript_claims": "§1.1；图1",
    "limitations": "结构示意图，不含实测数据；比例非工程真实尺寸",
}


def write_manifest() -> None:
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    model_hash = hashlib.sha256((SRC_DIR / "scenario_model.py").read_bytes()).hexdigest()
    rows = []
    for stem, caption in CAPTIONS.items():
        source_data, function_name, claims, limitations = TRACE_META[stem]
        source_data = _relpath(source_data)
        rows.append({"figure_id": stem.split("_")[0], "manuscript_figure": MANUSCRIPT_FIGURE[stem], "stem": stem, "caption": caption,
                     "png": f"figures_paper/{stem}.png", "pdf": f"figures_paper/{stem}.pdf",
                     "source_data": source_data,
                     "transformation": f"paper_figures.py::{function_name}; plot_sha256={script_hash}; model_sha256={model_hash}",
                     "supported_manuscript_claims": claims,
                     "limitations": limitations})
    rows.insert(0, SCHEMATIC_ROW)
    pd.DataFrame(rows).to_csv(OUT / "figure_manifest.csv", index=False, encoding="utf-8-sig")
    lines = ["# 论文图组与图注", "", "所有图均提供 PNG 预览和 PDF 矢量版本。", ""]
    for row in sorted(rows, key=lambda x: x["manuscript_figure"]):
        lines.extend([
            f"## 正文图{row['manuscript_figure']}（文件编号{row['figure_id']}）{row['stem']}", "", row["caption"], "",
            f"- 源数据：`{row['source_data']}`",
            f"- 变换：`{row['transformation']}`",
            f"- 支持的正文论断：{row['supported_manuscript_claims']}",
            f"- 局限：{row['limitations']}",
            f"- PNG：`{row['png']}`", f"- PDF：`{row['pdf']}`", ""])
    (OUT / "图注清单.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    set_paper_style()
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    profiles = pd.read_csv(RESULTS / "condition_profiles.csv")
    cluster_eval = pd.read_csv(RESULTS / "cluster_selection_metrics.csv")
    optimum = pd.read_csv(RESULTS / "optimal_controls_central_scenario.csv")
    seed_data = pd.read_csv(RESULTS / "optimization_seed_stability.csv")
    structural = pd.read_csv(RESULTS / "structural_sensitivity.csv")
    ablation = pd.read_csv(RESULTS / "field_priority_ablation_summary.csv")
    baselines = pd.read_csv(RESULTS / "power_model_baselines.csv")
    q4 = pd.read_csv(RESULTS / "question4_by_condition.csv")
    model = fit_power_model(df)

    fig01_outlet_diagnosis(df)
    fig02_timeseries(df)
    fig03_cluster_selection(cluster_eval)
    fig04_condition_map(df)
    fig05_profile_heatmap(profiles)
    fig05b_boundary_proximity(optimum)
    fig06_method_overview()
    fig07_power_prediction(df, model, baselines)
    fig08b_feature_importance(model)
    fig08_residuals(df, model)
    fig09_voltage_response(df, profiles, model)
    fig10_power(optimum, q4)
    grouped_controls(optimum, U_COLS, "11_optimal_voltage", "最优电压 (kV)")
    fig12b_control_profiles(optimum)
    grouped_controls(optimum, T_COLS, "12_optimal_rapping_period", "最优振打周期 (s)")
    fig13_emission(optimum)
    fig14_peak(profiles)
    fig15_seed_stability(seed_data)
    fig16_structural_sensitivity(structural)
    fig17_field_ablation(ablation)
    fig18_penalty(q4)
    write_manifest()
    print(f"Generated {len(CAPTIONS)} figures in {OUT}")


if __name__ == "__main__":
    main()
