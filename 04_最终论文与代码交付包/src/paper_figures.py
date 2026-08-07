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
from scenario_model import (ALPHA_CENTRAL, PhysicalPowerModel, RidgePowerModel,
                            T_COLS, U_COLS)


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
    "01_outlet_data_diagnosis": "出口浓度记录存在显著截顶且不覆盖目标区间。有效记录全部位于48.74～50.00 mg/Nm³，其中5491条恰好等于50，5与10 mg/Nm³目标区间内没有任何训练样本，因而无法由该列直接回归出目标区间附近的控制—排放关系。",
    "02_process_timeseries": "首24小时入口温度、入口浓度、烟气流量与总功率的分钟级变化。入口变量与功率具有连续变化和工况迁移特征，与图2中被压缩在单点附近的出口浓度形成对照。",
    "22_technical_roadmap": "全文技术路线。可识别性审计把论证分为两条证据强度不同的路径：实证数据层（实线）可由附件复算，条件式灰箱层（虚线）须经现场标定后方可采信；两条路径仅在支持域优化阶段合并，四问结论按其所属路径分别标注。",
    "04b_operating_evidence": "附件中可识别的另一半信息。(a)(b) 为两段持续异常工况及控制端的同步响应：入口浓度尖峰时抬高$U_1$、功率随之上冲，入口温度尖峰（比电阻高值区）时反而降压降功率；(c) 为整周记录中工况变量与控制设定的相关矩阵，显示入口浓度升高时缩短各场振打周期这一稳定规律。全部为历史操作的相关关系，不构成排放侧的因果证据。",
    "14_rapping_peak_mechanism": "振打周期同时牵动电耗与瞬时峰值。(a) 由每周期积灰量$\\propto L\\cdot T$推得单次再飞扬峰值随周期近似线性放大，而在单位面积灰量对应的再飞扬比例近似不随周期变化的前提下，单位时间再飞扬量的一阶近似与周期无关（水平点线）；(b) 振打环节在中位工况下占总功率约四成，把周期由中位延长到历史极值可回收约109 kW，代价是单次积灰量增加约18%。",
    "03_cluster_selection": "两项内部指标并不一致：轮廓系数在$k=4$取最大值，Calinski–Harabasz指标却在$k=5$取最大值。因此$k=4$并非唯一的\"数学最优\"聚类数，本文取$k=4$的依据是轮廓系数最高、四类簇心构成清晰的温度×流量结构，且四类比五类更便于控制解释。",
    "04_condition_map": "四类工况在入口浓度—温度平面上形成清晰分区，点的大小表示烟气流量，从而同时呈现三维工况信息。",
    "07_power_prediction": "总功率由控制量近乎确定地决定。物理设定$P=a_0+\\sum b_iU_i^2+\\sum c_i/T_i$在回顾性测试段的$R^2$与RMSE均显著优于均值基线及把振打周期作为线性项的岭回归设定。",
    "08b_feature_importance": "振打周期必须以$1/T$进入功率模型。(a) 扣除电场功率后的剩余功率对$T_1$呈明显双曲形状，线性设定（虚线）在两端系统性偏离；(b) 线性$T$设定在回顾性测试段留下约−10.8 kW的平均偏差，改为$1/T$后偏差降至约0，说明该偏差来自设定误差而非数据漂移。",
    "08_power_residuals": "改用$1/T$设定后，回顾性测试残差围绕零对称分布，原先高功率区负偏差扩大的现象消失，确认修正后的误差不再含系统性结构。",
    "05b_boundary_proximity": "单变量幅值合规不等于组合落在历史联合运行经验之内。(a) 八组解的马氏距离平方为经验97.5%外壳的2.4～7.0倍；(b) 10 mg/Nm³解的各场电压均未超出该工况历史最大值，5 mg/Nm³解则需超出最多约5%。",
    "19_emission_power_frontier": "排放—功率权衡前沿。限值由10逐档收紧至5 mg/Nm³时四类工况的最优功率单调上升，工况加权功率由约1973升至约2105 kW；两档均高于历史实际加权功率1783.2 kW，说明超低排放改造是以增电耗换减排，而非节能。",
    "11_optimal_voltage": "两档限值下的电压配置并非四场机械同步抬升：分配由单位功率去除收益$\\alpha_i/(b_iU_{i}^{ref\\,2})$与排序约束共同决定，工况3、4中$U_2$明显低于$U_1$，工况4的$U_4$被压至历史下界。",
    "13_emission_decomposition": "限值裕度由连续排放基值与振打峰值增量共同占用。柱顶虚线分别表示10和5 mg/Nm³情景限值；峰值项占用0.95～1.61 mg/Nm³，故约束必须按$C_{peak}=C_{base}+B$整体施加，而不能只约束$C_{base}$。",
    "18_condition_energy_penalty": "闭式估计与八变量数值最优解的一致性检验。(a) 四类工况的点均落在$y=x$线附近；(b) 相对偏差为−0.04%～+0.97%，全部落在±1%带内，说明按四场电压同比缩放导出的解析式可作为第四问功率代价的解析解释。",
    "15_search_convergence": "多起点SLSQP与同一可行域内4×10^5点随机搜索的对照。八个工况—限值组合中，47～53个起点全部收敛到同一解（功率一致到10^-7 kW以内）；随机搜索均未找到更优解，其最优值反而高出0.97%～5.17%，工况1的10 mg/Nm³组合更是一个可行样本都没采到。这构成数值稳定性的强证据，但不是全局最优的数学证明。",
    "16_sensitivity_distribution": "81组结构情景全部可行。峰值指数在本文推导值1.00附近变化几乎不改变功率增幅，相位叠加尺度与场权重形状的影响也有限；加权增幅为5.13%～7.55%，中位数6.40%，说明核心结论对灰箱形状设定不敏感。",
    "20_eta_sensitivity": "功率增幅由幂次$p$主导。$K\\propto U^p$时增幅近似正比于$2/p$：$p=1$为14.2%，$p=2$（Deutsch迁移速度形式）为6.8%，$p=3$为4.4%。其中$p=1$与$p=1.5$在5%电压裕度内无解，说明$p$同时决定代价大小与目标可达性，是首要待标定参数。",
    "17_sensitivity_heatmap": "场权重消融结果。三种去除贡献权重形状下，10→5 mg/Nm³所需的加权电压增量分布与按$\\alpha_i/(b_iU_{i}^{ref\\,2})$排序的解析优先级一致，说明优先级结论不由中心权重单独写入。",
}

TRACE_META = {
    "01_outlet_data_diagnosis": (str(DATA_FILE), "fig01_outlet_diagnosis", "§2.2；图2", "出口记录为量程截顶；不代表真实排放分布"),
    "02_process_timeseries": (str(DATA_FILE), "fig02_timeseries", "§2.2；图3", "只展示首24 h，不代表全周期分布"),
    "22_technical_roadmap": (str(ROOT / "10_修订后完整论文_终稿.md"), "fig22_roadmap", "§3.3；图4", "概念流程图，不含新的经验数据"),
    "04b_operating_evidence": (str(RESULTS / "historical_strategy_correlations.csv"), "fig04b_operating_evidence", "§4.2—4.3；图5", "相关系数为同期线性关联，不等于因果；两段事件为个案，样本量有限"),
    "14_rapping_peak_mechanism": (str(RESULTS / "rapping_power_tradeoff.csv"), "fig14_peak", "§4.5；图6", "峰值尺度$b_0$仍为工程先验；小时均值结论依赖再飞扬比例近似恒定的前提"),
    "03_cluster_selection": (str(RESULTS / "cluster_selection_metrics.csv"), "fig03_cluster_selection", "§5.1；图7", "聚类指标只评价统计分离度，不等于物理工况唯一性"),
    "04_condition_map": (str(DATA_FILE), "fig04_condition_map", "§5.1；图8", "为可视化抽样点；聚类实际使用全部训练样本"),
    "07_power_prediction": (str(RESULTS / "power_model_baselines.csv"), "fig07_power_prediction + PhysicalPowerModel", "§5.2；图9", "回顾性测试并非真正未触碰盲测"),
    "08b_feature_importance": (str(RESULTS / "power_model_baselines.csv"), "fig08b_specification", "§5.2；图10", "残差图按$T_1$展开，其余三场周期固定在中位数"),
    "08_power_residuals": (str(DATA_FILE), "fig08_residuals + PhysicalPowerModel", "§5.2；图11", "残差接近零均值，但仍非全新日期盲测"),
    "05b_boundary_proximity": (str(RESULTS / "support_boundary_audit.csv"), "fig05b_boundary_proximity", "§5.6；图12", "经验支持域不是设备安全边界；超出部分必须由现场阶跃试验确认"),
    "19_emission_power_frontier": (str(RESULTS / "emission_power_frontier.csv"), "fig19_frontier", "§5.7；图13", "各限值下均为灰箱情景解，前沿形状依赖排放先验"),
    "11_optimal_voltage": (str(RESULTS / "optimal_controls_central_scenario.csv"), "grouped_controls(U_COLS)", "§6；图14", "最优电压是情景试验起点，不是设备安全设定"),
    "18_condition_energy_penalty": (str(RESULTS / "question4_by_condition.csv"), "fig18_penalty", "§7.2；图15", "一致性只说明解析近似可用，不改变灰箱先验的条件性"),
    "13_emission_decomposition": (str(RESULTS / "optimal_controls_central_scenario.csv"), "fig13_emission", "§7.4；图16", "全部排放值为情景推演，不是实测"),
    "15_search_convergence": (str(RESULTS / "optimization_verification.csv"), "fig15_optimizer_verification", "§8.1；图17", "随机搜索对照只能证伪，不构成全局最优的数学证明"),
    "16_sensitivity_distribution": (str(RESULTS / "structural_sensitivity.csv"), "fig16_structural_sensitivity", "§8.2；图18", "扫描范围仍由显式情景网格决定"),
    "20_eta_sensitivity": (str(RESULTS / "p_exponent_reference.csv"), "fig20_p_sensitivity", "§8.2；图19", "$p$的取值区间为先验网格；需由现场阶跃试验标定"),
    "17_sensitivity_heatmap": (str(RESULTS / "field_priority_ablation_summary.csv"), "fig17_field_ablation", "§8.2；图20", "消融仍依赖同一灰箱结构与支持域"),
}

MANUSCRIPT_FIGURE = {
    "01_outlet_data_diagnosis": 2,
    "02_process_timeseries": 3,
    "22_technical_roadmap": 4,
    "04b_operating_evidence": 5,
    "14_rapping_peak_mechanism": 6,
    "03_cluster_selection": 7,
    "04_condition_map": 8,
    "07_power_prediction": 9,
    "08b_feature_importance": 10,
    "08_power_residuals": 11,
    "05b_boundary_proximity": 12,
    "19_emission_power_frontier": 13,
    "11_optimal_voltage": 14,
    "18_condition_energy_penalty": 15,
    "13_emission_decomposition": 16,
    "15_search_convergence": 17,
    "16_sensitivity_distribution": 18,
    "20_eta_sensitivity": 19,
    "17_sensitivity_heatmap": 20,
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
    """The two internal indices disagree; the figure has to show that openly."""
    k = cluster_eval["k"].to_numpy(int)
    sil = cluster_eval["silhouette"].to_numpy(float)
    ch = cluster_eval["calinski_harabasz"].to_numpy(float)
    k_sil, k_ch = int(k[sil.argmax()]), int(k[ch.argmax()])

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    for ax, y, best, label, color, marker in (
            (axes[0], sil, k_sil, "轮廓系数", PALETTE[0], "o"),
            (axes[1], ch / 1000.0, k_ch, "Calinski–Harabasz 指标 (×10³)", PALETTE[2], "s")):
        ax.plot(k, y, marker + "-", color=color, markersize=5.5, linewidth=1.7)
        ax.scatter([best], [y[k == best]], s=150, facecolor="none",
                   edgecolor=PALETTE[4], linewidth=1.8, zorder=4)
        ax.annotate("最大值 $k=%d$" % best, (best, y[k == best][0]),
                    textcoords="offset points", xytext=(0, 16), ha="center",
                    fontsize=8.6, color=PALETTE[4])
        ax.set_xlabel("聚类数 $k$")
        ax.set_ylabel(label)
        ax.set_xticks(k)
        headroom(ax, 0.24)
        finish_axes(ax, grid_axis="both")
    add_subpanel_label(axes[0], "a")
    add_subpanel_label(axes[1], "b")
    note(axes[0], "本文取 $k=4$", loc="lower right")
    note(axes[1], "该指标支持 $k=5$", loc="lower right")
    fig.tight_layout(w_pad=2.2)
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



def fig04b_operating_evidence(df: pd.DataFrame, events: pd.DataFrame,
                              strategy: pd.DataFrame) -> None:
    """The identifiable half of Question 1.

    The outlet column carries no signal, but the operator's response to the
    inlet conditions is plainly visible: two sustained excursions (panels a, b)
    and a stable correlation structure between conditions and settings (c).
    """
    fig = plt.figure(figsize=(8.4, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.60, wspace=0.62)

    windows = [
        ("2024-05-02 13:30", "2024-05-02 16:20", "C_in_gNm3", "入口浓度 (g/Nm³)", "a"),
        ("2024-05-06 12:30", "2024-05-06 15:30", "Temp_C", "入口温度 (°C)", "b"),
    ]
    for col_idx, (t0, t1, col, ylabel, tag) in enumerate(windows):
        ax = fig.add_subplot(gs[0, col_idx])
        add_subpanel_label(ax, tag, x=-0.14)
        win = df[(df["timestamp"] >= t0) & (df["timestamp"] <= t1)]
        minutes = (win["timestamp"] - win["timestamp"].iloc[0]).dt.total_seconds() / 60
        ax.plot(minutes, win[col], color=PALETTE[4], linewidth=1.4, label=ylabel.split(" ")[0])
        ax.set_ylabel(ylabel, color=PALETTE[4], fontsize=9)
        ax.tick_params(axis="y", labelcolor=PALETTE[4])
        ax.set_xlabel("事件窗口内时间 (min)")
        twin = ax.twinx()
        twin.plot(minutes, win["U1_kV"], color=PALETTE[0], linewidth=1.4, label="$U_1$")
        twin.plot(minutes, win["P_total_kW"] / 30.0, color=PALETTE[2],
                  linewidth=1.2, linestyle="--", label="总功率/30")
        twin.set_ylabel("$U_1$ (kV)　总功率/30 (kW)", fontsize=8.0)
        twin.spines["top"].set_visible(False)
        finish_axes(ax, grid_axis="none")
        handles = ax.get_lines() + twin.get_lines()
        ax.legend(handles, [h.get_label() for h in handles],
                  frameon=False, fontsize=7.4, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, 1.30))

    ax = fig.add_subplot(gs[1, :])
    add_subpanel_label(ax, "c", x=-0.055)
    cond_labels = {"r_Temp_C": "入口温度 $H$", "r_C_in_gNm3": "入口浓度 $C_{in}$", "r_Q_Nm3h": "烟气流量 $Q$"}
    full = strategy[strategy["window"] == "full_record"].set_index("control")
    matrix = full.loc[U_COLS + T_COLS, list(cond_labels)].to_numpy(float).T
    im = ax.imshow(matrix, cmap="RdBu_r", aspect="auto", vmin=-0.30, vmax=0.30)
    ax.set_xticks(range(matrix.shape[1]),
                  [f"$U_{i}$" for i in range(1, 5)] + [f"$T_{i}$" for i in range(1, 5)])
    ax.set_yticks(range(3), list(cond_labels.values()))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=8.6,
                    color="white" if abs(matrix[i, j]) > 0.20 else "#1F2933")
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    cbar.set_label("Pearson 相关系数", fontsize=8.4)
    ax.set_title("全记录期工况变量与控制设定的历史响应（可由附件识别）", fontsize=9.2, pad=8)
    tr = strategy[strategy["window"] == "train"].set_index("control")
    te = strategy[strategy["window"] == "retrospective_test"].set_index("control")
    ax.text(1.0, -0.32,
            "注：振打周期规律（$C_{in}\\uparrow\\Rightarrow T\\downarrow$）在四个时窗内一致；"
            "电压规律并不平稳——$r(C_{in},U_1)$ 在前5天为 %+.2f，在末日为 %+.2f。"
            % (float(tr.loc["U1_kV", "r_C_in_gNm3"]), float(te.loc["U1_kV", "r_C_in_gNm3"])),
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color=GRAY)
    save_figure(fig, OUT, "04b_operating_evidence")


def fig22_roadmap() -> None:
    """One-page technical roadmap.

    The paper's argument splits at the identifiability audit into an
    empirically identified branch and an explicitly conditional grey-box
    branch; the two are visually distinguished so a reviewer can see at a
    glance which conclusions carry which strength of evidence.
    """
    EMP, SCN = "#1F4E79", "#B45309"       # empirical / conditional
    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.2); ax.axis("off")

    def box(x, y, w, h, text, color, dashed=False, fill="#FFFFFF", fs=9.0):
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.10",
            linewidth=1.3, edgecolor=color, facecolor=fill,
            linestyle="--" if dashed else "-", zorder=2))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                color="#1F2933", zorder=3, linespacing=1.45)

    def arrow(x1, y1, x2, y2, color=GRAY):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=13, linewidth=1.1,
                                     color=color, zorder=1,
                                     shrinkA=2, shrinkB=2))

    box(5, 11.5, 5.6, 0.78, "现场原始数据　10080×14", GRAY, fill="#F3F4F6")
    arrow(5, 11.11, 5, 10.62)
    box(5, 10.2, 4.6, 0.80, "§2　可识别性审计", "#111827", fill="#E5E7EB")

    arrow(4.0, 9.80, 2.6, 9.25); arrow(6.0, 9.80, 7.4, 9.25)
    box(2.6, 8.55, 4.0, 1.30,
        "实证数据层（可复算）\n§4.1　$P=a_0+\\sum b_iU_i^2+\\sum c_i/T_i$\n"
        "§4.2—4.3　历史操作规律与异常工况", EMP)
    box(7.4, 8.55, 4.0, 1.30,
        "条件式灰箱层（推演）\n§4.4—4.5　$C_{peak}(U,T)=C_{base}+B$\n"
        "Deutsch–Anderson　锚点 50 mg/Nm³", SCN, dashed=True)

    arrow(2.6, 7.90, 4.2, 7.35); arrow(7.4, 7.90, 5.8, 7.35)
    box(5, 6.95, 5.2, 0.80, "§5.1、§5.3　典型工况划分　+　历史支持域", "#111827")

    arrow(5, 6.55, 5, 6.05)
    box(5, 5.65, 5.4, 0.80, "问题二　§5.5　10 mg/Nm³ 最低功率控制参数", EMP)
    arrow(5, 5.25, 5, 4.75)
    box(5, 4.35, 5.4, 0.80, "问题三　§5.4、§6　电场调节优先级判据", EMP)
    arrow(5, 3.95, 5, 3.45)
    box(5, 3.05, 5.4, 0.80, "问题四　§7　10→5 mg/Nm³ 功率与容量代价", SCN, dashed=True)

    arrow(5, 2.65, 5, 2.15)
    box(5, 1.55, 6.6, 1.10,
        "§5.6 外推审计　·　§8.2 敏感性分析\n§9.3 现场阶跃标定　·　独立盲测后闭环",
        GRAY, fill="#F3F4F6", fs=8.6)

    handles = [
        plt.Line2D([], [], color=EMP, linewidth=1.6, label="实证辨识：可由附件复算"),
        plt.Line2D([], [], color=SCN, linewidth=1.6, linestyle="--",
                   label="条件式推演：须现场标定后方可采信"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
              ncol=2, frameon=False, fontsize=8.4)
    fig.tight_layout()
    save_figure(fig, OUT, "22_technical_roadmap")


def fig05b_boundary_proximity(audit: pd.DataFrame) -> None:
    """How far outside historical joint operating experience each solution sits."""
    ordered = audit.sort_values(["condition_cluster", "limit_mgNm3"],
                               ascending=[True, False]).reset_index(drop=True)
    labels = [f"工况{int(c)}\n{int(l)} mg"
              for c, l in zip(ordered["condition_cluster"], ordered["limit_mgNm3"])]
    colors = [PALETTE[int(c) - 1] for c in ordered["condition_cluster"]]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    add_subpanel_label(ax, "a")
    x = np.arange(len(ordered))
    ax.vlines(x, 1.0, ordered["ratio_to_q975"], color=colors, linewidth=2.2, alpha=0.8)
    ax.scatter(x, ordered["ratio_to_q975"], color=colors, s=45,
               edgecolor="black", linewidth=0.5, zorder=3)
    ax.axhline(1.0, color=PALETTE[4], linestyle="--", linewidth=1.2, label="历史经验97.5%外壳")
    ax.set_xticks(x, labels, fontsize=7.6)
    ax.set_ylabel(r"$d_M^2/q_{0.975}$（倍）")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    headroom(ax, 0.16)
    finish_axes(ax)

    ax = axes[1]
    add_subpanel_label(ax, "b")
    bars = ax.bar(x, ordered["max_voltage_over_historical_max_pct"], color=colors, width=0.62)
    _label_bars(ax, bars, fmt="{:+.1f}%", dy=0.12)
    ax.axhline(0, color=PALETTE[4], linestyle="--", linewidth=1.2)
    ax.set_xticks(x, labels, fontsize=7.6)
    ax.set_ylabel("最高电压超出历史最大值 (%)")
    note(ax, "0%以下＝仍在历史电压包络内", loc="upper left")
    headroom(ax, 0.30)
    finish_axes(ax)
    fig.tight_layout(w_pad=1.9)
    save_figure(fig, OUT, "05b_boundary_proximity")


def fit_power_model(df: pd.DataFrame) -> PhysicalPowerModel:
    train = df[df["split"] == "train"]
    return PhysicalPowerModel().fit(train, train["P_total_kW"].to_numpy(float))


def fit_linear_period_model(df: pd.DataFrame) -> RidgePowerModel:
    """The misspecified reference: rapping period entered as a linear term."""
    train = df[df["split"] == "train"]
    metrics = pd.read_json(RESULTS / "power_model_metrics.json", typ="series")
    alpha = float(metrics["ridge_selected_alphas"]["quadratic_ridge_T"])
    model = RidgePowerModel(alpha=alpha)
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
    metrics = pd.read_json(RESULTS / "power_model_metrics.json", typ="series")
    ax.text(0.04, 0.94,
            "$R^2=%.3f$\nRMSE=%.2f kW\n偏差=%+.2f kW" % (float(metrics["retrospective_test_r2"]),
                                                        float(metrics["retrospective_test_rmse_kW"]),
                                                        float(metrics["retrospective_test_bias_kW"])),
            transform=ax.transAxes, va="top",
            bbox={"boxstyle":"round", "facecolor":"white", "edgecolor":LIGHT_GRAY})
    ax.set_xlabel("实际总功率 (kW)")
    ax.set_ylabel("预测总功率 (kW)")
    ax.legend(frameon=False, loc="lower right")
    finish_axes(ax, grid_axis="both")

    ax = axes[1]
    add_subpanel_label(ax, "b")
    labels = ["均值\n基线", "线性$U$\n线性$T$", "二次$U$\n线性$T$", "本文\n$U^2+1/T$"]
    model_order = ["mean_predictor", "linear_ridge_T", "quadratic_ridge_T", "physical_invT"]
    x = np.arange(len(model_order))
    val = baselines[baselines["split"] == "validation"].set_index("model").loc[model_order, "rmse_kW"]
    tst = baselines[baselines["split"] == "retrospective_test"].set_index("model").loc[model_order, "rmse_kW"]
    ax.bar(x-0.18, val, 0.36, color=PALETTE[5], label="验证集")
    ax.bar(x+0.18, tst, 0.36, color=PALETTE[0], label="回顾性测试")
    ax.set_xticks(x, labels, fontsize=7.6)
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


def fig08b_specification(df: pd.DataFrame, model: PhysicalPowerModel,
                         linear_model: RidgePowerModel, baselines: pd.DataFrame) -> None:
    """Why the rapping period must enter as 1/T rather than T.

    Panel (a) shows the residual of a voltage-only fit against the period:
    the curvature is the signature of a 1/T term.  Panel (b) contrasts the
    retrospective-test bias of both specifications.
    """
    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    add_subpanel_label(ax, "a")
    u = train[U_COLS].to_numpy(float)
    resid = train["P_total_kW"].to_numpy(float) - model.field_power(u) - model.intercept
    t1 = train["T1_s"].to_numpy(float)
    order = np.argsort(t1)
    ax.scatter(t1[order][::7], resid[order][::7], s=7, alpha=0.22,
               color=PALETTE[5], edgecolors="none", label="扣除电场功率后的残差")
    grid = np.linspace(t1.min(), t1.max(), 120)
    rap_other = float(np.sum(model.c[1:] / train[T_COLS[1:]].median().to_numpy(float)))
    ax.plot(grid, model.c[0] / grid + rap_other, color=PALETTE[4], linewidth=2.0,
            label=r"拟合 $c_1/T_1+\mathrm{const}$")
    slope = np.polyfit(t1, resid, 1)
    ax.plot(grid, np.polyval(slope, grid), "--", color=PALETTE[0], linewidth=1.6,
            label=r"线性 $T_1$ 设定")
    ax.set_xlabel("第1电场振打周期 $T_1$ (s)")
    ax.set_ylabel("剩余功率 (kW)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    finish_axes(ax, grid_axis="both")

    ax = axes[1]
    add_subpanel_label(ax, "b")
    names = {"quadratic_ridge_T": "线性 $T$ 设定", "physical_invT": "本文 $1/T$ 设定"}
    tst = baselines[baselines["split"] == "retrospective_test"].set_index("model")
    x = np.arange(2)
    bias = [float(tst.loc[k, "bias_kW"]) for k in names]
    rmse = [float(tst.loc[k, "rmse_kW"]) for k in names]
    ax.bar(x - 0.18, bias, 0.36, color=PALETTE[4], label="测试平均偏差")
    ax.bar(x + 0.18, rmse, 0.36, color=PALETTE[0], label="测试 RMSE")
    for xi, (b, r) in enumerate(zip(bias, rmse)):
        ax.text(xi - 0.18, b + (0.6 if b >= 0 else -1.6), f"{b:+.2f}", ha="center", fontsize=8)
        ax.text(xi + 0.18, r + 0.6, f"{r:.2f}", ha="center", fontsize=8)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.set_xticks(x, list(names.values()))
    ax.set_ylabel("回顾性测试 (kW)")
    ax.legend(frameon=False, fontsize=8)
    headroom(ax, 0.22)
    finish_axes(ax)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT, "08b_feature_importance")


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
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.tight_layout(h_pad=2.0, rect=(0, 0.06, 1, 1))
    figure_legend(fig, handles, labels, ncol=2, bottom=0.0)
    save_figure(fig, OUT, stem)



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
    """The rapping trade-off, derived rather than assumed.

    Cake mass per rap grows like L*T, so the per-event peak grows linearly with
    the period while the contribution to the hourly mean, L*T*(1/T) = L, does
    not depend on the period at all.  Panel (b) prices the same lever in power.
    """
    trade = pd.read_csv(RESULTS / "rapping_power_tradeoff.csv")
    ratio = np.linspace(0.75, 1.35, 200)
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.5), gridspec_kw={"width_ratios": [1.1, 1]})

    ax = axes[0]
    add_subpanel_label(ax, "a")
    for cluster in (1, 3, 4):
        pr = profiles[profiles["condition_cluster"] == cluster].iloc[0]
        peak = 1.2 * float(pr["load_ratio_to_train_median"]) * ratio
        ax.plot(100 * (ratio - 1), peak, color=PALETTE[cluster - 1],
                marker=["o", "s", "^", "D"][cluster - 1], markevery=40,
                label=COND_SHORT[cluster].replace("\n", " "))
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axvline(0, color=GRAY, linewidth=0.8)
    ax.plot(100 * (ratio - 1), np.full_like(ratio, 1.2 * float(
        profiles[profiles["condition_cluster"] == 3].iloc[0]["load_ratio_to_train_median"])),
        ":", color="#111827", linewidth=1.5, label="对小时均值的贡献（工况3，与$T$无关）")
    ax.set_xlabel("振打周期相对参考周期变化 (%)")
    ax.set_ylabel("单次再飞扬峰值代理 (mg/Nm³)")
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    headroom(ax, 0.18)
    finish_axes(ax, grid_axis="both")

    ax = axes[1]
    add_subpanel_label(ax, "b")
    x = np.arange(len(trade))
    bars = ax.bar(x, trade["rapping_power_kW"], color=[PALETTE[5], PALETTE[2], PALETTE[4]], width=0.6)
    _label_bars(ax, bars, fmt="{:.0f}", dy=6)
    for xi, row in trade.iterrows():
        if row["power_recovered_kW"] > 0:
            ax.text(xi, row["rapping_power_kW"] / 2,
                    "省%.0f kW\n积灰+%.0f%%" % (row["power_recovered_kW"],
                                              row["cake_mass_per_rap_change_pct"]),
                    ha="center", va="center", fontsize=7.6, color="white")
    ax.set_xticks(x, ["历史\n中位周期", "延长至\n历史P95", "延长至\n历史最大"], fontsize=8)
    ax.set_ylabel("振打环节功率 (kW)")
    note(ax, "中位工况下振打占总功率 %.1f%%" % float(trade.iloc[0]["rapping_share_pct"]),
         loc="upper right")
    headroom(ax, 0.24)
    finish_axes(ax)
    fig.tight_layout(w_pad=2.0)
    save_figure(fig, OUT, "14_rapping_peak_mechanism")


def fig15_optimizer_verification(verification: pd.DataFrame) -> None:
    """Multi-start SLSQP against a 4x10^5-point random search of the same set."""
    ordered = verification.sort_values(["condition_cluster", "limit_mgNm3"],
                                       ascending=[True, False]).reset_index(drop=True)
    labels = [f"工况{int(c)}\n{int(l)} mg"
              for c, l in zip(ordered["condition_cluster"], ordered["limit_mgNm3"])]
    x = np.arange(len(ordered))
    slsqp = ordered["slsqp_power_kW"].to_numpy(float)
    rnd = ordered["random_search_power_kW"].to_numpy(float)
    gaps = ordered["random_search_gap_pct"].to_numpy(float)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3), gridspec_kw={"width_ratios": [1.2, 1]})

    ax = axes[0]
    add_subpanel_label(ax, "a")
    lo = float(np.nanmin(slsqp)) * 0.94
    hi = float(np.nanmax([np.nanmax(slsqp), np.nanmax(rnd)])) * 1.04
    ax.set_ylim(lo, hi)
    ax.bar(x - 0.18, slsqp, 0.36, color=PALETTE[0], label="多起点SLSQP")
    ax.bar(x + 0.18, np.where(np.isfinite(rnd), rnd, lo), 0.36,
           color=LIGHT_GRAY, label="随机搜索最优")
    for xi, value in enumerate(rnd):
        if not np.isfinite(value):
            ax.annotate("随机搜索\n无可行样本", xy=(xi + 0.18, lo),
                        xytext=(xi + 0.55, lo + 0.30 * (hi - lo)),
                        fontsize=7.2, color=PALETTE[4], ha="left",
                        arrowprops={"arrowstyle": "->", "color": PALETTE[4], "linewidth": 0.9})
    ax.set_xticks(x, labels, fontsize=7.4)
    ax.set_ylabel("最优总功率 (kW)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    finish_axes(ax)

    ax = axes[1]
    add_subpanel_label(ax, "b")
    finite = np.isfinite(gaps)
    ax.bar(x[finite], gaps[finite], color=PALETTE[2], width=0.6)
    for xi, value in zip(x[finite], gaps[finite]):
        ax.text(xi, value + 0.12, f"{value:.2f}", ha="center", va="bottom", fontsize=7.4)
    ax.set_ylim(0, float(np.nanmax(gaps)) * 1.42)
    ax.set_xticks(x, labels, fontsize=7.4)
    ax.set_ylabel("随机搜索相对SLSQP的劣势 (%)")
    note(ax, "全部为正：随机搜索未找到更优解", loc="upper right")
    finish_axes(ax)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, OUT, "15_search_convergence")


def fig16_structural_sensitivity(structural: pd.DataFrame) -> None:
    valid = structural[structural["all_conditions_feasible"] == True]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), sharey=True)
    ax = axes[0]
    add_subpanel_label(ax, "a")
    exps = sorted(valid["peak_exponent"].unique())
    data = [valid[valid["peak_exponent"] == x]["weighted_power_increase_pct"] for x in exps]
    boxes = ax.boxplot(data, tick_labels=[f"{x:.2f}" for x in exps],
                       patch_artist=True, showfliers=False)
    for i, box in enumerate(boxes["boxes"]):
        box.set_facecolor(PALETTE[i % len(PALETTE)]); box.set_alpha(0.72)
    ax.set_xlabel(r"振打峰值指数 $\gamma$（本文推导值 1.00）")
    ax.set_ylabel("5相对10 mg/Nm³加权功率增幅 (%)")
    finish_axes(ax)

    ax = axes[1]
    add_subpanel_label(ax, "b")
    phases = sorted(valid["phase_scale"].unique())
    for profile, style, label, color in zip(
            ["equal", "weak_front", "central_front"], ["o-", "s--", "^:"],
            ["等权", "弱前级", "中心前级"], [PALETTE[0], PALETTE[2], PALETTE[4]]):
        medians = [valid[(valid["phase_scale"] == ph) & (valid["alpha_profile"] == profile)]
                   ["weighted_power_increase_pct"].median() for ph in phases]
        ax.plot(phases, medians, style, color=color, label=label)
    ax.set_xlabel(r"振打相位叠加尺度 $\rho$")
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
    """Does the closed-form estimate reproduce the full eight-variable optimum?

    Four near-equal bars of ~6.7% carry little information; what is worth
    showing is that the analytical approximation and the numerical optimum
    agree to within 1%.
    """
    q4 = q4.sort_values("condition_cluster")
    ana = q4["analytic_delta_power_kW"].to_numpy(float)
    num = q4["numeric_delta_power_kW"].to_numpy(float)
    rel = 100.0 * (ana - num) / num

    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.5), gridspec_kw={"width_ratios": [1, 1.05]})

    ax = axes[0]
    add_subpanel_label(ax, "a")
    lo, hi = min(ana.min(), num.min()) - 8, max(ana.max(), num.max()) + 8
    ax.plot([lo, hi], [lo, hi], "--", color=GRAY, linewidth=1.2, label="$y=x$")
    for idx in range(len(q4)):
        ax.scatter(num[idx], ana[idx], s=70, color=PALETTE[idx], zorder=3,
                   edgecolor="white", linewidth=0.8,
                   label=COND_SHORT[int(q4["condition_cluster"].iloc[idx])].replace("\n", " "))
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("数值最优解的 $\\Delta P$ (kW)")
    ax.set_ylabel("闭式估计的 $\\Delta P$ (kW)")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left")
    finish_axes(ax, grid_axis="both")

    ax = axes[1]
    add_subpanel_label(ax, "b")
    x = np.arange(len(q4))
    bars = ax.bar(x, rel, color=[PALETTE[i] for i in range(len(q4))], width=0.58)
    for bar, value in zip(bars, rel):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (0.04 if value >= 0 else -0.09),
                f"{value:+.2f}%", ha="center",
                va="bottom" if value >= 0 else "top", fontsize=8)
    ax.axhline(0, color=GRAY, linewidth=0.8)
    ax.axhspan(-1, 1, color=NEUTRAL_FILL, alpha=0.45, zorder=0)
    ax.text(len(q4) - 0.5, 0.86, "±1% 带", ha="right", va="top", fontsize=7.8, color=GRAY)
    ax.set_xticks(x, [COND_SHORT[int(c)] for c in q4["condition_cluster"]], fontsize=7.4)
    ax.set_ylabel("闭式估计相对数值解的偏差 (%)")
    ax.set_ylim(-1.2, 1.2)
    finish_axes(ax)
    fig.tight_layout(w_pad=2.2)
    save_figure(fig, OUT, "18_condition_energy_penalty")


def _relpath(path: str) -> str:
    """交付物中一律使用相对路径，避免写入本机绝对路径。"""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


TIKZ_ROWS = [{
    "figure_id": "01", "manuscript_figure": 1, "stem": "00_esp_schematic",
    "caption": "四电场电除尘器结构与变量定义",
    "png": "figures_paper/00_esp_schematic.pdf", "pdf": "figures_paper/00_esp_schematic.pdf",
    "source_data": "figures_paper/00_esp_schematic.tex",
    "transformation": "按赛题给定的设备结构示意以TikZ矢量重绘，xelatex独立编译",
    "supported_manuscript_claims": "§1.1；图1",
    "limitations": "结构示意图，不含实测数据；比例非工程真实尺寸",
}]


def fig19_frontier(frontier: pd.DataFrame, weighted: pd.DataFrame) -> None:
    """排放—功率权衡前沿：说明放宽限值换来的功率下降不是等排放下的节能。"""
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for idx, (cluster, grp) in enumerate(frontier.groupby("condition_cluster")):
        grp = grp.sort_values("limit_mgNm3")
        ax.plot(grp["limit_mgNm3"], grp["predicted_power_kW"], "o-",
                color=PALETTE[idx], markersize=4.5, linewidth=1.5,
                label=COND_SHORT[int(cluster)].replace("\n", " "))
    w = weighted.sort_values("limit_mgNm3")
    ax.plot(w["limit_mgNm3"], w["weighted_power_kW"], "s--", color="#111827",
            markersize=5, linewidth=2.0, label="工况加权")
    hist = float(frontier["historical_power_kW"].iloc[0]) if "historical_power_kW" in frontier else np.nan
    p10 = float(w[w["limit_mgNm3"] == 10.0]["weighted_power_kW"].iloc[0])
    p5 = float(w[w["limit_mgNm3"] == 5.0]["weighted_power_kW"].iloc[0])
    if np.isfinite(hist):
        ax.axhline(hist, color=GRAY, linestyle=":", linewidth=1.3)
        ax.text(9.9, hist - 14, f"历史实际加权功率 {hist:.0f} kW（排放约50 mg/Nm³）",
                fontsize=8, color=GRAY, ha="right")
    ax.annotate("", xy=(10.0, p10), xytext=(5.0, p5),
                arrowprops={"arrowstyle": "<->", "color": PALETTE[4], "linewidth": 1.3})
    ax.text(7.5, (p5 + p10) / 2 + 10,
            f"限值 10→5 mg/Nm³\n加权功率增 {p5 - p10:.0f} kW（{100*(p5-p10)/p10:.1f}%）",
            fontsize=8.2, color=PALETTE[4], ha="center")
    ax.set_xlabel("情景排放限值 (mg/Nm³)")
    ax.set_ylabel("支持域内最佳采样预测功率 (kW)")
    ax.set_xticks(sorted(frontier["limit_mgNm3"].unique()))
    ax.legend(frameon=False, ncol=2, fontsize=8.4, loc="upper right")
    headroom(ax, 0.16)
    finish_axes(ax, grid_axis="both")
    fig.tight_layout()
    save_figure(fig, OUT, "19_emission_power_frontier")


def fig20_p_sensitivity(p_ref: pd.DataFrame, prior: pd.DataFrame) -> None:
    """The increase is set by one calibratable scalar: the exponent p in K ~ U^p."""
    ref = p_ref.sort_values("p_exponent")
    fig, ax = plt.subplots(figsize=(6.9, 3.9))
    ax.plot(ref["p_exponent"], ref["analytic_increase_pct"], "-", color="#111827",
            linewidth=1.8, zorder=2, label=r"解析式 $P_f[(1+\Delta K/K)^{2/p}-1]/P_{10}$")
    feas = ref[ref["feasible_at_5pct_headroom"] == True]
    infeas = ref[ref["feasible_at_5pct_headroom"] != True]
    ax.scatter(feas["p_exponent"], feas["analytic_increase_pct"], s=62, color=PALETTE[0],
               edgecolor="white", linewidth=0.8, zorder=4, label="5%电压裕度内可达")
    ax.scatter(infeas["p_exponent"], infeas["analytic_increase_pct"], s=62,
               facecolor="white", edgecolor=PALETTE[4], linewidth=1.6, zorder=4,
               label="需更大电压裕度（不可达）")
    num = ref.dropna(subset=["numeric_increase_pct"])
    ax.scatter(num["p_exponent"], num["numeric_increase_pct"], marker="x", s=52,
               color=PALETTE[2], zorder=5, label="数值重解校验")
    for _, row in ref.iterrows():
        ax.annotate(f"{row['analytic_increase_pct']:.1f}%",
                    (row["p_exponent"], row["analytic_increase_pct"]),
                    textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=8.2, color=GRAY)
    ax.set_xlabel(r"去除指数—电压幂次 $p$（$K\propto U^p$；$p=2$ 为Deutsch迁移速度形式）")
    ax.set_ylabel("10→5 mg/Nm³ 加权功率增幅 (%)")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    headroom(ax, 0.24)
    finish_axes(ax, grid_axis="both")
    fig.tight_layout()
    save_figure(fig, OUT, "20_eta_sensitivity")


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
    rows = TIKZ_ROWS + rows
    rows.sort(key=lambda r: r["manuscript_figure"])
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
    audit = pd.read_csv(RESULTS / "support_boundary_audit.csv")
    verification = pd.read_csv(RESULTS / "optimization_verification.csv")
    structural = pd.read_csv(RESULTS / "structural_sensitivity.csv")
    ablation = pd.read_csv(RESULTS / "field_priority_ablation_summary.csv")
    baselines = pd.read_csv(RESULTS / "power_model_baselines.csv")
    q4 = pd.read_csv(RESULTS / "question4_by_condition.csv")
    frontier = pd.read_csv(RESULTS / "emission_power_frontier.csv")
    frontier_w = pd.read_csv(RESULTS / "emission_power_frontier_weighted.csv")
    prior = pd.read_csv(RESULTS / "question4_sensitivity.csv")
    p_ref = pd.read_csv(RESULTS / "p_exponent_reference.csv")
    events = pd.read_csv(RESULTS / "anomaly_events.csv")
    strategy = pd.read_csv(RESULTS / "historical_strategy_correlations.csv")
    model = fit_power_model(df)
    linear_model = fit_linear_period_model(df)

    fig01_outlet_diagnosis(df)
    fig02_timeseries(df)
    fig22_roadmap()
    fig04b_operating_evidence(df, events, strategy)
    fig14_peak(profiles)
    fig03_cluster_selection(cluster_eval)
    fig04_condition_map(df)
    fig07_power_prediction(df, model, baselines)
    fig08b_specification(df, model, linear_model, baselines)
    fig08_residuals(df, model)
    fig05b_boundary_proximity(audit)
    fig19_frontier(frontier, frontier_w)
    grouped_controls(optimum, U_COLS, "11_optimal_voltage", "最优电压 (kV)")
    fig18_penalty(q4)
    fig13_emission(optimum)
    fig15_optimizer_verification(verification)
    fig16_structural_sensitivity(structural)
    fig20_p_sensitivity(p_ref, prior)
    fig17_field_ablation(ablation)
    write_manifest()
    print(f"Generated {len(CAPTIONS)} figures in {OUT}")


if __name__ == "__main__":
    main()
