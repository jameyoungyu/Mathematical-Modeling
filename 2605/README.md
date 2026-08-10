# 2605 · 微构体中填充导电介质的仿真优化

2026 年第七届"华数杯"大学生数学建模竞赛 A 题（截止 2026-08-10 18:00）。
本目录为**赛后完整复现**，非赛内提交稿。

## 交付索引

```
01_题目与附件/                  原始赛题 PDF 与附件 xlsx
03_建模过程记录/notes/          题目拆解、数据审计结论、建模决策
04_最终论文与代码交付包/
├── 论文_微构体中填充导电介质的仿真优化.md   终稿
├── src/                       可运行源程序（见下）
├── results/                   全部结果 JSON（论文中每个数字的出处）
├── figures_paper/             正文图件（PNG 预览 + PDF 矢量）+ 图注清单
└── scripts/                   绘图字体配置与论文自检脚本
```

## 源程序与运行顺序

| 脚本 | 作用 | 产物 |
|---|---|---|
| `src/microstructure.py` | 几何与渗流内核（边界截断、最短距离、并查集判定） | — |
| `src/load_attachment.py` | 读附件、把碎片还原为母介质 | — |
| `src/audit_attachment.py` | 数据审计（周期盒、丢弃碎片、方向分布检验） | `results/data_audit.json` |
| `src/validate.py` | 几何内核的 6 项独立校验 | `results/validation.json` |
| `src/solve_p1.py` | 问题一：三个微构体的导通判定 | `results/p1_connectivity.json` |
| `src/solve_p2.py` | 问题二：四档体积分数的导通概率 | `results/p2_probabilities.json` |
| `src/solve_p3.py` | 问题三：P≥90% 的最低体积分数 | `results/p3_threshold.json` |
| `src/solve_p4.py` | 问题四：A/B 混填的最低成本配比 | `results/p4_cost_optimum.json` |
| `src/sensitivity.py` | 灵敏度与收敛性 | `results/sensitivity.json` |
| `src/paper_figures.py` | 全部正文图件 | `figures_paper/` |

```bash
cd 04_最终论文与代码交付包/src
python3 validate.py && python3 audit_attachment.py
python3 solve_p1.py && python3 solve_p2.py && python3 solve_p3.py && python3 solve_p4.py
python3 sensitivity.py && python3 paper_figures.py
```

依赖：Python ≥3.11、numpy、scipy、matplotlib、openpyxl。
随机性全部来自 `numpy.random.SeedSequence`，按 worker 分叉，结果与并行度无关、可逐位复现。

## 三条影响结论的数据发现

附件与题面的字面读法有出入，三条都在论文第 2 节给出了检验：

1. 附件的一行是**边界截断后的碎片**，不是一根介质：组 1/2/3 的母介质数为 7 / 28 / 354；
2. 组 1、组 2 的周期盒是 **10000×1000×1000 nm**（y、z 在 ±500 回绕），不是立方体；
3. 介质方向**不是各向同性**，而是以带电面法向为极轴的极角均匀分布
   （KS 检验：各向同性 p=6.7e-19 被拒绝，极角均匀 p=0.73 不被拒绝）。
   这一条使同一体积分数下的导通概率接近翻倍，是全题最敏感的建模口径。
