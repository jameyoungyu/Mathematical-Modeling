# 2605 · 微构体中填充导电介质的仿真优化

本项目用于校内数学建模练习，包含题目材料、建模记录、终稿、源程序与可复现结果。

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
| `src/solve_p4.py` | 问题四：A/B 混填候选搜索与直接验证 | `results/p4_cost_optimum.json` |
| `src/p4_backfill_probes.py` | 补记阶段 C 未通过的探测点 | 同上（追加字段） |
| `src/p4_break_even.py` | 介质 B 的盈亏平衡单价 | `results/p4_break_even.json` |
| `src/theory_check.py` | 排除体积判据、跃变中点、边际效率 | `results/theory_check.json` |
| `src/sensitivity.py` | 灵敏度与收敛性 | `results/sensitivity.json` |
| `src/sphere_mode_check.py` | 介质 B 越界处理口径对照 | `results/sphere_mode_check.json` |
| `src/cluster_stats.py` | 最大团簇占比（渗流序参量） | `results/cluster_stats.json` |
| `src/geometry_bracket.py` | 球柱体近似的严格上下界 | `results/geometry_bracket.json` |
| `src/p4_global_audit.py` | 问题四预算线角点审计与覆盖范围证书 | `results/p4_global_audit.json` |
| `src/p4_global_audit2.py` | 问题四未覆盖整数列的可断点续跑审计 | `results/p4_global_audit2.json` |
| `src/p4_marginal_recheck.py` | 预算线擦边点的大样本复核 | `results/p4_marginal_recheck.json` |
| `src/make_results_bundle.py` | 合并结果、装配论文附录 B | `results/results.json` |
| `src/paper_figures.py` | 全部正文图件 | `figures_paper/` |
| `src/build_paper_pdf.py` | 由 Markdown 终稿编译提交 PDF | `*_终稿.pdf` |

```bash
cd 04_最终论文与代码交付包/src
python3 validate.py && python3 audit_attachment.py
python3 solve_p1.py && python3 solve_p2.py && python3 solve_p3.py && python3 solve_p4.py
python3 p4_backfill_probes.py && python3 p4_break_even.py && python3 theory_check.py
python3 sensitivity.py && python3 sphere_mode_check.py && python3 cluster_stats.py
python3 geometry_bracket.py && python3 p4_global_audit.py && python3 p4_marginal_recheck.py
python3 p4_global_audit2.py  # 可断点续跑；完整逐列审计耗时较长
python3 paper_figures.py && python3 make_results_bundle.py
python3 build_paper_pdf.py                 # 提交稿 PDF（需 pandoc + texlive-xetex）
python3 build_paper_pdf.py --ai-details    # 支撑材料《AI工具使用详情》PDF
```

依赖：Python ≥3.11、numpy、scipy、matplotlib、openpyxl；编译 PDF 另需 pandoc 与
texlive-xetex / texlive-lang-chinese（正文用 Fandol，代码用 DejaVu Sans Mono，均由 TeX Live 提供）。
随机性全部来自 `numpy.random.SeedSequence`，固定分成 4 条随机流；worker 只负责调度。
因此在 numpy 与算法版本相同的前提下，改变并行度仍可逐位复现。

## 三条影响结论的数据发现

附件与题面的字面读法有出入，三条都在论文第 2 节给出了检验：

1. 附件的一行是**边界截断后的碎片**，不是一根介质：组 1/2/3 的母介质数为 7 / 28 / 354；
2. 组 1、组 2 的周期盒是 **10000×1000×1000 nm**（y、z 在 ±500 回绕），不是立方体；
3. 附件组 3 的介质方向**不是各向同性**，而是以带电面法向为极轴的极角均匀分布
   （KS 检验：各向同性 p=6.7e-19 被拒绝，极角均匀 p=0.73 不被拒绝）。
   这一条使同一体积分数下的导通概率接近翻倍，是全题最敏感的建模口径。
   **但它不作为主口径**：题面对问题二至四只写"方向随机"、未指定分布，
   故正文按最少假设取球面均匀，附件标定分布作为灵敏度口径全程并列给出
   （结果存于 `results/对照口径_polar_uniform/`）。
