# 阶段资产索引

以下路径相对归档根目录（现场为 ~/vision_diagnostics）。历史结果按原目录保留。核对先看报告及对应事件记录。

| 阶段 | 资料 | 用途 |
|---|---|---|
| E3.2 / E3.3 | e3_2/SUMMARY.md、metrics.csv；e3_3/E3.3.md、depth_stress.csv | 感知路线与深度压力场景 |
| E3.4 / E3.4b | e3_4/REPORT.md、observations.csv；e3_4b/d2.py、counterexamples.csv | 候选深度与归属边界 |
| E3.5 | e3_5/SUMMARY.md、trajectory.csv、events.csv、reproduce.sh | V1 和两条 YOLOE 路线的软件闭环、历史输入 |
| R4 | r4/SUMMARY.md、control_events.csv、metrics.csv、reproduce.sh | Final/V2 比较、异步门控 |
| R4.1 / R4.2 | r4_1/INTEGRATION.md、safety_regression.csv；r4_2/evidence/test_results.md、runs/、tests/verify.py | 最小授权修复及已集成版本的回归 |
| V4.0 | v4_0/ARCHITECTURE.md、INTERFACE_CONTRACT.md、RISK_REGISTER.md | 当前架构、目标契约、风险 |
| V4.1 | v4_1/INTERFACE_MATRIX.csv、results/acceptance_1/、results/acceptance_2/、tests/regression/data/manifest.json | A/B/C 矩阵、两轮原始事件、历史输入哈希 |
| H1.0 | h1_0/CAMERA_CONFIG.md、CHECKPOINT.md、samples/、timestamp_pairs.csv、calibration/ | D435 实测、同步样本、标定与停机记录 |
| V4.1a | v4_1a/runs/、evidence/、first_commit.patch、V4.1a-review.zip | 按版本导出构建、正负对照、可审查补丁 |
