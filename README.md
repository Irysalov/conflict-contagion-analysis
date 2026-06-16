# 冲突事件时间聚集性分析

基于INGARCH模型的冲突事件时间序列分析，检验冲突是否存在显著的时间聚集性（传染效应）。同时包含时空Hawkes模型的空间自激发效应可视化分析。

## 环境配置

```bash
conda create -n conflict python=3.10 -y
conda activate conflict
pip install -r requirements.txt
```

## 一、INGARCH时间序列分析（基础分析）

按以下顺序依次运行各模块：

### 1.1 INGARCH模型拟合

```bash
python hawkes.py
```

**输出**：
- `./output/hawkes/hawkes_results_discrete.json` — 参数估计结果（μ, α, β, 分支比, p值）

### 1.2 INGARCH模型诊断

```bash
python diagnostics.py
```

**输出**：
- `./output/diagnostics/palestine_israel_diagnostics_discrete.png` — 巴以冲突诊断图
- `./output/diagnostics/russia_ukraine_diagnostics_discrete.png` — 俄乌冲突诊断图

### 1.3 INGARCH结果可视化

```bash
python visualization.py
```

**输出**（保存至 `./output/visualization/`）：
- `branching_ratio_comparison.png` — 分支比对比图
- `parameter_comparison.png` — 参数对比图
- `palestine_israel_fitted_vs_actual.png` — 巴以拟合效果图
- `russia_ukraine_fitted_vs_actual.png` — 俄乌拟合效果图
- `palestine_israel_intensity_sequence.png` — 巴以强度序列图
- `russia_ukraine_intensity_sequence.png` — 俄乌强度序列图
- `results_summary.csv` — 结果汇总表
- `results_table.png` — 结果汇总表图片

### 1.4 INGARCH生成报告

```bash
python report.py
```

**输出**：
- `./report/report.md` — 完整分析报告（Markdown格式）

### 1.5 INGARCH辅助模块（非必须）

```bash
python read.py      # 验证数据加载是否正常
python eda.py       # 探索性数据分析（生成数据概览）
```

### 1.6 INGARCH完整主分析流程

```bash
python hawkes.py && python diagnostics.py && python visualization.py && python report.py
```

---

## 二、时空Hawkes模型分析（空间可视化扩展）

首先进入时空分析子目录：

```bash
cd spatio-temporal_Hawkes
```

> 以下所有命令均需在 `./spatio-temporal_Hawkes/` 目录下执行。

### 2.1 巴以冲突：自激发系数空间热力图

```bash
python PI_time.py
```

**说明**：对巴以冲突各行政区拟合单变量Hawkes模型，估计每个区域的自激发系数α，并以热力图形式展示其空间分布。

**依赖数据**：`../data/Palestine_Israel.csv`、GADM行政区划shapefile（`gadm41_*_shp`）

**输出**（保存至 `./S-T_output/`）：
- `Palestine_Israel_Hawkes1D_Alpha.png` — 自激发系数α空间热力图
- `Palestine_Israel_Hawkes1D_Beta.png` — 时间衰减系数β空间热力图
- `Palestine_Israel_Hawkes1D_Correlation.png` — 模型拟合相关性空间热力图
- `Palestine_Israel_Hawkes1D_Results.csv` — 各行政区估计结果（Alpha, Beta, Corr, P值）

### 2.2 俄乌冲突：自激发系数空间热力图（区级）

```bash
python RU_time.py
```

**说明**：对乌克兰各行政区拟合单变量Hawkes模型，估计每个区域的自激发系数α，并以热力图形式展示其空间分布。

**依赖数据**：`../data/Russia_Ukraine.csv`、`gadm_UKR/` 行政区划shapefile

**输出**（保存至 `./S-T_output/`）：
- `Ukraine_Hawkes1D_Alpha.png` — 乌克兰各区自激发系数α热力图
- `Ukraine_Hawkes1D_Beta.png` — 乌克兰各区时间衰减系数β热力图
- `Ukraine_Hawkes1D_Correlation.png` — 乌克兰各区模型拟合相关性热力图
- `Ukraine_Hawkes1D_Results.csv` — 各区估计结果（Alpha, Beta, Corr, P值）

### 2.3 巴以冲突：时空Hawkes模型（含空间传染）

```bash
python PI_timespace_paper_but_betaori_hfsdst.py
```

**说明**：考虑区域间空间传染效应的Hawkes模型。每个区域的强度不仅受自身历史影响，还受周围区域事件的加权影响。采用指数型空间核（基于哈弗辛距离，单位km），同时优化估计时间衰减β和空间带宽σ。空间核采用归一化权重（空间权重之和为1），模型在二级行政区尺度上运行。

**依赖数据**：`../data/Palestine_Israel.csv`、GADM行政区划shapefile（`gadm41_*_shp`）

**输出**（保存至 `./S-T_output/`）：
- `Palestine_Israel_Hawkes_Alpha.png` — 时空Hawkes自激发系数α热力图
- `Palestine_Israel_Hawkes_Beta.png` — 时间衰减系数β热力图
- `Palestine_Israel_Hawkes_Sigma.png` — 空间带宽σ热力图（单位：km）
- `Palestine_Israel_Hawkes_Correlation.png` — 模型拟合相关性热力图
- `Palestine_Israel_Spatio-temporal_Results.csv` — 各区估计结果（Alpha, Beta, Sigma_km, Corr, P值）

### 2.4 俄乌冲突：时空Hawkes模型（含空间传染）

```bash
python RU_timespace_paper_but_betaori_hfsdst.py
```

**说明**：与2.3类似，对乌克兰二级行政区运行时空Hawkes模型，空间核采用哈弗辛距离（单位：km），空间带宽σ以千米为单位进行优化。模型同时估计α、β、σ三个参数。

**依赖数据**：`../data/Russia_Ukraine.csv`、`gadm_UKR/` 行政区划shapefile

**输出**（保存至 `./S-T_output/`）：
- `Ukraine_Hawkes_Alpha.png` — 时空Hawkes自激发系数α热力图
- `Ukraine_Hawkes_Beta.png` — 时间衰减系数β热力图
- `Ukraine_Hawkes_Sigma.png` — 空间带宽σ热力图（单位：km）
- `Ukraine_Hawkes_Correlation.png` — 模型拟合相关性热力图
- `Ukraine_Hawkes_Spatio-temporal_Results.csv` — 各区估计结果（Alpha, Beta, Sigma_km, Corr, P值）

### 2.5 巴以冲突：全局时空Hawkes假设检验

```bash
python time_space_all_PI.py
```

**说明**：在全局尺度上（将所有行政区合并为一个系统）拟合时空Hawkes模型，通过似然比检验判断巴以冲突是否在统计上存在显著的空间自激发/传染效应。该模块执行H₀: α=0 vs H₁: α>0的假设检验，并输出全局参数估计及p值。由于该分析不依赖行政区边界，因此不会产生逐区域热力图。

**依赖数据**：`../data/Palestine_Israel.csv`、GADM行政区划shapefile（`gadm41_*_shp`）

**输出**（保存至 `./S-T_output/`）：
- `Israel_Palestine_Global_Hawkes_Test.png` — 全局拟合诊断图（时间序列 + 拟合曲线）
- 控制台输出：全局参数估计（α, β, σ）、似然比统计量、p值、检验结论

### 2.6 俄乌冲突：全局时空Hawkes假设检验

```bash
python time_space_all_RU.py
```

**说明**：与2.5类似，对乌克兰战区执行全局时空Hawkes假设检验，判断是否存在显著的空间自激发效应。该分析不依赖行政区边界，因此不会产生逐区域热力图。

**依赖数据**：`../data/Russia_Ukraine.csv`、`gadm_UKR/` 行政区划shapefile

**输出**（保存至 `./S-T_output/`）：
- `Ukraine_Global_Hawkes_Test.png` — 全局拟合诊断图（时间序列 + 拟合曲线）
- 控制台输出：全局参数估计（α, β, σ）、似然比统计量、p值、检验结论

---

## 项目结构

```
项目目录/
├── data/                           # 数据文件
│   ├── Palestine_Israel.csv
│   └── Russia_Ukraine.csv
├── output/                         # INGARCH输出目录
│   ├── hawkes/                     # 模型结果
│   ├── diagnostics/                # 诊断图表
│   ├── visualization/              # 可视化图表
│   └── eda/                        # EDA输出
├── report/                         # 报告输出
├── spatio-temporal_Hawkes/         # 时空Hawkes分析子目录
│   ├── gadm_UKR/                   # 乌克兰GADM行政区划数据
│   ├── gadm41_IRN_shp/             # 伊朗GADM数据
│   ├── gadm41_ISR_shp/             # 以色列GADM数据
│   ├── gadm41_LBN_shp/             # 黎巴嫩GADM数据
│   ├── gadm41_PSE_shp/             # 巴勒斯坦GADM数据
│   ├── gadm41_SYR_shp/             # 叙利亚GADM数据
│   ├── gadm41_YEM_shp/             # 也门GADM数据
│   ├── S-T_output/                 # 时空分析输出
│   ├── PI_time.py                  # 2.1 巴以热力图
│   ├── RU_time.py                  # 2.2 乌克兰热力图（区级）
│   ├── PI_timespace_paper_but_betaori_hfsdst.py    # 2.3 巴以时空Hawkes
│   ├── RU_timespace_paper_but_betaori_hfsdst.py    # 2.4 俄乌时空Hawkes
│   ├── time_space_all_PI.py        # 2.5 巴以全局假设检验
│   └── time_space_all_RU.py        # 2.6 俄乌全局假设检验
├── hawkes.py                       # 1.1 INGARCH拟合
├── diagnostics.py                  # 1.2 INGARCH诊断
├── visualization.py                # 1.3 INGARCH可视化
├── report.py                       # 1.4 生成报告
├── read.py                         # 1.5 数据加载验证
├── eda.py                          # 1.5 探索性数据分析
└── requirements.txt                # 依赖包列表
```

## 时空分析输出目录

`./spatio-temporal_Hawkes/S-T_output/` 目录下包含所有时空Hawkes分析的输出文件：

| 文件名 | 来源模块 | 说明 |
|--------|----------|------|
| `Palestine_Israel_Hawkes1D_Alpha.png` | PI_time.py | 巴以α热力图 |
| `Palestine_Israel_Hawkes1D_Beta.png` | PI_time.py | 巴以β热力图 |
| `Palestine_Israel_Hawkes1D_Correlation.png` | PI_time.py | 巴以相关性热力图 |
| `Palestine_Israel_Hawkes1D_Results.csv` | PI_time.py | 巴以各行政区结果 |
| `Ukraine_Hawkes1D_Alpha.png` | RU_time.py | 乌克兰α热力图 |
| `Ukraine_Hawkes1D_Beta.png` | RU_time.py | 乌克兰β热力图 |
| `Ukraine_Hawkes1D_Correlation.png` | RU_time.py | 乌克兰相关性热力图 |
| `Ukraine_Hawkes1D_Results.csv` | RU_time.py | 乌克兰各区结果 |
| `Palestine_Israel_Hawkes_Alpha.png` | PI_timespace_paper_but_betaori_hfsdst.py | 巴以时空Hawkes α |
| `Palestine_Israel_Hawkes_Beta.png` | PI_timespace_paper_but_betaori_hfsdst.py | 巴以时空Hawkes β |
| `Palestine_Israel_Hawkes_Sigma.png` | PI_timespace_paper_but_betaori_hfsdst.py | 巴以时空Hawkes σ |
| `Palestine_Israel_Hawkes_Correlation.png` | PI_timespace_paper_but_betaori_hfsdst.py | 巴以时空Hawkes 相关性 |
| `Palestine_Israel_Spatiotemporal_Results.csv` | PI_timespace_paper_but_betaori_hfsdst.py | 巴以时空Hawkes结果 |
| `Ukraine_Hawkes_Alpha.png` | RU_timespace_paper_but_betaori_hfsdst.py | 俄乌时空Hawkes α |
| `Ukraine_Hawkes_Beta.png` | RU_timespace_paper_but_betaori_hfsdst.py | 俄乌时空Hawkes β |
| `Ukraine_Hawkes_Sigma.png` | RU_timespace_paper_but_betaori_hfsdst.py | 俄乌时空Hawkes σ |
| `Ukraine_Hawkes_Correlation.png` | RU_timespace_paper_but_betaori_hfsdst.py | 俄乌时空Hawkes 相关性 |
| `Ukraine_Hawkes_Poisson_Results.csv` | RU_timespace_paper_but_betaori_hfsdst.py | 俄乌时空Hawkes结果 |
| `Israel_Palestine_Global_Hawkes_Test.png` | time_space_all_PI.py | 巴以全局假设检验图 |
| `Ukraine_Global_Hawkes_Test.png` | time_space_all_RU.py | 俄乌全局假设检验图 |

## 注意事项

- 时空Hawkes模块需要GADM shapefile支持，请确保 `gadm_*` 文件夹完整
- 所有时空分析输出默认保存在 `./spatio-temporal_Hawkes/S-T_output/`，运行前请确保该目录存在
- 时空分析数据路径为 `../data/`，请确保数据文件位于正确位置
- 部分时空模型计算量较大，运行时间可能较长
- 运行时空分析模块前，请先 `cd spatio-temporal_Hawkes` 进入子目录