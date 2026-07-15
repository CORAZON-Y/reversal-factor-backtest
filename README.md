# 题目介绍
单因子检验-根据研报完成以下任务。

1.载入数据（基础和市值行业等数据，只载入相关数据提高效率）

2.计算因子和收益率（因子使用数据截至t0日收盘价，因子为（t0日收盘/t-5日收盘）-1，(收益率t2日开盘/t1日开盘)-1）

3.计算收益率时剔除t1日涨跌停和停牌的股票，剔除st股票，剔除上市60天内的股票

4.合并数据（索引：时间，股票代码）

5.因子值标准化（zscore方案和rank方案）

6.计算因子中性化后的值（后续操作分别对这两种因子做一遍）

a)原始因子值

b)对市值行业中性化

7.计算因子IC值

8.因子IC值序列t检验（计算IC均值、标准差等）
9.分层回测（等权或流通市值加权，收益率累加计算，同时计算各类回测评价指标）
a)每个截面上对因子值排序分n层，计算每层收益并累加
b)对每个行业做a操作再累加（可选）
10.画出cumulative sum IC和分层回测的图
11.基于预测出的收益率，通过二次规划对因子值最小的100个股票进行个股上下限和行业约束

# A 股因子检验与组合优化项目

本项目分为两部分：

1. 第一问：单因子构造、IC 检验、分层回测。
2. 第二题：基于因子的收益率预测，并使用二次规划做组合优化回测。

项目默认使用虚拟环境中的 Python 运行：

```bash
.venv/bin/python
```

## 目录结构

```text
.
├── basic_data/              原始 parquet 数据
├── data/                    兼容旧入口和简单数据查看脚本
├── factor_analysis/         第一问：因子检验与分层回测代码
├── 第二题/                  第二题：收益率预测与组合优化代码
├── output/                  第一问运行结果
├── README.md                项目总说明
├── 项目说明.md              原始题目说明
└── 数据说明.md              原始数据字段说明
```

## 环境与依赖

当前项目依赖主要包括：

```text
pandas
numpy
scipy
matplotlib
pyarrow
```

安装依赖：

```bash
pip install -r requirements.txt
```

仓库中已有 `.venv/` 时，直接使用：

```bash
.venv/bin/python -m factor_analysis --help
```

如果能看到命令帮助，说明环境可用。

## 第一问：因子检验与分层回测

代码目录：

```text
factor_analysis/
```

### 因子定义

当前项目使用的是原始 5 日涨跌幅的负号：

```text
factor = -1 * (t0 日收盘价 / t-5 日收盘价 - 1)
```

取负号的原因是：原始 5 日涨跌幅在当前数据中表现为反向因子。取负号后，IC 和多空组合方向更符合“因子越大，未来收益越高”的解释。

未来收益率定义：

```text
return = t2 日开盘价 / t1 日开盘价 - 1
```

在计算收益率时，项目会过滤 t1 日不可交易股票：

```text
涨跌停
停牌
ST
上市未满 60 天
价格或收益率缺失
```

### 第一问模块说明

```text
factor_analysis/config.py          运行参数
factor_analysis/constants.py       文件名、字段名、状态常量
factor_analysis/data_loader.py     读取 daily、industry、ST、停牌数据
factor_analysis/dataset.py         构造因子、收益率、过滤样本
factor_analysis/factors.py         去极值、z-score、rank z-score
factor_analysis/neutralization.py  市值和行业中性化
factor_analysis/ic.py              IC、RankIC、t 检验
factor_analysis/backtest.py        分层回测
factor_analysis/plotting.py        作图
factor_analysis/outputs.py         保存结果
factor_analysis/pipeline.py        主流程编排
factor_analysis/cli.py             命令行入口
```

### 运行第一问

完整运行：

```bash
.venv/bin/python -m factor_analysis
```

只计算因子表，不跑 IC 和回测：

```bash
.venv/bin/python -m factor_analysis --factors-only
```

开启行业内分层回测：

```bash
.venv/bin/python -m factor_analysis --industry-layer
```

只做等权回测：

```bash
.venv/bin/python -m factor_analysis --weight equal
```

改成 10 分层：

```bash
.venv/bin/python -m factor_analysis --groups 10
```

兼容旧入口：

```bash
.venv/bin/python data/data_process.py
```

### 第一问输出

默认输出目录：

```text
output/
```

主要输出文件：

```text
processed_data.parquet              处理后的因子样本表
ic_series.csv                       每日 IC / RankIC
ic_summary.csv                      IC 汇总统计和 t 检验
group_returns.csv                   每日分层收益
group_cumulative_returns.csv        每日分层累计收益
backtest_summary.csv                分层回测指标
cumulative_ic.png                   累计 IC 图
cumulative_rank_ic.png              累计 RankIC 图
group_backtest_*.png                分层回测图
```

如果运行了 `--industry-layer`，结果表中会出现：

```text
scope = industry_layer
```

同时会生成类似：

```text
group_backtest_factor_zscore_industry_layer_equal.png
group_backtest_factor_zscore_industry_layer_float_mv.png
```

注意：当前“行业内分层”含义是每个行业内部先分层，再把所有行业同一层合并计算收益；不是每个行业单独输出一套回测。

## 第二题：收益率预测

代码位置：

```text
第二题/expected_return.py
```

本模块计算组合优化需要的股票预期收益 `mu`。

当前采用一因子预测模型，使用第一问中未经过行业中性化的 `factor_zscore`：

```text
expected_return_i,T+1 = beta_hat_T * factor_zscore_i,T
```

其中 `beta_hat_T` 来自历史截面回归系数的滚动平均：

```text
realized_return_i,t = beta_t * factor_zscore_i,t + error_i,t
beta_hat_T = mean(beta_{T-window}, ..., beta_{T-1})
```

这里使用 `shift(1)`，避免在预测 T 日时使用 T 日已经发生的未来收益。

### 运行收益率预测

先确保第一问已经生成：

```text
output/processed_data.parquet
```

然后运行：

```bash
.venv/bin/python 第二题/expected_return.py
```

默认参数：

```text
factor_col   = factor_zscore
window       = 60
min_periods  = 20
```

输出：

```text
第二题/expected_returns.parquet
第二题/expected_factor_returns.csv
```

`expected_returns.parquet` 中的核心字段：

```text
date
code
factor_zscore
beta_hat
expected_return
realized_return
```

其中 `expected_return` 就是后续组合优化使用的 `mu`。

## 第二题：组合优化回测

代码位置：

```text
第二题/portfolio_optimization.py
```

每日流程：

```text
1. 按 factor_zscore 从高到低选出前 100 只股票
2. 使用候选股票过去 60 天真实收益估计协方差矩阵
3. 对协方差矩阵做 shrinkage 稳定化
4. 在约束条件下最大化组合预期收益
5. 用真实 realized_return 计算当期组合收益
```

优化模型：

```text
maximize    mu' w
subject to  w' Sigma w <= max_variance
            sum(w) = 1
            0 <= w_i <= 0.05
            industry_weight_j <= 0.20
```

其中：

```text
mu       股票预期收益
w        股票权重
Sigma    候选股票协方差矩阵
```

### 运行组合优化

完整运行：

```bash
.venv/bin/python 第二题/portfolio_optimization.py
```

先试跑 5 个交易日：

```bash
.venv/bin/python 第二题/portfolio_optimization.py --start-date 2010-03-01 --max-dates 5 --output-dir 第二题/optimized_portfolio_test
```

调整组合方差上限：

```bash
.venv/bin/python 第二题/portfolio_optimization.py --max-variance 0.0003
```

默认输出目录：

```text
第二题/optimized_portfolio/
```

主要输出：

```text
portfolio_returns.csv       每日组合收益
portfolio_weights.parquet   每日持仓权重
portfolio_summary.csv       组合回测指标
optimization_report.csv     每日优化状态
```

## 推荐运行顺序

从头到尾完整复现：

```bash
# 1. 第一问：因子检验和分层回测
.venv/bin/python -m factor_analysis --industry-layer

# 2. 第二题：收益率预测
.venv/bin/python 第二题/expected_return.py

# 3. 第二题：组合优化回测
.venv/bin/python 第二题/portfolio_optimization.py
```

如果只想先确认代码能跑通，可以使用小样本：

```bash
.venv/bin/python 第二题/portfolio_optimization.py --start-date 2010-03-01 --max-dates 5 --output-dir 第二题/optimized_portfolio_test
```

## 提交说明

运行结果文件通常不需要提交。

`.gitignore` 已忽略：

```text
output/
第二题/expected_returns.parquet
第二题/expected_factor_returns.csv
第二题/optimized_portfolio/
第二题/optimized_portfolio_test/
.venv/
__pycache__/
```

如果要手动清理运行结果，可以删除：

```bash
rm -rf output
rm -f 第二题/expected_returns.parquet 第二题/expected_factor_returns.csv
rm -rf 第二题/optimized_portfolio 第二题/optimized_portfolio_test
```

## 常见修改位置

修改因子公式：

```text
factor_analysis/dataset.py
```

修改因子标准化：

```text
factor_analysis/factors.py
```

修改中性化逻辑：

```text
factor_analysis/neutralization.py
```

修改 IC 统计：

```text
factor_analysis/ic.py
```

修改分层回测：

```text
factor_analysis/backtest.py
```

修改收益率预测模型：

```text
第二题/expected_return.py
```

修改组合优化约束：

```text
第二题/portfolio_optimization.py
```
