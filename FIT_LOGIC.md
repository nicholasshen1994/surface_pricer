# surface_pricer Fit 逻辑说明

> 目标：单次 from-scratch 拟合 EDS SABR surface，参数化、清洗、惩罚项、边界与时间口径对齐 edslib CN（`ML.IDX.PRICING.MID`）。
>
> 入口：`surface_pricer.fitting.pipeline.fit_surface(snapshot, settings, progress=None)`
> CLI：`python -m surface_pricer fit`（旧命令 `python -m surface_pricer.fit_surface_snapshot` 已随包根 shim 一起移除，或直接跑 `apps/fit_surface.py`）

---

## 0. 总览

```
QuoteApiDataProvider.load(underlying)        # 取数（唯一与外部数据源耦合的地方）
        │  RawSnapshot
        ▼
fit_prepare.prepare_slices(snapshot, settings, market)
        │  [SliceData] + forward_overrides
        ▼
fit_engine.EDSSabrFitter(slices, settings, progress).fit(market)
        │  EDSSabrSurface + [SliceFitResult]
        ▼
fitter.fit_surface(...) -> FitResult
        │
        ├─ surface_fit_report.format_fit_report()   # report.txt / stdout
        ├─ vol_fit_plot.plot_fit_result()           # smile_*.png + term_structure.png
        └─ EDSSabrSurface.to_dict()                 # surface.json
```

| 模块 | 职责 |
| --- | --- |
| `providers.py` | `MarketDataProvider` 协议 + QuoteApi 实现 + 标的注册表 |
| `index_option_contracts.py` | CFFEX / SSE / SZSE 期权合约解析与到期日 |
| `fit_prepare.py` | 报价清洗 → parity forward → OTM IV → 无套利 → 权重 → `SliceData` |
| `fit_engine.py` | 顺序拟合：调度、初值、惩罚、优化、surface 组装 |
| `fitter.py` | 薄入口，输出 `FitResult` |
| `jaeckel.py` | `py_lets_be_rational` 封装（IV 反演） |
| `eds_slice.py` | 纯 Python EDS SABR slice（参数化与 wing 常数与 edslib 一致） |
| `surface.py` | `EDSSabrSurface` 容器与插值 |
| `fit_settings.py` | `FitSettings` 全部可调参数 |

数据容器：

- `RawSnapshot`：`underlying / valuation_datetime / spot / option_records / spot_records / future_price_by_expiry / rate_curve / calendar / trading_days_per_year / holiday_weight / diagnostics`
- `OptionQuoteRecord`：`underlying / expiry / strike / option_type / bid / ask / last / volume / open_interest / raw_code / exchange`
- `SliceData`：`expiry / tau / forward / discount_factor / strikes / ln_moneyness / vols / bid_vols / ask_vols / weights / option_types / diagnostics`
- `FitResult`：`surface / slices / spot / valuation_datetime / underlying / forward_overrides / settings / market / diagnostics`

---

## 1. 取数层（providers）

### 1.1 标的注册表

| Underlying | 类型 | 交易所 | 请求方式 | 标的行情 | 期货映射 |
| --- | --- | --- | --- | --- | --- |
| MO | 指数期权 | CFFEX (`F`) | `get_option_snapshots(prefix="MO")` | 指数 `000852.SH` | IM |
| IO | 指数期权 | CFFEX (`F`) | 同上 | 指数 `000300.SH` | IF |
| HO | 指数期权 | CFFEX (`F`) | 同上 | 指数 `000016.SH` | IH |
| 510500 / 588000 | ETF 期权 | 上交所 (`0`) | `get_category_snapshots("O", "0")` + 前缀过滤 | ETF 现价 | 无 |
| 159915 | ETF 期权 | 深交所 (`1`) | 同上 | ETF 现价 | 无 |

- 指数标的：期权链 + 对应期货快照（`IM/IF/IH`）+ 指数点位；期货价只作为 **spot 参考与诊断**，默认不参与拟合（forward 由 parity 推出）。
- ETF 标的：交易所期权 category 全量快照按代码前缀过滤；forward 一律从 parity 推出。
- `valuation_datetime` 由快照的 `trading_day` + 最新 `time` 推导；利率曲线默认 `ConstantRateCurve(rate)`。

### 1.2 合约解析

- CFFEX 指数期权：`MO2610-C-7600.CFE` → underlying/年月/CP/行权价；到期日 = 到期月**第三个周五**（遇节假日顺延）。
- 沪深 ETF 期权：`510500C2610M06000` → 6 位标的 + C/P + YYMM + M/A + 5 位行权价（÷1000）；到期日 = 到期月**第四个周三**（顺延）。
- 统一输出 `ParsedListedOption`；无法解析或被过滤的记录计入 `diagnostics`。

---

## 2. 数据准备（fit_prepare）

对每个到期日依次执行：

### 2.1 期限窗口过滤

- 剩余日历日 ≤ 0 → 丢弃；
- 日历日 > `max_expiry_calendar_days`（默认 730 ≈ 2Y）→ 丢弃；
- 交易日 < `min_expiry_business_days`（默认 2B）→ 丢弃。

### 2.2 MAD 大价差过滤

call 与 put 分别独立处理：

1. 剔除 `bid ≤ 0`、`ask ≤ 0`、`ask < bid` 的报价；
2. `spread = ask − bid`，`m = median(spread)`，`mad = mean(|spread − m|)`；
3. 若 `mad ≤ 0` 不剔除，否则剔除 `spread − m > 3 × mad` 的报价（`spread_mad_factor=3`）。

### 2.3 OTM 价格单调化

- call mid 必须随 strike 不增，put mid 必须随 strike 不减；
- 首个违规点，比较左右候选点相对邻点线性插值的残差，删除残差更大的点，循环至无违规。

### 2.4 Parity synthetic forward

清洗后的 call/put 报价中：

1. 存在公共 strike：取 `argmin |C_mid − P_mid|` 的 strike，`F = K + (C_mid − P_mid) / DF`；
2. 无公共 strike（fallback）：取最近的 call/put 对，初值 `F0 = (K_c + K_p)/2 + (C_mid − P_mid)/DF`，在 `[0.5·F0, 2·F0]` 内用 `least_squares` 最小化 `(IV_call(F) − IV_put(F))`（τ=1 反演）；
3. `forward_source="future"` 时直接使用快照期货价，缺失时回退 parity。

forward 写入 `FitResult.forward_overrides`（按 ISO 日期），使 vanilla 定价与拟合一致。

### 2.5 OTM 选择与 IV 反演

- `K < F` 用 put，`K ≥ F` 用 call；
- `bid_vol = IV(bid/DF)`、`ask_vol = IV(ask/DF)`（Jaeckel，`py_lets_be_rational`）；
- 反演失败（价格越界/非有限）→ 丢弃该点；`ask_vol < bid_vol` 时交换；
- `mid_vol = (bid_vol + ask_vol)/2`。

### 2.6 Jaeckel 无套利清洗（Clamping Down on Arbitrage）

以 mid vol 计算无贴现 call/put 价格（`df = 1`），按顺序删除违规点（每次收集完再统一删除，保持数组对齐）：

1. **左翼**：`P_i/K_i ≥ P_{i+1}/K_{i+1}` → 删 `i`（i = 0..n−2）；
2. **右翼**：`C_i ≥ C_{i−1}` → 删 `i`（从右往左）；
3. **内部**（i = 1..n−2）：
   - `K_i < F` 且 `P_i ≥ P_{i+1}` → 删；
   - `K_i > F` 且 `C_i ≥ C_{i−1}` → 删；
   - 否则蝶式检查：`max(put_butterfly, call_butterfly) < 0` → 删，
     其中 `butterfly = V_{i−1}/(K_i−K_{i−1}) − V_i/(K_i−K_{i−1}) − V_i/(K_{i+1}−K_i) + V_{i+1}/(K_{i+1}−K_i)`。

剩余点数 < `min_strikes_per_expiry`（默认 4）→ 丢弃该期限。

### 2.7 权重（归一化前）

| mode | 公式 |
| --- | --- |
| `vega`（默认） | `σ_i = vol_i·√τ`；`d1 = −ln(K_i/F)/σ_i + 0.5σ_i`；`w_i = φ(d1)·F·√τ` |
| `atm_vega` | 同上，但 `σ = atm_vol·√τ`（`atm_vol` 取 fwd 右侧最近 strike 的 vol） |
| `equal` | `w_i = 1` |
| `spread` | `w_i = 1 / max(|ask_i − bid_i|, 1e−8)` |

统一 `w_i = max(w_i, 1e−12)`，再 `w /= Σw`。

---

## 3. vol time 口径

与 edslib `DateUtil.dtcf`（calendar 版）一致：

```
vol_time = [ busdays(start, end)                        # 含 start、不含 end
           + (calendar_days − busdays) × holiday_weight # 非交易日折算比例
           + tod(end) − tod(start) ]                    # 当日时间分数；端点非交易日时乘 holiday_weight
         / trading_days_per_year
```

默认 `trading_days_per_year = 243`、`holiday_weight = 0.05`（CN）。该口径同时用于拟合与 vanilla/risk 定价。

---

## 4. 拟合引擎（fit_engine）

### 4.1 流动性评分（仅用于调度）

对每个切片：

```
atm_vol   = interp(F, strikes, vols)
σ         = atm_vol · √τ
grid      = F × linspace(e^{−2σ}, e^{2σ}, 51)        # 50 个价格桶
coverage  = 非空桶数 / 50
spread    = median(ask_vols − bid_vols)（若为 0 取 1）
liquidity = coverage / spread
```

### 4.2 依赖图与拟合顺序

- 每片 i 向前、向后各找**第一个 liquidity 更大**的片 j，作为依赖；
- Kahn 拓扑排序得到拟合顺序（流动性好的先拟合）；
- 拟合某片时，把已拟合且成功的相邻依赖片作为 `pre_slice`（更早期限）与 `next_slice`（更晚期限），用于 calendar 与 tenor 约束。

### 4.3 每片初始值

```
x0 = reference_surface 参数（若提供）否则 0 向量（6 参数）

若 zero_initialization：x0 = 0
否则：loss(x0) 为 NaN/Inf 或 > 0.5 → x0 = 0

若 x0[0] == 0 或 x0[1] == 0：                    # RR / butterfly 估计
    idx95/100/105 = searchsorted(strikes, F × 0.95/1.0/1.05)（含边界保护）
    skew = (vol[105] − vol[95]) / vol[100]       # 无差异时取 0.1
    conv = (vol[95] − 2·vol[100] + vol[105]) / vol[100]  # 无差异时取 0.1

x0 = clip(x0, bounds)
```

### 4.4 目标函数（惩罚项）

```
P(x) = MID·Σ w_i · (σ_i^fit − σ_i^mkt)²
     + OUT·Σ w_i · (max(bid_i − σ_i^fit, 0) + max(σ_i^fit − ask_i, 0))²
     + calendar_pre + calendar_next
     + tenor_pre + tenor_next            # next 因子默认 0
     + reference_penalty                 # 仅当提供 reference_surface
```

- `MID = 10`、`OUT = 100`（`mid_vol_penalty_factor` / `out_of_bid_ask_penalty_factor`）；
- **calendar**：在 `ln(K/F) = σ_atm·√τ · linspace(−2, 2, 11)` 网格上（权重 `w_i = φ(d1_i)/Σφ(d1_i)`，`d1_i = −grid_i + 0.5σ_atm√τ`）：
  - 对 pre：`x_i = var_other − var_cur`；对 next：`x_i = var_cur − var_other`；
  - `penalty = factor × Σ w_i · smooth(x_i)`，`smooth(x) = 0 (x<0)，exp(x) − x − 1 (x≥0)`，`factor = 1.0`；
- **tenor（pre）**：`0.0005(skew_pre − skew)² + 0.0005(conv_pre − conv)² + 0.0001(l1_pre − l1)² + 0.0001(r1_pre − r1)²`；
- **reference**（可选）：`0.001(skew_ref − skew)² + 0.001(conv_ref − conv)² + 0.0001(l1_ref − l1)² + 0.0001(r1_ref − r1)²`。

### 4.5 参数边界与顺序

参数顺序：`(skew, conv, left_skew_1, left_skew_2, right_skew_1, right_skew_2)`

| 参数 | 下界 | 上界 |
| --- | --- | --- |
| skew | −2 | 2 |
| conv | 0.001 | 2 |
| left / right skew1 | 0 | 10 |
| left / right skew2 | 0 | 40 |

（edslib CN 值，不随时间缩放。）

### 4.6 优化器

1. 依次尝试 `L-BFGS-B` → `SLSQP`（`optimizer_method_priority`）；
2. 选项：`maxiter = max_iterations`（默认 **500**）、`ftol = 1e-9`、L-BFGS-B 另加 `gtol = 1e-9`；
3. 返回 `success=True` 且参数有限才接受；
4. 收敛后 `|param| ≤ 1e-4` 的参数置 0；
5. 全部方法失败 → 该期限拟合失败，从 surface 中剔除（进度里标记 FAILED）。

### 4.7 surface 组装

对每片成功结果：

```
scale_i     = max(0.3, √τ_i)                       # param_scaling_floor
surface 参数 = raw_param × scale_i                 # 与 edslib 输出口径一致
atm_vols    = interp(F, strikes, vols)
stickiness_ratio = 0                               # sticky to moneyness
```

构造 `EDSSabrSurface(init_date, init_spot, expiry_dates, atm_vols, skews, convs, left_skews_1, left_skews_2, right_skews_1, right_skews_2, calendar, trading_days_per_year, holiday_weight)`。

### 4.8 每片结果（SliceFitResult）

`params（raw）/ fitted（EDSSabrSlice）/ atm_vol / rmse / weighted_rmse / out_of_bid_ask / optimizer_method / iterations / function_evaluations / elapsed_seconds / is_override / is_synthetic / override_source`

`is_override=True` 表示该期限有手工固定值（`optimizer_method="pinned"` 为全固定）；`is_synthetic=True` 表示该期限是 fit 之后追加的人工期限（无市场报价，见 §11）。

---

## 5. 输出层

- **报告** `format_fit_report`：
  `Expiry | Fwd | Tau | ATM | skew | conv | L1 | L2 | R1 | R2 | RMSE | WRMSE | N | OutBA`
  （参数为缩放后的 surface 值；`--report-rows N` 可附每期报价明细）
- **图表** `plot_fit_result`：每期限一张 `smile_<expiry>.png`（bid/ask 散点 + fitted 曲线），`term_structure.png`（ATM、skew/conv、wing 参数）。
- **JSON**：`surface.json`（`EDSSabrSurface.to_dict()`），可直接用于 vanilla/risk。
- **进度**：`fitter`/`engine` 通过 `progress` 回调输出阶段日志与逐片日志（CLI 默认打印，`--quiet` 关闭）。

---

## 6. FitSettings 速查

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `trading_days_per_year` | 243 | vol time 分母 |
| `holiday_weight` | 0.05 | 非交易日折算比例 |
| `min_expiry_business_days` | 2 | 最短期限（交易日） |
| `max_expiry_calendar_days` | 730 | 最长期限（日历日） |
| `spread_mad_factor` | 3.0 | MAD 大价差阈值 |
| `min_strikes_per_expiry` | 4 | 每期最少有效报价点 |
| `forward_source` | `parity` | `parity` / `future` |
| `weight_mode` | `vega` | `vega` / `atm_vega` / `equal` / `spread` |
| `max_iterations` | 500 | 优化器迭代上限 |
| `gradient_tolerance` / `function_tolerance` | 1e-9 | 收敛容差 |
| `optimizer_method_priority` | `L-BFGS-B, SLSQP` | 依次尝试 |
| `param_zero_threshold` | 1e-4 | 小幅参数归零 |
| `zero_initialization` | False | 强制零初值 |
| `mid_vol_penalty_factor` | 10.0 | mid 残差权重 |
| `out_of_bid_ask_penalty_factor` | 100.0 | 出盘口权重 |
| `calendar_penalty_factor` | 1.0 | calendar 惩罚系数 |
| `arb_check_points` / `arb_check_std_range` | 11 / 2.0 | calendar 网格 |
| `sticky_to_pre_tenor_*` | 0.0005 / 0.0005 / 0.0001 / 0.0001 | 相邻期限粘性（skew/conv/skew1） |
| `sticky_to_next_tenor_*` | 0 | 后向粘性（edslib CN 为 0） |
| `reference_surface` | None | 非空时启用 sticky-to-reference |
| `sticky_to_reference_*` | 0.001 / 0.001 / 0.0001 / 0.0001 | 参考面粘性 |
| `param_bounds` | CN 边界（见 4.5） | 参数边界 |
| `param_scaling_floor` | 0.3 | 输出缩放下限 |
| `override_config` | None | 手工覆盖 + 合成期限配置（见 §11）；None 时与历史版本完全一致 |

---

## 7. CLI

```powershell
cd C:\Code\edslib
python -m surface_pricer fit                    # 用 apps/fit_surface.py 顶部 DEFAULT_* 配置
python -m surface_pricer fit --underlying IO --index 000300.SH
python -m surface_pricer fit --maxiter 200 --quiet
# 手工固定若干期限后重新拟合，并延伸远月期限（详见 §11）
python -m surface_pricer fit --pin "2027-06-18:atm_vol=0.215,skew=0.04"
python -m surface_pricer fit --override-file overrides.json --extend-tenors
```

- 默认参数集中在 `apps/fit_surface.py` 顶部：`DEFAULT_UNDERLYING / DEFAULT_INDEX / DEFAULT_RATE / DEFAULT_WEIGHT_MODE / DEFAULT_MAX_ITERATIONS / DEFAULT_REPORT_ROWS`。其中 **`DEFAULT_RATE` 只在 `--ir-curve none` 时生效**（2026-10）：默认 `--ir-curve latest` 下，拟合的折扣因子与 parity 远期都取自 rate curve（`MarketState.discount_factor` → `rate_curve`），manifest 里记的 `rate` 是**该曲线在 3M 处的 zero**（`flat_rate()`），供之后 `--ir-curve none` 的报价回退使用。
- 密码读取：真实环境变量 > `--env-file` > `surface_pricer/.env` > 仓库根 `.env`（`QUOTE_GATEWAY_PASSWORD` / `CICC_QUOTE_PASSWORD`）。
- 输出目录：`surface_pricer/output/<underlying>_<YYYYmmdd_HHMMSS>/`。
- 手工覆盖与期限延伸：`--pin "EXPIRY:FIELD=VALUE[,...]"`（可重复）、`--override-file PATH`、`--extend-tenors` / `--no-extend-tenors`、`--synthetic-end-tenor` / `--synthetic-months` / `--synthetic-week`；生效配置落盘为 `overrides.json`（见 §11）。

---

## 8. 性能特征

- `EDSSabrSlice` 的网格积分与概率项已向量化：单次构造约 **0.07s**（初版约 0.5s，提速 ~7 倍，数值回归测试不变）。
- 单期限拟合（`maxiter=500`）约 **5~30s**，六期限几分钟内完成。
- 进度日志示例：

```
[1/6] fitting 2026-10-16 | 15 quotes | fwd=7537.384 | tau=0.0841
[1/6] 2026-10-16 done | rmse=0.00002 | out-of-ba=0 | L-BFGS-B nit=13 nfev=112 | 5.8s
```

---

## 9. 与 edslib 的对应关系

**对齐项**：MAD 清洗、单调化、parity forward、OTM 选择、Jaeckel IV、Jaeckel 无套利、vega 权重、流动性排序 + 依赖图、mid×10 / 出盘口×100、calendar 惩罚、pre-tenor 粘性、参数边界、L-BFGS-B→SLSQP、`|p|≤1e-4` 归零、`max(0.3,√τ)` 缩放、vol time 口径、CN 默认设置（243/0.05、vega、maxiter=500）。

**有意差异**：

1. 单次 from-scratch：`sticky_to_pre_param` 默认关闭，仅在传入 `reference_surface` 时启用（edslib 常态是以上一版参数为初值做增量拟合）；
2. 不含 edslib 的 borrow 曲线 bootstrap / 期货 borrow 混合 / `VolRegulator` 后处理（这些属于 edslib 参数管理链路，不属于 standalone 拟合）；
3. 不实现 `GLOBAL` 拟合模式与 `staged fitting`（edslib CN 默认均为关闭）；
4. `SKEW_2_SHORT_TENOR_THRESHOLD`（4M）在 edslib 中仅作用于 GLOBAL 模式与 `zero_param_*_skew_2=True` 的市场，CN 顺序拟合不触发，故未移植；
5. edslib 的 `_find_valid_strike_range`（VIX 式区间）是死代码，未移植。
6. 手工调整 refit：edslib 走"手工 surface → 合成 vanilla（±2.5 vol 点盘口，`source=manual`）→ 标准 fit"的软注入，并在出版阶段用 `mix_vol_surface(attrs=[])`（只补新期限，不覆盖已有期限）与 `mix_manual_vol`（只覆盖超出拟合最长期限的期限）做期限混合；本工具改为硬固定 + 曲面延伸（见 §11），仅复刻 edslib 的"6/12 月第三个周五"合成期限规则。

---

## 10. 已知限制与后续

1. ETF 合约解析按公开规则实现（6 位标的 + C/P + YYMM + M/A + 5 位行权价）；若网关返回 8 位合约编号，只需扩展 `providers.py` 的解析分支。
2. ETF 期权链来自交易所 category 全量快照 + 前缀过滤，请求量较大；若网关支持按标的条件请求可进一步收敛。
3. `sticky_to_next_tenor_*` 因子默认 0（对齐 edslib CN），如需后向平滑直接调大。
4. 后续扩展：期权 pricer（雪球/ACC 等）、离线 JSON provider、calendar/borrow 曲线联动。

---

## 11. 手工覆盖与期限延伸

把自己判断更准的 ATM / smile 参数以**硬固定**方式注入下一次拟合，并可选择把曲面延伸到场内没有报价的更远期限（近月市场数据无法改善远月时的常规补救手段）。

实现集中在 `surface_pricer/overrides.py`，引擎侧改动在 `fit_engine.py`（固定维度剔除）、`fitter.py`（fit 后延伸与编排）、`surface.py`（`pillar_index` / `set_pillar`）。

### 11.1 语义：硬固定（不是软注入）

- 被覆盖的**字段**从优化变量中删除：`_fit_one_slice` 只把未固定的维度交给 `minimize`（`expand()` 在每次目标函数求值前把固定值填回完整参数向量）。输出曲面对应位置**严格等于**给定值，不受 `param_zero_threshold` 归零、惩罚项与相邻期限粘性影响。
- 只覆盖部分字段时，同一期限其余字段照常拟合；**全部字段被覆盖时完全跳过优化器**（`optimizer_method="pinned"`、`iterations=0`、`function_evaluations=0`）。
- 被固定期限照常写入 `fitted`，因此仍作为已完成期限参与相邻期限的 `calendar_penalty` 与 `sticky_to_pre/next_tenor_*` 惩罚，相邻期限不会失去参照。
- ATM 说明：优化变量只有 6 个 smile 参数，ATM 由市场中间价在 forward 处线性插值（`_slice_atm_vol`）；覆盖 ATM 即改写该返回值，同时重建以 ATM 为输入的套利检查网格。

### 11.2 参数口径

对外（JSON / CLI / `metrics` / 报告）统一使用 **surface 层存储值**，即 `EDSSabrSurface.to_dict()` 与报告里的同一口径（已乘 `max(0.3, sqrt(tau))`）。内部换算为 slice 层（除以同一 scale，`overrides.surface_to_slice_values`）后交给优化器，并按 `FitSettings.param_bounds`（CN：skew(-2,2)、conv(0.001,2)、L1/R1(0,10)、L2/R2(0,40)）校验换算后的值，越界抛出带期限与允许范围的 `ValueError`。

### 11.3 新增期限与自动合成期限

场内无报价的期限无法进入 `prepare_slices`（会被期限窗口与 `min_strikes_per_expiry` 过滤），因此在 `EDSSabrFitter.fit` 之后处理：

1. `EDSSabrSurface.rebuild(原 pillar ∪ 新增 pillar)`：ATM 用总方差插值、6 个 smile 参数按 vol time 线性插值；**原有 pillar 数值保持不变**（插值在自己的 pivot 上精确）。
2. 再把用户显式给定的字段写回新增 pillar（未给出的字段保留插值值）。
3. 开启 `extend_synthetic_tenors` 时按 edslib 规则追加期限：**每年 6 月与 12 月的第三个周五**（遇非交易日按 Modified Following 顺延），一直延伸到 `估值日 + synthetic_end_tenor`（默认 `3Y`）所在年份的年底，且只保留比现有最长期限更远的日期。

新增期限在 `FitResult.slices` 中生成一条空报价记录（`is_synthetic=True`、`rmse=0`、`optimizer_method="synthetic"`、`slice_info.strikes` 为空），报告标注 `[SYNTH]`；被手工固定的期限标注 `[MANUAL]`；`metrics` 输出 `fixed` / `synthetic` 开关（0/1）便于程序化核对。

### 11.4 配置文件与 CLI

```json
{
  "overrides": [
    {"expiry": "2027-06-18", "atm_vol": 0.215, "skew": 0.04},
    {"expiry": "18M", "left_skew_1": 0.5}
  ],
  "extend_synthetic_tenors": true,
  "synthetic_months": [6, 12],
  "synthetic_week": 3,
  "synthetic_end_tenor": "3Y"
}
```

```powershell
# 直接固定一个期限（ATM + skew），其余期限照常拟合
python -m surface_pricer fit --pin "2027-06-18:atm_vol=0.215,skew=0.04"

# 读配置文件，并延伸到更远的期限
python -m surface_pricer fit --override-file overrides.json --extend-tenors

# 只覆盖、不延伸
python -m surface_pricer fit --override-file overrides.json --no-extend-tenors
```

- 期限写法：ISO 日期（`2027-06-18`）或 tenor（`18M` / `3Y` / `90D`）；解析后按交易日历顺延到下一个交易日。与场内已有期限匹配时视为"固定已有期限"，否则视为"新增期限"。
- `--pin` 可重复；`--override-file` 与 `--pin` 合并，出现重复期限直接报错。
- 未传任何覆盖参数时 `FitSettings.override_config=None`，代码路径与数值结果与历史版本完全一致（由 `tests/test_overrides.py::test_empty_override_config_matches_default_run` 保护）。
- 生效配置落盘到输出目录的 `overrides.json`，便于复核与复现。

### 11.5 与 edslib 手工调整机制的差异

edslib 是**软注入**，且分两段：

| 环节 | edslib 做法 | 本工具 |
| --- | --- | --- |
| 手工值进入拟合 | `marked_surface_vanilla_generator.generate_vanillas_from_marked_surface` / `SurfaceToVanBuilder` 把手工 surface 变成带 `spread=0.05`（±2.5 vol 点）盘口的 synthetic vanilla，以 `VolSourceType.MANUAL` 写入 `DBFitVolTradedVanilla`；`import_vanilla_options` 优先读 DB（有数据的 ticker 不再取实时快照），缺失的才回落到快照 | 直接从参数固定，不经过任何报价层；手工值不会被权重/出盘口惩罚/相邻期限粘性拉动 |
| 已有期限 | 依赖上述合成报价参与拟合，本身不强制等于手工值 | 该期限的固定字段严格等于给定值 |
| 新增/超长期限 | `VolRegulator.shift_tenor` 对齐期限日期；`mix_vol_surface(attrs=[])` 只插入新期限、不覆盖已有期限；`UnifiedVolBorrowRegulator.mix_manual_vol` 只覆盖 `> 拟合最长期限` 的手工期限 | fit 后 `rebuild` 插值填充 + 手工值写回（等价于 `mix_manual_vol` 的思路），并按 edslib 的 6/12 月第三个周五规则自动生成期限 |

一句话：edslib 让手工意见"和市场数据一起竞价"，本工具让手工意见"直接说了算"，两者在期限延伸规则上保持一致。

