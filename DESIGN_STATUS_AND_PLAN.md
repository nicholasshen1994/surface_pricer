# Surface Pricer Design, Status, And Plan

## 1. 目标与设计原则

`surface_pricer` 是一套 standalone 的指数/ETF 单币种工具链，覆盖四个功能：

1. **数据抓取**：QuoteApi 快照 → 原始行情记录（`RawSnapshot`）；
2. **vol fit**：EDS SABR 曲面拟合，完全对齐 edslib CN（`ML.IDX.PRICING.MID`）口径；
3. **定价与风险**：欧式 vanilla 的 NPV 与 Greeks（含 bucketed）；
4. **真实合约估值**：手工条款表（JSON/CSV/Excel）→ 存续期与敲入敲出处理 → 组合报表。

设计原则：

1. **分层单向依赖**：`core → marketdata → fitting → pricing → portfolio`；`reporting` / `io` / `apps` 是最上层适配与入口，下层不 import 上层。
2. **standalone**：不 import edslib 主工程任何包；Jaeckel 反演经 `py_lets_be_rational`（MIT）。
3. **单次 from-scratch 拟合**：不做盘中重复 fit；`FitSettings.reference_surface` 可选启用 sticky-to-reference；手工覆盖（hard pin）见 §8。
4. 模块之间只通过明确的数据容器交互：`RawSnapshot → SliceData → FitResult / EDSSabrSurface / MarketState`。
5. 公共 API 由 `surface_pricer/__init__.py` 再导出，`from surface_pricer import ...` 不受目录调整影响。

## 2. 目录结构

```
surface_pricer/
├── __init__.py            # 公共 API 再导出
├── __main__.py            # python -m surface_pricer {fit|price|price-tool|build-ir-curve|build-borrow-curve}
├── api.py                 # 稳定的高层门面（price_vanilla / price_json ...）
├── price_tool.py          # 单合约查询：某次 fit run + strike/tenor/类型 → NPV + Greeks
├── fit_surface_snapshot.py        # [deprecated] → apps.fit_surface
├── core/                  # 基础设施（无业务依赖）
│   ├── calendars.py       # 中国日历 DateHelper
│   ├── daycount.py        # BusinessCalendar / year_fraction / add_tenor / shift_tenor
│   ├── curves.py          # ConstantRateCurve / PiecewiseRateCurve / forward
│   ├── ir_curve.py        # CNY FR007 曲线：QuantLib bootstrap（edslib 口径，仅生成阶段用 QL）
│   ├── borrow_curve.py    # 借券反解 + OU 尾部外推（纯 numpy，见 §11）
│   ├── market.py          # MarketState（spot / curves / surface / forward_overrides）
│   └── math/              # black.py / implied_vol.py / jaeckel.py
├── marketdata/            # ① 数据抓取
│   ├── data.py            # RawSnapshot / OptionQuoteRecord / SpotRecord
│   ├── providers.py       # MarketDataProvider 协议 + QuoteApiDataProvider
│   ├── registry.py        # UNDERLYING_SPECS / UnderlyingSpec（MO/IO/HO/ETF）
│   ├── gateway.py         # QuoteApi 底层客户端
│   ├── mapping.py         # 交易所后缀映射
│   ├── listed_contracts.py# CFFEX/ETF 合约解析 + 到期日规则
│   ├── future_inputs.py   # 期权到期日 → 股指期货快照映射
│   ├── forwards.py        # 期货优先 + 期权 parity 兜底合成远期
│   ├── rate_inputs.py     # 利率导出（FR007 + IRS）解析 → RateInputs
│   └── offline_quotes.py  # 离线报价容器
├── fitting/               # ② vol fit
│   ├── settings.py        # FitSettings / CN_PARAM_BOUNDS
│   ├── prepare.py         # 清洗 / parity forward / OTM IV / 权重 → SliceData
│   ├── engine.py          # EDSSabrFitter（调度 + 惩罚 + 优化 + 硬固定）
│   ├── pipeline.py        # fit_surface / FitResult
│   ├── overrides.py       # 手工覆盖与合成期限（见 §8）
│   ├── surface.py         # EDSSabrSurface
│   └── eds_slice.py       # EDSSabrSlice
├── pricing/               # ③ 定价与风险
│   ├── contracts.py       # VanillaContract（exo 合约后续加入）
│   ├── results.py         # PricingResult / RiskSettings
│   ├── vanilla.py         # VanillaPricer（Black-76 on forward + surface IV）
│   ├── greeks.py          # calculate_greeks（§6 口径）
│   └── exotics/           # 预留：ExoticPricer 协议 + register/get_pricer
├── portfolio/             # ④ 真实合约估值
│   ├── terms.py           # TradeTerms / ObservationRecord / load_terms（JSON/CSV/XLSX）
│   ├── schedule.py        # 存续期状态、观察日、KI/KO 推导与校验
│   ├── valuation.py       # 逐合约估值 + 组合汇总
│   └── report.py          # CSV / JSON 报表 + 文本摘要
├── reporting/             # fit_report.py / plots.py / quote_report.py（单合约渲染）
├── io/                    # serialization.py（JSON 适配）+ fit_runs.py（output 运行记录）
├── apps/                  # 入口层
│   ├── _common.py         # .env 加载 / progress 回调
│   ├── build_borrow_curve.py  # python -m surface_pricer build-borrow-curve（§11）
│   ├── build_ir_curve.py  # python -m surface_pricer build-ir-curve（§12）
│   ├── fit_surface.py     # python -m surface_pricer fit
│   └── price_trades.py    # python -m surface_pricer price
├── data/                  # 桌面导出的利率数据（interest_rate.csv）
└── tests/                 # 13 个测试文件（见 §9）
```

## 3. 数据流

```
QuoteApiDataProvider.load(underlying)                marketdata
        -> RawSnapshot (option records / spot / futures / rate / calendar)
fitting.prepare_slices(snapshot, settings, market)   fitting
        -> [SliceData] + forward_overrides
fitting.EDSSabrFitter(slices, settings).fit(market)  fitting
        -> EDSSabrSurface + [SliceFitResult]（含 [MANUAL]/[SYNTH] 标记）
fitting.pipeline.fit_surface(...) -> FitResult
        -> reporting.fit_report / reporting.plots / surface.json

io.market_from_surface(surface.json) -> MarketState  io + core
pricing.vanilla.VanillaPricer        -> NPV           pricing
pricing.greeks.calculate_greeks      -> NPV + Greeks  pricing
portfolio.load_terms + value_portfolio              portfolio
        -> trades.csv / trades.json（逐合约 + 组合汇总）
```

数据准备顺序严格对齐 edslib：

1. `remove_large_price_spread_by_mad`：call/put 分别剔除 `spread - median > 3 * mean|spread - median|` 的报价；
2. OTM mid 单调化（put 递增、call 递减，局部残差决定删点）；
3. parity synthetic forward：优先取 `|C_mid - P_mid|` 最小的公共 strike，`F = K + (C_mid - P_mid) / DF`；
4. OTM 选择：`K < F` 用 put、`K >= F` 用 call；
5. Jaeckel IV 反演得到 bid/mid/ask vol；
6. Jaeckel `Clamping Down on Arbitrage`：删除垂直/蝶式违规点；
7. 权重：`vega`（默认）、`atm_vega`、`equal`、`spread`，归一化。

## 4. 与 edslib CN 的拟合对齐

- **调度**：流动性 `coverage / median(spread)` + 依赖图 + Kahn 拓扑排序，逐片拟合并传入 pre/next slice。
- **惩罚项**：mid 残差 ×10；出 bid/ask 残差 ×100；calendar 方差单调惩罚（11 点网格、`nd1` 权重、`exp(x)-x-1` 平滑、因子 1.0）；pre-tenor 参数粘性（skew/conv 0.0005、skew1 0.0001）。
- **初值**：缺省零参数；loss NaN 或 > 0.5 则归零；再用 `(vol@1.05F - vol@0.95F)/vol@F` 估 skew、蝶式差分估 conv。
- **参数边界**（CN）：skew (-2, 2)、conv (0.001, 2)、L1/R1 (0, 10)、L2/R2 (0, 40)。
- **优化器**：L-BFGS-B（`max_iterations=500`、gtol/ftol=1e-9），失败切 SLSQP；收敛后 `|param| <= 1e-4` 归零。
- **输出**：每期限参数乘以 `max(0.3, sqrt(tau))` 写入 `EDSSabrSurface`，`stickiness_ratio = 0`。
- **时间口径**：`(交易日 + (日历日 - 交易日) * holiday_weight) / trading_days_per_year`，默认 243 与 0.05（对齐 `DateUtil.dtcf`）。

## 5. 定价

`VanillaPricer` 与 edslib 的 `AnalyticalVanilla` 同构：

1. `forward = market.forward(expiry)`（优先读 `forward_overrides`）；
2. `vol = surface.implied_vol(expiry, strike, current_forward, initial_forward=surface.init_spot)`：
   参数除以 `max(0.3, sqrt(tau))`、`ref_strike = init_fwd^ratio · fwd^(1-ratio)`（sticky 处理）；
3. `npv = notional × black_price(forward, strike, tau, vol, df, call_put)`。

## 6. Greeks 口径（对齐 edslib risk，`greeks/greeks.py`）

| Greek | bump 对象 | 差分 | 报告单位 | 备注 |
| --- | --- | --- | --- | --- |
| `delta` | spot（相对 ±1%） | 中心 | dNPV/dSpot | |
| `delta_cash` | 同上 | 中心 | `delta × spot` | edslib `Delta($)` |
| `delta_n` | 同上 | 中心 | `delta / spot` | edslib `Delta Shares` |
| `gamma` | spot（相对 ±1%） | 二阶中心 | d²NPV/dSpot² | edslib 展示 `%Gamma = dollar_gamma / 100` |
| `vega` | 曲面平行（±0.5 vol 点） | 中心 | 每 1 vol 点（/100） | |
| `volga` | 曲面平行（±0.5 vol 点） | 二阶中心 | 每 (1 vol 点)²（/10000） | |
| `vanna` | spot × vol 交叉 | 4 状态交叉 | 每 1 vol 点（/100） | |
| `theta` | 估值日 +1 天 | 前向 | 每日 NPV 变化 | 不含票息现金流 |
| `rho` | rate 曲线平行（±1bp） | 中心 | 每 1%（/100） | edslib `get_greek_scaling()=100` |
| `rhoq` | borrow 曲线平行（±1bp） | 中心 | 每 1%（/100） | |
| `bucketed_vega` | 逐 vol pillar | 中心 / 后向 | 每 1 vol 点 | `point_by_point`（默认）/ `cumulative_backward` |
| `bucketed_rhoq` | 逐 borrow pillar / tenor | 中心 | 每 1% | 常数曲线按 edslib 默认网格 1M…2Y |
| `bucketed_rho` | 逐 rate pillar | 中心 | 每 1% | 仅折线 curve 有多个 bucket |
| `bucketed_delta` | 由 `bucketed_rhoq` 折算 | — | cash | 现货 `delta_cash` 按借券敏感度占比分摊到各桶（`delta = rhoq × 100 / (−τ)`，有效换算时间 `τ = −100·Σrhoq / delta_cash`）；加总恒 = `delta_cash` |

`PricingResult.metadata["greek_convention"]` 会写出本次运行的实际口径。

## 7. CLI 用法

统一启动器（从仓库根运行）：

```powershell
cd C:\Code\edslib
python -m surface_pricer                                  # 帮助
python -m surface_pricer fit --underlying IO --index 000300.SH
python -m surface_pricer fit --pin "2027-06-18:atm_vol=0.215,skew=0.04" --extend-tenors
python -m surface_pricer price --terms trades.json --surface surface.json --rate 0.015
```

### 7.1 `fit`

- 默认参数集中在 `apps/fit_surface.py` 顶部：`DEFAULT_UNDERLYING / DEFAULT_INDEX / DEFAULT_RATE / DEFAULT_WEIGHT_MODE / DEFAULT_MAX_ITERATIONS / DEFAULT_REPORT_ROWS`。
- 密码读取：真实环境变量 > `--env-file` > `surface_pricer/.env` > 仓库根 `.env`（`QUOTE_GATEWAY_PASSWORD` / `CICC_QUOTE_PASSWORD`）。
- 输出目录：`surface_pricer/output/<underlying>_<YYYYmmdd_HHMMSS>/`，含 `report.txt`、`smile_*.png`、`term_structure.png`、`surface.json`、`manifest.json`、`overrides.json`。
- **fit 运行记录**：每次 fit 在 `surface_pricer/output/` 维护 `index.json`（历次运行清单，新→旧）与 `latest.json`（最近一次）；每期目录内的 `manifest.json` 记录 underlying / 估值日 / spot / rate / 期限列表 / 设置摘要 / fit 指标 / 文件清单。这套 JSON 由 `io/fit_runs.py` 统一读写，`price_tool` 通过它按名字选择某一次拟合结果；后续若改数据库只需替换该模块。
- 手工覆盖：`--pin` / `--override-file` / `--extend-tenors`（见 §8）。
- 支持的标的：

| Underlying | 类型 | 交易所 | 标的行情 | 期货映射 |
| --- | --- | --- | --- | --- |
| MO | 指数期权 | CFFEX | 000852.SH | IM |
| IO | 指数期权 | CFFEX | 000300.SH | IF |
| HO | 指数期权 | CFFEX | 000016.SH | IH |
| 510500 | ETF 期权 | 上交所 | 510500.SH | - |
| 588000 | ETF 期权 | 上交所 | 588000.SH | - |
| 159915 | ETF 期权 | 深交所 | 159915.SZ | - |

### 7.2 `price`（真实合约估值）

条款表（JSON / CSV / XLSX，`portfolio/terms.py`）：

```json
{"trades": [
  {"trade_id": "TRD-001", "underlying": "MO", "product_type": "vanilla",
   "booked_date": "2026-06-01", "start_date": "2026-06-01",
   "expiry_date": "2027-06-18", "call_put": "call", "strike": 7500.0,
   "strike_type": "absolute", "notional": 1000000,
   "ki_barrier": null, "ko_barrier": null, "ki_flag": false, "ko_flag": false,
   "coupon": null, "observations": [{"date": "2026-06-18", "spot": 7480.0}]}
]}
```

- CSV / Excel 用同名列表头；`observations` 单元格填 JSON 数组字符串；`.xlsx` 需 `openpyxl`（可选依赖）。
- 生命周期：`not_started`（估值日 < 起始日，不给值）/ `active`（正常定价）/ `expired`（估值日 ≥ 到期日，计 0 且排除）。
- 敲入敲出：同时给出障碍价与观察历史时按历史推导（`ko: spot >= ko_barrier`、`ki: spot <= ki_barrier`）并与表格 flag 交叉校验，不一致直接报错；只给 flag 时以表格为准。
- 输出：`trades.csv`（逐合约一行，含全部 Greek 列与 bucketed JSON 列）与 `trades.json`（明细 + 组合汇总 + bucket 汇总）。
- `--no-risk` 只定价不算 Greek；`--no-buckets` 跳过 bucketed vega / rhoQ / delta（加速）。

### 7.3 `price-tool`（单合约查询）

```powershell
python -m surface_pricer.price_tool --fit latest --strike 7500 --tenor 3M --call
python -m surface_pricer.price_tool --fit MO_20260928_150000 --strike 7500 --tenor 3M --put --json
python -m surface_pricer.price_tool --fit latest --strike 100 --strike-type percentage --tenor 6M --put --notional 1000000
python -m surface_pricer.price_tool --list-runs
python -m surface_pricer.price_tool --fit latest --interactive
```

- `--fit`：`latest`（默认）/ run 名（唯一前缀即可）/ run 目录 / `surface.json` 路径。
- 合约输入：`--strike` +（`--tenor` 3M/1Y/90D 或 `--expiry` YYYY-MM-DD）+ `--call` / `--put`（或 `--option-type`）；可选 `--notional`、`--strike-type`、`--spot/--rate/--borrow` 覆盖、`--no-buckets`（加速）、`--json`（机器可读）。
- tenor 经 `core.daycount.shift_tenor`（含交易日顺延）换算到期日，必须晚于估值日，否则报错。
- 输出：market 摘要（forward / df / implied vol）、`npv`、全部 10 个 Greek，以及 bucketed vega / rhoQ / delta。
- `--interactive` 进入循环输入：`<strike> <tenor|expiry> <call|put> [notional]`，空行或 `q` 退出。

### 7.4 旧命令（deprecated）

```powershell
python -m surface_pricer.fit_surface_snapshot      # → python -m surface_pricer fit
```

`fit_surface_snapshot.py` 保留为转发 shim，便于既有脚本平滑迁移；原独立 EOD 脚本 `update_option_surface_daily.py` 已从本包移除（该功能在其他代码库维护）。

## 8. 手工覆盖与期限延伸（hard-pinned pillars）

- `--pin "2027-06-18:atm_vol=0.215,skew=0.04"` 直接把手工值**硬固定**：被覆盖的维度从优化器变量中剔除，输出严格等于给定值（部分覆盖时其余维度照常拟合，全固定时跳过优化器）。
- 参数口径为 surface 层存储值（= `surface.json` 中的值），内部按 `max(0.3, sqrt(tau))` 换算到 slice 层并校验 `FitSettings.param_bounds`。
- 无报价的更远期限在 fit 之后用 `EDSSabrSurface.rebuild` 插值补出；`--extend-tenors` 额外按 edslib 规则（每年 6/12 月第三个周五，直到估值日 + 3Y）自动生成期限。
- 报告与 `metrics` 用 `[MANUAL]` / `[SYNTH]` 标记手工与合成期限。
- 与 edslib 的差异：edslib 是"手工 surface → 合成 vanilla（±2.5 vol 点，`source=manual`）→ 标准 fit"的软注入；本工具是硬固定，仅期限延伸规则与其一致（详见 `FIT_LOGIC.md` §11）。

## 9. 测试

```powershell
python -m pytest surface_pricer/tests -q      # 76 passed
```

| 测试文件 | 覆盖 |
| --- | --- |
| `test_eds_slice.py` / `test_jaeckel.py` | EDS slice 数值回归、Jaeckel 往返与 Brent 交叉验证 |
| `test_option_contracts.py` | CFFEX/ETF 合约解析与到期日 |
| `test_prepare.py` | MAD 清洗、parity / future forward、OTM 选择 |
| `test_fitter.py` | flat / skewed smile 的合成拟合 |
| `test_overrides.py` | 手工覆盖配置、合成期限规则、硬固定、无覆盖回归 |
| `test_surface_and_vanilla.py` | surface 插值与 vanilla 定价回归 |
| `test_greeks.py` | delta/gamma/vega/volga/vanna/rho/rhoq 与 Black-76 闭式解对照、bucketed 口径 |
| `test_portfolio.py` | 条款载入、存续期状态、KI/KO 校验、notional 线性、组合汇总、报表 |
| `test_apps.py` / `test_env_loader.py` | `price` 端到端落盘、启动器分发、`.env` 加载 |
| `test_price_tool.py` | fit run 记录（surface/manifest/index/latest）、run 解析（latest/名字/前缀/路径）、`price-tool` 查询与 JSON 输出 |

## 10. 已知限制与后续计划

1. **exo 定价预留**：`pricing/exotics` 已定义 `ExoticPricer` 协议与 `register_pricer` / `get_pricer`；`portfolio.valuation` 通过 `product_type` 分发，新增产品不需要改动 portfolio 与 CLI。本期支持 `vanilla`。
2. `bucketed_delta` 的有效换算时间 τ 由定价合约标定（`−100·Σrhoq / delta_cash`）：除远期通道（`−d lnF/dq`）外，vol/moneyness 通道与现货差分的步长口径都被吸收，保证各桶加总 = 报告的 `delta_cash`；edslib 的逐桶 dcf 口径在"桶=到期日 + ATM"时一致。现货 delta 默认 ±1% 相对 bump（edslib 惯例），OTM 时中心差分误差可达 ~2%（`RiskSettings.delta_bump_pct` 可调小）。
3. `theta` 为纯时间衰减（不含票息/现金流扣减）；exo 接入时再补现金流处理。
4. `.xlsx` 条款表需要 `openpyxl`；缺失时可用 CSV。
5. fit 结果目前以 JSON 存放在 `surface_pricer/output/`（`io/fit_runs.py` 统一读写）；接口已收口，后续切换数据库只改这一层，`fit` 与 `price-tool` 不动。
6. 原独立 EOD 脚本（`update_option_surface_daily.py`，依赖主工程 `dbfuncs` / `env`）已从本包移除，该功能在其他代码库维护；`surface_pricer` 保持 standalone，不再含任何主工程依赖入口。
7. bucketed rate / borrow 口径：平曲线先按 bucket 网格重建（edslib `rebuild_ql_curve_by_tenors`）再逐 pillar 单点 bump，本身带 pillar 的曲线（`build-borrow-curve` 产物）直接单点 bump；曲线折现对"起点落在估值日当天"的情形按日期口径处理（无日内 stub），否则近端 pillar 会经 `zero_rate(start)` 的左端外推产生伪敏感性。

## 11. 利率与借券曲线构建（build-ir-curve / build-borrow-curve）

CNY FR007 利率曲线与借券曲线是定价输入（`MarketState.rate_curve` / `borrow_curve`）。两条曲线由生成阶段命令产出，定价运行时只读 JSON（`PiecewiseRateCurve`），**不需要 QuantLib**。

### 11.1 利率曲线（`build-ir-curve`）

- 输入：`data/interest_rate.csv`（桌面导出，单位百分比）：`FR007.IR` 定盘 + `FR007S<tenor>.IR` 各期限 IRS par rate。
- 解析（`marketdata/rate_inputs.py`）：每列取**最新观测**、单位 /100、剔除"全历史恒定"的坏列（当前文件中的 2M=3.34 / 6Y=3.85 / 20Y=1.92 / 30Y=1.85）与空列（7D/14D），结果存 `RateInputs`（支持 JSON 往返）。
- 构建（`core/ir_curve.py`）：与 edslib `CNY-FR007` 同口径 —— `ql.SwapRateHelper`（季度 fixed、ACT/365F、BDC following、calendar `ql.China(ql.China.IB)`、浮动指数 `CNY-FR007-3M`（settlement 1、ACT/365F）、`Pillar.LastRelevantDate`、spread 0）+ FR007 1W fixing 注入 + `ql.PiecewiseLinearZero`（曲线 DCC ACT/360，开启外推）。
- 产物：`output/ir_curve.json`（pillar 日期 / 天数 / zero rate / par 复核），`max_par_residual` 自检（当前 2e-13）。
- QuantLib 只在这一步需要；定价侧用 `IRCurvePillars.to_piecewise_curve()` 得到线性 zero 插值的 `PiecewiseRateCurve`。

### 11.2 借券曲线（`build-borrow-curve`）

- 取数：行情网关快照（`QuoteApiDataProvider`）→ `RawSnapshot`。
- 远期（`marketdata/forwards.py`，照搬 `CNBorrowRateFitter`）：期货优先；缺期货的到期日用期权 put/call parity 兜底 `F = (C−P)/DF + K`（取 |C−P| 最小的 strike）。
- 反解（`core/borrow_curve.py`）：`q(T) = f(0,T) − ln(F/S)/dcf(0,T)`，`f` 为利率曲线的连续复利资金成本（`DF = exp(−f·dcf)`），时间刻度与 `PiecewiseRateCurve` 一致（ACT/365F）；剔除 3 天内到期（edslib `MIN_DAYS_TO_EXPIRY`）。
- 尾部外推（照搬 `OUBorrowCalibrator.extend_curve`）：`instantaneous_f0` + `integrate_ou_forward` + `zero_rate_at`，pillar 为**季度第三周五**（CFFEX 到期日规则，非交易日顺延），上限 `--horizon-years`（默认 3Y）。无历史标定时 `kappa` 取 edslib 对数先验中心 1.0、`mu` 取观测均值、`f0` 用最后两点瞬时远期并夹在 `q_anchor ± 0.02`。
- 产物：`output/borrow_curve.json`。

### 11.3 口径注意

1. 反解出的 `q` 是"期货贴水"隐含成本，含股息 + 实际借券。分红接口已预留（`cum_div_factors`，见 §11.4），当前未传 → `q` 偏高（含股息）。
2. 定价自洽性不受影响：用同一套 `(r, q)` 重建的 forward 与市场期货一致。
3. 利率导出（9/28）与行情快照（9/29）的估值日可能差一天，曲线在两者之间插值。
4. 若要让 OU 尾部用**历史标定**的 `kappa`（而非先验），需要历史借券序列输入（edslib 用 150 天定 tenor 的历史曲线 + ≥120 对转换样本）。

### 11.4 接进定价链路

`fit` 与 `price-tool` 都接受这两个曲线文件（`io/curve_files.py` 负责加载并校验文件类型）：

```powershell
python -m surface_pricer price-tool --fit latest --strike 7500 --tenor 3M `
    --ir-curve output/ir_curve.json --borrow-curve output/borrow_curve.json
python -m surface_pricer fit --underlying MO `
    --ir-curve output/ir_curve.json --borrow-curve output/borrow_curve.json
```

- 加载后替换原来的 `ConstantRateCurve`：`--ir-curve` 顶替 `--rate`，`--borrow-curve` 顶替 `--borrow`；
- `fit` 侧经 `QuoteApiDataProvider(rate_curve=..., borrow_curve=...)` 注入快照，`RawSnapshot.borrow_curve` 由 `build_market_state` 带进 `MarketState`；`manifest.json` 的 `extra` 会记录两个文件路径；
- 曲线锚点与 run 的估值日不一致时向 **stderr** 打 WARNING（不阻断，`--json` 输出保持干净）；
- `price-tool` 的 `QUICK_DEFAULTS` 也支持这两项（填文件路径即可让"直接运行"带上真实曲线）。

**分红（预留未启用）**：`build_borrow_curve(..., cum_div_factors={expiry: D})` 会把结果换成纯借券（`q_pure = q_total + ln(D)/dcf`），并在 pillar 上记录用到的 `D`；此时定价端需要在 forward 上乘回 `D`（`F = S·exp((r−q_pure)·t)·D`），这条链路尚未接入，等有分红数据再做。
