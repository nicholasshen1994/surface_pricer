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
├── __main__.py            # python -m surface_pricer {fit|price|price-tool|autocall|build-json|price-json|slide|build-ir-curve|build-borrow-curve}
├── config/                # barrier_shift.json：障碍位移定价规则的统一标准（§12）
├── api.py                 # 稳定的高层门面（price_vanilla / price_json ...）
├── core/                  # 基础设施（无业务依赖）
│   ├── calendars.py       # 中国日历 DateHelper
│   ├── daycount.py        # BusinessCalendar / year_fraction / add_tenor / shift_tenor / month_grid
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
│   ├── option_contracts.py# 挂牌期权合约表（Wind → data/option_contracts.json 的本地缓存 + 快照 join）
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
├── pricing/               # ③ 定价与风险（按功能分层，见 §2.1）
│   ├── results.py         # PricingResult / RiskSettings（共享容器）
│   ├── risk/              # 风险层：bumps.py（市场 bump 原语）+ diff.py（bump-and-revalue 驱动）+ slide.py（现价梯子，§14）
│   ├── models/            # 曲面派生的模型系数：localvol.py（Dupire，两引擎共用，§12）
│   ├── numerics/          # 与产品无关的数值原语：fdm.py（三对角 / 对数网格 / θ 格式）
│   ├── rules/             # 定价规则层：barrier_shift.py（配置 + 覆盖优先级 + 展开，§12）
│   ├── vanilla/           # 产品：欧式香草（contract.py / spec.py / pricer.py / greeks.py，§13.2）
│   └── exotics/           # ExoticPricer 协议 + 注册表（__init__.py）
│       └── autocall/      # 产品：雪球 —— 引擎在产品包内
│           ├── contract.py   # AutocallContract（原始条款）
│           ├── schedule.py   # AutocallSchedule + build_schedule + from_dict（位移唯一施加点：expand_shift）
│           ├── grid.py       # TimeGrid / build_time_grid / smooth_indicator
│           ├── cashflows.py  # coupon_ratio / ko_cash_flow / expiry_cash_flow
│           ├── mc.py         # Monte Carlo 引擎（公共随机数 + 障碍平滑，§12）
│           └── pde.py        # 1D 有限差分引擎（敲入双状态，§12）
├── portfolio/             # ④ 真实合约估值
│   ├── terms.py           # TradeTerms / ObservationRecord / load_terms（JSON/CSV/XLSX）
│   ├── schedule.py        # 存续期状态、观察日、KI/KO 推导与校验
│   ├── valuation.py       # 逐合约估值 + 组合汇总
│   └── report.py          # CSV / JSON 报表 + 文本摘要
├── reporting/             # fit_report.py / plots.py / quote_report.py / autocall_report.py / slide_report.py（§14）
├── io/                    # serialization.py（JSON 适配）+ fit_runs.py（output 运行记录）
├── apps/                  # 入口层（全部 CLI 集中在此）
│   ├── _common.py         # .env 加载 / progress 回调
│   ├── _market.py         # fit run → MarketState 的共享装配（price-json / slide / build-json 共用）
│   ├── build_borrow_curve.py  # python -m surface_pricer build-borrow-curve（§11）
│   ├── build_ir_curve.py  # python -m surface_pricer build-ir-curve（§11）
│   ├── build_json.py      # python -m surface_pricer build-json（只生成载荷 JSON，定价交给 price-json，§15）
│   ├── fit_surface.py     # python -m surface_pricer fit
│   ├── price_json.py      # python -m surface_pricer price-json / slide（喂 JSON 出 NPV/希腊值、现价梯子；QUICK_DEFAULTS 在其文件顶部，§7.6、§14）
│   └── autocall_pricer.py # python -m surface_pricer autocall-pricer（反解票息：给定目标 NPV 求平坦年化票息，条款同 build_json、输出可定价载荷）
├── data/                  # 桌面导出的利率数据（interest_rate.csv）
└── tests/                 # 23 个测试文件（见 §9）
```

### 2.1 `pricing/` 分层约定（2026-09 重排）

原来的 `pricing/` 根目录把**产品**（香草条款 / 引擎 / 风险）、**共享设施**（结果容器）、**雪球专用**（局部波动率 / 有限差分 / 位移规则）混在同一层，`exotics/` 里产品定义与两个引擎也同层并列；下一个产品（barrier / accumulator）无处安放。现按功能切开，依赖只向下：

| 层 | 目录 | 职责 | 不允许出现 |
| --- | --- | --- | --- |
| 容器 | `pricing/results.py` | `PricingResult` / `RiskSettings` | 任何产品逻辑 |
| 风险 | `pricing/risk/` | 市场 bump 原语（`bumps.py`）+ 通用差分驱动（`diff.py`） | 产品条款 |
| 模型 | `pricing/models/` | 由曲面派生的系数（`localvol.py`） | 产品 / 引擎 |
| 数值 | `pricing/numerics/` | 三对角、对数网格、θ 格式 | 产品语义 |
| 规则 | `pricing/rules/` | 定价规则（`barrier_shift.py`），在引擎之前展开 | 引擎调用 |
| 产品 | `pricing/vanilla/`、`pricing/exotics/<product>/` | 条款 + 生效条款 + 现金流 + 引擎 | 引用另一个产品 |

- 一个产品包**自带引擎**（`exotics/autocall/mc.py`、`pde.py`），对外只暴露 `build_schedule` / 现金流 / pricer 工厂；`exotics/__init__.py` 仍是唯一的插件协议与注册表（`get_pricer("autocallable")` 的路径不变，`portfolio/valuation.py` 无需改动）。
- `exotics/autocall` 的对外导入路径保持不变（包 `__init__` 再导出），所以 `from surface_pricer.pricing.exotics.autocall import AutocallContract` 这类写法不受拆分影响。
- 迁移映射（旧 → 新）：`contracts.py → vanilla/contract.py`、`vanilla.py → vanilla/pricer.py`、`greeks.py → vanilla/greeks.py` + `risk/bumps.py`、`shift.py → rules/barrier_shift.py`、`localvol.py → models/localvol.py`、`fdm.py → numerics/fdm.py`、`exotics/bump.py → risk/diff.py`、`exotics/autocall.py → exotics/autocall/{contract,schedule,grid,cashflows}.py`、`exotics/{mc,pde}.py → exotics/autocall/{mc,pde}.py`、`price_tool.py → apps/price_vanilla.py`。旧路径**不留任何转发模块**（包根的 `price_tool.py` 薄壳已删除），全部引用一次性改到位；`python -m surface_pricer price-tool` 是唯一入口，IDE Run 直接跑 `apps/price_vanilla.py`（自举守卫见 §7.5）。

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
pricing.vanilla.pricer.VanillaPricer -> NPV           pricing
pricing.vanilla.greeks.calculate_greeks -> NPV + Greeks pricing
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

条款（`strike` + `strike_type`）先由 `resolve_spec()` 折成 **`VanillaSpec`** —— 绝对行权价、绝对到期日，加上面这组市场映射 —— 引擎与希腊值都只读它。`strike_type` 因此只解析一次：spot bump 量的是**固定行权价**的敏感度，不会拿百分比重新定一次行权价（否则 delta 里会混入"行权价跟着现货走"的假项）。载荷字段释义见 §13.2。

## 6. Greeks 口径（对齐 edslib risk，`greeks/greeks.py`）

| Greek | bump 对象 | 差分 | 报告单位 | 备注 |
| --- | --- | --- | --- | --- |
| `delta` | spot（相对 ±1%，`RiskSettings.delta_bump_pct`） | 中心 `(NPV(S+h) − NPV(S−h)) / (2h)`，`h = spot × 1%` | dNPV/dSpot | |
| `delta_cash` | 同上（共用同一对 bump） | 同上 | `delta × spot`，等价 `(NPV(S+h) − NPV(S−h)) / (2 × 1%)` | edslib `Delta($)`；= dNPV/d(ln S)，即"现货相对变动 100% 时 NPV 变化多少"；除以 notional 就是"名义倍数" |
| `delta_n` | 同上 | 同上 | `delta / spot` | edslib `Delta Shares`；标的价格每涨 1 个点，NPV 变化多少 |
| `gamma` | spot（相对 ±1%） | 二阶中心 | d²NPV/dSpot² | edslib 展示 `%Gamma = dollar_gamma / 100` |
| `gamma_cash` | spot（相对 ±1%） | 二阶中心（与 `gamma` 同一对 bump） | `gamma × spot² / 100` | 即"每 1% 现货变动的 NPV 变化"，与 `delta_cash` 并排读；雪球与香草同口径 |
| `vega` | 曲面平行（±0.5 vol 点） | 中心 | 每 1 vol 点（/100） | |
| `volga` | 曲面平行（±0.5 vol 点） | 二阶中心 | 每 (1 vol 点)²（/10000） | 雪球与香草同口径（2026-10）；**复用 vega 的 vol 状态**，与 `vega` 同价；不进 `all` |
| `vanna` | spot × vol 交叉 | 4 状态交叉 | 每 1 vol 点（/100） | 雪球与香草同口径（2026-10）；**额外 4 次估值**，用的 σ± 曲面与 vega/volga 共用（不额外建表）；不进 `all` |
| `theta` | 估值日 +1 天 | 前向 | 每日 NPV 变化 | 不含票息现金流 |
| `rho` | rate 曲线平行（±1bp） | 中心 | 每 1%（/100） | edslib `get_greek_scaling()=100` |
| `rhoq` | borrow 曲线平行（±1bp） | 中心 | 每 1%（/100） | |
| `bucketed_vega` | 逐 vol pillar | 中心 / 后向 | 每 1 vol 点 | `point_by_point`（默认）/ `cumulative_backward` |
| `bucketed_rhoq` | 逐 borrow pillar / tenor | 中心 | 每 1% | 常数曲线按 edslib 默认网格 1M…2Y |
| `bucketed_rho` | 逐 rate pillar | 中心 | 每 1% | 仅折线 curve 有多个 bucket |
| `bucketed_delta` | 由 `bucketed_rhoq` 折算 | — | cash | 现货 `delta_cash` 按借券敏感度占比分摊到各桶（`delta = rhoq × 100 / (−τ)`，有效换算时间 `τ = −100·Σrhoq / delta_cash`）；加总恒 = `delta_cash` |

`PricingResult.metadata["greek_convention"]` 会写出本次运行的实际口径。

- **`delta_cash` / `gamma_cash` 与差分步长的关系**（`risk/diff.py`）：`delta_cash = (NPV_up − NPV_down) / (2 × delta_bump_pct)`、`gamma_cash = (NPV_up − 2·NPV + NPV_down) / delta_bump_pct² / 100`。也就是说它们是"相对步长归一化"的数，**与 spot 的绝对水平无关**（同一笔合约在 5,000 与 9,000 上，同样的斜率给同样的 `delta_cash`）。
- **注意 1% 是一扇窗**：默认中心差分把 **±1% 内的结构平均掉**。数字型跳变（贴着 KO/KI 的窄带）、真实存在的尖峰，在 1% 步长下都会被摊平；要看障碍边的真实斜率，要么在库内把 `RiskSettings.delta_bump_pct` 收到 0.1%（`delta_cash` 的定义不变），要么在障碍两侧加密档位。雪球梯子上"KO 边 delta 跳变"的宽度就是这条的直接后果（§12.6 的实测）。
- **敲出（已结算）合约的希腊值为 0**：`greeks_schedule` 短路，只折现敲出现金；`PricingResult.zero_greeks(selection)` 把本次请求的名字显式置 0（详见 §12.3 第 9 条）。

## 7. CLI 用法

统一启动器（从仓库根运行）：

```powershell
cd C:\Code\edslib
python -m surface_pricer                                  # 帮助
python -m surface_pricer fit --underlying IO --index 000300.SH
python -m surface_pricer fit --pin "2027-06-18:atm_vol=0.215,skew=0.04" --extend-tenors
python -m surface_pricer build-json --product autocall --output-name contract.json   # §15
python -m surface_pricer price-json contract.json --greeks delta,gamma_cash         # §7.6
python -m surface_pricer build-json --product autocall --output-name snowball.json   # 只生成载荷（§15）
```

### 7.1 `fit`

- 默认参数集中在 `apps/fit_surface.py` 顶部：`DEFAULT_UNDERLYING / DEFAULT_INDEX / DEFAULT_RATE / DEFAULT_WEIGHT_MODE / DEFAULT_MAX_ITERATIONS / DEFAULT_REPORT_ROWS`。其中 **`DEFAULT_RATE` 只在 `--ir-curve none` 时生效**（2026-10）：默认 `--ir-curve latest` 下，拟合的折扣因子与 parity 远期都取自 rate curve（`MarketState.discount_factor` → `rate_curve`），manifest 里记的 `rate` 是**该曲线在 3M 处的 zero**（`flat_rate()`），供之后 `--ir-curve none` 的报价回退使用 —— 不再是程序默认的 0.015。
- 密码读取：真实环境变量 > `--env-file` > `surface_pricer/.env` > 仓库根 `.env`（`QUOTE_GATEWAY_PASSWORD` / `CICC_QUOTE_PASSWORD`）。
- 输出目录：`surface_pricer/output/vol_fit/<指数代码>/<指数代码>_<YYYYmmdd_HHMMSS>/`（`--underlying MO --index 000852.SH` ⇒ `vol_fit/000852/000852_20261008_101541/`；没配 `--index` 时退化为期权代码），含 `report.txt`、`smile_*.png`、`term_structure.png`、`surface.json`、`manifest.json`、`overrides.json`。
- **输出根与三个 run 文件夹（2026-10）**：`--output-dir`（fit）与 `--output-root`（price-json / build-json）指的是**同一个输出根**（默认 `surface_pricer/output`），三类产物各占一个子文件夹、互不覆盖，也都不再覆盖历史：

  | 文件夹 | 内容 | 写入方 | 读取方（默认值） |
  | --- | --- | --- | --- |
  | `vol_fit/<指数>/` | 每一个 run 目录 + **该指数的** `index.json` / `latest.json` | `fit` | `--fit latest`（`io/fit_runs.py`） |
  | `ir_curve/` | `ir_curve_<stamp>.json` + `index.json` + `latest.json`（全体共用一条曲线，不分指数） | `build-ir-curve` | `--ir-curve latest`（`io/curve_runs.py`） |
  | `borrow_curve/<指数>/` | `borrow_curve_<stamp>.json` + `index.json` + `latest.json` | `build-borrow-curve`（按其 `--underlying` 对应的指数归档） | `--borrow-curve latest`（取当前标的所属指数的那一个文件夹） |
  | `local_vol/<指数>/` | `lv_<hash>.json`：Dupire 系数 + **生成它的输入**与时间 | `price-json` / `slide`（首次报价时） | 输入相同 ⇒ 直接读表（`io/local_vol_cache.py`） |

  **先指数文件夹，再明细**（2026-10）：`vol_fit/000852/` 里是 000852 的每一个 run 与它自己的 `index.json` / `latest.json`，`borrow_curve/510500/` 里是 510500 的每一条 borrow —— 指数写在路径上（文件名不用背），于是"比较 / 清理某个指数的历史"就是对一个文件夹操作。改造前直接放在 `vol_fit/` / `borrow_curve/` 根下的 run 仍然可读（回退到根指针，含早期的单一名字与 `{"000852": …}` 映射两种形态）；再往下新增一律写进指数文件夹。

  `local_vol/` 是**局部波动率表缓存**（2026-10）：**按指数分档**（`local_vol/000852/lv_<hash>.json`）—— 表是某个指数曲面的 Dupire、用该指数的融券曲线贴现，所以两个指数不共用一个目录、更不共用一个文件；文件名是输入的 sha1 前 16 位，文件里存着曲面 / 两条曲线 / 时间网格 / 锚点 / 指数 / 离散化与 `created_at`，所以"这份表还能不能用"打开文件就能判断；`--no-local-vol-cache`（或块里 `local_vol_cache=False`）则每次重建。**四项输入 ir / borrow / vol / spot 任一改变都是另一个 key（重建）**：vol 取 run 的 `surface.json` 内容哈希，ir 与 borrow 取解析后的曲线文件（名字 + 内容哈希）——borrow 本身按指数分档，所以换指数就是换文件。**现货锚（spot）既进 key 也写进文件**（`inputs.spot_anchor` 与 `table.spot_anchor`）：系数按对数 moneyness 相对锚点表达，锚不同就是另一个模型 —— 读回时文件层先校验表自身的锚与 key 一致（不一致/缺失即视为未命中并就地重写），引擎层再校验表锚与本次市场的锚一致（`stale` 计数并重建），因此换 spot（`--spot`、梯子以外的场景）不会误用旧表；同一风险运行 / 梯子内锚点是钉住的（基准 spot），仍然只用一张表。

- **fit 运行记录**：每次 fit 在 `vol_fit/` 维护 `index.json`（历次运行清单，新→旧）与 `latest.json`（**按指数分档**：`{"000852": "000852_20261008_153937", "000300": "..."}`，2026-10 起 —— 两个指数就是两张曲面，`--fit latest` 必须知道问的是哪一个，否则 000852 的报价会落到刚 fit 完的 000300 曲面上；旧的单指针 `{"run": ...}` 仍可读，映射到该 run 自己的指数）。`price-json` 用**载荷的 `underlying`** 去解析（`build-json` 用条款里的标的换算成指数代码），命中同指数的 run；该指数没有指针时只在**同指数**的历次 run 里取最新（绝不跨指数）；完全没有 ⇒ **报错**并列出已知的指数（原先的"文件夹里最新那个"是全局的，等于悄悄换了曲面）；显式点名 run 时若其标的与载荷不同则打 stderr WARNING；每期目录内的 `manifest.json` 记录 **指数标的**（`underlying`）/ 估值日 / spot / rate / 期限列表 / 设置摘要 / fit 指标 / 文件清单，外加**期权来源** `option_underlying`（如 `MO`）与 `index_underlying`。run 用**指数**命名与标识，是因为定价、障碍、行情都活在指数空间，而同一指数可以来自不同期权产地（000905 用 510500 的期权拟合）——`FitRun.describe()` 显示为 `000852.SH (MO)`（§12.2 的类别匹配按裸代码，两种写法都能命中）。这套 JSON 由 `io/fit_runs.py` 统一读写，`price_tool` 通过它按名字选择某一次拟合结果；后续若改数据库只需替换该模块。
- 手工覆盖：`--pin` / `--override-file` / `--extend-tenors`（见 §8）。
- **挂牌期权合约表（2026-10）**：SSE/SZSE 的 ETF 期权在行情网关里**只有交易所的数字合约 ID**（`10012493.SH`，`resp_stk_code`），没有行权价 / 到期 / 认购认沽 —— 这正是 510500 一直 fit 不出来的原因。`data/option_contracts.json`（**单文件、全量**）由 `python -m surface_pricer fetch-contracts --underlying 510500[,MO]` 从 Wind `WINDDF.CHINAOPTIONDESCRIPTION` 拉取（键 = 合约的 Wind 码 = 快照的 `resp_stk_code` + 交易所后缀，`S_INFO_SCCODE` = venue 的聚合码 `510500OP.SH` / `MO.CFE`）；**两条 provider 路径（ETF 与 CFFEX）都先查它**，查不到才回退到代码解析。定价路径**从不连数据库**：只有出现新合约时才跑一次 fetch，且合并是**增量**的（已有的行原样保留、`updated_at` 记录时间）。凭据放 `surface_pricer/.env`（`WIND_DB_*`，git-ignored）；该服务器老于 python-oracledb thin 模式支持范围，脚本会自动加载 Instant Client（`ORACLE_CLIENT_LIB_DIR` 可指定）。实测：510500 全量 1,790 条 + MO 3,260 条，ETF 链 126 条报价全部解析成功（修复前 0 条）。
- 支持的标的：

| Underlying | 类型 | 交易所 | 标的行情 | 期货映射 |
| --- | --- | --- | --- | --- |
| MO | 指数期权 | CFFEX | 000852.SH | IM |
| IO | 指数期权 | CFFEX | 000300.SH | IF |
| HO | 指数期权 | CFFEX | 000016.SH | IH |
| 510500 | ETF 期权 | 上交所 | 510500.SH | - |
| 588000 | ETF 期权 | 上交所 | 588000.SH | - |
| 159915 | ETF 期权 | 深交所 | 159915.SZ | - |

### 7.2 `price`（**已移除**，2026-10）

组合台账估值（`--terms JSON/CSV/XLSX` → 逐笔 + 组合汇总 + bucket 汇总）整条链已删除：`apps/price_trades.py`、`portfolio/` 包（terms/schedule/valuation/report）与 `tests/test_portfolio.py`。生产流程改为**载荷级**：`build-json` 出每份合约的载荷 → `price-json` / `slide` 做单份估值与现价梯子（批量循环由外层脚本或定时任务做，见 §7.6）。下面保留该 CLI 的条款表格式与口径说明，作为**历史参照**。

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

### 7.3 `price-tool`（**已移除**，2026-10）

香草报价改走 `build-json --product vanilla`（出 `vanilla_spec` 载荷）+ `price-json`（定价、选希腊值）——**载荷是唯一接口**，`price-json` 不做条款解析（§7.6、§13.2）。原名下的旋钮现在由 `price_json.QUICK_DEFAULTS`（报价侧的希腊值/引擎/市场覆盖）与 `build_json.QUICK_DEFAULTS["vanilla"]`（条款）承载；`--interactive` 取消。下面保留的是该 CLI 曾经的旗标与口径，作为**语义参照**。

```powershell
python -m surface_pricer price-tool --fit latest --strike 7500 --tenor 3M --call
python -m surface_pricer price-tool --fit MO_20260928_150000 --strike 7500 --tenor 3M --put --json
python -m surface_pricer price-tool --fit latest --strike 100 --strike-type percentage --tenor 6M --put --notional 1000000
python -m surface_pricer price-tool --fit latest --strike 7500 --tenor 3M --spec-out spec.json   # 导出已解析合约
python -m surface_pricer price-tool --fit latest --spec spec.json                              # 改完喂回来（§13.2）
python -m surface_pricer price-tool --list-runs
python -m surface_pricer price-tool --fit latest --interactive
```

- `--fit`：`latest`（默认）/ run 名（唯一前缀即可）/ run 目录 / `surface.json` 路径。
- 合约输入：`--strike` +（`--tenor` 3M/1Y/90D 或 `--expiry` YYYY-MM-DD）+ `--call` / `--put`（或 `--option-type`）；可选 `--notional`、`--strike-type`、`--spot/--rate/--borrow` 覆盖、`--no-buckets`（加速）、`--json`（机器可读）。
- JSON 层：`--spec-out [FILE]` 导出 `vanilla_spec` 载荷（**省略文件名 = `test.json`**，`-` 打印），`--spec FILE`（`-` 读 stdin）直接定价；带 `--spec` 时条款 flag 不再是输入（`--strike/--tenor` 可以省略），字段释义见 §13.2。写文件后在 stderr 打印一行 `note: wrote …`，不会静默成功。
- tenor 经 `core.daycount.shift_tenor`（含交易日顺延）换算到期日，必须晚于估值日，否则报错。
- 输出：先**回显已解析载荷**（`vanilla_spec`，§13.2；长数组不省略），再打 market 摘要（forward / df / implied vol）、`npv`、全部 Greek（含 `gamma_cash`），以及 bucketed vega / rhoQ / delta。
- `--greeks LIST` 收窄风险跑：默认 `all`（平行希腊值 + 分桶，向后兼容），`none` 只做一次解析定价、不做任何 bump；`volga` / `vanna` 是二阶交叉差分，**雪球与香草都有**（2026-10；两边都需显式点名、不进 `all`，见 §13.3 与 §12.6 的成本）。
- `--interactive` 进入循环输入：`<strike> <tenor|expiry> <call|put> [notional]`，空行或 `q` 退出（循环内不回显载荷，避免刷屏）。

### 7.4 旧命令（已移除）

包根不再保留任何转发 shim：`python -m surface_pricer.fit_surface_snapshot` 与 `surface_pricer/price_tool.py` 都已删除，统一走启动器 `python -m surface_pricer fit` / `price-json`（或 IDE 直跑 `apps/<name>.py`，见 §7.6）。原独立 EOD 脚本 `update_option_surface_daily.py` 也已从本包移除（该功能在其他代码库维护）。2026-10 起 `price_autocall` / `price_vanilla` 两个 app 也一并移除（见 §7.3、§7.5）。

### 7.5 `autocall`（**已移除**，2026-10）

雪球走同一条链：`build-json`（块 `QUICK_DEFAULTS["autocall"]`，§15）出 `autocall_schedule` 载荷 → `price-json` / `slide` 定价（quick block 在 `price_json.py` 顶部，§7.6）。原 CLI 独有的三件事：`--compare`（MC vs PDE 并排）取消 —— 需要时两行库调用（同一份 schedule 分别喂 `AutocallPDE` / `AutocallMonteCarlo`）；`--interactive` 取消；"条款直接报价 + `--spec-out`"由 `build-json` + `price-json` 覆盖（载荷落盘 = `build-json` 的输出）。历史回放仍是库函数 `apply_history`（§12.3 第 4 条）：**台账日期在生成载荷前回放好**（`knocked_in_date` / `knocked_out_at` 写进载荷），引擎只读这两个日期并把敲入日与估值日比较派生成状态。下面保留该 CLI 的旗标与语义说明，作为**语义参照**。

```powershell
python -m surface_pricer autocall --fit latest --tenor 1Y --coupon 0.20
python -m surface_pricer autocall --fit latest --tenor 1Y --method pde --json
python -m surface_pricer autocall --fit latest --tenor 1Y --compare        # MC vs PDE 并列
python -m surface_pricer autocall --fit latest --tenor 1Y --no-shift       # 原始条款、只定价
python -m surface_pricer autocall --fit latest --tenor 1Y --coupon 0.10,0.11,0.12,0.13 --rebate 0.05
python -m surface_pricer autocall --fit latest --tenor 1Y --ki-frequency expiry --ki-strike 0.9
python -m surface_pricer autocall --fit latest --tenor 1Y --greeks delta,gamma,gamma_cash
python -m surface_pricer autocall --fit latest --tenor 1Y --greeks vega,bucketed_vega  # + 分桶
python -m surface_pricer autocall --fit latest --tenor 1Y --day-count act/act  # 票息计息基准
python -m surface_pricer autocall --fit latest --tenor 1Y --spec-out spec.json  # 导出已解析合约
python -m surface_pricer autocall --fit latest --tenor 1Y --spec-out          # 省略文件名 -> test.json
python -m surface_pricer autocall --spec spec.json                        # 改完再喂回来（§12.7）
# 存续期估值：计息仍从 --start（期初）算整段，过去发生的事情要"说清楚"而不是猜
python -m surface_pricer autocall --start 2025-10-05 --tenor 1Y --valuation-date 2026-01-05 \
    --start-spot 7330.29 --assume-alive --greeks delta,gamma_cash
python -m surface_pricer autocall --start 2025-10-05 --tenor 1Y --valuation-date 2026-01-05 \
    --start-spot 7330.29 --history fixings.json --ki-frequency observation_dates
python -m surface_pricer autocall --list-runs                              # 可用 fit run
python -m surface_pricer autocall --interactive                            # 逐行试价
```

- 条款：`--underlying/--start/--tenor|--expiry/--obs-freq M|Q|S|A|--obs-dates/--ko/--ki/--ki-frequency daily|expiry|observation_dates/--ki-strike/--coupon/--rebate/--day-count/--notional/--protection/--ki-gearing/--settlement-days`。周期网格里落在节假日/周末的观察日**顺延到下一个工作日**（与 `build-json` 同一条规则，§15；显式 `--obs-dates` 原样使用）。`--coupon` 支持**每个观察日一个**（逗号分隔的 step-up / step-down，如 `0.10,0.11,0.12`，数量等于观察日个数）；`--rebate` 是"不敲出也不敲入"腿的年化票息（默认取最后一个观察日的票息），与敲出票息可以不同；`--ki-frequency expiry` 是欧式敲入（只在到期日观察，常规 ACN，edslib 的 `DateFrequency.AT_EXPIRY`）；`--ki-strike` 给敲入亏损腿换一个行权价（比率，默认 1.0 = 从起始价算盈亏，< 1 即 OTM 雪球）。边界口径另给 `--ko-boundary` / `--ki-boundary`（`inclusive`（默认，碰到即触发）/ `exclusive`（严格穿过），§12.3）。
- 票息计息：`--coupon/--rebate` 是**年化**率，实际收取 `1 + 年化率 × accrual(--start → 观察日)`。`--day-count act/365f`（默认，自然日/365）/ `act/360` / `act/act`（ISDA，跨年按各自年长 365/366）三者可选，敲出与 rebate 两条腿同基准（§13.1）。**计息起点永远是 `--start`（期初），不是估值日**：存续期估值照样按整段计息，theta（估值日 +1D）不会因此多算或少算一天。
- 存续期估值（`--start` 早于估值日）：构建合约时**不再直接拒绝**，但要求把"已经发生过什么"说清楚——`--start-spot`（障碍锚必须是期初价）+ 下面二者之一：`--history fixings.json`（`{"2025-11-05": 90.0, ...}`，按过去观察日逐个回放，缺一天就报错）或 `--assume-alive`（声明过去无敲出/敲入）。也可以直接喂已解析载荷（`--spec`，其 `knocked_in_before` / `knocked_out_at` 就是账本状态）。三者都不给 ⇒ 报错退出，绝不默认"什么都没发生"。
- 引擎：`--method mc|pde`、`--paths/--greek-paths/--seed/--mc-steps/--pde-nodes/--pde-steps/--pde-theta/--theta-days`。`--paths` 同时决定定价与风险跑的路径数（否则调 `--paths` 会在算希腊值时静默失效），要区分时显式给 `--greek-paths`。PDE 的时间离散是**按观察窗口固定细分**（`--pde-steps`，默认 6 个等 vol-time 子步），原 `--pde-time-step`（全局步长上限）已删除：它只在窗口宽于 ~1.2 vol-time 年时才生效，与该参数重复（§12.6）。
- 位移（定价规则）：`--shift-config`（默认包内 `config/barrier_shift.json`）、`--no-shift`、`--ko-shift/--ki-shift`（标量或 `mode:value`）。
- 边界口径（合约条款，与位移无关）：`--ko-boundary` / `--ki-boundary`（`inclusive`（默认）= 碰到即触发、`exclusive` = 严格穿过），会随 `--spec-out` 写进载荷的 `knock_out.boundary` / `knock_in.boundary`（§12.3、§13.1）。
- 当日判定口径（估值输入，**不**写进载荷）：`--trigger-basis contractual|effective` —— `contractual`（默认）用真实条款线判今天（EOD、入账口径），`effective` 用位移后生效线判（盘中口径：spot 在两条线之间时仍算未敲入，希腊值停在模型状态、不用马上对冲；§12.3 第 8 条）。
- 输出：文本模式只打**文本报表**（原始条款 / 位移来源 / **生效绝对障碍** / NPV / 希腊值 / 方法元信息）——已解析载荷**不再回显到控制台**，要看/要留请用 `--spec-out`（`autocall_schedule`，§13.1；`daily` 这类规则型的监控网格不写进载荷，只有 `custom` 网格可能很长，`--spec-out` 始终拿全量）；机器输出走 `--json`（不打报表，载荷在 `contract` 块里）。`--greeks LIST` 选择要计算的希腊值：`all` / `none`（或省略）/ `delta / delta_cash / delta_n / gamma / gamma_cash / vega / theta / rho / rhoq`，**默认 `none`（只给 NPV，不做任何 bump）**；名字拼错直接报错退出，不会静默少算；`volga / vanna`（**2026-10 起雪球引擎也有**，与香草同 stencil、同报告单位，见上表；不进 `all`）：`volga` 复用 vega 的那对 ±0.5 vol 点状态（**+0 次估值、+0 张局部波动率表**），`vanna` 再加 4 次交叉估值（σ± 的曲面与 vega/volga 共用，仍不额外建表）。只跑一个 bump 对的组合最省：`delta`（含 `delta_cash/delta_n`）与 `gamma`（含 `gamma_cash`）共用同一对 spot bump，`vega/volga` 共用同一对 vol bump，`rho/rhoq/theta` 各自一对。**分桶希腊值单独列名**：`bucketed_vega / bucketed_delta / bucketed_rhoq / bucketed_rho`（别名 `buckets` = 四个全算），口径与香草一致（§12.6），且**不进 `all`** —— 每个桶都是一对估值，其中 vol 桶还会重建局部波动率表，代价见 §12.6 的成本说明。
- `--list-runs` 列出已存 fit run；`--interactive` 进入 `autocall>` 循环：`<tenor|expiry> [ko] [ki] [coupon] [obs_freq] [method]`，每行输出单行摘要（生效障碍 + NPV + 标准误 + 被选中的希腊值），市场只解析一次；`--greeks` 选择对整个循环生效。
- **无参运行（quick block）**：随 CLI 移除，条块的落点改为 `build_json.QUICK_DEFAULTS["autocall"]`（条款 → 载荷，§15）与 `price_json.QUICK_DEFAULTS`（报价侧全部参数，§7.6）。两处的键名都必须是 argparse 的 **dest（下划线）**，写错在启动时直接报错（`_apply_quick_defaults` 校验键名）；相对**输出**路径按**包目录**解析、不按 cwd（`quick_output_path`，IDE Run 的工作目录谁也猜不到），写文件后在 stderr 打一行绝对路径 `note: wrote …`。
- **反解票息**：`python -m surface_pricer autocall-pricer`（`apps/autocall_pricer.py`）。条款与 `build_json.QUICK_DEFAULTS["autocall"]` 同款（`--terms FILE` 可逐键覆盖；块里写 `coupon`/`rebate` 或拼错键名都会被拒），`notional` 默认 **1**、`start_spot` 默认取**估值现货**（起息日默认估值日），给 `--target`（占名义本金比例，如 `0.95`）后用 bracket + 二分反解**票息阶梯里被标成未知的那一段**。票息沿用 `build_json` 的阶梯词汇 `{名义期数: 年化利率}`（标量 = 全体同率、列表 = 每期一个；期数从起息日按月数，**`guaranteed_period` 不改变编号**），把要解的那一段写成 `null`：2Y 月度「前 12 期 10%、解后 12 期」= `{1: 0.10, 13: null}`，反过来「解前段、后段固定 10%」= `{1: null, 13: 0.10}`；只允许一个 `null`（一次二分只解一个利率，标两个直接报错），一个都没有则"没得解"（也报错）。未知段从标记的期数起一直到下一个键（或最后一个期数），`--coupon-min/max/tol` 都是**这一段**的利率；整段落在锁定期内（一个观察日都不覆盖）也报错 —— 那种票息不影响价格，二分无解。报告同时给名义期数与它实际影响的观察日（`periods 5-24 (of 24, 3 locked up) | moves 2027-03-09 -> 2028-10-09`）。**rebate 默认跟随最后一个期数的票息**：解尾段时它跟着走，解前段时它留在固定尾段上。`--rebate-gap` 是**以那最后一段票息为基准的年化加差**（不是比例、也不是相对被解的那一段）：`-0.005` = 比最后一段低 50bp（最后一段 10% ⇒ rebate 9.5%）、`0` = 完全跟随；条款里显式写 `rebate` 则以它为准（此时 gap 被忽略，报表注 `(stated)`）。**负 gap 会把搜索下界抬到 `|gap|`**（解尾段时下界 0% 会要求负 rebate，合约层拒绝；抬升后从"还不至于负 rebate"的最低票息起搜，确实无解则报 `no room for rebate_gap`）。报表会把来源写清：`rebate 9.500000% annual (the last period's coupon -0.5000%)`。每次试算都重建条款（含位移规则 —— 房屋规则可能按票息定位移，冻结首份 schedule 会解出另一个合约）并在**同一张 local vol 表**上定价（`--method pde` 默认：确定值最适合二分；MC 有噪声）。解出的合约写成正常 `autocall_schedule` 载荷（`--out`，块默认 `output/autocall_priced.json`），可直接喂 `price-json` / `slide`；目标够不到时报错并给出"能到达的 NPV"（低于 0% 票息价值时提示票息需为负），绝不返回一个编出来的利率。
- **所有 CLI 都支持直接跑文件**（IDE Run 按钮）：`python surface_pricer/apps/<name>.py`。每个入口文件顶部有自举守卫（`__package__` 为空时把**仓库根**加入 `sys.path` 再转交包内模块），因此不依赖 cwd 或 `PYTHONPATH`；新增 app 时必须照抄（回归测试会静态校验）。
- theta 的日期间隔由 `--theta-days` 控制；日度 theta 的读法与两引擎差异见 §12.6。
- 架构与配置标准见 §12。

### 7.6 `price-json`（喂 JSON 出结果）

```powershell
python -m surface_pricer price-json contract.json --greeks delta,gamma_cash
python -m surface_pricer price-json C:\work\contract.json --greeks all --json   # 绝对路径最稳（按原样读）
python -m surface_pricer price-json - --method pde --pde-nodes 401 < contract.json
python -m surface_pricer price-json contract.json --slide --greeks delta,gamma --slide-step 5% --csv ladder.csv  # 现价梯子（§14）
```

- 这是"给定 JSON，返回 NPV + 我选的希腊值"的入口：载荷（§13 的 `vanilla_spec` / `autocall_schedule`，或整份 `--json` 报价）用**位置参数**给（`price-json <载荷路径>`；2026-10 起 `--spec` 已并入这一个参数），载荷的 `kind` 决定用哪个定价器，`--fit` 提供市场，`--greeks` 选风险跑。
- **路径按原样读**（2026-10 起）：给什么读什么（相对路径按 cwd，绝对路径最稳），**不再回退查找**（原先会再去 `surface_pricer/` 与 `output/` 里按文件名找）——写错路径立即报错并回显试过的那个路径，不会悄悄定价同名的另一个文件。`-` 表示读 stdin。
- **不做任何条款解析**：载荷写什么就按什么定价（行权价、障碍、票息、观察日全部以载荷为准；`strike_type` 只是溯源，`strike_input` 2026-10 起不在载荷里，带它就报错）。**`kind` 必填**（2026-10）：`price-json` 不再按字段猜定价器，"有 `spot0` + `observations` 就是雪球"这种推断已删除——缺 `kind` 直接报错，手写载荷必须自己声明。
- 希腊值选择：`--greeks` 缺省时读载荷自带的 `"greeks"`（字符串或数组；`--json` 报价里的 `greeks` 是结果表，不会被当成选择），都没有就只算 NPV —— **`QUICK_DEFAULTS` 里的默认值也是 `"none"`**（无参运行 / IDE Run 默认只出 NPV，要希腊值显式打开）。
- **无参运行 = 用块**（IDE Run 按钮 / `python -m surface_pricer price-json`）：文件顶部 `QUICK_DEFAULTS` 列出了**全部**参数（载荷、fit run、希腊值与引擎/网格、市场覆盖、当日判定口径、slide 与输出），键名是 argparse 的 dest（下划线），写错在启动时直接报错。相对**输出**路径（`csv`）按**包目录**解析（IDE Run 的 cwd 不可靠，`quick_output_path`）；`payload` **写绝对路径**（按原样读，读不到就报错），块里给的就是一份可直接跑的样例（`C:\Code\edslib\surface_pricer\output\autocall.json`）。`slide` 子命令无参运行同样吃这个块，但"铺梯子"这个模式本身由子命令保留（块里的 `slide` 键在那条路径上被忽略）。
- **`--greeks none` 的报表只有两行**（2026-10）：`npv` 与 `method : pde`。网格细节（`nodes/steps/theta`）与 `grid delta (cross-check)` 只在**要希腊值时**打印 —— 风险跑里网格是你的选择、值得留痕，NPV-only 报价里它们只是噪声（`--json` 的 `metadata` 照旧全量保留，机器读的字段不裁）。`--borrow` 的默认值也改成 `None`（与 `--spot`/`--rate` 一致）：`None` = 不覆盖（用 run 的，run 没记 flat borrow 就是 0），只有 `--borrow-curve none` 时这个数才生效。
- **局部波动率表缓存**（2026-10）：`local_vol_cache`（块）/ `--no-local-vol-cache`（CLI）。表是报价里最贵的一块（2Y 逐日敲入 ≈ 6.8s），缓存把它变成一次性成本 —— `output/local_vol/lv_<hash>.json` 存系数 + 生成它的输入（fit run 与 `surface.json` 的 sha1、两条曲线的文件名与 sha1、估值日、锚点、离散化、时间网格）+ `created_at`；命中直接读表（同一份 000300 雪球：15.8s → 5.3s），未命中建表并落盘。**greeks 与 slide 共用同一张表**：表的锚点钉在**基准 spot**（与风险跑同一口径），所以一次风险跑、整条 spot 梯子都只付一次；曲面不同（另一个 run、或 vol bump）不会误命中，`--no-local-vol-cache` 关掉整个 store。来源写在 **stderr** 一行：`local vol : 1 built, 0 from cache, lv_xxx.json (written 2026-10-08 18:11:35)`（`--json` 的 stdout 不受影响）。
- 引擎与市场旋钮：`--method mc|pde`、`--paths/--seed/--pde-nodes/--pde-theta/--theta-days`、`--full-bucket-grid`（关掉桶网格的按交易裁剪/长端合并，回到"一 pillar 一桶"，§12.6）。**`--greek-paths` 已并入 `--paths`**（2026-10）：一份路径数同时给价格与希腊值（配对公共随机数下把希腊值的路径调小只会加噪声，而两个旋钮会让人以为"调了其一"却白调）。`--pde-theta` 是 PDE 的**时间离散权重**（`0.5` = Crank–Nicolson，默认、时间二阶；`1.0` = 全隐式 Euler、一阶但更稳、顺便压障碍折点），换它只在收敛性检查里有意义：两种权重收敛到同一个值，差异就是离散误差。**`--mc-steps` / `--pde-steps` 已删除**（2026-10）：日频敲入时时间网格就是监控网格、这两个旋钮完全不参与（实测 6 与 12 逐位相同），其余情形内部固定用默认细分 6（§12.6）——把它留在命令行只会制造"我调了却没变"的错觉。`--ir-curve/--borrow-curve` 默认 **`latest`**（`ir` = 该文件夹最新一次；`borrow` = **当前标的所属指数**文件夹里的最新一次）、也可给**路径**（按原样读）或 `none`（退回 `--rate/--borrow` 平值）——三种写法之外没有第四种，`latest` 找不到对应条目直接报错（提示先跑 `build-*-curve`，见 §11.4）；`borrow` 按指数分档（`borrow_curve/<指数>/`），换指数不会落到别人的表上。**同一个 fit run + 同一份载荷，任意入口（`price-json` / `slide` / 库直调）给出同一个 NPV**（历史实测 967,601.42 / delta 91.829454 逐位一致）。
- **估值日**：`--valuation-date YYYY-MM-DD` 覆盖 fit run 的估值日（`--spot/--rate/--borrow/--ir-curve/--borrow-curve/--calendar-file` 同理，全走 `apps/_market.py` 同一套市场构造）。载荷里的 `valuation_date` 只是**溯源**，定价一律按 `--fit` + `--valuation-date` 重算市场侧，所以 7 月生成的载荷可以按 9 月定价（报价里的 `valuation_date` 是**市场**那一天）。但载荷**本身**必须与估值日相容：雪球载荷中任何观察日 ≤ 估值日会被 `from_dict` 拒绝（`every observation must be in the future`）—— 历史不能猜，要存续期估值就让载荷本身是那个视角：用 `build-json` 在目标日生成（块的 `valuation_date` / CLI `--valuation-date`），（存续期载荷：`build-json` 的块里写 `start` / `start_spot`，账本日期 —— `knocked_in_date` / `knocked_out_at` —— 在生成前用 `apply_history` 算好写进载荷，§12.3 第 4 条），再 `price-json mid.json --valuation-date X`。香草没有这个限制（只有"到期日必须晚于估值日"）。
- 文本模式同样先回显载荷再出报表；`--json` 输出与另两个入口同构（`contract` 块 + `npv` + `greeks` + `bucketed` + `method`）。
- **`--slide`（§14）**：把同一次运行变成**逐档现价梯子** —— `--slide-range` / `--slide-step` / `--slide-spots` 给档位，`--greeks` 给每档要算的希腊值，`--csv` 写表、`--progress` 报进度；`slide` 命令是 `price-json --slide` 的别名。

## 8. 手工覆盖与期限延伸（hard-pinned pillars）

- `--pin "2027-06-18:atm_vol=0.215,skew=0.04"` 直接把手工值**硬固定**：被覆盖的维度从优化器变量中剔除，输出严格等于给定值（部分覆盖时其余维度照常拟合，全固定时跳过优化器）。
- 参数口径为 surface 层存储值（= `surface.json` 中的值），内部按 `max(0.3, sqrt(tau))` 换算到 slice 层并校验 `FitSettings.param_bounds`。
- 无报价的更远期限在 fit 之后用 `EDSSabrSurface.rebuild` 插值补出；`--extend-tenors` 额外按 edslib 规则（每年 6/12 月第三个周五，直到估值日 + 3Y）自动生成期限。
- 报告与 `metrics` 用 `[MANUAL]` / `[SYNTH]` 标记手工与合成期限。
- 与 edslib 的差异：edslib 是"手工 surface → 合成 vanilla（±2.5 vol 点，`source=manual`）→ 标准 fit"的软注入；本工具是硬固定，仅期限延伸规则与其一致（详见 `FIT_LOGIC.md` §11）。

## 9. 测试

```powershell
python -m pytest surface_pricer/tests -q      # 328 passed, 1 skipped
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
| `test_env_loader.py` | `.env` 的查找与优先级（真实环境变量 > `--env-file` > 包内 > 仓库根） |
| `test_apps.py` | apps 目录契约：每个入口的自举守卫（`__package__` + 仓库根）、launcher 只分发到存在的模块（删 app 漏改分发的回归）、退役命令（`price-tool` / `autocall` / `price`）被拒、`price_json.QUICK_DEFAULTS` 面板完整 |
| `test_shift_config.py` | 位移配置加载与报错、生效日/gap/标的类别分档、覆盖优先级、展开语义（逐步累积、cap/floor、已过观察日保留） |
| `test_autocall_schedule.py` | 条款校验、相对/绝对展开等价性、生效障碍锚定、历史回放（按原始障碍判定）、现金流与保护本金边界、**整段计息 + theta 不变性 + `day_count` 基准 + 缺 `start_date` 报错**、注册别名 |
| `test_localvol.py` | 平曲面退化为平值波动率、无 smile 期限结构的解析 Dupire 对照、套利诊断与裁剪、外推与缓存 |
| `test_autocall_mc.py` | 零波动解析极限（本金+票息 / 敲出 / 敲入 put / 保护本金）、可复现性、路径取整、希腊值确定性、已敲出退化 |
| `test_autocall_pde.py` | 三对角与网格单测、θ 步进的解析检验、确定性 payoff 极限、网格 delta 与 bump delta 交叉校验 |
| `test_autocall_engines.py` | 两条引擎的生效条款一致性断言、含 smile / 无平滑互校、收敛性、极端障碍与单调性 |
| `test_price_json.py` | `price-json`：香草/雪球载荷、希腊值与分桶选择、路径解析、`--slide` 梯子与 CSV、quick block 与逐 flag 等价 |
| `test_risk_selection.py` | `--greeks` 选择（别名、按需 bump、`all` 不含分桶）、分桶桶位与曲线 bump 记录、LV 表缓存 |
| `test_autocall_buckets.py` | 雪球分桶希腊值：加总恒等式（sum(vega 桶)=平行 vega、sum(delta 桶)=delta_cash、粗网格下 sum(rhoq 桶)≈平行 rhoq）、两引擎互校、**桶网格策略**（≤1Y 一 pillar 一桶 / 长端按年合并 / 越过 horizon 的丢掉但保留跨过它的那根 / 显式 grid 原样 / `bucket_group_after=None` 还原旧网格）与结果元数据 |
| `test_vanilla_spec.py` | 香草解析层：百分比/远期百分比折绝对行权价、行权价冻结（不随 bump 重定）、JSON 往返与手改、`rebased`、`api.price_json` 路由 |
| `test_price_json.py` | `price-json` 端到端：手写载荷（香草/雪球）、位置参数与相对路径兜底、载荷自带 `greeks` 选择、整份报价回喂、未知 kind / 缺文件 / 错名字 / 载荷给两次的报错、启动器分发 |
| `test_slide.py` | 现价梯子：梯子构造与基准档、逐档是完整重定价（行权价冻结 / 基准档=普通报价 / 希腊值随档位变化）、**档位 = `--spot` 单独报价**（香草与雪球）、表格/CSV/JSON 输出、分桶丢弃与非法档位报错、启动器 `slide` 分发 |
| `test_build_json.py` | 生成器：块→载荷（可直接喂 `price-json`）、锁定期（首个观察日=`start+N`个月后第一个网格日、首期按整段计息）、逐期降敲（档位精确 + 包内 KO 规则被置 none）、票息阶梯（按观察期编号展开 + 三类报错）、`ki_strike`、落盘位置（相对路径→包目录、`-` 打印）、vanilla 块、启动器分发 |

## 10. 已知限制与后续计划

1. **exo 定价**：`pricing/exotics` 的 `ExoticPricer` 协议 + `register_pricer` / `get_pricer` 已被 autocallable（别名 `snowball` / `autocall`）落地，`portfolio.valuation` 通过 `product_type` 分发保持不变；架构、位移规则与两条引擎见 §12。尚未实现：memory coupon、凤凰/期初看跌/`no_ko_if_ki` 等可选 leg、多标的（worst-of）、MC 的 pathwise/LR 希腊值。
2. `bucketed_delta` 的有效换算时间 τ 由定价合约标定（`−100·Σrhoq / delta_cash`）：除远期通道（`−d lnF/dq`）外，vol/moneyness 通道与现货差分的步长口径都被吸收，保证各桶加总 = 报告的 `delta_cash`；edslib 的逐桶 dcf 口径在"桶=到期日 + ATM"时一致。现货 delta 默认 ±1% 相对 bump（edslib 惯例），OTM 时中心差分误差可达 ~2%（`RiskSettings.delta_bump_pct` 可调小）。
3. vanilla 的 `theta` 为纯时间衰减；exo 的 `theta` 为"估值日 +1 天"的 bump-and-revalue（含票息与观察日推进，MC 在网格不变时复用同一批随机数），两者口径不同，见各自 `metadata["greek_convention"]`。
4. `.xlsx` 条款表需要 `openpyxl`；缺失时可用 CSV。
5. fit 结果目前以 JSON 存放在 `surface_pricer/output/`（`io/fit_runs.py` 统一读写）；接口已收口，后续切换数据库只改这一层，`fit` 与报价入口不动。
6. 原独立 EOD 脚本（`update_option_surface_daily.py`，依赖主工程 `dbfuncs` / `env`）已从本包移除，该功能在其他代码库维护；`surface_pricer` 保持 standalone，不再含任何主工程依赖入口。
7. bucketed rate / borrow 口径：平曲线先按 bucket 网格重建（edslib `rebuild_ql_curve_by_tenors`）再逐 pillar 单点 bump，本身带 pillar 的曲线（`build-borrow-curve` 产物）直接单点 bump；曲线折现对"起点落在估值日当天"的情形按日期口径处理（无日内 stub），否则近端 pillar 会经 `zero_rate(start)` 的左端外推产生伪敏感性。
8. autocallable 的性能与路径数选择见 §12.6：单次估值 ≈ 固定 6.5s（局部波动率表）+ 0.041s/千路径（MC）或 1.7s（PDE 求解）；`--greeks` 按需付费。批量估值前建议先做局部波动率采样（`EDSSabrSlice._vol_at_moneyness`）的向量化。

## 11. 利率与借券曲线构建（build-ir-curve / build-borrow-curve）

CNY FR007 利率曲线与借券曲线是定价输入（`MarketState.rate_curve` / `borrow_curve`）。两条曲线由生成阶段命令产出，定价运行时只读 JSON（`PiecewiseRateCurve`），**不需要 QuantLib**。

### 11.1 利率曲线（`build-ir-curve`）

- 输入：`data/interest_rate.csv`（桌面导出，单位百分比）：`FR007.IR` 定盘 + `FR007S<tenor>.IR` 各期限 IRS par rate。
- 解析（`marketdata/rate_inputs.py`）：每列取**最新观测**、单位 /100、剔除"全历史恒定"的坏列（当前文件中的 2M=3.34 / 6Y=3.85 / 20Y=1.92 / 30Y=1.85）与空列（7D/14D），结果存 `RateInputs`（支持 JSON 往返）。
- 构建（`core/ir_curve.py`）：与 edslib `CNY-FR007` 同口径 —— `ql.SwapRateHelper`（季度 fixed、ACT/365F、BDC following、calendar `ql.China(ql.China.IB)`、浮动指数 `CNY-FR007-3M`（settlement 1、ACT/365F）、`Pillar.LastRelevantDate`、spread 0）+ FR007 1W fixing 注入 + `ql.PiecewiseLinearZero`（曲线 DCC ACT/360，开启外推）。
- 产物：`output/ir_curve/ir_curve_<YYYYmmdd_HHMMSS>.json`（pillar 日期 / 天数 / zero rate / par 复核）+ 同目录 `latest.json` / `index.json`（`--out FILE` 可跳过 run 记账、只写指定文件）；`max_par_residual` 自检（当前 2e-13）。
- QuantLib 只在这一步需要；定价侧用 `IRCurvePillars.to_piecewise_curve()` 得到线性 zero 插值的 `PiecewiseRateCurve`。

### 11.2 借券曲线（`build-borrow-curve`）

- 取数：行情网关快照（`QuoteApiDataProvider`）→ `RawSnapshot`。
- 远期（`marketdata/forwards.py`，照搬 `CNBorrowRateFitter`）：期货优先；缺期货的到期日用期权 put/call parity 兜底 `F = (C−P)/DF + K`（取 |C−P| 最小的 strike）。
- 反解（`core/borrow_curve.py`）：`q(T) = f(0,T) − ln(F/S)/dcf(0,T)`，`f` 为利率曲线的连续复利资金成本（`DF = exp(−f·dcf)`），时间刻度与 `PiecewiseRateCurve` 一致（ACT/365F）；剔除 3 天内到期（edslib `MIN_DAYS_TO_EXPIRY`）。
- 尾部外推（照搬 `OUBorrowCalibrator.extend_curve`）：`instantaneous_f0` + `integrate_ou_forward` + `zero_rate_at`，pillar 为**季度第三周五**（CFFEX 到期日规则，非交易日顺延），上限 `--horizon-years`（默认 3Y）。无历史标定时 `kappa` 取 edslib 对数先验中心 1.0、`mu` 取观测均值、`f0` 用最后两点瞬时远期并夹在 `q_anchor ± 0.02`。
- 产物：`output/borrow_curve/<指数>/borrow_curve_<YYYYmmdd_HHMMSS>.json` + 同目录 `index.json` / `latest.json`（归档到 `--underlying` 对应指数：MO → `000852`、510500 → `510500`，`--index` 可覆盖）。`--out FILE` 同上（跳过记账）。`--ir-curve` 默认 `latest`（读曲线库），只有 `none` 或 `--rebuild-ir-curve` 才回退到 CSV 重新 bootstrap —— **找不到不再静默 bootstrap**：那会让 borrow 建立在一份不是你指定的利率曲线上。
- **无参运行 = 编辑 `QUICK_DEFAULTS`**（2026-10，与 `price-json` / `build-json` 同款）：`underlying` 决定建哪个标的的曲线（`MO / IO / HO / 510500 / 588000 / 159915`），连同 `index` / `ir_curve` / `rebuild_ir_curve` / `rate_file` / `horizon_years` / `min_days_to_expiry` / `no_extend` / `out` / `output_root` 一起在块里选；**传任何命令行参数即关闭该块**（脚本行为不变），块里写错 key（如 `output-root`）会直接报错而不是静默无效。IDE 直接运行该文件同样走块（`_no_cli_arguments()`）。

### 11.3 口径注意

1. 反解出的 `q` 是"期货贴水"隐含成本，含股息 + 实际借券。分红接口已预留（`cum_div_factors`，见 §11.4），当前未传 → `q` 偏高（含股息）。
2. 定价自洽性不受影响：用同一套 `(r, q)` 重建的 forward 与市场期货一致。
3. 利率导出（9/28）与行情快照（9/29）的估值日可能差一天，曲线在两者之间插值。
4. 若要让 OU 尾部用**历史标定**的 `kappa`（而非先验），需要历史借券序列输入（edslib 用 150 天定 tenor 的历史曲线 + ≥120 对转换样本）。

### 11.4 接进定价链路

`fit`、`price-json`、`build-json`、`build-borrow-curve` 都接受这两个曲线值（`io/curve_files.py` 加载并校验文件类型，`io/curve_runs.py` 负责"哪一份"）：

```powershell
python -m surface_pricer build-ir-curve           # -> output/ir_curve/ir_curve_<stamp>.json + latest.json
python -m surface_pricer build-borrow-curve       # 默认 --ir-curve latest -> output/borrow_curve/...
python -m surface_pricer fit --underlying MO      # 默认吃上面两个 latest，run 落在 output/vol_fit/
python -m surface_pricer price-json contract.json --fit latest `
    --ir-curve latest --borrow-curve latest       # 显式写出默认值
python -m surface_pricer price-json contract.json `
    --ir-curve output/ir_curve/ir_curve_20261008_171818.json   # 指定某一次 run（按原样读）
python -m surface_pricer price-json contract.json --ir-curve none --borrow-curve none  # 平值 --rate/--borrow
```

**三种写法就是全部**：`latest`（该文件夹里最新一次 run）、一个**路径**（按原样读，不做回退查找）、`none`（平值）。空文件夹 + `latest` ⇒ 报错并提示先跑 `build-*-curve`；`fit` 的默认同样是 `latest`（要用平值就显式 `none`）。**融券曲线按指数分档**（2026-10）：两个指数两条 borrow（各自从自己的期货 / 期权链隐含），存放在各自的文件夹 `borrow_curve/<指数>/` 下，`latest` 问的是"**当前标的所属指数**的最新一次"，该指数没有文件夹 ⇒ 报错并列出已有的指数，绝不借邻指数的（`--borrow-curve none` 是显式回退平值）；`ir_curve` 是全体共用的一条 CNY 曲线，`latest` 与指数无关。

- 加载后替换原来的 `ConstantRateCurve`：`--ir-curve` 顶替 `--rate`，`--borrow-curve` 顶替 `--borrow`；
- `fit` 侧经 `QuoteApiDataProvider(rate_curve=..., borrow_curve=...)` 注入快照，`RawSnapshot.borrow_curve` 由 `build_market_state` 带进 `MarketState`；`manifest.json` 的 `extra` 记录**解析后的**两个文件路径（不是 `latest` 这个字面值），所以每个 fit run 都能追溯到底用了哪份曲线；
- 曲线锚点与 run 的估值日不一致时向 **stderr** 打 WARNING（不阻断，`--json` 输出保持干净）——**告警里带文件名**（`--ir-curve (ir_curve_20261008_171818.json) was built for 2026-09-28 but the valuation date is 2026-10-08`），一份报价存下来即可自证用的是哪份曲线；
- `price-json` / `build-json` 的 `QUICK_DEFAULTS` 也支持这两项（`ir_curve` / `borrow_curve`：默认 `latest`）。

**分红（预留未启用）**：`build_borrow_curve(..., cum_div_factors={expiry: D})` 会把结果换成纯借券（`q_pure = q_total + ln(D)/dcf`），并在 pillar 上记录用到的 `D`；此时定价端需要在 forward 上乘回 `D`（`F = S·exp((r−q_pure)·t)·D`），这条链路尚未接入，等有分红数据再做。

## 12. exo 定价系统（autocallable：MC + PDE）

### 12.1 三层结构

条款层 → 规则层 → 生效层 → 引擎层，单向依赖，**位移与引擎彻底解耦**：

```
AutocallContract（原始条款：相对障碍 + 票息 + 观察日 + 可选 shift 覆盖）
        │
        ├── pricing/rules/barrier_shift.py   ← config/barrier_shift.json（统一标准）+ 合约覆盖 + CLI 覆盖
        │        resolve_shift(contract, config) → BarrierShiftSpec(mode/value/cap/floor/stepwise/step_offset/source)
        ▼
build_schedule(contract, market, ko_shift, ki_shift, shift_config, contractual) → AutocallSchedule
        ★ 位移的唯一落地点：relative / additive 一律在此展开为**绝对障碍水平**
        ▼
exotics/autocall/mc.py 与 exotics/autocall/pde.py：只消费 AutocallSchedule，内部不存在 relative/shift 概念
```

- **为什么这样切**：位移改变的是"合约实际生效的障碍"，属于定价对象；放进引擎参数会让同一合约在不同方法下出现不同价格、互校失效；只在合约里存相对值则每个引擎都要重复展开逻辑。
- **两个引擎的入参完全相同**：`price_schedule(contract, schedule, market, settings)`；`crosscheck` 直接断言两边 metadata 的 `observation_dates / ko_levels / ki_levels / spot0 / ko_shift / ki_shift / barrier_smooth` 相等。

### 12.2 位移配置标准（`config/barrier_shift.json`，结构对齐 edslib）

| 字段 | 含义 |
| --- | --- |
| `templates` | 参与位移的产品类型白名单（默认 autocallable / snowball / autocall；对应 edslib 的 SNOWBALL/PHOENIX/TRIGGER/FCN） |
| `underlying_classes.index_like` | 视为指数/ETF 的标的名单，决定 KI 默认档。现含期货/期权代码（MO/IO/HO/IF/IH/IC/IM）+ 指数代码（000016/000300/000688/000852/000905）+ ETF 代码（510300/510500/159919），且**按裸代码匹配**（忽略 `.SH`/`.SZ`/`.CFE` 后缀与大小写） |
| `defaults.ko_shift` | `rule: coupon_fraction` + `fraction: -0.1`（= edslib `-annual_coupon/10`），`mode: relative`，`stepwise: true` |
| `defaults.ki_shift` | `rule: underlying_class`：指数/ETF `relative -1.25%`、其他 `additive -3%`（= edslib `GeneralBarrierShift`） |
| `effective_dates[].from` | 生效日期分档：取"不晚于合约起始日的最近生效日"（同 `IndexoBarrierShift._get_barrier_shift_config`），块内按字段覆盖 defaults |
| `ko_shift.rule: gap_tiers` | 按最大累计票息 gap（`年化票息 × notional × 年数`）分档，取第一条 `gap < max_gap` 的 `value`（同 edslib `_get_max_gap_size`） |
| `cap` / `floor` | 对**累积位移**的上下限（同 edslib `calculate_absolute_shift_size` 的 clamp） |
| `stepwise` / `step_offset` | 是否逐期累积（末期达到全值）与前几期不位移（同 edslib `shift(..., step_offset)`） |

- 位移值一律在**障碍比例空间**（`1.0` = 100% 起始现货）：`additive -0.03` 把 75% 的 KI 移到 72%，`relative -1.25%` 缩放到原水平的 98.75% —— 与 edslib 的 ratio 口径一致。
- **类别按裸代码匹配**（`_ticker_code`，忽略 `.SH` 之类后缀与大小写）。这条是为"**用 MO 的期权拟合、给 000852 的雪球定价**"准备的：两种写法（`MO` 与 `000852.SH`）都落在 `index_like`，KI 一律取 `relative -1.25%`。ETF 代码进名单是因为 000905 之类没有场内期权，得用 510500 的期权拟合 —— 同一指数可有不同期权产地，所以 fit run 的 `underlying` 记**指数**、`option_underlying` 记产地（§7.1）。
- 覆盖优先级（高 → 低）：**CLI 参数 → 合约 `shift_override` → 配置生效日档位 → 配置默认 → 内置 none**；每个规格带 `source`（`cli` / `contract-override` / `config:<date>:ko_shift` / `none`）供报表审计。
- `--no-shift`（等价 edslib `CONTRACTUAL_PAYOFF`）按原始条款定价。

### 12.3 生效语义（`build_schedule` 固定下来的约定）

1. 障碍绝对水平锚定 **`spot0` = 合约起始日现货**（默认 `anchor=start_spot`，可切 `valuation_spot`），保证 spot/vol/curve bump 时障碍不变、希腊值反映同一合约。
2. `relative` 基于原始比例水平计算、`additive` 直接加在比例上，都在 `spot0` 之上生效 → 位移规则不随市场变化。
3. **已过观察日保留原始水平**：`expand_shift(start_index=...)` 只对未过观察日应用位移（同 edslib `on_or_before(ds_date).merge(shifted.after(ds_date))`），累积序列仍锚定合约起点。
4. 历史观察日是**台账数据、不是定价状态**：由调用方在引擎之外用 `autocall.history.apply_history(contract, market)` 回放（**用原始未位移障碍**，并按该侧 `ko_boundary` / `ki_boundary` 判定，缺失 fixing 即报错不静默；回放返回**敲入日期**与敲出日期）。引擎只接收 `knocked_in_date` / `knocked_out_at` 两个**台账日期**，并把敲入日派生成 `knocked_in_before = 日期 ≤ 估值日` —— 日期晚于估值日即"尚未敲入"，所以**把估值日挪到敲入日之前就能重估敲入前**（载荷不动，`rebased` 也会重算这个派生量）。`build_schedule`/`from_dict` 校验日期落在合约生命内（start < date ≤ expiry）且落在监控网格上（daily = 业务日或观察日、expiry = 到期日、custom 以载荷为准），KO 日必须是过去的观察日；已敲出 → 退化为单一折现现金流。读回旧的 `knocked_in_before` 键直接报错，要求改写成日期。
5. **观察频率**：敲出只看观察日；**敲入默认逐交易日观察**（`ki_frequency="daily"`，对齐市场与 edslib 的 `ki_dates = SCHEDULE(..., 'DAILY')`），`ki_dates` 展开为每个交易日的绝对水平（每期位移水平作用到该期所有交易日），观察日本身必被监控；`ki_frequency="observation_dates"` 保留"敲入与敲出同日观察"的简化口径。时间网格因此就是监控网格（见 §12.4），两个引擎在每个监控节点施加敲入约束。
6. 现金流：**敲入不是终态**（两个引擎都把敲出条件继续施加在"已敲入"支路上 —— 已敲入的路径之后摸到敲出价仍然敲出、拿票息，这是合约本意，也是 KI 之下 delta 不"平"的来源）；**同一观察日敲出优先**（敲出在回测中后于敲入施加）；敲出 = 本金 + 起始至该观察日的累计票息（act/365，`settlement_days` 后支付）；未敲入未敲出到期 = 本金 + 全额票息；已敲入且**未再敲出**、到期结算 = `max(protected_principal, 1 − gearing·(1 − min(S_T/S_0, 1)))`（**无 rebate、无票息、封顶在 1**）；到期日恰为观察日时，敲入/敲出判定并入终值条件。
7. **边界口径分两侧**（`ko_boundary` / `ki_boundary`）：`inclusive`（默认，碰到即触发 —— KO 用 `>=`、KI 用 `<=`）/ `exclusive`（严格穿过）。写进载荷（`knock_out.boundary` / `knock_in.boundary`，§13.1）；**只认这两个词**（2026-10：`>=` / `gt` / `touch` / `strict` 等别名全部报错——同一个口径不该有两种拼法；缺键 = 默认 `inclusive`，那不是别名）。它对**确定性判定**（历史回放、当日判定）是硬比较；引擎内部的路径型指示器仍是平滑的（"正好停在障碍上"在连续模型里是零测事件），故该口径只在关掉障碍平滑时才逐位体现。
8. **当日判定**（`schedule.resolve_today`，两条入口都调）：估值日也是 fixing 日。① 当天有观察日 ⇒ 用**调用方输入的 spot**（`--spot`；不给就是 run 的 `init_spot`，§7.1）对**未位移条款价**、按该侧 `boundary` 硬判定：触发 ⇒ 该观察日结算（现金 = 该期 `ko_cash_flow`，此后只剩折现）；未触发 ⇒ 视为"已观察并通过"，由 `rebased` 从前瞻列表剔除。② 日频监控的敲入把**当天**算在内（当天是交易日，或当天本就是观察日）⇒ 触发即置 `knocked_in_date`（= 估值日）与派生的 `knocked_in_before`。因此 `from_dict` **接受**落在估值日的观察日（此前直接报 "every observation must be in the future"）。它按**基准 spot** 判一次，希腊值 / slide 档位里不随档位重判（差异是"触发日推迟一天"的衰减）。**口径开关 `trigger_basis`**（估值输入、不是条款）：`contractual`（默认）拿**未位移条款价**判今天 —— EOD、历史回放（`replay_history` 用原始障碍，§12.3 第 4 条）与入账都用它；`effective` 拿**位移后生效价**判 —— 盘中口径：spot 已穿真实线但未穿生效线时仍算"未敲入"，希腊值停在定价模型所在的状态，直到 EOD 那次 `contractual` 估值把判定做实（入账口径永远 `contractual`）。落点：`resolve_today(..., basis=)`、`from_dict/build_schedule(..., trigger_basis=)`（两个引擎构造器同名参数透传，`resolve_trigger_basis` 统一校验，非法值报错不猜）；CLI `--trigger-basis {contractual,effective}`（§7.5、§14）；结果随渲染供审计（文本报表 `today : basis=...`、JSON `effective.trigger_basis`）。
9. **敲出后希腊值全 0**：`is_settled` 是两个引擎 `greeks_schedule` 的第一条短路 —— NPV 照报（= 该笔敲出现金 × 折现），`PricingResult.zero_greeks(selection)` 把本次请求的希腊值显式置 0、分桶表清空，**一个 bump 都不花**（§6）。

### 12.4 两条引擎

| | Monte Carlo（`exotics/autocall/mc.py`） | PDE（`exotics/autocall/pde.py`） |
| --- | --- | --- |
| 动态 | Dupire 局部波动率，保鞅步进 `S_{i+1} = S_i·(F_{i+1}/F_i)·exp(−½σ²dt + σ√dt·Z)`；**远期比例每一步都应用**（网格含无 vol-time 的节点时不丢漂移） | 同一份局部波动率系数；对数网格 + θ 格式（默认 Crank-Nicolson），三对角 O(n) 求解 |
| 网格 | 共用的 `build_time_grid`：观察日必为节点；**每日敲入时监控日即网格节点**（1Y ≈ 245 步），否则段内子步按 vol time 等分（`mc_steps_per_observation`，默认 6） | 同一张时间网格；段内步数 = `pde_steps_per_observation`（默认 6，按 vol time 等分；原全局步长上限 `pde_time_step` 已废，见 §12.6）；有效障碍作为 crucial level 精确落在空间节点上 |
| 障碍 | KO/KI 指示器共享 `smooth_indicator`（smoothstep，带宽 = 1% × 水平） | 同一平滑函数；KO 为观察日的价值条件、**KI 用"未敲入/已敲入"双状态**在回推中转移（edslib 辅助合约的等价实现） |
| 稳定性 | 公共随机数：确定性 Sobol（固定种子）矩阵按时间网格缓存，**全部希腊值 bump 复用同一批随机数**；估值日推进若网格不变也复用 | 确定性，无需降噪 |
| 希腊值 | 共享 `risk/diff.bump_greeks`（平行）+ `risk/buckets.bucket_greeks`（分桶，与香草同一套口径，§12.6） | 同一套平行 / 分桶驱动器，bump 后复用首解的空间网格与局部波动率表 |
| 输出 | `npv` + `std_error` + `paths/steps/seed`；希腊值用较少路径（`mc_greek_paths`，默认 16384）在 CRN 下差分 | `npv` + `nodes/steps/theta` + 网格直接求导的 `grid_delta` 交叉校验 |

### 12.5 与 edslib 的对应

| edslib | 本包 |
| --- | --- |
| `apps/ODTS/indexo_barrier_shift_config.json`（生效日 + gap 档 + floor） | `config/barrier_shift.json`（同构，去掉 book/desk 维度） |
| `apps/ODTS/special_barrier_shift_infos.json`（trade_id 台账覆盖） | `AutocallContract.shift_override` + CLI `--ko-shift/--ki-shift` |
| `ki_dates = SCHEDULE(start, expiry, 'DAILY', …)`（**逐交易日观察敲入**） | `ki_frequency="daily"` → `AutocallSchedule.ki_dates` + 时间网格逐步长到交易日 |
| `DateFrequency.AT_EXPIRY` / `ONCE`（欧式敲入，标准 ACN） | `ki_frequency="expiry"`（2026-10 起只认 `daily` / `expiry` / `observation_dates` 三个词，`at_expiry`/`maturity`/`obs` 别名删除）→ `ki_dates = (expiry,)`，网格只在到期节点测敲入 |
| `BROADIE_GLASSERMAN` 宏 + `optimize_ki_observation`（把每日 KI 收敛到周期日并乘 BG 因子 `exp(βσ(√Δt−1/√n))`，β≈0.5826） | 已实现为可选项（不默认）：保留周期网格 + 给 KI 障碍乘 BG 因子；实测在月度雪球上 **BG 近似只覆盖真实每日效应的约一半**（−0.83% vs −1.72%），所以默认走真实每日网格 |
| `utils/barrier_shift.py`（GeneralBarrierShift 默认档、KI 的 relative/additive、gap 分档） | `pricing/rules/barrier_shift.py` 的内置规则 |
| `EDSPL.py::BARRIER_SHIFT` / `calculate_absolute_shift_size`（逐步累积 + cap/floor + 仅未来生效） | `expand_shift` + `build_schedule` |
| `PDEEngineAutocall`（KO 约束 + KI 辅助合约 + 观察日强制时间节点） | `exotics/autocall/pde.py` 的价值条件 + 双状态 |
| `MCEngineScriptedPayoff` + `MCDYLocalVol` + QMC Sobol | `exotics/autocall/mc.py` + `pricing/models/localvol.py` + `scipy.stats.qmc` |
| `AUTOCALL_BARRIER_SHIFT_FACTOR` 等合约字段 | `BarrierShiftSpec`（只带 `source` 溯源，不写回条款） |

**未移植（有意）**：券商台账规则（book/desk/trade_id 特例、年末统一 bump）、`GapRiskPrewash`、Broadie–Glasserman 离散监控调整、antithetic / 矩匹配、pathwise / LR 希腊值。

### 12.6 验证与现状

- 互校：真实曲面（含 smile + 借券曲线）上 `--compare` 差异约 **0.2%–0.7%**、折合 2 倍左右蒙特卡洛标准误；平曲面合成市场互校 < 1.5%。
- **希腊值口径（五条冻结约定，两引擎共用）**：① PDE 的空间网格与 MC 的随机数矩阵在整轮风险中只构建一次 —— 网格若跟着 bumped forward 平移，delta 会被网格位移抵消（实测 bump delta ≈ 1 而同解的网格 delta ≈ 56，自相矛盾）；② 局部波动率表以其 `spot_anchor` 为基准冻结（sticky-moneyness），现货 bump 只移动查询点而不重建 `sigma_lv` 表（否则 delta 从 ~40 掉到 ~2）；③ MC 路径起点固定为 `market.spot`（`F(0) = S` 的无套利恒等），避免曲线锚点的半天 stub 污染首段比例；④ **局部波动率表按 `(曲面, 时间网格)` 在整轮风险中复用**（`LocalVolCache`）：spot / rate / borrow bump 只动漂移与查询点，系数不动；只有 vol bump（新曲面）与 theta（新网格）才重建 —— 建表 6.5s（1Y 日度监控、83 片）占单次估值 ~75%，这条决定整轮风险的**固定成本**；⑤ **MC 随机数矩阵按区间对齐复用**：估值日 +1 会让日度监控网格丢掉首个区间（249→248 步，且首节点是估值时间戳 `15:34:28` 而非午夜），按列位置复用会把 day-0 的冲击错配到 day-1 上、重新抽样则让 theta 变成两个独立估计之差（各带 ±700/天 的标准误）。现在缓存记住区间顺序、按**日期**比对后切片 ⇒ 同一合约 65k 路径下 MC theta 从 −1,205（纯噪声）变为 +980，与 PDE 的 +919 同号同量级。修完后真实市场上 MC/PDE 的 `delta / vega / rhoQ` 差异进入 ~10%，`delta` 与 PDE 网格 delta 方向一致。
- **时间网格（2026-09 修正，2026-10 简化）**：段步数下限 `pde_steps_per_observation`（与 MC 的 `mc_steps_per_observation` 同义，默认 6）是唯一旋钮。历史上还有一个步长**上限** `pde_time_step`：只有上限时月度观察（段长 ≈ 0.083 vol-time 年 < 0.2）会退化成 **1 步/月**，PDE 的 NPV 偏低约 0.5%、theta 偏低约 40%（实测 595 vs MC 904）；加上下限后两引擎跑在**同一张网格**上，NPV 差收敛到 **0.008%**。上限随后删除 —— 它只在窗口宽于 ~1.2 vol-time 年（长年期年度观察）时才生效，与下限重复。另外标准雪球默认 `--ki-frequency daily`，此时网格**就是业务日监控网格**（245~249 步），两个参数都不参与（实测 `--pde-steps 6` 与 `12` 结果逐位相同），它们只在 `observation_dates` 敲入模式或 MC 子步下起作用。**2026-10：`--mc-steps` / `--pde-steps` 两个 CLI 与 quick-block 旋钮已删除**（内部仍固定用默认下限 6）—— 见 §7.6。
- **节点位置按 vol time 等分（同日修正）**：EDS 的 vol time 按交易日计数（国庆整周贡献为 0），而旧实现把节点按**日历天**等分取整。于是估值日 +1 天会重铺首段、并把首段最后一步的方差分配改变约 30%（3 个交易日 → 2 个），1 天 theta 量到的主要是网格自己（PDE 1d 在合理设置下从 +595 跳到 −1,096）。现在节点按 **vol time 等分**落位、布局锚定在 segment 上：除首段外，bump 后网格逐字复用。回归测试见 `test_time_grid_is_laid_out_by_vol_time_and_survives_a_one_day_bump`。
- **theta 的正确读法**：月度观察雪球的日度 theta 本身是"锯齿"的 —— 一个**交易日**会把首个观察日的 vol time 缩短 1/16（≈6.25%，2026-09-29→09-30 正逢国庆周），而休市日只有 act/365 的票息累积、扩散时钟不前进。所以 `--theta-days 1` 混了两种日子；比较两引擎请用 `--theta-days 5~7`。区间对齐（见上 ⑤）后同一合约的日度 theta：PDE（nodes 1243，确定性）**+919/天**，MC 65k 路径 **+980/天**（差 7%），而旧实现给 **−1,205/天** —— 说明此前的 MC/PDE theta 差异主要是**未配对噪声**而非方法差异。残余差异来自格式差异（log-Euler vs Crank-Nicolson）与 vol-time 起点平移。
- 解析极限（零波动/单观察日/极端障碍/全额保护/敲出退化）与收敛性由单测锁定（§9）。
- **分桶希腊值（2026-10 新增，雪球与香草共用一套）**：原来只有香草算 bucket，雪球的 `bucketed_*` 永远为空；现在桶原语上提到 `pricing/risk/buckets.py`（`bucket_greeks` 驱动器 + `vega_bucket/curve_bucket/convert_bucketed_delta`，香草改为调用它，行为不变），两个雪球引擎都在 `greeks()` 里回填 `PricingResult.bucketed_*`。桶口径与香草完全一致：**vega 桶 = 曲面期限**、**rhoQ/rho 桶 = 曲线 pillar**（平曲线先按 edslib 的 1M…2Y 网格重建再逐点 bump）、**`bucketed_delta` = 现货 `delta_cash` 按 rhoQ 份额分摊**（构造上精确加总）。
  - 互校（真实市场：6 个曲面期限、15 个借券 pillar、MC 4096 路径）：`sum(bucketed_vega) = −5,729` vs 平行 `vega = −5,736`（**0.12%**）、`sum(bucketed_rhoQ) = −3,241` vs `rhoq = −3,246`（**0.13%**）；残差来自逐点 bump 与平行 bump 的 smile 插值差异 + MC 噪声。PDE 侧同样通过（确定性）。
  - **成本**：分桶是 opt-in（`--greeks buckets`，不随 `all`），因为每个桶是一对估值，而 **vol pillar bump 会让局部波动率表重建**（6 个期限 ⇒ 12 张表 ≈ 78s）；曲线桶便宜 —— 曲面未变 ⇒ 复用同一张表，只有求解/路径成本。真实合约实测：PDE `bucketed_vega + bucketed_rhoq` ≈ **177s**。
  - **桶网格按交易裁剪 + 长端合并（2026-10，仿 edslib 的默认 tenor 网格思路）**：**自动**曲线桶网格（借券 / 利率；显式 pin 的 grid 原样使用）现在走 `risk.buckets.bucket_grid`：① 越过**交易最后收付日**（`horizon`，引擎传 `schedule.expiry_payment_date`、香草传 `spec.expiry_date`）的 pillar 直接丢掉，只保留**跨过它的那根**（局部插值下它之后的区段对更早的日期没有权重，所以这一步是精确而非近似；`linear_zero` / flat-forward 成立，非局部三次曲线请用 `--full-bucket-grid`）；② **1Y 以内**每根 pillar 一个桶（近端细节与敏感度都在这里）；③ 1Y 以外**按年合并**成一个桶，组内 pillar 一次性一起 bump（这些曲线是 `rates[i] += h` 的加性 bump ⇒ 加总恒等式仍成立）。`RiskSettings.bucket_group_after` 控制 ②③（`None` = 一 pillar 一桶，即改造前），`RiskSettings`/CLI `--full-bucket-grid` 是逃生门。桶网格策略会写进结果元数据（`bucket_grid`: pillars / buckets / dropped / group_after / horizon），报表在 `bucketed :` 下方打一行 `bucket grid: …`，`--json` 的 `bucket_grid` 也在（雪球放在 `method` 块里）。
  - 实测（真实 2Y 月度雪球 + 真实 14-pillar 借券曲线，`--greeks bucketed_delta`，PDE nodes=301 steps=3）：**14 桶 88.5s → 7 桶 54.6s**（越过到期日的 4 根丢掉、≥1Y 的 3 根并成 1 桶）。剩下的固定成本是基值 + 局部波动率表 + 现货对（≈21s），桶数再降时收益递减 ⇒ 下一步该做的是**桶态用粗网格**与**桶之间并行**（§10 的"希腊值并行化"）。
  - 桶网格目前只走默认（曲面期限 / 曲线 pillar），CLI 不给自定义桶；库内可直接调 `risk.buckets.bucket_greeks(..., bucketed_vega_pillars=[...], bucketed_delta_pillars=[...])`（香草 `calculate_greeks` 已暴露同样两个参数）。
  - 组合批量（`price` + `portfolio/`）那条链已移除（§7.2）；批量要跑桶态就让外层脚本对每份载荷把桶名拼进 `--greeks`（会显著变慢，故不作为默认）。
- **生产参数与耗时（1Y 月度观察雪球、真实曲面、KO 103 / KI 70 / coupon 20%、2026-10 实测）**：
  - 固定成本：局部波动率表 **6.5s/张**（1Y 日度监控 = 83 片）；整轮风险的表数量取决于 bump 种类（spot/rate/borrow 复用同一张，vol 与 theta 各重建）—— `--greeks delta,gamma,vega` 3 张、全套 4 张。
  - 边际成本：MC ≈ **0.041s / 千路径 / 次估值**（249 步）；PDE 求解 ≈ 1.7s / 次（601 节点）。
  - 二阶希腊值（2026-10）：`volga` 复用 vega 的 vol 状态（**+0 估值、+0 张表**）；`vanna` **+4 次估值**，其 σ± 曲面与 vega/volga 共用（默认三者 bump 都是 0.5 vol 点，因此不额外建表）⇒ `--greeks delta,gamma,vega,volga` 与不含 `volga` 同价，再加 `vanna` 只多 4 次求解（PDE ≈ 7s / MC 65536 路径 ≈ 11s）。
  - 整轮耗时：MC 65536 路径 + `delta,gamma,vega` ≈ **36s**；PDE 601 节点 + 全套希腊值 ≈ **44s**；PDE 只算 `delta,gamma,theta` ≈ 20s。
- **MC 路径数收敛与噪声**（同一 seed、Sobol 序列嵌套，delta/gamma/vega + NPV 一整轮）：

  | 路径数 | se(NPV) | NPV | delta | gamma | vega | 耗时 |
  | --- | --- | --- | --- | --- | --- | --- |
  | 8,192 | 2,031 (0.21%) | 988,457 | 76.5 | −0.235 | −6,134 | 26s |
  | 32,768 | 1,028 (0.10%) | 984,195 | 79.1 | −0.171 | −5,866 | 30s |
  | 65,536 | 727 (0.074%) | 984,468 | 82.2 | −0.186 | −6,084 | 36s |
  | 131,072 | 517 (0.053%) | 983,864 | 81.5 | −0.183 | −5,942 | 49s |
  | 65,536 × 3 seed（离散度） | — | ±1,516 (0.15%) | ±3.2 (4%) | ±0.05 (29%) | ±149 (2.5%) | — |
  | PDE 1243 节点（确定性参照） | — | 983,074 | 79.14 | −0.154 | −5,901 | 48s |

  - `se(NPV)` 严格按 `1/√N` 收敛（每 4 倍路径减半），**32k~64k 路径即把 NPV 误差压到 0.1% 以内**（与 edslib 的 65535 默认一致）；vega 在 64k 下 ±2.5%，**delta ±4%、gamma ±29%** —— MC 的 gamma 在任何可承受的路径数下都不可用（要 ±5% 需再乘 25 倍路径）。
  - 所以**分工**：NPV 用 MC（32k~64k 路径，可与 PDE 交叉验证：实测两法差 0.08% ≈ 1.5 倍标准误）；**希腊值优先用 PDE**（601~1243 节点，全套 44s、确定性、无 seed 噪声），或 MC 只取 delta/vega 并接受 4% / 2.5% 的噪声。`--greeks` 让这两件事分开付费。

- **障碍位移的单位统一（2026-10，载荷路径的关键修正）**：`from_dict` 原来把**绝对条款价**直接喂给"以障碍比例表达"的位移规则 ⇒ `additive -0.03` 只减了 0.03 个指数点（5,396.17 → 5,396.14），与 term-sheet 路径（在比例上展开、再乘回 `spot0`）不一致 —— 同一份载荷经两条入口会得到不同生效价。现在两侧都在比例空间展开（`build_schedule` 手上就是比率，`from_dict` 先除以 `spot0` 再乘回）。实测同一份 000852 雪球（2Y、KI 65%、年化票息 12.96%、notional 1.952e8、2026-09-30 市场、PDE 601 节点、全程无 bump）：修正前 `ki_eff = 5,396.14` / NPV **177,086,427.82**；修正后按载荷写明的 `additive -3%` 口径 `ki_eff = 5,147.12` / NPV **179,398,125.60**；按 `index_like` 的 `relative -1.25%`（现为默认口径，§12.2）`ki_eff = 5,328.72`。回归测试 `test_a_payload_and_its_term_sheet_agree_on_an_additive_shift` 用**非 100 的锚点**钉住两条入口给出同一有效障碍（100 的锚点会掩盖这个 bug）。
- **KI 上方为什么没有 delta 尖峰（2026-10 排查结论）**：先排除三种数值解释 —— ① 分辨率：同一市场 201→601→1201→2401 节点的现值收敛到 0.01% 内；② 差分步长：`delta_bump_pct` 从 1% 收到 0.1%，KI 处 delta 从 0.778 变到 0.774（× 名义）；③ 障碍平滑：连平滑一起关掉也只到 0.826（+6%）。结论是**结构性的**：KI 之上这份合约近似"一只行权价 100% 的空头看跌"（敲入后到期付 `min(S_T/S0, 1)`），其 delta `1 − N(d1)` ≈ 0.77~0.83，所以越贴近 KI，delta 反而越**低**；能量集中在 KO 带。51 档梯子（601 节点、60%~110% × `spot0`、每 1%）实测：`delta` 峰 **208.0M @ 84%**（1.07 × 名义）、`|gamma|` 峰 **20.9M @ 100%**（KO 数字边）、`|vega|` 峰 **584k @ 88%**（KO 带中部，84.5%~94.5%）；宽峰来自 6,600~7,100 的"敲出/收息腿"。要看到贴障碍的 delta 尖峰，需要 KI 与亏损行权价重合（`ki_strike = ki`），本结构隔了 35%。
- **梯子/希腊值读法**：`delta_cash / notional` 即"名义倍数"；`delta_cash` 按相对步长归一化，1% 中心差分是"一扇 ±1% 的窗"（§6）—— 窄于 ±1% 的结构（数字边）会被平均，复核请用 0.1% 步长或在障碍两侧加密档位。敲出（已结算）后 `greeks_schedule` 短路：NPV 照报、全部希腊值 0（§12.3 第 9 条）。

### 12.7 已解析合约的 JSON 层（term sheet / CLI ↔ 引擎）

引擎**只吃 `AutocallSchedule`**：绝对障碍、每个观察日一条票息、敲入监控网格、亏损腿三项（strike/gearing/floor）、notional、以及 rebate 的到期比例（由年化率折算，折算公式与读载荷共用）。所有"条款换算"（降敲、每期票息不同、锁定期、位移规则、欧式/每日敲入、BG 近似…）都在 `contract.py` + `schedule.py` 里做完，引擎与报表都不再解释条款。这个对象就是 JSON 层：

* `AutocallSchedule.to_dict()` 导出 **`kind=autocall_schedule`、`version=1`** 的载荷（`observations[*] = {date, ko(条款绝对价), coupon_rate(年化)}`、`knock_in = {frequency, strike(绝对), gearing, protected_principal, level(条款绝对价)}`，`frequency="custom"` 才给 `dates`/`levels`；`rebate`(年化)、`notional`、`settlement_days`、`day_count`、`knocked_out_at`/`knocked_out_cash`，以及**位移规则** `shift.{ko,ki}` + `shift.elapsed`）。逐字段释义见 §13.1；
* `AutocallSchedule.from_dict(payload, market)` 读回来：只取**合约侧**数据，vol time / 折现 / 支付日由当时市场重算（所以昨天的载荷今天能用），**位移在这里施加**（与 `build_schedule` 共用 `expand_shift`，`elapsed` 交代存续期已摊期数），因此手改条款价或改位移规则都会立刻改变生效价。它同时接受裸载荷和 `--json` 的整份报价（`{"contract": …}`）；
* `AutocallSchedule.rebased(market)` = "同一份合约换一个市场"（希腊值 bump 用）：市场侧字段重算，新估值日之前的观察日与监控日按 `build_schedule` 的同一规则丢弃；
* 载荷的**导出** = `build-json` 的 `--output-name`（`-` 打印到 stdout，§15）；**定价** = `price-json <载荷路径>`（`-` 读 stdin），可配 `--greeks/--method/--paths` 全套；两个雪球引擎都有 `price_schedule()` / `greeks_schedule()` 对应入口。
* 手写例子（OTM 行权价 + 欧式敲入 + 四段递增票息，实测 MC 967,867 vs PDE 967,123 = 0.08%）见 §7.5 的条款示例（照 §13.1 翻成载荷即可）；"不敲出"就是把 `ko` 设成远高于现价的水平，锁定期就是不列那些观察日。**逐字段释义见 §13.1**，与香草的同类载荷（§13.2）对称。

**现金流口径（`cashflows.py`，纯函数读 schedule）**：

| 腿 | 何时 | 金额（每单位 notional） |
| --- | --- | --- |
| 敲出 `ko_cash_flow(schedule, i)` | 观察日 `i` 触发 | `1 + coupon_rate[i] * accrual(start, obs_i)`（每期自己的票息） |
| rebate `rebate_cash_flow(schedule)` | 到期且既未敲出也未敲入 | `rebate_ratio = 1 + rebate_rate * accrual(start, expiry)`，默认 `rebate_rate` = 最后一个观察日票息 |
| 敲入 `expiry_cash_flow(schedule, S, True)` | 到期且敲入 | `max(protected, 1 - gearing * (1 - min(S / (ki_strike * spot0), 1)))` |

应计从 **`start_date`（期初）** 起算、按载荷的 `day_count`（`act/365f` 默认 / `act/360` / `act/act`）计息，**不随估值日移动**（存续期估值照样整段计息，theta bump 不动现金流；缺 `start_date` 直接报错而不是退回估值日）。`--rebate`/`--coupon` 给的都是**年化率**。`ki_frequency="expiry"` 时 `ki_dates = (expiry,)`，网格只在到期节点施加敲入约束（两个引擎都在到期节点用到期 KI 水平做状态混合，见 §12.4）。
- **局部波动率构造（2026-09 修正）**：profile 显示瓶颈不在 `localvol.py`，而在它调用的两个数学基元 —— ① `black_price` 用 `scipy.stats.norm.cdf`（走 `rv_continuous`，标量一次 ~12µs，且触发百万级 `numpy.array` 调用）→ 改 `scipy.special.ndtr`（同值，快 ~10×）；② `EDSSabrSlice._vol_at_moneyness` 的 价格→IV 反演走 `brentq`（每次约 59 次 Black 求值）→ 先走 **Jaeckel 解析反演**（`implied_vol_jaeckel`，仅越界时回退 brentq）。单切片 ~300ms → ~70ms，整套测试 51s → 33s。再加 `DupireLocalVol(slice_step=0.01)`：系数沿时间光滑，节点比 0.01 vol-time 年更密时按 vol time 抽稀并线性插值（实测价格偏差 +0.0005%；网格本来稀疏则完全不抽稀）——这是每日观察网格（~250 时间节点）能跑得动的前提。进一步方向：切片 `_calibrate` 向量化、希腊值并行化。

## 13. JSON 载荷字段字典（`autocall_schedule` / `vanilla_spec`）

两个产品共用同一套架构：**条款 → 已解析合约（JSON）→ 引擎**。载荷里放的是"引擎实际用到的数字"，不是等待解释的条款；读取方（`from_dict`）只取**合约侧**字段，市场侧字段按传入的 `MarketState` 重算，因此昨天导出的文件今天可以照着重新定价，而手改合约字段（行权价、障碍、票息、观察日）会立刻改变报价。

| | 雪球（autocall） | 香草（vanilla） |
| --- | --- | --- |
| 原始条款 | `AutocallContract`（比率 + 规则） | `VanillaContract`（`strike` + `strike_type`） |
| 解析层（唯一读条款的地方） | `build_schedule()`：施加位移、展开绝对障碍、丢弃已过观察日 | `resolve_spec()`：把 `strike_type` 折成绝对行权价 |
| 已解析对象 = JSON | `AutocallSchedule`（`kind="autocall_schedule"`，§13.1） | `VanillaSpec`（`kind="vanilla_spec"`，§13.2） |
| 引擎入口 | `price_schedule()` / `greeks_schedule()` | `price_spec()` / `calculate_greeks_spec()` |
| 换市场（希腊值 bump） | `schedule.rebased(market)` | `spec.rebased(market)` |
| CLI（写出 / 读入） | `autocall --spec-out/--spec` | `price-tool --spec-out/--spec` |
| 通吃两类载荷的入口 | `price-json --spec FILE --greeks LIST`（§7.6） | 同左 |

**共同约定**

1. `kind` + `version` 自描述，且 **`kind` 必填**（2026-10：读取方不再靠字段猜类型，缺失即报错）；`-` 表示 stdin / stdout。
2. `--json` 的整份报价可以**原样回喂**：读取方识别 `contract` 包装并自动拆开（§13.3）。
3. 载荷是**条款原样 + 规则**，且**全绝对**：障碍写**位移前**的条款价（`observations[].ko`、`knock_in.level`，比率只存在于条款侧 `AutocallContract` / CLI flag），位移以**机器可读规则**写在 `shift` 里（`mode`/`value`/`stepwise`/`cap`/`floor`/`source`，见 §13.1），由引擎在定价时施加 —— 一份载荷于是读起来像条款表："103% 敲出、按 4 期逐步下移 0.5%"，而不是同一价位两个版本。改 `ko`/`level` 改的是条款，改 `shift` 改的是规则，两者都立刻反映到价格上。费率也统一口径：`coupon_rate` 与 `rebate` 都是**年化率**，不是"本金+票息"的总额。
4. 载荷只写**合约侧**数据：市场侧（vol time / 折现 / 远期 / 监控日期）一律由读取方按当时的市场与规则重算，`valuation_date` 之前的观察日与监控日按 `build_schedule` 的同一规则**丢弃**（`rebased` 亦然；`custom` 网格也按同一窗口过滤，但保留每一天自己的价位）。
5. 解析后没有未来观察日 ⇒ 报错；已结算合约用 `knocked_out_at` / `knocked_out_cash` 表达（`knocked_out_cash` 直接采用，不再由票息重算）。

### 13.1 `autocall_schedule`（雪球）

| 字段 | 类型 | 释义 |
| --- | --- | --- |
| `kind` / `version` | string / int | 固定 `"autocall_schedule"` / `1` |
| `underlying` | string | 标的**指数代码**（如 `000852.SH`；标识用，不参与定价，但**决定位移类别** —— §12.2 按裸代码匹配）。期权产地不写在这里：fit run 用 `option_underlying` 记（如 `MO`），§7.1 |
| `product_type` | string | 默认 `autocallable` |
| `spot0` | float | 障碍锚（期初价）；`ko`/`ki`/`strike` 的比率基准 |
| `anchored_on` | string | `start_spot`（合约期初价）/ `valuation_spot`（估值日现价） |
| `start_date` | ISO | **票息应计起点（期初）**：即使估值日在存续期中，每条腿都按 `start_date → 观察日` 整段计息；缺这个字段 `from_dict` 直接报错（默认成估值日就会少计息） |
| `day_count` | string | 计息基准：`act/365f`（默认，自然日/365）/ `act/360` / `act/act`（ISDA，跨年按 365/366 分段）；敲出与 rebate 同基准 |
| `valuation_date` | ISO | 估值时点（保留日内时间） |
| `expiry_date` | ISO | 到期日 |
| `notional` | float | 现金流乘数 |
| `settlement_days` | int | 收付延迟（日历天）：支付日 = 观察日 + 延迟，只影响折现 |
| `observations[]` | array | **仅未来**观察日，按时间升序 |
| `observations[].date` | ISO date | 观察日 |
| `observations[].ko` | float | 该期**条款（位移前）**敲出价，绝对价；实际生效价 = 本值按 `shift.ko` 逐步施加（§13.1 末"规则 vs 位置"） |
| `observations[].coupon_rate` | float | **该期**年化敲出票息（逐期不同即 step-up / step-down） |
| `knock_in.frequency` | string | 监控规则：`daily`（逐交易日）/ `expiry`（欧式，仅到期观察）/ `observation_dates`（与敲出同日）/ **`custom`**（网格就在载荷里，见下两行） |
| `knock_in.level` | float | **条款（位移前）**敲入价，绝对价，适用所有观察期；规则型频率只写这一个数，监控日期与逐期生效价都由读取方重算 |
| `knock_in.dates` | ISO date[] | **仅 `frequency: "custom"`**：显式监控日 |
| `knock_in.levels` | float[] | **仅 `frequency: "custom"`**：与 `dates` 一一对应的监控价（**按原样使用，不再叠位移** —— 你要什么就是什么） |
| `knock_out.boundary` | string | **顶层块**（与 `knock_in` 平级）：敲出边界口径 —— `inclusive`（默认，现货 ≥ 该期敲出价即触发）/ `exclusive`（严格大于）；缺省键 = `inclusive`。历史回放与当日判定按它做**硬比较**（§12.3 第 7、8 条） |
| `knock_in.boundary` | string | 敲入边界口径：`inclusive`（默认，现货 ≤ 敲入价即触发）/ `exclusive`（严格小于）；缺省键 = `inclusive` |
| `shift.ko` / `shift.ki` | object | **位移规则**：`mode`（`none`/`relative`/`additive`）、`value`（总位移，**障碍比例空间**：`relative` 是 `-0.0125` 这样的比例，`additive` 是同空间的比例点，`-0.03` = −3% 起始现货）、`stepwise`（按观察期逐步摊到整段）、`cap`/`floor`（累计位移上下限）、`source`（`config:<生效日>:ko_shift` / `contract-override` / `cli`，溯源）。缺省键表示默认值（不逐步、无上下限） |
| `shift.elapsed` | int | 该载荷的**第一期之前已经走过几期**（仅 `stepwise` 需要，等于 0 时省略）：逐步位移的分母是整段期数，所以存续期导出的载荷必须交代已经摊掉几期，读回来才能复现同样的价位 |
| `knock_in.strike` | float | 亏损腿行权价，**绝对价**（= 期初价 × 条款比率；等于 `spot0` 时从期初价算盈亏，低于 `spot0` 即 OTM 雪球）。与 `ko` / `levels` 一样是绝对空间，比率只是条款侧的写法 |
| `knock_in.gearing` | float | 亏损放大倍数 |
| `knock_in.protected_principal` | float | 亏损腿保底（占 notional 比率） |
| `rebate` | float | "既不敲出也不敲入"腿的**年化率**——与 `coupon_rate` 同一口径（`--rebate` 也是年化），默认取最后一期票息；到期实付比例由它推出：`1 + rebate × accrual(start_date, expiry_date)`（同 `day_count`）。≥ 1.0 的写法会被拒（那是旧版"总收付比"的口径） |
| `knocked_in_date` | ISO date / null | **敲入发生的日期**（ledger 记录；`apply_history` 或你的台账脚本写入）。状态由它与估值日比较**派生**：日期晚于估值日 = 尚未敲入 —— 所以把估值日挪到它之前就能重估"敲入前"，载荷一个字不用改。旧的 `knocked_in_before` 键会被拒（要求改成日期） |
| `knocked_out_at` | ISO date / null | 历史敲出观察日（已结算合约；由支付日减 `settlement_days` 反推） |
| `knocked_out_cash` | float / null | 该次敲出的现金额（已结算合约；`from_dict` 直接采用） |
| `shift.notes` | string[] | 解析备注（锚点、位移来源、监控日数量等） |

**规则 vs 显式网格**：`daily` / `expiry` / `observation_dates` 都是**规则**——载荷只写 `frequency` + `level`，监控日期由读取方用 `contract.monitoring_grid()` 从日历与观察日重算（`daily` 时 1Y 约 245 天），所以一份日度监控雪球的载荷只有 ~60 行而不是 ~530 行；`custom` 才带 `dates` / `levels`（两者都按同一窗口规则只保留估值日之后的网格，且按原样使用）。读取方对"规则名下却带了显式网格"的旧写法也照 `custom` 处理，因此旧载荷仍可加载，重写时自动收敛成短格式。`from_dict` 还会拒绝把 `strike` 写成比率的载荷（绝对价 = 期初价 × 比率，比率写法会差一个 `spot0` 量级）。

**位移怎么施加**：位移值活在**障碍比例空间**（`1.0` = 100% 起始现货），所以两条路都是"换算到比例 → `pricing.rules.barrier_shift.expand_shift(spec, 比例, elapsed=shift.elapsed)` → 乘回 `spot0`"：term sheet 建表（`build_schedule`）手上就是条款比率，读载荷（`from_dict`）则先把绝对条款价除以 `spot0`。`relative` 把每期比例乘以 `1+位移`、`additive` 在比例上加位移（`-0.03` = −3% 起始现货，**不是** −0.03 个价格点 —— 直接把位移加到绝对价上会差一个 `spot0` 量级）；`stepwise` 把总 `value` 摊到整段观察期上（分母是**条款期数**，`elapsed` 说明载荷之前已经摊了几期，`rebased` 丢观察日时会同步累加），累计位移过 `cap`/`floor` 截断。手改 `shift` 就是改定价规则（实测 −0.5% → −3%：首期生效敲出价 7,540.76 → 7,493.57，NPV 968,566.28 → 967,747.09）。

现金流口径（`cashflows.py`，每单位 notional）：敲出 `1 + coupon_rate[i] × accrual(start, obs_i)`；到期未触发 `rebate_ratio = 1 + rebate × accrual(start, expiry)`（`cashflows.rebate_ratio()` 一个公式，建表与读载荷共用）；到期敲入 `max(protected, 1 - gearing × (1 - min(S_T / (strike × spot0), 1)))`。两条腿的费率都是年化，口径一致。`accrual` 用载荷的 `day_count`、从 `start_date` 起算**整段**——估值日在不在存续期中都一样，也正因如此「估值日 +1 天」的 theta bump 不会挪动任何一条腿的现金流（回归测试分别锁住这两条）。

### 13.2 `vanilla_spec`（香草）

| 字段 | 类型 | 释义 |
| --- | --- | --- |
| `kind` / `version` | string / int | 固定 `"vanilla_spec"` / `1` |
| `underlying` | string | 标的（标识用） |
| `product_type` | string | 默认 `vanilla` |
| `option_type` | string | `call` / `put` |
| `notional` | float | 乘在 NPV 上（delta/gamma/vega 同比例） |
| `valuation_date` | ISO | 估值时点 |
| `expiry_date` | ISO | 绝对到期日 |
| `strike` | float | **绝对**行权价 —— 引擎实际使用，改价改这里 |
| `strike_type` | string | `absolute` / `percentage`（×spot）/ `fwd_percentage`（×forward）；仅溯源（改它不重定价）。**只有这三个词**（2026-10）——`spot_percentage` / `forward_relative` 之类别名、以及任何未知名一律报错，不再默默按 `absolute` 处理 |

（`strike_input` —— 调用方原始输入的比例 —— **2026-10 从载荷移除**：一份已解析的载荷只有一个行权价，带 `strike_input` 的载荷/报价会被拒绝并提示删键或重新生成。）
**载荷里没有市场字段**：`spot` / `forward` / `discount_factor` / vol time（`year_fraction`）/ `initial_forward` 全部由定价引擎在定价时从当时的 `MarketState` 取，载荷只写合约基本要素。这样一份载荷可以跨日复用（换一条曲线、换一天估值都成立），也不会误导成"这些数被冻结了"——被冻结的是**交易**（行权价、到期日、期权类型），不是市场。

定价口径：`npv = notional × black_price(forward, strike, τ, vol, discount_factor, option_type)`，其中四项都由市场现算：

| 引擎侧输入 | 来源 | 含义 |
| --- | --- | --- |
| `forward` | `market.forward(expiry)` | 到期远期 `F`（Black-76 的 F，含 `forward_overrides` 优先） |
| `discount_factor` | `market.discount_factor(expiry)` | 到期折现因子（乘在期权现值上） |
| `τ`（vol time） | `market.year_fraction(expiry)` | **波动率时间**：EDS 口径 `(交易日 + 非交易日 × holiday_weight) / trading_days_per_year`，对齐 edslib `DateUtil.dtcf`；和 Black 公式里的 `vol²τ` 配套。**不是** act/365 应计 —— 计息（rho/Q）与票息用的是 `core.daycount.year_fraction(..., basis="act/365f")` 那条线 |
| `initial_forward` | `market.forward(expiry, spot=surface.init_spot)` | 曲面锚定期初价对应的远期，SABR 查表用（sticky 处理的分母） |
| `vol` | `surface.implied_vol(...)` | 曲面插值出的隐含波动率，**永远现算**（vol bump 才会移动它） |

`api.price_json()` 两种载荷都吃：带 `kind="vanilla_spec"`（或 `{"contract": …}` 包装）⇒ 按写好的数字定价；只给 `strike` + `strike_type` 的原始形式 ⇒ 先按市场解析一次再定价。

### 13.3 `--json` 报价包装

两个 CLI 的 `--json` 输出都是"载荷 + 本次结果"，因此可以整份存下来、手改 `contract` 块再回喂：

| 键 | 雪球 `autocall --json` | 香草 `price-tool --json` |
| --- | --- | --- |
| `contract` | §13.1 的 `autocall_schedule` | §13.2 的 `vanilla_spec` |
| `effective` | 生效障碍 / 观察日 / 票息率 / 支付日 / `rebate_rate` + `rebate_ratio` / 位移规则（与载荷同形） / 是否已结算（报表镜像：载荷是条款价，这里是施加位移后的价） | — |
| `npv` | ✓ | ✓ |
| `forward` / `discount_factor` / `implied_vol` / `strike` / `year_fraction` | —（文本报表里有 spot / forward / df；exotic 结果的 `strike` / `implied_vol` 是占位 0） | 顶层（Black-76 的输入输出） |
| `greeks` | 平行希腊值（`delta / delta_cash / delta_n / gamma / gamma_cash / vega / theta / rho / rhoq`，未请求的为 `null`；`volga` / `vanna` 需显式点名） | 同（`volga` / `vanna` 两边都有，2026-10） |
| `bucketed` | `bucketed_vega` / `bucketed_delta` / `bucketed_rhoq` / `bucketed_rho`（未请求为空 `{}`） | 同 |
| `method` | 引擎参数（method / paths / nodes / steps / seed / std_error …） | — |
| `greek_convention` | 本次希腊值口径 | 同 |

文本模式（不带 `--json`）里，`price-json` 对**香草**载荷会**先回显已解析载荷**再出报表，方便直接看一眼"这次到底按什么条款定价"（`echo_payload` 渲染，超过 6 个元素的长数组只留首尾并在 stderr 说明；要全量就落盘 —— 载荷本来就是 `build-json` 写出的那份）；**雪球不回显**——它的报表已经含原始条款与生效障碍（§7.6）。

## 14. Spot slide（`slide` / `price-json --slide`）

"标的走到那里，这个仓位值多少、希腊值是多少" —— 逐档**完整重定价**，不是从基准报价做一阶外推。实现分三层：`pricing/risk/slide.py`（梯子 + 行容器，与产品无关）、`reporting/slide_report.py`（文本表 / JSON / CSV）、`apps/price_json.py` 的 `--slide`（把载荷接到引擎上）。

```powershell
python -m surface_pricer slide --fit latest var.json --greeks delta,gamma_cash,vega --slide-range 20% --slide-step 10%
python -m surface_pricer price-json spec.json --slide --slide-spots 7000,7500,8000 --greeks delta,gamma --json --csv ladder.csv
python -m surface_pricer slide snowball.json --method pde --pde-nodes 201 --slide-step 15% --progress
```

```
slide      : 5 rungs | span=+-20.00% | step=10.00% | greeks=delta,gamma_cash,vega
fit run    : MO_20260930_111643 | MO | 2026-09-30 11:16:43 | spot=7330.2916 | 6 expiries
contract   : vanilla_spec | MO | strike=7,696.8062 | expiry=2027-03-30 | notional=1
market     : base spot=7,330.2916 | method=analytic
convention : delta: bump-and-revalue dNPV/dSpot
             gamma_cash: d2NPV/dSpot2 * spot^2 / 100 (NPV change per (1% spot move)^2)
             vega: surface parallel bump, reported per 1 vol point

      spot     bump        npv        delta  gamma cash       vega
5,864.2333  -20.00%  23.893039  0.038971298   20.716766  3.8619738
6,597.2624  -10.00%  76.170414   0.11702683   73.528037  9.2425052
7,330.2916   +0.00%  221.05868   0.29501061   162.69153  17.168661   <- base
8,063.3208  +10.00%  522.49254   0.52776299   199.12876  21.478482
8,796.3499  +20.00%  983.64049    0.7174262   158.31151  19.623686
```

- **每档都是一次真实估值**：`market.clone(spot=level)` 后重新定价 —— 香草 `spec.rebased(moved)` + `calculate_greeks_spec`，雪球 `schedule.rebased(moved)` + 引擎。所以档位落进敲入区时看到的凸性是真实凸性，且每个希腊值都是在**该档**差分出来的（delta 在档位上量，不在基准点量）。
- **交易不动**：香草的行权价是已解析的绝对数，雪球的障碍锚在 `spot0`，而 `rebased` 只移动市场侧（估值日、vol time、折现因子、KI 监控网格），所以 `strike` / `ko` / `ki` / 观察日跨档完全不变（测试锁定）。
- 梯子：`--slide-range`（半幅，`0.30` 与 `30%` 都行，默认 30%）+ `--slide-step`（默认 5%）⇒ 默认 13 档；`--slide-spots` 给绝对档位（更好读），它与 `--slide-range/--slide-step` **互斥**（2026-10：同时给报错，不再静默让 spots 赢——照 `--json` + `--csv -` 的先例）。**基准档一定在表里**（步长除不尽时补上），`bump = spot/base − 1`，基准行标 `<- base`。
- **次日判定口径（盘中 / EOD）**：`slide` 与单次报价共用 `--trigger-basis`（§12.3 第 8 条）。盘中用 `effective`：今天这条 fixing 按**位移后的生效线**读，spot 在两条线之间时整张梯子都还是"未敲入"状态（基准 spot 判一次、所有档位与希腊值共用 —— 这就是"不用马上卖到敲入后水平"的量化口径）；EOD 用 `contractual` 复算即得法律判定。注意档位本身永远吃**位移后**的障碍（引擎只认生效价），开关只动"今天算不算已触发"。
- **梯子与表缓存**（2026-10）：档位只改 spot，表的锚点钉在**基准 spot**（风险跑口径），所以整条梯子共用**一张**局部波动率表 —— 冷启动建一次（≈6.8s），之后从 `output/local_vol/` 直接读（3 档实测 11.8s，含进程启动与 3 次 PDE 求解）。
- 希腊值仍由 `--greeks` 选（与单次报价同一套选择器）。**分桶希腊值不进梯子**：每个桶是一对 bump，逐档出桶表既贵又没有信息量，给了就丢弃并打印 note（单次报价照旧有桶）。`--progress` 把每档打到 stderr —— PDE + 希腊值一档约 1~3 秒，长梯子不会静悄悄。
- 输出三种：默认文本表（表头带 fit run / 合约块 / 基准现价 / 引擎 / 希腊值口径，列宽按内容对齐）；`--json` 给机器人读的 `spot_slide` 载荷；`--csv FILE` 写表（`-` 则把 CSV 打到 stdout 并**不出表格**）。`--json` 与 `--csv -` 同时用会报错（都占 stdout）。`slide` 命令 = `price-json --slide`。
- **一致性（单测锁定）**：某档的 `npv / delta / gamma` 与"同一份载荷用 `--spot 该档位` 单独报价"逐位一致（香草解析式 1e-12；PDE 完全相同），基准档等于普通报价 —— 也就是说梯子里的每个数都可以用另一个入口复现。

`--json` 的行字段（`rows[]`）：

| 字段 | 类型 | 释义 |
| --- | --- | --- |
| `spot` | float | 该档现价（绝对水平） |
| `bump` | float | 相对基准现价的变动（`spot / base − 1`，基准档 = 0） |
| `npv` | float | 该档估值（含 notional） |
| `<greek>` | float | 该档按 `--greeks` 选出的希腊值，键名与报价一致（`delta` / `gamma_cash` / `vega` …）；**未请求的键不出现** |

外层字段：`kind="spot_slide"` / `run` / `contract`（被定价的那份已解析载荷，可直接回喂）/ `base_spot` / `span` / `step`（给了 `--slide-spots` 时为 `null`）/ `greeks` / `method` / `rows`。

## 15. JSON 生成器（`build-json`）

把"生成载荷"和"定价"彻底分开：`build-json` 只做前者 —— 把文件顶部 `QUICK_DEFAULTS` 块解析成 `autocall_schedule` / `vanilla_spec`（§13）写进文件；定价全部交给 `price-json` / `slide`。这里**不做任何估值**（不拟合、不算希腊值），所以出载荷是毫秒级，而一次报价可能要几分钟。**2026-10 起 `price_autocall` / `price_vanilla` 已移除**（§7.3、§7.5）：`build-json` 是唯一的载荷出口，"只要 JSON"的流程从生成到定价只有这一条链。

```powershell
python -m surface_pricer build-json                                     # 用块里的 autocall 条款
python -m surface_pricer build-json --product vanilla --output-name var.json
python -m surface_pricer build-json --fit MO_20260928_150000 --output-dir out2 --json
# 生成 → 定价 → 现价梯子
python -m surface_pricer build-json --output-name snowball.json
python -m surface_pricer price-json snowball.json --greeks delta,gamma_cash
python -m surface_pricer slide snowball.json --slide-step 10%
```

```
payload    : C:\Code\edslib\surface_pricer\output\snowball.json
contract   : autocall_schedule | MO | notional=1,000,000
terms      : 2026-09-30 -> 2028-10-09 | 22 observation(s) | guaranteed=3M | accrual=act/365f | spot0=7,330.2916
knock-out  : 7,403.5945 (101.0000%) -> 6,633.9139 (90.5000%) | shift=none, levels as written
knock-in   : 4,544.7808 (62.0000%) | expiry | loss strike 7,330.2916 (100.0000%) | gearing 1 | protection 0.0000%
coupon     : 19.0000% #1-10 -> 10.0000% #11-22 | rebate=10.0000% (annual)
```

- **条款只在块里**（`QUICK_DEFAULTS["autocall"]` / `["vanilla"]`）：命令行只给产品、fit run、输出位置与市场覆盖，所以"这次到底生成什么"只有一个地方可看。相对路径（`output_dir`）按**包目录**解析（IDE Run 的 cwd 不可靠），绝对路径照写；`--output-name -` 打到 stdout。
- 无参运行 = 用块（IDE Run 按钮）；`--json` 额外把载荷打到 stdout；`--list-runs` 列出可用的 fit run。
- **每个日期都过日历**：观察日网格（以及显式 `expiry`）里落在节假日/周末的日期**顺延到下一个工作日**，`month_grid(..., calendar=...)` 与 `BusinessCalendar.next_business_day()` 负责，报表的 `dates` 行说明顺延了几个；顺延只会往后 ⇒ 网格仍然严格递增，两个日期撞到一起会合并（期数可能比原始网格少）。顺序是"先按合约日做锁定期裁剪、再顺延"——锁定期边界上的那一天不会因为顺延出窗口就被当成观察日。网格规则只有这一处实现（2026-10 前 `price_autocall` 也委托它，那个入口已移除），所以"同一套条款 ↗ 同一份载荷"不存在两个入口对不上的可能。
- 上面的 `knock-out` 行是**生效值**（位移已施加），而载荷里存的仍是条款价 + 位移规则（§13.1）—— 两个数字都能对上账；`stepdown_size` 走显式档位时规则被置 `none`，报表直接写 `shift=none, levels as written`。
- 生成物直接喂 `price-json` / `slide`：`price-json <载荷路径>`（路径按原样读，§7.6），所以"生成一次、随处定价"都成立。
- `core/daycount.month_grid()` 现在是**唯一**的观察日网格实现，锁定期与网格口径不会在两个入口之间漂移。

### 15.1 块里的 autocall 专属字段

| 字段 | 类型 | 释义 |
| --- | --- | --- |
| `guaranteed_period` | int | **不观察的整月数**：首个观察日 = `start + guaranteed_period` 个月之后、网格上的第一个日期（按**合约日**裁剪，之后才顺延节假日）。它同时**按整段计息**（accrual 从 `start` 起，§13.1），所以锁 3 个月时第一次敲出拿的是 3 个月票息，不是一期 —— 实测 25 期月度网格 + 3 个月锁定 ⇒ 22 个观察日、首个在 `start + 101 天` |
| `stepdown_size` | float | **逐期降敲**：`0.005` = 每次敲出观察把 KO 降 0.5%（`ko_i = ko - (i-1) x step`，第 1 个观察不降）。写成逐期显式档位（写进 `observations[].ko`），同时把包内的 stepwise KO 规则置为 `none`，两套规则不会叠加 |
| `coupon` | float / list / dict | 年化票息：标量（每期相同）、逐期列表（长度须 = **名义期数**）、或**按名义期数的阶梯** `{1: 0.19, 11: 0.10}` = 第 1 期起 0.19、第 11 期起 0.10。阶梯必须从 1 开始、编号不得超过名义期数。编号一律数**名义期数**（从起息日按月数）：**`guaranteed_period` 不改变编号**（2026-10）—— 锁定期只是让合约少观察几期，不重排期号，所以 `{1: 0.10, 13: None}`（反解用 `null`，见 §7.5）在锁 3 个月与没有锁定时含义相同；落在锁定期内部的档位从**第一个被观察到的期数**起生效 |
| `ki_strike` | float | 敲入亏损腿行权价（锚的比率）：1.0 = 从期初价算盈亏，0.9 = OTM 雪球。载荷里 `knock_in.strike` 是绝对价（= 比率 x `spot0`） |

其余键与 `AutocallContract` 字段一一对应：`underlying`（None → run 的标的；现在 run 的 `underlying` 就是**指数**，§7.1）、`start`（None → 估值日）、`tenor` / `expiry`、`obs_freq` `M/Q/S/A`、`ko`、`ki`、`ki_frequency` `daily/expiry/observation_dates`、`ko_boundary` / `ki_boundary`（`inclusive`（默认）/`exclusive`，§12.3）、`ki_gearing`、`protection`、`settlement_days`、`rebate`（None → 最后一期票息）、`day_count`、`notional`、`start_spot`、`no_shift`。`start_spot` 只在 `start` 早于估值日（存续期载荷）时需要，缺了会报错，而不是拿今天的现货顶上。**块顶部的 `spot` / `valuation_date` 默认 `None`**（= 用当时 run 的 `init_spot` 与估值日），存续期报价把 `start` 写成期初日、`start_spot` 写成期初价 —— 这样"输入一个 spot 报价"与"锚点在期初"两件事互不干扰（§12.7 的 `build_schedule` 约定）。vanilla 块就是 `strike` + `strike_type` + `tenor|expiry` + `option_type` + `notional`。

一律"报错而不是猜"（退出码 2 + 明确信息）：锁定期吃掉所有观察日、降敲把 KO 打到 ≤ 0、票息列表长度不符、阶梯不从 1 开始或编号超期数、缺 tenor/expiry、缺 `start_spot` —— 都是拒绝生成，不产出一个"看起来对"的文件。

