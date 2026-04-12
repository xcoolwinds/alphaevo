# 当前能力状态

本文档回答 3 个高频问题：

1. 现在是否已经具备“策略优化 / 自我验证 / 自我进化”能力？
2. 现在是否支持设置迭代轮数、采样数、模型、数据源等参数？
3. 现在对大盘、板块、外围突发事件等外部因素的支持处于什么阶段？

## 结论

- 已具备“策略生成 → 回测验证 → 多维评估 → 失败归因 → 自动改写 → 再验证”的研究闭环。
- 已具备运行时参数控制能力，尤其是进化轮数、方法、样本数、日期范围、数据适配器、LLM 模型、反思模型、人类提示约束。
- 对大盘、benchmark、市场环境、板块因素已具备可运行支持；对真实新闻流、宏观突发事件、外围黑天鹅事件仍未完整接入，当前主要依赖价格/量能 proxy 与上下文合成。
- 事件 / 新闻上下文现在已经有统一 provider 抽象和逐日注入通道；内置适配器尚未提供真实 feed，默认仍走 proxy fallback。
- 评估报告现在会显式给出 event/news context 的 provider/proxy 覆盖率，避免事件型结果被误读。

## 1. 已具备的核心能力

### 1.1 策略优化 / 自我验证 / 自我进化

当前主流程已经具备：

- 策略 DSL 解析、序列化、版本化存储
- 数据采样、历史数据抓取、回测执行
- 多维评估，包括胜率、平均收益、回撤、Sharpe、benchmark 对比、anti-overfit
- 失败案例提取与 LLM 反思
- 基于反思结果的小步改写与多轮进化
- *(内部实验)* 多岛并行进化与课程式进化——模块已实现，尚未暴露为公开 CLI 命令

可以把能力边界理解为：

- `alphaevo run`: 单轮研究闭环，自我验证
- `alphaevo evolve`: 单 lineage 多轮优化
- `alphaevo evolve-islands`: 多岛并行进化，多样性探索
- `alphaevo evolve-curriculum`: 渐进难度训练 (easy → medium → hard → reality)

这说明系统已经不是“只会回测”，而是具备完整的研究 Agent 主链。

### 1.2 目前不属于本项目能力边界的部分

当前还不是：

- 实盘交易执行系统
- 实时风控中台
- 完整的宏观 / 新闻 / 事件驱动决策系统
- 面向社区协作的公共策略平台

这些方向可以继续扩展，但不应和当前已完成能力混淆。

## 2. 参数控制能力

### 2.1 已支持的 CLI 运行时参数

当前已经可以在 CLI 层直接设置关键运行参数。

#### `alphaevo run`

支持：

- `--samples`: 本轮采样股票数量
- `--start` / `--end`: 回测日期范围，支持只给一侧
- `--output`: 报告输出目录
- `--adapter`: 数据源覆盖，例如 `yfinance` / `akshare` / `dsa`

#### `alphaevo evolve`

支持：

- `--rounds`: 进化轮数
- `--method`: `llm` / `param_search` / `hybrid`
- `--samples`: 每轮采样股票数
- `--start` / `--end`: 进化期所用数据窗口

- `--adapter`: 数据源覆盖
- `--model`: 主 LLM 模型覆盖
- `--reflect-model`: 反思专用模型覆盖

#### *(内部实验)* `IslandEvolution` (Python API)

> 以下参数通过 Python API 传入，尚未暴露为 CLI 命令。

支持：

- `islands`: 岛屿数量
- `generations`: 代数
- `rounds_per_gen`: 每代每个岛的进化轮数
- `max_symbols`
- `date_range`

#### `alphaevo evolve-curriculum`

CLI 命令参数：

- `strategy_id` (positional): 待训练的策略 ID
- `--adapter`: 数据源覆盖
- `--model`: LLM 模型覆盖

课程式阶段参数目前在代码里由默认 curriculum 配置控制，不是 CLI 公开参数。

### 2.2 配置层已有但需注意的参数

配置模型里还定义了以下进化控制项：

- `evolution.max_rounds`
- `evolution.max_changes_per_round`
- `evolution.min_signal_count`
- `evolution.complexity_limit`

其中：

- `max_changes_per_round`、`min_signal_count`、`complexity_limit` 已直接参与进化安全护栏。
- `max_rounds` 目前作为配置项存在，但当前主执行路径主要由 CLI / API 显式传入的 `rounds`、`generations`、`rounds-per-gen` 控制，不是一个已经强制生效的全局硬上限。

这意味着当前系统已经“能设置迭代轮数等参数”，只是配置层和运行时层的职责还没有完全统一。

## 3. 对大盘、板块、外围事件的支持现状

### 3.1 已支持的部分

当前系统已经有以下“环境感知”能力：

- `preferred_regime`: 策略元数据可声明适用市场环境
- `RegimeDetector`: 可基于指数 / OHLCV 数据检测趋势、波动、恐慌、震荡等 regime
- `benchmark` 对比：研究结果会和 buy-and-hold 做同窗对比
- L2 指标支持 benchmark / 板块上下文，例如：
  - `relative_strength_20d`
  - `sector_heat_rank`
  - `sector_heat_rising_days`
  - `intra_sector_strength_rank_pct`
- `IndicatorContext` 已为 benchmark、板块、个股信息、市场上下文预留承载结构

因此，对“大盘环境”和“板块相对位置”的考虑，不是零能力，而是已经有基础设施。

### 3.2 当前尚未完整具备的部分

对以下因素，目前还不能称为“完整支持”：

- 外围突发事件
- 新闻流
- 宏观事件时间线
- 政策事件
- 盘中风险突发
- 跨市场联动冲击

原因不是模型层完全没有设计，而是数据链路还没真正把这些信息持续、可靠地灌进研究主流程。

当前 L3 指标如：

- `negative_news_score`
- `news_sentiment_score`
- `days_since_event`
- `price_above_pre_event`
- `sector_fund_flow_positive`
- `already_overreacted`

当前更准确的说法是“proxy + 中性兜底”。也就是说：

- 没有真实事件 feed 时，系统会用价格缺口、量能放大、事件前价格锚点等 proxy 继续运行
- 一旦后续接入 provider，逐日事件上下文可以直接注入主流程，不需要再改回测语义层
- 但它仍然不能真正理解“昨晚美联储超预期加息”这类外部冲击的语义层含义

### 3.3 对问题 2 的直接回答

如果问题是：

- “能不能部分考虑大盘、板块、市场环境？”

答案是：能，已经有一部分能力。

如果问题是：

- “能不能系统性处理外围突发事件、新闻冲击、宏观风险传播？”

答案是：现在还不能算完整具备，最多算接口和指标层已经预留。

## 4. 当前最适合的使用方式

当前最适合把 AlphaEvo 用在：

- 基于 OHLCV 的技术型 / 结构型策略研究
- 趋势、回踩、均值回归、部分板块轮动策略
- 策略参数优化、回测验证、失败归因、自动迭代
- 需要“人能读懂”进化轨迹和 YAML 版本变化的研究流程

当前不建议过度承诺的方向：

- 纯事件驱动策略
- 强依赖新闻 / 宏观 / 舆情输入的策略
- 实时交易与实盘自动执行

## 5. 如果要继续补强，优先级建议

建议按下面顺序增强：

1. 继续强化 self-evolution 的研究记忆，把 Pattern Library / Experience Store / Meta-Learner 的结果真正暴露到评估报告和进化结果里。
2. 引入统一的事件 / 新闻适配器，形成可缓存、可回放、可对齐时间窗的事件数据层，让 L3 从 proxy 走向真实金融语义层。
3. 给事件型策略单独建立 canonical 评测协议，并继续补更严格的日历型 walk-forward / stress-window protocol。
4. 补 portfolio / risk layer，把“单策略进化”扩展成“策略池与组合研究”。
5. 最后再考虑 Web 工作台，不要让入口形式先于能力建设。

更完整的能力优先路线见 [`capability_roadmap.md`](capability_roadmap.md)。
