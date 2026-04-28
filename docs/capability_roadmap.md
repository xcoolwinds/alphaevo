# Capability-First 路线图

更新时间: 2026-04-28

## 结论

AlphaEvo 下一阶段的主线应该是 **能力增强**，不是优先做 Web 界面。

原因很直接：

- 当前项目最有差异化的地方是“策略自我迭代进化”，不是入口形式。
- 如果研究能力、评估可信度、经验沉淀还没继续做深，过早做 Web 只会把 CLI 包上一层壳。
- 先把研究引擎做强，后续无论接 CLI、API 还是 Web，价值都会更高。

因此建议的优先顺序是：

1. 强化策略策划、显式买卖语义与参数/退出优化
2. 强化研究闭环能力
3. 强化真实数据与事件语义能力
4. 强化评估可信度与组合层能力
5. 最后再做 Web 研究工作台

---

## 1. 当前最值得继续做强的能力

### 1.0 显式买卖语义与参数优化

当前最优先的能力是把“卖出/退出”从默认止损止盈中拆出来，变成可表达、可回测、可优化的研究对象。

已推进：

- DSL v0.4 新增 `exit.triggers`，支持跌破均线、RSI 过热等显式卖出触发器。
- 回测引擎支持 `ExitReason.SIGNAL`，在持仓期间命中 `exit.triggers` 后按当前 close 平仓。
- 新增 `alphaevo optimize`，可在同一批历史数据上搜索 `entry / params / indicator / exit / stoploss / takeprofit / holding` 空间。
- `alphaevo optimize` 新增显式合格门槛：`--objective win_rate --min-win-rate 0.5 --min-avg-return 0 --min-profit-loss-ratio 1.0 --max-drawdown 0.35 --min-signals 30`，避免只按综合评分包装不合格策略，也避免用负期望或低盈亏比刷胜率。
- 优化搜索新增 `--evaluation-mode fast` 与 `--full-eval-top`，先用交易级指标快速筛选，再完整复评头部候选。
- DSL v0.5 新增 `entry.triggers` / `entry.guards`，把真正买点与硬过滤拆开，避免信号黏滞。
- 新增同 K 线止损/止盈冲突的 `fill_policy`：`conservative`、`optimistic`、`close_first`。
- 退出优化报告新增卖点诊断：卖早、卖晚、止损有效性、止盈截断收益。

下一步：

- 让 `strategy draft` / `strategy revise` 的规则解析覆盖更多口语化买点、卖点和风险约束。
- 在 `optimize` 中加入 walk-forward / stress-window 作为可选质量门，而不是只看单窗口排序。
- 将卖点诊断结果反向喂给 LLM 反思，让下一轮 mutation 有明确的“卖早/卖晚”证据。

### 1.1 Self-Evolution 研究记忆

重点不是再加一次性生成能力，而是把“学到的东西”持续积累下来。

应优先强化：

- `PatternLibrary` 在评估报告、排行榜、研究日志中的可见性
- `ExperienceStore` 对反思和 mutation 的约束能力
- `MetaLearner` 的状态感知能力
- family / category / regime 维度的经验迁移

目标：

- 系统不只是“能改”，而是“越改越有经验”

### 1.2 事件 / 新闻 / catalyst 语义层

当前 event/news 指标已经从常量 fallback 提升到了价格/量能 proxy，但还不够。

下一步应补：

- 统一 event/news adapter
- 可缓存、可回放、可对齐时间窗的事件数据层
- catalyst registry（政策、财报、宏观、行业事件）
- 反思阶段的 retrieval-augmented event context

目标：

- L3 不再主要靠 proxy，而是真正形成金融语义层

### 1.3 Canonical 评估协议

如果没有更严格的评估协议，进化结果会越来越像局部最优，而不是研究资产。

建议优先补：

- canonical walk-forward 模式
- 更明确的 train / val / test 时间隔离
- regime holdout / stress window 测试
- category-specific evaluation tracks

目标：

- 提高进化结果的可信度与可比性

### 1.4 Portfolio / Risk Layer

当前系统强在单策略研究，但还没有形成完整组合研究能力。

建议下一阶段补：

- strategy-to-portfolio aggregation
- exposure / correlation / concentration controls
- category / sector / regime risk budget
- benchmark-aware portfolio comparison

目标：

- 从“单策略自我进化”扩展到“策略池与组合研究”

### 1.5 Alpha Factory 与动态因子治理

Alpha Factory 已有基础，但要成为核心能力还需要治理层。

建议补：

- 更严格的 factor validation protocol
- IC / IR / turnover / stability gates
- factor lineage 与淘汰机制
- factor 与 strategy family 的关联分析

目标：

- 因子不只是“能生成”，而是“能进入研究资产体系”

---

## 2. 建议的阶段优先级

### P0: 强化研究可见性

- 已完成：把 `top_patterns`、经验摘要、失败模式写进评估/进化报告
- 已完成：在 `evolve` 输出中展示 meta-learning、family lessons、pattern reuse
- 已完成：让用户能直接看到“系统为什么这么改”

### P1: 强化真实世界数据能力

- 已完成统一事件 / 新闻 provider 抽象；下一步是补真实 provider 实现
- benchmark / sector / event 的统一上下文构建
- 事件型策略单独评测协议

### P2: 强化评估可信度

- 已显式输出并参数化 walk-forward protocol；下一步是日历型 canonical walk-forward protocol
- 已显式输出 regime holdout
- 已显式输出 stress-window benchmark；下一步是更严格的 stress-window benchmark protocol
- 组合层风险指标

### P3: 强化研究资产飞轮

- pattern / factor / experience 的统一治理
- family/category/regime 维度的经验迁移
- 更强的 meta-learning

### P4: 最后再做 Web

到这一步再上 Web，才更像“研究工作台”而不是“命令行按钮化”。

---

## 3. 对当前项目最重要的产品叙事

AlphaEvo 应该反复强调的不是：

- “我也有界面”
- “我也能聊天”
- “我也能生成策略”

而是：

- 我能把策略当作研究对象
- 我能自动验证、自动归因、自动改写
- 我能把经验沉淀成组织资产

这才是最难复制、也最值得持续加深的能力。
