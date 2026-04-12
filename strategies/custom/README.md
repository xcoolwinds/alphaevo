# 自定义策略目录

将你的策略 YAML 文件放在此目录中。

格式与 `strategies/builtin/` 中的内置策略相同。

如果自定义策略与内置策略同名，自定义策略会覆盖内置策略。

也可以通过环境变量指定其他目录：

```env
ALPHAEVO_STRATEGY_DIR=./my_strategies
```
