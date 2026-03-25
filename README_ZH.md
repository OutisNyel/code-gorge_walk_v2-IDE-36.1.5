# code-gorge_walk_v2-IDE-36.1.5

这是一个面向腾讯开悟平台 `gorge_walk_v2` 场景的强化学习代码包，内置五类算法包：

- Dynamic Programming
- Monte Carlo
- Q-Learning
- SARSA
- DIY 模板

它不是一个完整自包含的 Python 独立项目，更准确地说，它是一个面向平台运行时的算法代码包，外加一个本地冒烟验证入口。

## 这个仓库是什么

官方腾讯 Arena 文档中的标准 KaiwuRL 项目通常包含：

- 一个 `agent` 包
- 一组共享配置
- 一个 `train_test.py` 校验脚本

这个仓库是该模板的一个变体：

- 它不是只保留一个 `agent/` 目录，而是按算法拆成多个包：
  - `agent_dynamic_programming`
  - `agent_monte_carlo`
  - `agent_q_learning`
  - `agent_sarsa`
  - `agent_diy`
- 它依赖外部脚本在运行时切换当前算法
- `train_test.py` 的用途更像“代码正确性冒烟测试”，用于在正式平台训练前先完成一次最小训练闭环检查

因此，单看某个静态配置文件，往往不能准确反映真实运行时状态。

## 仓库结构

```text
.
|-- train_test.py
|-- kaiwu.json
|-- conf/
|   |-- algo_conf_gorge_walk_v2.toml
|   |-- app_conf_gorge_walk_v2.toml
|   |-- configure_app.toml
|   `-- map_data/
|-- agent_dynamic_programming/
|-- agent_monte_carlo/
|-- agent_q_learning/
|-- agent_sarsa/
`-- agent_diy/
```

关键目录和文件说明：

- `train_test.py`
  - 冒烟验证入口
  - 负责选择算法
  - 启动 learner 和 aisrv
  - 通过日志与模型产物判断训练是否成功
- `conf/algo_conf_gorge_walk_v2.toml`
  - 维护算法名到 agent 类和 workflow 的映射
- `conf/app_conf_gorge_walk_v2.toml`
  - 应用侧策略配置
- `conf/map_data/F_level_*.json`
  - 动态规划算法使用的状态转移图数据
- `agent_<name>/agent.py`
  - 智能体实现
- `agent_<name>/algorithm/algorithm.py`
  - 学习规则实现
- `agent_<name>/workflow/train_workflow.py`
  - 训练循环实现
- `agent_<name>/feature/definition.py`
  - 样本处理与奖励塑形
- `agent_<name>/conf/train_env_conf.toml`
  - 训练环境配置

## 环境与任务

根据官方文档：

- 地图大小：`64 x 64`
- 动作空间：
  - `0 = UP`
  - `1 = DOWN`
  - `2 = LEFT`
  - `3 = RIGHT`
- 智能体视野：以自身为中心的 `5 x 5`
- 任务目标：
  - 到达终点
  - 尽量减少步数
  - 可选收集宝箱

默认训练环境配置为：

- 起点：`[29, 9]`
- 终点：`[11, 55]`
- `treasure_random = false`
- `treasure_count = 0`
- `treasure_id = []`
- `max_step = 2000`

官方规则中的得分逻辑包括：

- 到达终点获得终点分
- 使用更少步数获得步数奖励
- 每个宝箱带来额外积分
- 超时任务总分为 `0`

一个非常重要的区分是：

- `score` 是环境表现指标
- `reward` 是训练用的强化学习信号，可以在代码里单独塑形

## 支持的算法

| 算法 | 状态编码 | 学习方式 | 是否依赖训练采样 | 保存产物 |
| --- | --- | --- | --- | --- |
| Dynamic Programming | 仅位置 | 值迭代/策略迭代 | 不依赖在线采样式训练 | 策略表 |
| Monte Carlo | 仅位置 | First-Visit 整回合控制 | 是 | 策略表 |
| Q-Learning | 位置 + 宝箱比特 | Off-policy TD | 是 | Q 表 |
| SARSA | 位置 + 宝箱比特 | On-policy TD | 是 | Q 表 |
| DIY | 用户自定义 | 用户自定义 | 用户自定义 | 用户自定义 |

说明：

- `dynamic_programming` 直接使用 `conf/map_data/F_level_1.json` 中的已知转移数据
- `monte_carlo`、`q_learning`、`sarsa` 依赖与环境交互采样
- `diy` 当前只是骨架模板，不能直接视为可运行算法

## 状态与特征设计

官方文档把智能体相关数据划分为三类：

- `ObsData`：预测输入
- `ActData`：预测输出
- `SampleData`：训练输入

本仓库在表格型算法中对这套接口做了简化实现。

### 仅位置状态

使用算法：

- `agent_dynamic_programming`
- `agent_monte_carlo`

编码方式：

- `pos_feature = x * 64 + z`
- 状态总数：`64 * 64 = 4096`

### 位置 + 宝箱状态

使用算法：

- `agent_q_learning`
- `agent_sarsa`

编码方式：

- 先计算 `pos_feature = x * 64 + z`
- 再从 `organs` 中提取 10 位宝箱状态
- 最终合成为：
  - `feature = 1024 * pos_feature + treasure_binary`

状态总数：

- `64 * 64 * 1024`

这和官方环境文档、智能体文档中的组合编码设计是一致的。

## 训练与评估语义

官方框架约定：

- `predict`
  - 训练时使用
  - 可以包含探索策略
- `exploit`
  - 评估时使用
  - 通常应选择贪心或确定性动作
- `learn`
  - 消费 `SampleData`
- `observation_process`
  - 将环境原始输出转换为 `ObsData`
- `action_process`
  - 将 `ActData` 转为环境可接受动作
- `save_model`
  - 写出文件名中包含 `model.ckpt-{id}` 的模型文件
- `load_model`
  - 重新加载模型状态

在这个仓库里：

- `predict` 和 `exploit` 是明确分开的
- 尤其对 epsilon-greedy 算法来说，训练行为和评估行为并不相同

## 训练工作流

官方工作流文档要求训练循环大致执行：

1. 用 `usr_conf` 重置环境
2. 获取观测
3. 调用 `agent.predict`
4. 调用 `agent.action_process`
5. 调用 `env.step`
6. 计算 reward
7. 收集轨迹帧
8. 调用 `sample_process`
9. 调用 `agent.learn`
10. 按周期保存模型

本仓库遵循了这个模式，但各算法略有不同：

- `dynamic_programming`
  - 不走在线轨迹训练，而是直接基于地图转移数据求解
- `monte_carlo`
  - 以整条轨迹为训练单位
- `q_learning`
  - 以单步更新 Q 表
- `sarsa`
  - 以单步更新 Q 表，并依赖实际下一动作

## 运行时选择细节

这里有一个需要特别注意的点：

- `train_test.py` 当前把 `algorithm_name` 设为 `q_learning`
- `conf/app_conf_gorge_walk_v2.toml` 的静态初始值仍可能是 `dynamic_programming`

这不一定代表实现错误。更可能的设计意图是：训练开始前由外部脚本把实际运行配置同步过去。

所以，不能只看某一个配置文件来判断当前真正运行的是哪个算法。

## 外部依赖

这个仓库依赖更大的 Kaiwu 运行环境，包括：

- `kaiwudrl`
- `common_python`
- 各类 `tools` 模块
- shell 脚本，例如：
  - `tools/stop.sh`
  - `tools/modelpool_start.sh`
  - `tools/change_sample_server.sh`
  - `/root/tools/change_algorithm_all.sh`

因此，单独克隆本仓库通常不足以完整跑通训练。

## 单机与分布式训练

根据 `gorge_walk_v2` 官方文档：

- `dynamic_programming`
- `monte_carlo`
- `q_learning`
- `sarsa`

这四种内置算法都只支持单机训练。

如果要做分布式训练，官方预期是从 `agent_diy` 出发自行实现。

这也意味着：

- 官方分布式训练里常见的 `SampleData2NumpyData` / `NumpyData2SampleData` 路径，在这几个内置表格型算法中并未实现
- 想加分布式支持，不是改个配置就够了，而是要补整套接口

## 监控与日志

官方文档把监控分成四类：

- `basic`
- `algorithm`
- `env`
- `diy`

一些重要的内置基础指标包括：

- `train_global_step`
- `predict_succ_cnt`
- `sample_production_and_consumption_ratio`
- `episode_cnt`
- `load_model_succ_cnt`
- `sample_receive_cnt`

对这个仓库的建议是：

- 保留现有 workflow 中的监控上报逻辑
- 不要重写平台日志系统，否则监控面板可能无法正确统计错误日志
- 如果需要自定义指标，优先使用平台提供的 `diy_1` 到 `diy_5`

## 模型保存限制

官方平台限制：

- 最多保存频率：`2 次 / 分钟`
- 总保存次数限制：
  - Dynamic Programming：`10`
  - Monte Carlo：`100`
  - Q-Learning：`100`
  - SARSA：`100`

因此，不要在 workflow 里加入高频模型保存逻辑。

## 评估约束

官方评估规则指出：

- 评估调用的是 `agent.exploit`
- 评估时起点和终点固定：
  - 起点：`[29, 9]`
  - 终点：`[11, 55]`
- 评估时主要可变的是宝箱配置和 `max_step`

## 开发建议

- 如果修改了特征编码，需要同步修改：
  - `STATE_SIZE`
  - Q 表或策略表尺寸
  - `observation_process`
  - 模型持久化格式
- 如果修改 reward shaping，需要一并检查：
  - 收敛逻辑
  - 指标含义
  - `score` 与 `reward` 的区分
- 如果修改 workflow，建议保留：
  - `usr_conf` 处理
  - 样本收集
  - 容灾逻辑
  - 监控上报
  - 周期性模型保存
- 如果新增算法，需要同步更新：
  - `conf/algo_conf_gorge_walk_v2.toml`
  - 运行时切换逻辑
  - 文档

## 当前局限

- 没有完整依赖声明文件，例如 `requirements.txt` 或 `pyproject.toml`
- 外部运行时脚本不在当前仓库中
- `agent_diy` 尚未完成
- 没有发现自包含的自动化测试体系
- 官方模型开发文档主要面向神经网络，而本仓库主体是表格型方法

## 适合怎么用

如果你已经有可用的 Kaiwu 环境，这个仓库适合：

- 对比 `gorge_walk_v2` 上的多种表格型强化学习算法
- 研究官方环境接口如何映射到具体 agent 代码
- 基于模板继续孵化自己的算法

如果你的目标是把它当成独立项目使用，那就需要先补齐缺失的运行时、平台模块和脚本。
