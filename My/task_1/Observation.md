# Training Profile
q_learing aglo
step penalty -1
wall-hit penalty -10
no treasure
LEARNING_RATE = 0.8
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 10000

# Training Metrics
## 分数统计
![alt text](image.png)
宝藏和通关分数
在初始轮为270多分，训练结束后上升到470多分

## 步数统计
![alt text](image-1.png)
从开始的1300多步，在中间某轮突然下降到346步然后趋于平稳
从来没有撞到2000 step 的threshold

## Reward
![alt text](image-2.png)
从初轮的-2300 稳步上升至341

# Training Time
最大时长30mins
实际只花了1min

# Training Result
训练收敛
251 episodes


# Training Evaluation
测试超时，不停撞墙


# Training Profile
q_learing aglo
step penalty -1
wall-hit penalty -10
5 random treasure
LEARNING_RATE = 0.8
GAMMA = 0.9
EPSILON = 0.1
EPISODES = 10000

# Training Metrics
![alt text](image-3.png)
![alt text](image-4.png)
## 分数统计
总分和宝箱得分都震荡缓慢上升
初始总分300
宝箱100
结束总分370
宝箱200

## 最大步数反复撞2000steps threshold

## 收集宝箱的数量
震荡上升
1.05 -> 2.15

## Reward
震荡上升
-2300 -> -1600

# Training Time
最大时长30mins
实际只花了30mins

# Training Result
训练Not收敛
训练了约2500000轮

# Training Evaluation
测试超时，不停撞墙

