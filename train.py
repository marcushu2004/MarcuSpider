import gymnasium as gym
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback # 引入保存回调
import logic

# 1. 创建环境
env = logic.SpiderEnv()

# 2. 定义保存回调：每 20,000 步自动存一次模型
checkpoint_callback = CheckpointCallback(
  save_freq=100000,
  save_path='./models/',
  name_prefix='marcuspider_v2'
)

# 3. 定义模型
model = MaskablePPO(
    "MlpPolicy",
    env,
    learning_rate=1e-4,
    n_steps=2048,
    batch_size=128,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
    verbose=1,
    tensorboard_log="./spider_tensorboard/"
)

# 4. 开始训练 (加上 callback)
print("开始训练，模型将自动保存到 ./models/ 文件夹...")
try:
    model.learn(
        total_timesteps=1000000,
        progress_bar=True,
        log_interval=10,
        callback=checkpoint_callback # 【关键】绑定保存逻辑
    )
except KeyboardInterrupt:
    print("检测到手动停止，正在紧急保存当前权重...")
    model.save("marcuspider_v2_emergency_stop")

# 5. 正常结束保存
model.save("marcuspider_v2_final")