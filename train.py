from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
import logic

env = make_vec_env(logic.SpiderEnv, n_envs=8)

checkpoint_callback = CheckpointCallback(
  save_freq=100_000,
  save_path='./models/',
  name_prefix='marcuspider'
)

model = MaskablePPO(
    "MlpPolicy",
    env,
    device="cuda",
    learning_rate=2e-4,
    n_steps=4096,
    batch_size=1024,
    ent_coef=0.01,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
    verbose=1,
    tensorboard_log="./spider_tensorboard/"
)

model.learn(
    total_timesteps=1000000,
    tb_log_name="spider_v2",
    log_interval=1,
    progress_bar=True,
    callback=checkpoint_callback
)
model.save("marcuspider_final")
