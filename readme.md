# 🕷️ Marcuspider — Spider Solitaire Reinforcement Learning Agent
项目简介 / Project Overview

Marcuspider 是一个基于强化学习的 蜘蛛纸牌（Spider Solitaire，单花色）AI 项目。
该项目使用自定义环境模拟蜘蛛纸牌规则，并通过 Maskable PPO 训练智能体，在合法动作约束下学习完成整局游戏。

Marcuspider is a Reinforcement Learning agent for Spider Solitaire (single-suit).
It implements a custom OpenAI Gym–style environment and trains an agent using Maskable PPO to solve the game under strict rule-based action constraints.

🎯 项目目标 / Goals

实现一个 规则正确、可扩展 的蜘蛛纸牌强化学习环境

在 单花色 Spider Solitaire 设定下训练稳定可用的 AI

支持：

合法动作掩码（Action Masking）

完整序列（A–K）自动判定与移除

翻牌、发牌等真实规则

提供 实战辅助脚本，用于在真实游戏中测试模型决策

🧠 核心方法 / Core Approach

强化学习算法 / RL Algorithm

Maskable PPO（来自 sb3-contrib）

动作掩码确保智能体只选择合法操作

状态表示 / Observation

每列最多 30 张牌

每张牌编码：

点数（1–13，或 -1 表示盖牌）

是否翻开（face_up）

单花色设定下不编码花色（suit）

动作空间 / Action Space

0–99：将第 src 列的可移动序列移动到第 dest 列

100：发牌（每列一张）

🃏 游戏规则建模 / Game Rules Modeled

单花色蜘蛛纸牌（Single-Suit Spider Solitaire）

允许移动：

连续递减

全部翻开

同花色（单花色情况下恒成立）

自动规则处理：

合法移动后自动翻牌

完整 K → A（13 张） 序列自动移除

发牌前必须保证所有列非空

🗂️ 项目结构 / Project Structure
Marcuspider/
├── logic.py              # 强化学习环境（Spider Solitaire 规则核心）
├── train.py              # 模型训练脚本（Maskable PPO）
├── verify_V3.py          # 实战同步测试脚本（与真实游戏交互）
├── verify_real_game.py   # （实验中）真实游戏测试脚本
├── testGPU.py            # GPU 可用性测试
├── models/               # 训练得到的模型（本地生成）
└── README.md

🧪 verify_V3.py 说明 / About verify_V3.py

verify_V3.py 是一个 人机协作测试脚本，用于：

将 AI 的动作决策应用到真实蜘蛛纸牌游戏中

由用户输入真实游戏中的翻牌 / 发牌点数

在不破坏环境逻辑的前提下验证模型行为

⚠️ 注意：该脚本仅用于测试与验证，不影响训练环境 logic.py 的正确性。

🏗️ 当前状态 / Current Status

✅ 单花色 Spider Solitaire 环境规则已完成并验证

✅ 模型可稳定训练并给出合理策略

🚧 实战测试脚本（verify_V3）仍在持续优化中

🔮 未来可扩展至：

多花色 Spider

更精细的奖励设计

完全自动化的真实游戏识别

🚀 未来计划 / Future Work

支持 2 / 4 花色 Spider Solitaire

引入更贴近胜率的奖励函数

自动识别真实游戏画面（CV + RL）

长期策略分析与可视化