import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class SpiderEnv(gym.Env):
    def __init__(self, num_suits=1):
        super(SpiderEnv, self).__init__()

        self.num_suits = num_suits
        # 10列，每列假设最大堆叠30张（保险起见）
        # Observation: (列, 深度, 特征) -> 特征包括: [点数, 是否正面]
        self.observation_space = spaces.Box(
            low=-1, high=13, shape=(600,), dtype=np.int8
        )

        # 动作空间：从 i 列移动到 j 列 (10*10=100) + 发牌 (1)
        self.action_space = spaces.Discrete(101)

        self.reset()

    def get_action_mask(self):
        """
        返回一个长度为 101 的布尔数组
        True 代表合法，False 代表非法
        """
        mask = np.zeros(101, dtype=bool)

        # 检查 100 种移动组合 (0-99)
        for action in range(100):
            src_idx = action // 10
            dest_idx = action % 10
            is_valid, _ = self._check_move(src_idx, dest_idx)
            if is_valid:
                mask[action] = True

        # 检查发牌动作 (100)
        if self._can_deal():
            mask[100] = True

        return mask

    def _create_deck(self):
        # 蜘蛛纸牌共104张牌
        # 单花色：13个点数 * 8组
        deck = []
        full_set = list(range(1, 14)) * 8
        for val in full_set:
            deck.append({'val': val, 'suit': 0, 'face_up': False})
        random.shuffle(deck)
        return deck

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.deck = self._create_deck()
        self.columns = [[] for _ in range(10)]

        # 初始发牌：前4列每列6张，后6列每列5张
        for i in range(10):
            num_cards = 6 if i < 4 else 5
            for _ in range(num_cards):
                card = self.deck.pop()
                self.columns[i].append(card)
            # 每列最后一张翻开
            self.columns[i][-1]['face_up'] = True
        self.last_action = None  # 新增：记录上一个动作

        return self._get_obs(), {}

    def _get_obs(self):
        # 将对象列表转换为 NumPy 矩阵喂给 AI
        obs = np.zeros((10, 30, 2), dtype=np.int8)
        for i, col in enumerate(self.columns):
            for j, card in enumerate(col):
                if j < 30:
                    obs[i, j, 0] = card['val'] if card['face_up'] else -1
                    obs[i, j, 1] = 1 if card['face_up'] else 0
        return obs.flatten()

    def step(self, action):
        # 全局步数税
        # 每一帧都扣除微小分数，逼迫 AI 尽快行动，不磨洋工
        reward = -0.05

        terminated = False
        truncated = False
        info = {"msg": ""}

        # 处理发牌动作 (100)
        if action == 100:
            if self._can_deal():
                self._deal_cards()
                reward += 20.0  # 发牌依然是正面反馈
                self.last_action = 100
            else:
                reward -= 10.0  # 非法发牌重罚
            return self._get_obs(), reward, terminated, truncated, info

        # 解析移动动作
        src_idx = action // 10
        dest_idx = action % 10

        is_valid, movable_seq = self._check_move(src_idx, dest_idx)

        if is_valid:
            # --- 条件 5：检测退回步数并重罚 ---
            if self.last_action is not None and self.last_action < 100:
                last_src = self.last_action // 10
                last_dest = self.last_action % 10
                # 如果把牌从 A 移回 B，而上一步刚从 B 移到 A
                if src_idx == last_dest and dest_idx == last_src:
                    reward -= 15.0  # 给予重罚，打破死循环
                    info["msg"] = "back_forth_penalty"

            # 执行移动
            num_to_move = len(movable_seq)
            src_had_hidden = any(not c['face_up'] for c in self.columns[src_idx])

            self.columns[src_idx] = self.columns[src_idx][:-num_to_move]
            self.columns[dest_idx].extend(movable_seq)

            # 记录有效动作
            self.last_action = action

            # --- 奖励结算 ---
            reward += 1.0  # 基础移动奖

            # 核心奖励 1：翻开隐藏牌 (50.0)
            if self.columns[src_idx] and not self.columns[src_idx][-1]['face_up']:
                self.columns[src_idx][-1]['face_up'] = True
                reward += 50.0

            # 核心奖励 2：创造出空列 (30.0)
            if len(self.columns[src_idx]) == 0 and src_had_hidden:
                reward += 30.0

            # 核心奖励 3：叠放奖 (5.0)
            # 只有目标列原本有牌时才给，鼓励“连接”而非单纯移动到空位
            if len(self.columns[dest_idx]) > len(movable_seq):
                reward += 5.0

            # 核心奖励 4：完成 A-K 序列 (建议提高到 300)
            if self._remove_complete_sequence(dest_idx):
                reward += 300.0
                if all(len(c) == 0 for c in self.columns) and len(self.deck) == 0:
                    reward += 1000.0
                    terminated = True
        else:
            # 非法动作惩罚 (Action Masking 开启时理论上不会触发)
            reward -= 2.0
            self.last_action = None

        self.current_step += 1
        if self.current_step >= 1000:
            truncated = True

        return self._get_obs(), reward, terminated, truncated, info

    def action_masks(self):
        # 这里的逻辑就是你之前实现的掩码逻辑
        mask = np.zeros(101, dtype=bool)
        for action in range(100):
            src_idx = action // 10
            dest_idx = action % 10
            # 这里的 _check_move 是你类里的判定函数
            is_valid, _ = self._check_move(src_idx, dest_idx)
            mask[action] = is_valid

        mask[100] = self._can_deal()
        return mask

    def _check_move(self, src_idx, dest_idx):
        """判断移动合法性并返回可移动的序列"""
        if src_idx == dest_idx: return False, []
        src_col = self.columns[src_idx]
        dest_col = self.columns[dest_idx]

        if not src_col: return False, []

        # 获取源列末尾同花色连续序列
        # 蜘蛛纸牌规则：只有同花色连续递减序列才能整体移动
        movable_seq = []
        for i in range(len(src_col) - 1, -1, -1):
            card = src_col[i]
            if not card['face_up']: break

            if not movable_seq:
                movable_seq.insert(0, card)
            else:
                last_added = movable_seq[0]
                if card['val'] == last_added['val'] + 1 and card['suit'] == last_added['suit']:
                    movable_seq.insert(0, card)
                else:
                    break

        # 目标列检查：
        # 规则：序列的第一张牌必须比目标列最后一张牌小 1（不限花色）
        if not dest_col:
            return True, movable_seq  # 目标为空，随便移
        else:
            if movable_seq[0]['val'] == dest_col[-1]['val'] - 1:
                return True, movable_seq

        return False, []

    def _remove_complete_sequence(self, col_idx):
        """检查并移除 13张连贯同花色的牌"""
        col = self.columns[col_idx]
        if len(col) < 13: return False

        # 检查最后13张是否是同花色 13, 12, ..., 1
        if col[-1]['val'] != 1: return False

        count = 1
        suit = col[-1]['suit']
        for i in range(len(col) - 2, len(col) - 14, -1):
            if col[i]['face_up'] and col[i]['val'] == col[i + 1]['val'] + 1 and col[i]['suit'] == suit:
                count += 1
            else:
                break

        if count == 13:
            self.columns[col_idx] = col[:-13]
            # 移除后可能需要再次翻牌
            if self.columns[col_idx] and not self.columns[col_idx][-1]['face_up']:
                self.columns[col_idx][-1]['face_up'] = True
            return True
        return False

    def _can_deal(self):
        """规则：发牌时所有列不能为空"""
        return len(self.deck) >= 10 and all(len(c) > 0 for c in self.columns)

    def _deal_cards(self):
        for i in range(10):
            card = self.deck.pop()
            card['face_up'] = True
            self.columns[i].append(card)

    def render(self):
        # 简单的字符界面打印，方便 Debug
        for i, col in enumerate(self.columns):
            display = [f"{c['val'] if c['face_up'] else '?'}" for c in col]
            print(f"Col {i}: {' '.join(display)}")
        print(f"Deck remaining: {len(self.deck)}")


if __name__ == "__main__":
    env = SpiderEnv()
    obs, _ = env.reset()

    for step_num in range(1000):
        # 获取合法动作掩码
        mask = env.get_action_mask()
        legal_actions = np.where(mask == True)[0]

        if len(legal_actions) == 0:
            print("无路可走，游戏结束")
            break

        # 随机选一个合法动作
        action = np.random.choice(legal_actions)
        obs, reward, terminated, truncated, info = env.step(action)

        if step_num % 50 == 0:
            print(f"Step: {step_num}, Action: {action}, Reward: {reward}")
            env.render()

        if terminated or truncated:
            print("轮次结束")
            obs, _ = env.reset()