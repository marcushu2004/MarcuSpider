import numpy as np
from sb3_contrib import MaskablePPO
from logic import SpiderEnv


def translate(v: int) -> str:
    mapping = {1: "A", 11: "J", 12: "Q", 13: "K", 0: "空", -1: "盖"}
    return mapping.get(v, str(v))


def build_obs(columns):
    obs = np.zeros((10, 30, 2), dtype=np.int8)
    for i, col in enumerate(columns):
        for j, card in enumerate(col[:30]):
            obs[i, j, 0] = card["val"] if card["face_up"] else -1
            obs[i, j, 1] = 1 if card["face_up"] else 0
    return obs.flatten()


def remove_complete_sequence(columns, col_idx):
    """对齐 logic.py：若目标列末尾形成同花色 K..A(13->1) 共13张则移除，并自动翻牌。单花色 suit 恒为0。"""
    col = columns[col_idx]
    if len(col) < 13:
        return False

    # 末尾必须是A
    if col[-1]["val"] != 1:
        return False

    suit = col[-1]["suit"]
    # 检查倒数13张是否 face_up 且 13..1
    for k in range(13):
        c = col[-1 - k]
        expected = 1 + k
        if (not c["face_up"]) or c["suit"] != suit or c["val"] != expected:
            return False

    # 移除这13张
    columns[col_idx] = col[:-13]

    # 移除后如果顶部是盖牌，真实游戏会翻开（需要你输入点数）
    if columns[col_idx] and not columns[col_idx][-1]["face_up"]:
        columns[col_idx][-1]["face_up"] = True
        return True, True  # removed, flipped_needed
    return True, False


def require_int(prompt: str):
    """强制输入整数（不允许空）"""
    while True:
        s = input(prompt).strip()
        try:
            return int(s)
        except ValueError:
            print("⚠️ 请输入整数。")


def optional_int(prompt: str):
    """允许空回车；返回 None 表示跳过"""
    s = input(prompt).strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        print("⚠️ 输入无效，已当作跳过。")
        return None


def print_top(columns):
    tops = []
    for i in range(10):
        if not columns[i]:
            tops.append("空")
        else:
            c = columns[i][-1]
            tops.append(translate(c["val"]) if c["face_up"] else "盖")
    print("当前各列顶牌：", " | ".join([f"{i}:{t}" for i, t in enumerate(tops)]))


def live_test():
    model = MaskablePPO.load("marcuspider_v2_final")
    columns = [[] for _ in range(10)]

    print("\n=== Marcuspider 智能同步助手 V3（单花色） ===")
    print("初始化：请输入每列“可见的顶牌点数”。若该列看不到牌/为空请输入 0。\n")

    for i in range(10):
        val = require_int(f"第 {i} 列可见点数(0=空): ")
        if val > 0:
            # 单花色蜘蛛：初始每列总牌数 前4列6张、后6列5张，顶牌翻开 => 盖牌数分别为5和4
            cover_count = 5 if i < 4 else 4
            for _ in range(cover_count):
                columns[i].append({"val": -1, "suit": 0, "face_up": False})
            columns[i].append({"val": val, "suit": 0, "face_up": True})

    temp_env = SpiderEnv()
    temp_env.columns = columns

    while True:
        print("\n" + "-" * 60)
        print_top(columns)

        # 构建观察并拿 mask
        obs = build_obs(columns)
        temp_env.columns = columns
        action_masks = temp_env.action_masks()

        action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)

        # 发牌
        if action == 100:
            print("\n" + "!" * 18 + " AI 指令：发牌（每列一张） " + "!" * 18)
            for i in range(10):
                new_v = require_int(f"第 {i} 列发出的新牌点数: ")
                columns[i].append({"val": new_v, "suit": 0, "face_up": True})

            # 发牌后也可能立刻形成 A-K（少见但可处理）
            for i in range(10):
                removed, flipped = remove_complete_sequence(columns, i) if len(columns[i]) >= 13 else (False, False)
                if removed:
                    print(f"✅ 第 {i} 列完成 A-K 序列，已自动收走 13 张。")
                    if flipped:
                        v = require_int(f"第 {i} 列收走后翻开了新牌，请输入点数: ")
                        columns[i][-1] = {"val": v, "suit": 0, "face_up": True}
            continue

        # 移动动作
        src, dest = action // 10, action % 10
        temp_env.columns = columns
        is_valid, movable_seq = temp_env._check_move(src, dest)

        if not is_valid:
            print("\n⚠️ AI 给了非法动作（理论上不该发生，除非内存不同步）。")
            cmd = input("输入 s 强制同步顶牌 / 回车跳过 / q退出: ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                v = require_int(f"请手动输入 第 {src} 列当前顶牌点数(0=空): ")
                if v == 0:
                    columns[src] = []
                else:
                    if columns[src]:
                        columns[src][-1] = {"val": v, "suit": 0, "face_up": True}
                    else:
                        columns[src].append({"val": v, "suit": 0, "face_up": True})
            continue

        num_to_move = len(movable_seq)
        head_val = movable_seq[0]["val"] if movable_seq else -1
        print(f"\n>>> AI 指令：将 第{src}列 从 [{translate(head_val)}] 开始的 {num_to_move} 张牌 移到 第{dest}列")

        # 1) 执行“整段搬运”（对齐 logic.py）
        columns[src] = columns[src][:-num_to_move]
        columns[dest].extend(movable_seq)

        # 2) 源列翻牌同步：若源列顶部现在是盖牌，真实游戏会翻开，需要你输入点数
        if columns[src] and not columns[src][-1]["face_up"]:
            columns[src][-1]["face_up"] = True
            v = require_int(f"第 {src} 列翻开了新牌，请输入点数: ")
            columns[src][-1] = {"val": v, "suit": 0, "face_up": True}
        elif not columns[src]:
            print(f"ℹ️ 第 {src} 列已空。")

        # 3) 目标列完成序列检测：自动收走 A-K
        if len(columns[dest]) >= 13:
            removed, flipped = remove_complete_sequence(columns, dest)
            if removed:
                print(f"✅ 第 {dest} 列完成 A-K 序列，已自动收走 13 张。")
                if flipped:
                    v = require_int(f"第 {dest} 列收走后翻开了新牌，请输入点数: ")
                    columns[dest][-1] = {"val": v, "suit": 0, "face_up": True}

        # 4) 允许你可选地“纠正”源列顶牌（有些情况下你操作时可能发生叠放/自动变化）
        fix = input("如需手动修正某列顶牌，输入列号(0-9)，否则回车继续；q退出: ").strip().lower()
        if fix == "q":
            break
        if fix.isdigit():
            idx = int(fix)
            if 0 <= idx <= 9:
                v = require_int(f"请输入 第 {idx} 列当前顶牌点数(0=空): ")
                if v == 0:
                    columns[idx] = []
                else:
                    if columns[idx]:
                        columns[idx][-1] = {"val": v, "suit": 0, "face_up": True}
                    else:
                        columns[idx].append({"val": v, "suit": 0, "face_up": True})


if __name__ == "__main__":
    live_test()
