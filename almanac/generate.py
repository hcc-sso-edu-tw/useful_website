#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate.py — Rectangle Filler 關卡產生器

流程：
  1. 依世界設定（盤面大小、灰色格子比例）為每一關均勻隨機放置灰色格子。
  2. 以精確演算法（位元遮罩 + 迭代加深 + 分支定界）求出「最少長方形數量」。
  3. 輸出成 levels.js（window.LEVELS = {...}）。

兩種求解模式（--mode）：
  cover     （預設）長方形可以互相重疊，只要整個非灰色區域被覆蓋即可。
            => 最小集合覆蓋問題，候選集合為「最大空矩形」。
  partition 長方形不可重疊（精確覆蓋 / exact cover）。
            => 最小矩形分割問題，以「凹角 - 最大獨立弦 - 洞 + 1」公式
               配合二分匹配在多項式時間內精確求解。

兩種模式皆使用相同的下界：
  找一組「任兩格都不可能被同一個長方形包住」的格子，
  每一格都需要一個不同的長方形，因此其數量即為下界。

求解沒有時間上限，每一關都會一直搜尋到證明最佳為止。

用法：
  python3 generate.py --seed 42 --output levels.js
"""

import argparse
import json
import os
import random
import sys
import time
from multiprocessing import Pool

# --------------------------------------------------------------------------
# 世界設定
# --------------------------------------------------------------------------

WORLDS = [
    {"name": "Meadow",      "size": 6,  "icon": "🌿"},
    {"name": "Forest",      "size": 7,  "icon": "🌲"},
    {"name": "Canyon",      "size": 8,  "icon": "🏜️"},
    {"name": "Glacier",     "size": 9,  "icon": "🧊"},
    {"name": "Volcano",     "size": 10, "icon": "🌋"},
    {"name": "Lagoon",      "size": 11, "icon": "🏝️"},
    {"name": "Savannah",    "size": 12, "icon": "🦁"},
    {"name": "Tundra",      "size": 13, "icon": "❄️"},
    {"name": "Nebula",      "size": 14, "icon": "🌌"},
    {"name": "Singularity", "size": 15, "icon": "🕳️"},
]
LEVELS_PER_WORLD = 20

# int.bit_count 需要 Python 3.10+，舊版本退回字串計數
if hasattr(int, "bit_count"):
    def popcount(x):
        return x.bit_count()
else:
    def popcount(x):
        return bin(x).count("1")


# --------------------------------------------------------------------------
# 灰色格子的產生
# --------------------------------------------------------------------------

def gray_ratio(world_index):
    """World 1 為 1/14，World 10 為 1/12，中間線性內插。"""
    t = world_index / (len(WORLDS) - 1)
    return 1 / 14 + (1 / 12 - 1 / 14) * t


def gray_count(size, world_index):
    """依世界比例計算灰色格子數量（至少 1 個）。"""
    return max(1, round(size * size * gray_ratio(world_index)))


def place_gray_cells(size, count, rng):
    """
    以「最佳候選點法 (Mitchell's best-candidate)」放置灰色格子：
    每放一格，先隨機抽出 m 個候選點，選離既有灰格最遠者。
      - m 每關隨機（3~10），小 m 較隨機、大 m 較均勻，增加關卡多樣性。
      - 優先選擇不與既有灰格相鄰（含斜角）的位置，且不會重複。
    回傳排序後的 [(r, c), ...]。
    """
    cells = [(r, c) for r in range(size) for c in range(size)]
    m = rng.randint(3, 10)
    chosen = [rng.choice(cells)]
    chosen_set = set(chosen)

    while len(chosen) < count:
        # 允許的位置：與已選格子的切比雪夫距離 >= 2（不相鄰）
        allowed = [p for p in cells
                   if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 2
                          for q in chosen)]
        if not allowed:  # 極端情況：放寬為「只要不重複」
            allowed = [p for p in cells if p not in chosen_set]
        candidates = rng.sample(allowed, min(m, len(allowed)))

        def min_dist2(p):
            return min((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 for q in chosen)

        best = max(candidates, key=min_dist2)
        chosen.append(best)
        chosen_set.add(best)

    return sorted(chosen)


def canonical_form(grays, size):
    """
    回傳灰格配置在 8 種旋轉/翻轉下的最小表示，
    用來避免產生「只是旋轉或鏡射」的重複關卡。
    """
    forms = []
    for t in range(8):
        pts = []
        for r, c in grays:
            if t & 4:
                r, c = c, r
            if t & 1:
                r = size - 1 - r
            if t & 2:
                c = size - 1 - c
            pts.append((r, c))
        forms.append(tuple(sorted(pts)))
    return min(forms)


# --------------------------------------------------------------------------
# 矩形枚舉與位元遮罩工具
# --------------------------------------------------------------------------

def bits(x):
    """依序產生整數 x 中所有為 1 的位元索引。"""
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def rect_mask(size, r1, c1, r2, c2):
    """把矩形 (r1,c1)-(r2,c2)（含端點）轉成位元遮罩，格子索引 = r*size + c。"""
    row = ((1 << (c2 - c1 + 1)) - 1) << c1
    m = 0
    for r in range(r1, r2 + 1):
        m |= row << (r * size)
    return m


def enumerate_rects(size, grays, maximal_only):
    """
    枚舉盤面上所有不含灰格的矩形，回傳 [(r1,c1,r2,c2), ...]。
    maximal_only=True 時只保留「最大空矩形」（四個方向都無法再擴張）。
    利用二維前綴和 O(1) 判斷矩形是否為空。
    """
    blocked = [[0] * size for _ in range(size)]
    for r, c in grays:
        blocked[r][c] = 1
    ps = [[0] * (size + 1) for _ in range(size + 1)]
    for r in range(size):
        for c in range(size):
            ps[r + 1][c + 1] = (blocked[r][c] + ps[r][c + 1]
                                + ps[r + 1][c] - ps[r][c])

    def empty(r1, c1, r2, c2):
        return (ps[r2 + 1][c2 + 1] - ps[r1][c2 + 1]
                - ps[r2 + 1][c1] + ps[r1][c1]) == 0

    out = []
    for r1 in range(size):
        for r2 in range(r1, size):
            for c1 in range(size):
                for c2 in range(c1, size):
                    if not empty(r1, c1, r2, c2):
                        break  # c2 再變大也一定不空
                    if maximal_only:
                        if r1 > 0 and empty(r1 - 1, c1, r2, c2):
                            continue
                        if r2 < size - 1 and empty(r1, c1, r2 + 1, c2):
                            continue
                        if c1 > 0 and empty(r1, c1 - 1, r2, c2):
                            continue
                        if c2 < size - 1 and empty(r1, c1, r2, c2 + 1):
                            continue
                    out.append((r1, c1, r2, c2))
    return out


def free_universe(size, grays):
    """所有非灰格的位元遮罩（需被覆蓋的格子集合）。"""
    gray_set = set(grays)
    u = 0
    for r in range(size):
        for c in range(size):
            if (r, c) not in gray_set:
                u |= 1 << (r * size + c)
    return u


# --------------------------------------------------------------------------
# 下界：互斥格子集合
# --------------------------------------------------------------------------

def make_lower_bound(masks, universe):
    """
    建立下界函式 lb(unc)。
    conflict[c] = 所有包含格子 c 的矩形所覆蓋的格子聯集。
    若 d 不在 conflict[c] 中，則沒有任何矩形能同時包住 c 與 d。
    對剩餘未覆蓋格子 unc，貪婪地挑出一組彼此互斥的格子，
    其數量即為還需要的最少矩形數的下界。
    """
    conflict = {c: 0 for c in bits(universe)}
    for m in masks:
        for c in bits(m & universe):
            conflict[c] |= m

    def lb(unc):
        rem = unc
        cnt = 0
        while rem:
            best_c, best_k = -1, 1 << 30
            for c in bits(rem):
                k = popcount(conflict[c] & rem)
                if k < best_k:
                    best_c, best_k = c, k
                    if k == 1:  # 只和自己衝突，不可能更小
                        break
            cnt += 1
            rem &= ~conflict[best_c]
        return cnt

    return lb


# --------------------------------------------------------------------------
# 模式一：最小集合覆蓋（長方形可重疊）
# --------------------------------------------------------------------------

def reduce_instance(masks, universe):
    """
    集合覆蓋的化簡（保證不改變最佳解大小）：
      1. 格子支配：若覆蓋格子 a 的矩形集合 ⊆ 覆蓋格子 b 的矩形集合，
         則只要覆蓋了 a 就必定覆蓋 b，b 可以刪除。
      2. 矩形支配：限縮到剩餘格子後，被其他矩形包含的矩形可以刪除。
    反覆進行直到不再變化。
    """
    masks = list(set(masks))
    while True:
        changed = False
        cells = list(bits(universe))
        # cs[c] = 覆蓋格子 c 的矩形索引集合（以位元遮罩表示）
        cs = {c: 0 for c in cells}
        for j, m in enumerate(masks):
            for c in bits(m & universe):
                cs[c] |= 1 << j

        removed = 0
        for b in cells:
            for a in cells:
                if a == b:
                    continue
                ca, cb = cs[a], cs[b]
                if ca & cb == ca and (ca != cb or a < b):
                    removed |= 1 << b
                    break
        if removed:
            universe &= ~removed
            changed = True

        restricted = [m for m in set(m & universe for m in masks) if m]
        restricted.sort(key=popcount, reverse=True)
        kept = []
        for m in restricted:
            if not any(m & k == m for k in kept):
                kept.append(m)
        if len(kept) != len(masks):
            changed = True
        masks = kept
        if not changed:
            return masks, universe


def greedy_cover(masks, universe):
    """貪婪法求集合覆蓋的上界：每次選能覆蓋最多未覆蓋格子的矩形。"""
    unc, cnt = universe, 0
    while unc:
        best = max(masks, key=lambda m: popcount(m & unc))
        unc &= ~best
        cnt += 1
    return cnt


def min_cover(masks, universe):
    """
    精確求最小集合覆蓋，回傳最少矩形數。
    作法：迭代加深 —— 從下界 k 開始逐步嘗試，第一個可行的 k 就是最佳解。
    DFS 每層挑「候選矩形最少」的未覆蓋格子來分支，並用下界剪枝、
    用 failed 表記憶「此狀態在 k 步內無解」。
    """
    lb_fn = make_lower_bound(masks, universe)
    opts = {c: [i for i, m in enumerate(masks) if m >> c & 1]
            for c in bits(universe)}
    ub = greedy_cover(masks, universe)
    lo = lb_fn(universe)
    failed = {}          # unc -> 已知「用 k 個矩形無法覆蓋」的最大 k

    def dfs(unc, k):
        if unc == 0:
            return True
        if k == 0:
            return False
        if failed.get(unc, -1) >= k:
            return False
        if lb_fn(unc) > k:
            failed[unc] = k
            return False
        # 分支格子：候選矩形最少者
        pick = min(bits(unc), key=lambda c: len(opts[c]))
        cands = sorted(opts[pick], key=lambda i: -popcount(masks[i] & unc))
        for i in cands:
            if dfs(unc & ~masks[i], k - 1):
                return True
        failed[unc] = k
        return False

    for k in range(lo, ub):
        if dfs(universe, k):
            return k
    return ub


def solve_cover(size, grays):
    """集合覆蓋模式：以最大空矩形為候選，化簡後求解。"""
    rects = enumerate_rects(size, grays, maximal_only=True)
    masks = [rect_mask(size, *r) for r in rects]
    masks, universe = reduce_instance(masks, free_universe(size, grays))
    return min_cover(masks, universe)


# --------------------------------------------------------------------------
# 模式二：最小矩形分割（長方形不可重疊）
# --------------------------------------------------------------------------

def partition_by_chords(size, grays):
    """
    以多項式時間公式求最小矩形分割（Lipski / Ohtsuki 定理）：
        最少矩形數 = R - G - H + 1
      R：凹角（反射頂點）數，即某格點四周 4 格中恰好 1 格被擋（灰格或盤外）。
      G：互不相交的「弦」最多有幾條。弦是連接兩個同線凹角、且完全穿過
         空白區內部的水平/垂直線段。水平弦與垂直弦相交形成二分圖，
         最大獨立集 = 弦總數 - 最大匹配（König 定理）。
      H：洞的數量（未與盤面邊界相連的灰格群）。
    定理只適用於「非退化」的區域：灰格不可只以對角相接，且空白區必須連通。
    若遇到退化情形則回傳 None，由呼叫者改用搜尋法。
    """
    gs = set(grays)

    def blocked(r, c):
        return r < 0 or c < 0 or r >= size or c >= size or (r, c) in gs

    # 凹角；同時偵測對角相接的退化情形
    reflex = []
    for y in range(size + 1):
        for x in range(size + 1):
            b = [blocked(y - 1, x - 1), blocked(y - 1, x),
                 blocked(y, x - 1), blocked(y, x)]
            k = sum(b)
            if k == 2 and ((b[0] and b[3]) or (b[1] and b[2])):
                return None
            if k == 1:
                reflex.append((y, x))

    def components(cells_pred, lo, hi):
        """在 [lo,hi)^2 範圍內以 4 連通計算滿足條件的格子群數。"""
        seen, comps = set(), 0
        for r in range(lo, hi):
            for c in range(lo, hi):
                if (r, c) in seen or not cells_pred(r, c):
                    continue
                comps += 1
                stack = [(r, c)]
                seen.add((r, c))
                while stack:
                    cr, cc = stack.pop()
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc),
                                   (cr, cc + 1), (cr, cc - 1)):
                        if (lo <= nr < hi and lo <= nc < hi
                                and (nr, nc) not in seen
                                and cells_pred(nr, nc)):
                            seen.add((nr, nc))
                            stack.append((nr, nc))
        return comps

    # 空白區必須連通
    if components(lambda r, c: not blocked(r, c), 0, size) != 1:
        return None
    # 洞數 = 含外框的擋格群數 - 1（外框自成一群）
    holes = components(blocked, -1, size + 1) - 1

    # 弦：同一水平線(或垂直線)上兩個凹角之間，兩側的格子全為空白
    horiz, vert = [], []
    for i, (y1, x1) in enumerate(reflex):
        for (y2, x2) in reflex[i + 1:]:
            if y1 == y2:
                xa, xb = sorted((x1, x2))
                if all(not blocked(y1 - 1, x) and not blocked(y1, x)
                       for x in range(xa, xb)):
                    horiz.append((y1, xa, xb))
            elif x1 == x2:
                ya, yb = sorted((y1, y2))
                if all(not blocked(y, x1 - 1) and not blocked(y, x1)
                       for y in range(ya, yb)):
                    vert.append((x1, ya, yb))

    # 水平弦與垂直弦（含端點接觸）相交 => 二分圖的邊
    adj = [[j for j, (x, ya, yb) in enumerate(vert)
            if xa <= x <= xb and ya <= y <= yb]
           for (y, xa, xb) in horiz]
    match_v = [-1] * len(vert)

    def augment(u, visited):
        for v in adj[u]:
            if v in visited:
                continue
            visited.add(v)
            if match_v[v] == -1 or augment(match_v[v], visited):
                match_v[v] = u
                return True
        return False

    matching = sum(augment(u, set()) for u in range(len(horiz)))
    max_independent = len(horiz) + len(vert) - matching
    return len(reflex) - max_independent - holes + 1


def solve_partition(size, grays):
    """
    不可重疊模式的求解：優先使用弦公式（瞬間完成、必為最佳）；
    遇到退化配置才改用下方的搜尋法。
    """
    result = partition_by_chords(size, grays)
    if result is not None:
        return result
    return solve_partition_search(size, grays)


def solve_partition_search(size, grays):
    """
    以搜尋精確求最小矩形分割（退化情形的備援），回傳最少矩形數。
    列優先順序下第一個未覆蓋的格子，必定是某個矩形的左上角，
    因此每層只需枚舉「左上角在該格」且完全落在未覆蓋區內的矩形。
    下界仍使用最大空矩形的互斥格子法（對分割同樣成立）。
    """
    universe = free_universe(size, grays)
    max_masks = [rect_mask(size, *r)
                 for r in enumerate_rects(size, grays, maximal_only=True)]
    lb_fn = make_lower_bound(max_masks, universe)

    # 依左上角分組的所有矩形，面積大者優先
    by_tl = {}
    for r1, c1, r2, c2 in enumerate_rects(size, grays, maximal_only=False):
        area = (r2 - r1 + 1) * (c2 - c1 + 1)
        by_tl.setdefault(r1 * size + c1, []).append(
            (area, rect_mask(size, r1, c1, r2, c2)))
    for lst in by_tl.values():
        lst.sort(key=lambda t: -t[0])
        lst[:] = [m for _, m in lst]

    # 貪婪上界：每次在第一個空格放最大且不重疊的矩形
    unc, ub = universe, 0
    while unc:
        c = (unc & -unc).bit_length() - 1
        for m in by_tl[c]:
            if m & unc == m:
                unc &= ~m
                ub += 1
                break

    lo = lb_fn(universe)
    failed = {}

    def dfs(unc, k):
        if unc == 0:
            return True
        if k == 0:
            return False
        if failed.get(unc, -1) >= k:
            return False
        if lb_fn(unc) > k:
            failed[unc] = k
            return False
        c = (unc & -unc).bit_length() - 1
        for m in by_tl[c]:
            if m & unc == m and dfs(unc ^ m, k - 1):
                return True
        failed[unc] = k
        return False

    for k in range(lo, ub):
        if dfs(universe, k):
            return k
    return ub


# --------------------------------------------------------------------------
# 單關求解（供 multiprocessing 呼叫）與主程式
# --------------------------------------------------------------------------

def solve_task(task):
    """解一關。task = (world, level, size, grays, mode)。"""
    world, level, size, grays, mode = task
    start = time.monotonic()
    solver = solve_cover if mode == "cover" else solve_partition
    steps = solver(size, grays)
    return world, level, steps, time.monotonic() - start


def build_levels(seed):
    """
    產生全部關卡的盤面（不含 minimumSteps）。
    每關使用由 (seed, world, level) 衍生的獨立亂數，結果可重現；
    同一世界內若出現（含旋轉/鏡射）重複配置會重抽。
    """
    levels = []
    for w, world in enumerate(WORLDS):
        size = world["size"]
        count = gray_count(size, w)
        seen = set()
        for l in range(1, LEVELS_PER_WORLD + 1):
            rng = random.Random(f"{seed}:{w}:{l}")
            while True:
                grays = place_gray_cells(size, count, rng)
                key = canonical_form(grays, size)
                if key not in seen:
                    seen.add(key)
                    break
            levels.append({"world": w, "level": l, "size": size,
                           "grayCells": [list(p) for p in grays]})
    return levels


def write_js(path, levels):
    """把結果寫成 window.LEVELS = {...}; 的 JavaScript 檔。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write("window.LEVELS = {\n  worlds: [\n")
        f.write(",\n".join("    " + json.dumps(w, ensure_ascii=False)
                           for w in WORLDS))
        f.write("\n  ],\n  levels: [\n")
        rows = []
        for lv in levels:
            obj = {"world": lv["world"], "level": lv["level"],
                   "size": lv["size"], "grayCells": lv["grayCells"],
                   "minimumSteps": lv["minimumSteps"]}
            rows.append("    " + json.dumps(obj, separators=(",", ":")))
        f.write(",\n".join(rows))
        f.write("\n  ]\n};\n")


def main():
    ap = argparse.ArgumentParser(description="Rectangle Filler 關卡產生器")
    ap.add_argument("--seed", type=int, default=42, help="亂數種子（預設 42）")
    ap.add_argument("--output", default="levels.js",
                    help="輸出檔名（預設 levels.js）")
    ap.add_argument("--mode", choices=["cover", "partition"], default="cover",
                    help="cover=長方形可重疊（預設）；partition=不可重疊")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1,
                    help="平行處理的行程數（預設為 CPU 數）")
    args = ap.parse_args()

    levels = build_levels(args.seed)
    tasks = [(lv["world"], lv["level"], lv["size"],
              [tuple(p) for p in lv["grayCells"]], args.mode)
             for lv in levels]
    index = {(lv["world"], lv["level"]): lv for lv in levels}

    t0 = time.monotonic()
    done = 0

    def record(res):
        """寫回一關的結果並顯示進度。"""
        nonlocal done
        w, l, steps, dt = res
        index[(w, l)]["minimumSteps"] = steps
        done += 1
        if done % 20 == 0 or dt > 30:
            print(f"  已完成 {done}/{len(tasks)} 關"
                  f"（{time.monotonic() - t0:.1f}s）", file=sys.stderr)

    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            for res in pool.imap(solve_task, tasks, chunksize=1):
                record(res)
    else:
        for task in tasks:
            record(solve_task(task))

    write_js(args.output, levels)
    print(f"完成：{len(levels)} 關 → {args.output}"
          f"（模式 {args.mode}，耗時 {time.monotonic() - t0:.1f}s，皆為最佳解）",
          file=sys.stderr)


if __name__ == "__main__":
    main()