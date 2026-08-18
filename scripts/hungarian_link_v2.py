# -*- coding: utf-8 -*-
"""
hungarian_link_v2.py — 升級版匈牙利配對:加入「外觀簽名」約束

配對時同時考慮:
  - 中心距離
  - 物理尺寸 (size)
  - 主軸方向 (PCA 第一主軸方向)

設計:
  cost[i, j] = α * 中心距離 + β * 尺寸差 + γ * 主軸方向差
  超過 max_cost 視為拒絕配對
"""
import numpy as np
import math
from scipy.optimize import linear_sum_assignment


def feat_distance(feat1, feat2):
    """綜合距離:
       - 中心 euclid = feat.center
       - size L1 = |feat1.size - feat2.size|
       - PCA 第一軸 cosine distance = 1 - |<v1, v2>|
    """
    c1 = feat1["center"]
    c2 = feat2["center"]
    d_cent = math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2)

    s1 = feat1["size"]
    s2 = feat2["size"]
    d_size = (abs(s1[0]-s2[0]) + abs(s1[1]-s2[1]) + abs(s1[2]-s2[2])) / 3.0

    p1 = np.array(feat1["pca_axes"][0])
    p2 = np.array(feat2["pca_axes"][0])
    cos = float(np.dot(p1, p2))
    d_pca = 1.0 - abs(cos)   # 0..1, 0 表示同向

    return d_cent, d_size, d_pca


def hungarian_match_v2(prev_features, curr_features,
                        w_center=1.0, w_size=0.3, w_pca=0.5,
                        max_cost=2.5):
    """
    prev_features, curr_features: list of dict (含 center, size, pca_axes)
    返回:
      matched: list of (prev_idx, curr_idx, cost_dict)
      unmatched_curr: list of curr_idx
    """
    if not prev_features or not curr_features:
        return [], list(range(len(curr_features)))

    n = len(prev_features)
    m = len(curr_features)
    cost = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            d_c, d_s, d_p = feat_distance(prev_features[i], curr_features[j])
            c = w_center * d_c + w_size * d_s + w_pca * d_p
            if c > max_cost:
                c = 1e9
            cost[i, j] = c

    row, col = linear_sum_assignment(cost)

    matched = []
    for r, c in zip(row, col):
        if float(cost[r, c]) <= max_cost:
            d_c, d_s, d_p = feat_distance(prev_features[r], curr_features[c])
            matched.append((r, c, {
                "cost": float(cost[r, c]),
                "d_center": d_c,
                "d_size": d_s,
                "d_pca": d_p,
            }))

    matched_curr = {c for _, c, _ in matched}
    unmatched_curr = [c for c in range(m) if c not in matched_curr]
    return matched, unmatched_curr
