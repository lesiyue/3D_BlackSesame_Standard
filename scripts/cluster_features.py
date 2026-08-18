# -*- coding: utf-8 -*-
"""
cluster_features.py — 對一組點計算形狀描述

  - center, aabb, size, height
  - 點數 count
  - 主軸方向 (PCA eigenvalues, eigenvectors)
"""
import numpy as np


def cluster_features(points):
    """points: list of (x, y, z)
    返回:
      center     (centroid)
      aabb_min, aabb_max
      size       (dx, dy, dz)
      height     (aabb_max.z - aabb_min.z, 由浮動 parquet)
      pca_axes   主軸方向, 3 個 eigenvector
      pca_eigv   三個 eigenvalue
      count      點數
    """
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) == 0:
        return None
    xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
    min_pt = np.array([xs.min(), ys.min(), zs.min()])
    max_pt = np.array([xs.max(), ys.max(), zs.max()])
    center = (min_pt + max_pt) / 2.0
    size = max_pt - min_pt
    height = float(size[2])
    count = len(pts)

    # PCA: 對零中心化點雲算協方差矩陣
    centered = pts - pts.mean(axis=0)
    if len(pts) >= 3:
        try:
            cov = np.cov(centered.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            # 排序: 從大到小
            order = np.argsort(eigvals)[::-1]
            eigvals = eigvals[order]
            eigvecs = eigvecs[:, order]
            pca_axes = eigvecs.T   # 3 行, 每行為一個主軸方向
            pca_eigv = eigvals
        except Exception:
            pca_axes = np.eye(3)
            pca_eigv = np.zeros(3)
    else:
        pca_axes = np.eye(3)
        pca_eigv = np.zeros(3)

    return {
        "center": center.tolist(),
        "min": min_pt.tolist(),
        "max": max_pt.tolist(),
        "size": size.tolist(),
        "height": height,
        "count": count,
        "pca_eigvals": pca_eigv.tolist(),
        "pca_axes": pca_axes.tolist(),
    }
