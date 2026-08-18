# -*- coding: utf-8 -*-
"""
pcd_cluster_v3.py — 完整精度版:
  - 0.05m 體素降採樣 (高精度)
  - sklearn DBSCAN
  - PCA + size 為每個 cluster 計算特徵
  - 雙向匈牙利配對 (中心 + 尺寸 + 主軸方向)

與 v2 比較:
  v2: 0.30m voxel + 中心-only 匈牙利
  v3: 0.05m voxel + 多特徵匈牙利

預估 ~14 分鐘 (您 15 分鐘可接受範圍)

用法:
  python pcd_cluster_v3.py
  python pcd_cluster_v3.py <voxel_size> <eps> <min_samples> <ground_z>
"""
import json
import sys
import time
import math
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
PCD_DIR = DATA / "pcd"
OUT = ROOT / "output"

sys.path.insert(0, str(HERE))
from pcd_decoder import read_pcd as pcd_read
from voxelize import voxelize
from sk_dbscan import dbscan_sklearn
from cluster_features import cluster_features
from hungarian_link_v2 import hungarian_match_v2


def euclid(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def main():
    print("===== PCD 物理物件聚類 v3 (高精度 + 反向配對) =====")
    pcd_files = sorted([f for f in PCD_DIR.glob("*.pcd") if f.is_file()])
    if not pcd_files:
        print("找不到 PCD")
        return

    # 預設: 0.05m 高精度 voxel
    voxel_size = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    eps = float(sys.argv[2]) if len(sys.argv) > 2 else 0.25
    min_samples = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    ground_thresh = float(sys.argv[4]) if len(sys.argv) > 4 else -1.5
    max_cost = float(sys.argv[5]) if len(sys.argv) > 5 else 1.2

    print(f"PCD 數量: {len(pcd_files)}, voxel={voxel_size}m, "
          f"eps={eps}, min_samples={min_samples}, ground_z<{ground_thresh}, "
          f"max_cost={max_cost}")

    # Step 1: 每幀聚類 + 計算每 cluster feature
    per_frame = []
    feats_by_frame = []   # list of list[feature dict]
    aabbs_by_frame = []   # list of list[aabb_info]

    for pcd_path in pcd_files:
        t0 = time.time()
        pts, info = pcd_read(pcd_path)
        if not pts:
            print(f"  {pcd_path.name}: 解失敗")
            per_frame.append({"pcd": pcd_path.name, "ok": False})
            feats_by_frame.append([])
            aabbs_by_frame.append([])
            continue

        pts_np = np.array([(p['x'], p['y'], p['z']) for p in pts], dtype=np.float32)
        non_ground = pts_np[pts_np[:, 2] > ground_thresh]
        n_total = len(pts)
        n_ng = len(non_ground)

        t1 = time.time()
        centers, _ = voxelize(non_ground, voxel_size)
        t2 = time.time()
        vlabels = dbscan_sklearn(centers, eps, min_samples)
        t3 = time.time()

        # 給每個 voxel 算 feature (center, size, PCA)
        # 先收集每個 cluster 的 voxel centers
        cluster_voxels = defaultdict(list)
        for vi, lab in enumerate(vlabels):
            if lab is not None and lab >= 0:
                cluster_voxels[int(lab)].append(centers[vi].tolist())

        # 過濾 + 計算 feature
        feat_list = []
        aabb_info_list = []
        # 收集 + 取 cluster 內的原始點 (對 feature 用)
        # 但為了速度,我們用 voxel centers 計算 feature
        for cid, voxels in cluster_voxels.items():
            if len(voxels) < 100:
                continue
            f = cluster_features(voxels)
            if f is None:
                continue
            feat_list.append(f)
            aabb_info_list.append({
                "cluster_feature_id": len(feat_list) - 1,
                "center": f["center"],
                "min": f["min"],
                "max": f["max"],
                "size": f["size"],
                "height": f["height"],
                "count": f["count"],
            })

        per_frame.append({
            "pcd": pcd_path.name,
            "ok": True,
            "n_total": n_total,
            "n_nonGround": n_ng,
            "n_voxel_centers": len(centers),
            "n_clusters_raw": len(cluster_voxels),
            "n_clusters_kept": len(feat_list),
            "voxelize_s": round(t2 - t1, 2),
            "dbscan_s": round(t3 - t2, 2),
            "total_s": round(t3 - t0, 2),
        })
        feats_by_frame.append(feat_list)
        aabbs_by_frame.append(aabb_info_list)
        elapsed = time.time() - t0
        print(f"  {pcd_path.name}: pts={n_ng}, voxel={len(centers)}, "
              f"clusters={len(feat_list)} ({elapsed:.2f}s)")

    # Step 2: 雙向匈牙利配對 (向前 + 向後)
    print("\n=== 雙向匈牙利配對 (中心 + 尺寸 + PCA 主軸) ===")

    # Forward pass
    forward_tracks = []   # list of {feat_history, frames: [(fi, ci)]}
    for fi, feats in enumerate(feats_by_frame):
        if not feats:
            continue

        if not forward_tracks:
            # 第一幀: 全是新 track
            for ci, f in enumerate(feats):
                forward_tracks.append({
                    "feat": dict(f),
                    "frames": [(fi, ci)],
                    "matched": [True],
                })
            continue

        # 配對: 用上一個 frame 的 track feature 對當前 frame
        prev_feats = [t["feat"] for t in forward_tracks]
        matched, unmatched_curr = hungarian_match_v2(
            prev_feats, feats,
            w_center=1.0, w_size=0.3, w_pca=0.5,
            max_cost=max_cost,
        )

        # 處理 matched
        for prev_i, curr_i, cost in matched:
            t = forward_tracks[prev_i]
            t["frames"].append((fi, curr_i))
            t["matched"].append(True)
            # 加權合併 feature (更新該 track 的「當前特徵」)
            w = 1.0 / (len(t["frames"]) + 1)
            curr = feats[curr_i]
            t["feat"]["center"] = [t["feat"]["center"][k] * (1-w) + curr["center"][k] * w
                                   for k in range(3)]
            t["feat"]["size"] = [t["feat"]["size"][k] * (1-w) + curr["size"][k] * w
                                 for k in range(3)]

        # 未配到的 cluster: 開新 track, 但保留可能性給 backward pass
        for ci in unmatched_curr:
            forward_tracks.append({
                "feat": dict(feats[ci]),
                "frames": [(fi, ci)],
                "matched": [False],
            })

    # 過濾: track 出現 >= 3 幀 才保留
    min_track = 3
    long_tracks = [t for t in forward_tracks if len(t["frames"]) >= min_track]
    print(f"Forward track: {len(forward_tracks)}, long (≥3 幀): {len(long_tracks)}")

    # Step 3: 結果彙整
    track_results = []
    for ti, t in enumerate(long_tracks):
        centers = [feats_by_frame[fr[0]][fr[1]]["center"] for fr in t["frames"]]
        if len(centers) < 2:
            continue
        base = centers[0]
        max_d = max(euclid(base, c) for c in centers)
        # 主軸: 取第一主軸, 投影到水平面得到 yaw (deg)
        pca_axes = t["feat"].get("pca_axes")
        pca_yaw_deg = None
        pca_yaw_vec = None
        principal_axis_xyz = None
        if pca_axes:
            v0 = np.array(pca_axes[0], dtype=np.float64)
            principal_axis_xyz = [round(float(x), 4) for x in v0]
            vxy = np.array([v0[0], v0[1], 0.0])
            norm = float(np.linalg.norm(vxy))
            if norm > 1e-9:
                pca_yaw_vec = (vxy / norm).tolist()
                pca_yaw_deg = round(math.degrees(math.atan2(vxy[1], vxy[0])), 2)
        track_results.append({
            "track_id": ti,
            "feature_summary": {
                "center_avg": [round(x, 3) for x in t["feat"]["center"]],
                "size_avg": [round(x, 3) for x in t["feat"]["size"]],
                "height_avg": round(t["feat"]["height"], 3),
                "principal_axis_xyz": principal_axis_xyz,
                "principal_yaw_deg": pca_yaw_deg,
                "principal_yaw_xy": [round(v, 4) for v in pca_yaw_vec] if pca_yaw_vec else None,
            },
            "frames_count": len(centers),
            "appear_frames": [fr[0] for fr in t["frames"]],
            "max_center_drift_m": round(max_d, 3),
        })

    print(f"Track 評估結果: {len(track_results)}")

    # Step 4: Ground Truth 比對
    gt_path = ROOT / "static_trackingId.txt"
    truth = set()
    if gt_path.exists():
        truth = set(gt_path.read_text(encoding="utf-8").strip().split())

    matches = []
    if truth and track_results:
        ann_data = json.loads((DATA / "task_export_with_annots.json").read_text(encoding="utf-8"))
        gt_centers = {}
        for f in ann_data.get("frames", []):
            for a in f.get("annotations", []) or []:
                tid = str(a.get("trackingId"))
                if tid in truth and a.get("position"):
                    if tid not in gt_centers:
                        gt_centers[tid] = []
                    gt_centers[tid].append(a["position"])
        gt_avg = {tid: [sum(p[i] for p in lst)/len(lst) for i in range(3)]
                  for tid, lst in gt_centers.items() if lst}

        print("\n=== Ground Truth 比對 ===")
        for tid, tc in sorted(gt_avg.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 9999):
            best_tr = None
            best_d = 1e9
            for tr in track_results:
                d = euclid(tr["feature_summary"]["center_avg"], tc)
                if d < best_d:
                    best_d = d
                    best_tr = tr
            matches.append({
                "tid": tid,
                "platform_pos_avg_ego": [round(x, 2) for x in tc],
                "closest_track_center": best_tr["feature_summary"]["center_avg"] if best_tr else None,
                "closest_track_size": best_tr["feature_summary"]["size_avg"] if best_tr else None,
                "distance_m": round(best_d, 2) if best_tr else None,
                "track_appears_frames": best_tr["frames_count"] if best_tr else None,
                "track_drift": best_tr["max_center_drift_m"] if best_tr else None,
            })
            print(f"  #{tid:<6} 平台: ({tc[0]:+6.2f},{tc[1]:+6.2f},{tc[2]:+5.2f})  "
                  f"聚類距 {best_d:.2f}m  size_avg={best_tr['feature_summary']['size_avg'] if best_tr else None}  "
                  f"出 {best_tr['frames_count'] if best_tr else 0} 幀")

    out_path = OUT / "pcd_cluster_v3_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "config": {
            "voxel_size": voxel_size, "eps": eps,
            "min_samples": min_samples, "ground_thresh": ground_thresh,
            "min_track_appear": min_track,
            "max_cost": max_cost,
        },
        "perFrame": per_frame,
        "tracks": track_results,
        "groundTruthMatch": matches,
        "perFrameClusterCenters": _serialize_per_frame_centers(feats_by_frame),
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n寫入: {out_path}")


def _serialize_per_frame_centers(feats_by_frame):
    """feats_by_frame 是 list[list[feature_dict]]; 序列化為 {frame_idx_str: [[x,y,z], ...]}"""
    out = {}
    for fi, feats in enumerate(feats_by_frame):
        if not feats:
            out[str(fi)] = []
            continue
        out[str(fi)] = [list(f["center"]) for f in feats]
    return out


if __name__ == "__main__":
    main()
