"""
叠幀精算 ID 尺寸
從指定幀段讀取 pcd 點雲和 ID 框，按 velo_pose_lidar 疊加到第 0 幀 ego 座標系
對每個 GT ID 用 OBB 計算 length/width/height
結果寫到 output/id_size_refined.json

scale 保持原始 float64 精度，不做 round。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PCD_DIR = DATA_DIR / "pcd"
OUTPUT_DIR = ROOT / "output"


def read_gt_file(path: Path) -> set:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8").strip()
    return set(text.split()) if text else set()


def load_velo_pose(frame):
    return np.array(frame["lioJson"]["velo_pose_lidar"], dtype=np.float64)


def get_velo_pose_at_frame(task_data, frame_idx):
    """velo_pose_lidar 在所有 frame 中都是同一個 15 個矩陣的列表，取第 frame_idx 個"""
    return np.array(task_data["frames"][0]["lioJson"]["velo_pose_lidar"][frame_idx], dtype=np.float64)


def read_pcd(path: Path) -> np.ndarray:
    """讀取 pcd 文件，返回 Nx3 numpy 數組"""
    pcd = o3d.io.read_point_cloud(str(path))
    return np.asarray(pcd.points, dtype=np.float64)


def compute_obb_scale(points: np.ndarray):
    """用 open3d 計算 OBB，返回 (length, width, height) 順序與 extent 一致"""
    if len(points) < 10:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_oriented_bounding_box()
    extent = obb.extent  # (length, width, height) in OBB's local frame
    return (float(extent[0]), float(extent[1]), float(extent[2]))


def get_id_box_at_frame(frame, tid):
    """從 frame.annotations 中找指定 ID 的框"""
    for ann in frame["annotations"]:
        if str(ann["trackingId"]) == str(tid):
            return ann
    return None


def collect_gt_points(task_data, gt_ids, frame_range):
    """對每個 GT ID，收集其在指定幀段內、ID 框附近的點雲
    半徑策略：取第一幀原始 scale 對角線 × 2，但上限 8m（避免被錯亂框放大）
    統一座標系：第 0 幀的 ego
    """
    T0_inv = np.linalg.inv(get_velo_pose_at_frame(task_data, 0))
    select_frame = task_data["frames"][0]["lioJson"]["select_frame"]

    # 先取每個 ID 的基準 scale（第一幀）
    base_scales = {}
    for tid in gt_ids:
        for fi in range(len(task_data["frames"])):
            ann = get_id_box_at_frame(task_data["frames"][fi], tid)
            if ann is not None:
                base_scales[tid] = np.array(ann["scale"], dtype=np.float64)
                break

    id_points = {tid: [] for tid in gt_ids}

    for frame_idx in frame_range:
        frame = task_data["frames"][frame_idx]
        T_k = get_velo_pose_at_frame(task_data, frame_idx)
        T_world = T0_inv @ T_k  # lidar_k → ego_0

        pcd_num = select_frame[frame_idx]
        pcd_path = PCD_DIR / f"{int(pcd_num):05d}.pcd"
        if not pcd_path.exists():
            print(f"  警告：第 {frame_idx} 幀 pcd 不存在 ({pcd_path})")
            continue

        pts_lidar = read_pcd(pcd_path)
        # 點雲 lidar_k → ego_0
        pts_homo = np.hstack([pts_lidar, np.ones((len(pts_lidar), 1))])
        pts_world = (T_world @ pts_homo.T).T[:, :3]

        for tid in gt_ids:
            ann = get_id_box_at_frame(frame, tid)
            if ann is None:
                continue
            center_lidar = np.array(ann["position"], dtype=np.float64)
            center_world = (T_world @ np.array([center_lidar[0], center_lidar[1], center_lidar[2], 1.0]))[:3]
            # 半徑：基準 scale 對角線 × 2，上限 8m，下限 1.5m
            base_scale = base_scales.get(tid, np.array([1.0, 1.0, 1.0]))
            diagonal = float(np.linalg.norm(base_scale))
            radius = min(diagonal * 2.0, 8.0)
            radius = max(radius, 1.5)

            dists = np.linalg.norm(pts_world - center_world, axis=1)
            mask = dists < radius
            id_points[tid].append(pts_world[mask])

    return id_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True, help="起始幀索引")
    parser.add_argument("--end", type=int, required=True, help="結束幀索引（含）")
    parser.add_argument("--task-export", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-ratio", type=float, default=3.0,
                        help="OBB scale 超過原始 × 此值視為異常，回退原始")
    args = parser.parse_args()

    task_path = Path(args.task_export) if args.task_export else DATA_DIR / "task_export_with_annots.json"
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "id_size_refined.json"

    if not task_path.exists():
        print(f"找不到 {task_path}")
        sys.exit(1)

    print(f"讀取 {task_path}")
    with open(task_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    gt_ids = read_gt_file(DATA_DIR / "static_trackingId.txt")
    if not gt_ids:
        print("static_trackingId.txt 為空，無 GT 可處理")
        sys.exit(1)

    frame_range = list(range(args.start, args.end + 1))
    total_frames = len(task_data["frames"])
    frame_range = [f for f in frame_range if 0 <= f < total_frames]
    if not frame_range:
        print(f"錯誤：幀段 {args.start}-{args.end} 超出範圍 (0-{total_frames-1})")
        sys.exit(1)
    print(f"GT IDs: {sorted(gt_ids)}")
    print(f"叠幀幀段: {frame_range}（總幀數 {total_frames}）")

    id_points = collect_gt_points(task_data, gt_ids, frame_range)

    # 取每個 ID 的基準 scale（用於合理性檢查）
    base_scales = {}
    for tid in gt_ids:
        for fi in range(total_frames):
            ann = get_id_box_at_frame(task_data["frames"][fi], tid)
            if ann is not None:
                base_scales[tid] = np.array(ann["scale"], dtype=np.float64)
                break

    results = {}
    for tid in sorted(gt_ids):
        pts_list = id_points[tid]
        if not pts_list:
            print(f"  ⚠ ID {tid} 在指定幀段無點雲")
            continue
        pts = np.vstack(pts_list)
        print(f"  ID {tid}: {len(pts)} 點")
        scale = compute_obb_scale(pts)
        if scale is None:
            print(f"  ⚠ ID {tid} 點數不足，跳過")
            continue

        # 合理性檢查：超過原始 ×max_ratio 視為異常，回退原始
        base = base_scales.get(tid)
        outlier = False
        if base is not None:
            for i in range(3):
                if scale[i] > base[i] * args.max_ratio or scale[i] < base[i] / args.max_ratio:
                    outlier = True
                    break
        if outlier and base is not None:
            print(f"  → scale 異常 ({scale[0]:.3f}, {scale[1]:.3f}, {scale[2]:.3f})，回退原始 ({base[0]:.3f}, {base[1]:.3f}, {base[2]:.3f})")
            final_scale = (float(base[0]), float(base[1]), float(base[2]))
        else:
            final_scale = scale
            print(f"  → scale = ({scale[0]:.6f}, {scale[1]:.6f}, {scale[2]:.6f})")

        results[tid] = {
            "trackingId": tid,
            "scale": [final_scale[0], final_scale[1], final_scale[2]],
            "point_count": int(len(pts)),
            "frame_range": [args.start, args.end],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"完成：{len(results)}/{len(gt_ids)} 個 ID 尺寸已精算")
    print(f"輸出：{output_path}")


if __name__ == "__main__":
    main()
