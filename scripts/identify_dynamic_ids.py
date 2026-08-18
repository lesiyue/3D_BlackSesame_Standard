"""
流程 B：動態優先
根據 ID 框在 task_export_with_annots.json 中跨幀的位置變化判定動/靜
規則（按優先級）：
  1. ID 在 static_trackingId.txt → 強制 static (GT)
  2. ID 在 dynamic_trackingId.txt → 強制 dynamic (GT)
  3. ID 跨幀最大位移 > threshold → dynamic
  4. className 屬於動態傾向類別 → 傾向 dynamic（輔助）
  5. 否則 → static

輸出：output/dynamic_priority_classification.json
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# className 傾向動態的類別（輔助判斷，僅在位移接近閾值時生效）
DYNAMIC_CLASS_HINTS = {
    "Pedestrian", "Similar_pedestrian",
    "Motorcycle_with_rider", "Motorcycle_without_rider",
    "Bicycle_without_rider",
    "Tricycle_with_rider", "Tricycle_without_rider",
    "Stroller",
}


def read_gt_file(path: Path) -> set:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8").strip()
    return set(text.split()) if text else set()


def load_velo_pose(frame):
    """讀取 velo_pose_lidar 4x4 矩陣"""
    return np.array(frame["lioJson"]["velo_pose_lidar"], dtype=np.float64)


def transform_point(T, p):
    """4x4 矩陣變換 3D 點（齊次）"""
    p_homo = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
    return (T @ p_homo)[:3]


def collect_id_positions(task_data):
    """收集每個 ID 在所有幀的世界座標位置
    統一基準：第 0 幀的 ego 座標系
    """
    T0_inv = np.linalg.inv(load_velo_pose(task_data["frames"][0]))
    id_positions = defaultdict(list)  # trackingId -> [(frame_idx, position_world), ...]
    id_classnames = {}

    for frame_idx, frame in enumerate(task_data["frames"]):
        T_k = load_velo_pose(frame)
        T_world = T0_inv @ T_k  # lidar_k → ego_0
        for ann in frame["annotations"]:
            tid = str(ann["trackingId"])
            pos_lidar = np.array(ann["position"], dtype=np.float64)
            pos_world = transform_point(T_world, pos_lidar)
            id_positions[tid].append((frame_idx, pos_world))
            id_classnames[tid] = ann.get("className", "")

    return id_positions, id_classnames


def compute_displacement(positions):
    """計算最大位移（相對於第一個位置）"""
    if len(positions) < 2:
        return 0.0
    first = positions[0][1]
    max_disp = 0.0
    for _, p in positions:
        disp = np.linalg.norm(p - first)
        max_disp = max(max_disp, disp)
    return float(max_disp)


def classify_ids(task_data, threshold, static_gt, dynamic_gt):
    """對每個 ID 進行動/靜判定"""
    id_positions, id_classnames = collect_id_positions(task_data)

    results = []
    static_count = 0
    dynamic_count = 0

    for tid, positions in id_positions.items():
        # 規則 1 & 2：GT 白名單
        if tid in static_gt:
            label = "STATIC"
            reason = "GT 白名單"
        elif tid in dynamic_gt:
            label = "DYNAMIC"
            reason = "GT 白名單"
        else:
            # 規則 3：位移閾值
            max_disp = compute_displacement(positions)
            classname = id_classnames.get(tid, "")
            if max_disp > threshold:
                label = "DYNAMIC"
                reason = f"位移 {max_disp:.3f}m > {threshold}m"
            else:
                # 規則 4：className 輔助（邊界情況，位移接近閾值時觸發）
                edge_low = threshold * 0.5
                if classname in DYNAMIC_CLASS_HINTS and max_disp > edge_low:
                    label = "DYNAMIC"
                    reason = f"位移 {max_disp:.3f}m（className={classname} 輔助）"
                else:
                    label = "STATIC"
                    reason = f"位移 {max_disp:.3f}m ≤ {threshold}m"

        if label == "STATIC":
            static_count += 1
        else:
            dynamic_count += 1

        results.append({
            "trackingId": tid,
            "label": label,
            "reason": reason,
            "max_displacement": compute_displacement(positions),
            "className": id_classnames.get(tid, ""),
            "frame_count": len(positions),
        })

    return results, static_count, dynamic_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="位移閾值（米），> 此值視為動態")
    parser.add_argument("--task-export", type=str, default=None,
                        help="task_export_with_annots.json 路徑")
    parser.add_argument("--output", type=str, default=None,
                        help="輸出 JSON 路徑")
    args = parser.parse_args()

    task_path = Path(args.task_export) if args.task_export else DATA_DIR / "task_export_with_annots.json"
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "dynamic_priority_classification.json"

    if not task_path.exists():
        print(f"❌ 找不到 {task_path}")
        sys.exit(1)

    print(f"▶ 讀取 {task_path}")
    with open(task_path, "r", encoding="utf-8") as f:
        task_data = json.load(f)

    static_gt = read_gt_file(DATA_DIR / "static_trackingId.txt")
    dynamic_gt = read_gt_file(DATA_DIR / "dynamic_trackingId.txt")
    print(f"▶ 靜態 GT: {sorted(static_gt)} ({len(static_gt)} 個)")
    print(f"▶ 動態 GT: {sorted(dynamic_gt)} ({len(dynamic_gt)} 個)")
    print(f"▶ 位移閾值: {args.threshold}m")

    results, static_count, dynamic_count = classify_ids(
        task_data, args.threshold, static_gt, dynamic_gt
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "workflow": "B",
        "method": "dynamic_priority",
        "displacement_threshold": args.threshold,
        "static_gt": sorted(static_gt),
        "dynamic_gt": sorted(dynamic_gt),
        "static_count": static_count,
        "dynamic_count": dynamic_count,
        "total_count": len(results),
        "ids": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ 完成：{static_count} STATIC + {dynamic_count} DYNAMIC = {len(results)} 總計")
    print(f"   輸出：{output_path}")


if __name__ == "__main__":
    main()
