# -*- coding: utf-8 -*-
"""
ego_motion_estimator.py — 從 box 軌跡估算 ego-motion (自車運動)

核心假設:
  - 多數標註框是世界靜止物體
  - 它們在 lidar-frame 下的位移 = -ego_motion
  - 用所有 box 的 frame-to-frame 位移取中位數/眾數來估算

輸入: task_export_with_annots.json (frames[].annotations[].position)
輸出: 每幀的 ego_motion 向量 (dx, dy, dz) 以及累積補償向量
"""
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

sys.stdout.reconfigure(encoding='utf-8')


def load_boxes(combined_path):
    """讀取所有 (frameIdx, tid, position)"""
    data = json.loads(Path(combined_path).read_text(encoding='utf-8'))
    by_frame = defaultdict(dict)  # fi -> {tid: pos}
    for f in data.get('frames', []):
        fi = f.get('frameIndex', -1)
        for a in f.get('annotations', []) or []:
            tid = a.get('trackingId')
            pos = a.get('position')
            if tid is not None and pos and len(pos) >= 3:
                by_frame[fi][str(tid)] = [float(pos[0]), float(pos[1]), float(pos[2])]
    return by_frame


def estimate_ego_motion(by_frame, min_boxes=10):
    """
    估算每幀相對前一幀的 ego motion (自車在 lidar-frame 下的位移)
    
    返回:
        ego_motions: list of (dx, dy, dz) 長度 = max_frame, ego_motions[0] = (0,0,0)
        cum_compensate: list of 累積補償向量 (將 lidar-frame 座標轉為 pseudo-world)
    """
    frames = sorted(k for k in by_frame.keys() if k >= 0)
    if len(frames) < 2:
        return [(0.0, 0.0, 0.0)] * len(frames), [(0.0, 0.0, 0.0)] * len(frames)

    ego_motions = [(0.0, 0.0, 0.0)]  # frame 0 無前一幀
    
    for i in range(1, len(frames)):
        fi_prev = frames[i-1]
        fi_curr = frames[i]
        prev_boxes = by_frame[fi_prev]
        curr_boxes = by_frame[fi_curr]
        
        # 找共同出現的 tid
        common_tids = set(prev_boxes.keys()) & set(curr_boxes.keys())
        if len(common_tids) < min_boxes:
            # 不夠樣本，沿用上一幀估計
            ego_motions.append(ego_motions[-1])
            continue
        
        displacements = []
        for tid in common_tids:
            p_prev = prev_boxes[tid]
            p_curr = curr_boxes[tid]
            dx = p_curr[0] - p_prev[0]
            dy = p_curr[1] - p_prev[1]
            dz = p_curr[2] - p_prev[2]
            displacements.append((dx, dy, dz))
        
        # 用中位數抗干擾 (動態物體會是離群值)
        arr = np.array(displacements, dtype=np.float32)
        med = np.median(arr, axis=0)
        ego_motions.append((float(med[0]), float(med[1]), float(med[2])))
    
    # 累積補償向量：要把 lidar-frame 位置 "扣除" ego 累積位移
    # 即 pseudo_world = lidar_pos - cum_ego
    cum = [(0.0, 0.0, 0.0)]
    for em in ego_motions[1:]:
        cx, cy, cz = cum[-1]
        cum.append((cx + em[0], cy + em[1], cz + em[2]))
    
    return ego_motions, cum


def compensate_positions(by_frame, cum_compensate):
    """
    將所有 box 位置補償到 pseudo-world 座標
    返回: {tid: {fi: compensated_pos}}
    """
    frames = sorted(k for k in by_frame.keys() if k >= 0)
    result = defaultdict(dict)
    for fi in frames:
        comp = cum_compensate[fi] if fi < len(cum_compensate) else cum_compensate[-1]
        for tid, pos in by_frame[fi].items():
            result[tid][fi] = [
                pos[0] - comp[0],
                pos[1] - comp[1],
                pos[2] - comp[2],
            ]
    return result


def analyze(combined_path, min_boxes=10):
    by_frame = load_boxes(combined_path)
    ego_motions, cum_comp = estimate_ego_motion(by_frame, min_boxes)
    compensated = compensate_positions(by_frame, cum_comp)
    
    # 統計每個 tid 的補償後漂移
    tid_stats = {}
    for tid, frames_dict in compensated.items():
        if len(frames_dict) < 2:
            continue
        positions = [frames_dict[fi] for fi in sorted(frames_dict.keys())]
        base = positions[0]
        max_d = max(math.sqrt(sum((p[i]-base[i])**2 for i in range(3))) for p in positions)
        max_dz = max(abs(p[2]-base[2]) for p in positions)
        tid_stats[tid] = {
            'appear': len(positions),
            'frames': sorted(frames_dict.keys()),
            'max_drift': round(max_d, 3),
            'max_dz': round(max_dz, 3),
            'positions': {str(fi): [round(p[0],3), round(p[1],3), round(p[2],3)] 
                          for fi, p in frames_dict.items()},
        }
    
    return {
        'ego_motions_per_frame': [[round(x,3) for x in em] for em in ego_motions],
        'cum_compensation': [[round(x,3) for x in c] for c in cum_comp],
        'track_stats': tid_stats,
    }


def main():
    import sys
    if len(sys.argv) < 2:
        print("用法: python ego_motion_estimator.py <combined.json> [min_boxes=10]")
        sys.exit(1)
    
    combined_path = Path(sys.argv[1])
    min_boxes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    rep = analyze(combined_path, min_boxes)
    
    out_dir = Path(__file__).resolve().parent.parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ego_motion_estimate.json"
    out_path.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding='utf-8')
    
    print("===== Ego Motion 估算 =====")
    print(f"輸入: {combined_path.name}")
    print(f"幀數: {len(rep['ego_motions_per_frame'])}")
    print(f"Ego motion/幀 (中位數):")
    for i, em in enumerate(rep['ego_motions_per_frame']):
        print(f"  f{i}: dx={em[0]:+7.3f} dy={em[1]:+7.3f} dz={em[2]:+7.3f}")
    print(f"\n累積補償向量 (前5幀):")
    for i, c in enumerate(rep['cum_compensation'][:5]):
        print(f"  f{i}: cx={c[0]:+8.3f} cy={c[1]:+8.3f} cz={c[2]:+8.3f}")
    print(f"\n追蹤 ID 數: {len(rep['track_stats'])}")
    
    # 列出漂移最小的前 10 個 (疑似靜止)
    sorted_tids = sorted(rep['track_stats'].items(), key=lambda x: x[1]['max_drift'])
    print(f"\n補償後漂移最小 TOP 10 (疑似靜止):")
    for tid, st in sorted_tids[:10]:
        print(f"  #{tid:<6} appear={st['appear']:>2}  drift={st['max_drift']:.3f}m  dz={st['max_dz']:.3f}m")
    
    print(f"\n寫入: {out_path}")


if __name__ == '__main__':
    main()