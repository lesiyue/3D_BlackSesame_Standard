# fix_static_classifier_v4.py — GT-corrected SLAM 靜止判定 (v5 邏輯)
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import math

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "output"

def mat(m): return np.array(m, dtype=np.float64).reshape(4, 4)

def q_mul(q1, q2):  # xyzw
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def mat_to_quat(R):  # 3x3 -> xyzw
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2,1] - R[1,2]) / S
        qy = (R[0,2] - R[2,0]) / S
        qz = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        qw = (R[2,1] - R[1,2]) / S
        qx = 0.25 * S
        qy = (R[0,1] + R[1,0]) / S
        qz = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        qw = (R[0,2] - R[2,0]) / S
        qx = (R[0,1] + R[1,0]) / S
        qy = 0.25 * S
        qz = (R[1,2] + R[2,1]) / S
    else:
        S = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        qw = (R[1,0] - R[0,1]) / S
        qx = (R[0,2] + R[2,0]) / S
        qy = (R[1,2] + R[2,1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw])

def load_combined():
    data = json.load(open(DATA / "task_export_with_annots.json", encoding='utf-8'))
    lio_raw = data['frames'][0]['lioJson']
    lio = json.loads(lio_raw) if isinstance(lio_raw, str) else lio_raw
    return data, lio

def build_slam_transforms(lio):
    V = [mat(m) for m in lio['velo_pose_lidar']]
    T_lidar_ego = mat(lio['lidar2ego'])
    T_ego_lidar = np.linalg.inv(T_lidar_ego)
    inv_V0 = np.linalg.inv(V[0])
    return V, T_lidar_ego, T_ego_lidar, inv_V0

def compute_slam_error_from_gt(lio, raw, gt_ids):
    V, T_lidar_ego, T_ego_lidar, inv_V0 = build_slam_transforms(lio)
    gt_ids = set(gt_ids)
    gt_tracks = defaultdict(list)
    for f in raw['frames']:
        fi = f['frameIndex']
        for a in f.get('annotations', []) or []:
            tid = str(a.get('trackingId'))
            if tid in gt_ids and a.get('position'):
                gt_tracks[tid].append((fi, np.array(a['position'])))
    
    gt_mean_world = {}
    for tid, lst in gt_tracks.items():
        world_poss = []
        for fi, pos_ego in lst:
            pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
            pos_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
            world_poss.append(pos_world)
        gt_mean_world[tid] = np.mean(world_poss, axis=0)
    
    slam_error = {}
    for fi in range(len(V)):
        errors = []
        for tid in gt_ids:
            for f, pos_ego in gt_tracks.get(tid, []):
                if f == fi:
                    pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
                    obs_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
                    errors.append(obs_world - gt_mean_world[tid])
                    break
        slam_error[fi] = np.mean(errors, axis=0) if errors else np.zeros(3)
    return slam_error, V, T_lidar_ego, T_ego_lidar, inv_V0

def correct_box_to_ego0(fi, pos_ego, slam_error, V, T_lidar_ego, T_ego_lidar, inv_V0):
    pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
    obs_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
    true_world = obs_world - slam_error[fi]
    pos_lidar0 = inv_V0[:3,:3] @ true_world + inv_V0[:3,3]
    pos_ego0 = T_lidar_ego[:3,:3] @ pos_lidar0 + T_lidar_ego[:3,3]
    return pos_ego0

def ego0_to_egofi(fi, pos_ego0, slam_error, V, T_lidar_ego, T_ego_lidar):
    pos_lidar0 = T_ego_lidar[:3,:3] @ pos_ego0 + T_ego_lidar[:3,3]
    corrected_world = V[0][:3,:3] @ pos_lidar0 + V[0][:3,3]
    raw_world_fi = corrected_world + slam_error[fi]
    pos_lidar = np.linalg.inv(V[fi])[:3,:3] @ (raw_world_fi - V[fi][:3,3])
    pos_ego = T_lidar_ego[:3,:3] @ pos_lidar + T_lidar_ego[:3,3]
    return pos_ego

def main():
    data, lio = load_combined()
    raw = data
    
    gt_ids = {'222','376','428','441','960'}
    slam_error, V, T_lidar_ego, T_ego_lidar, inv_V0 = compute_slam_error_from_gt(lio, raw, gt_ids)
    
    print("SLAM pose error (World frame) per frame:")
    for fi in range(len(V)):
        e = slam_error[fi]
        print(f"  f{fi}: dx={e[0]:+.3f} dy={e[1]:+.3f} dz={e[2]:+.3f}")
    
    # Ground planes
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ground_estimator import estimate_ground_per_frame
    select_frame = lio['select_frame']
    ground_planes = estimate_ground_per_frame(DATA / "pcd", list(range(15)), select_frame)
    
    # Collect all tracks -> corrected Ego0
    tracks = defaultdict(list)
    cls_map = {}
    for f in raw['frames']:
        fi = f['frameIndex']
        for a in f.get('annotations', []) or []:
            tid = str(a.get('trackingId'))
            if tid and a.get('position'):
                pos_ego = np.array(a['position'])
                pos_ego0 = correct_box_to_ego0(fi, pos_ego, slam_error, V, T_lidar_ego, T_ego_lidar, inv_V0)
                tracks[tid].append((fi, {
                    'pos_ego0': pos_ego0,
                    'pos_ego_orig': pos_ego,
                    'quaternion': np.array(a['quaternion']),
                    'scale': np.array(a['scale']),
                    'className': a.get('className') or a.get('userData',{}).get('cls',''),
                }))
                cls_map[tid] = a.get('className') or a.get('userData',{}).get('cls','')
    
    # Class prior sizes from STATIC candidates
    static_sizes = defaultdict(list)
    for tid, lst in tracks.items():
        if len(lst) >= 5:
            arr = np.array([b['pos_ego0'] for _, b in lst])
            base = arr[0]
            if np.std(np.linalg.norm(arr - base, axis=1)) < 0.5:
                for _, b in lst:
                    static_sizes[cls_map.get(tid,'')].append(b['scale'])
    class_prior = {cls: {'median': np.median(np.array(s), axis=0)} for cls, s in static_sizes.items() if s}
    
    # Static / Moving separation
    static_tracks = {}
    moving_tids = set()
    for tid, lst in tracks.items():
        if len(lst) < 5:
            moving_tids.add(tid)
            continue
        arr = np.array([b['pos_ego0'] for _, b in lst])
        base = arr[0]
        dists = np.linalg.norm(arr - base, axis=1)
        if np.std(dists) < 0.3 and np.max(dists) < 1.0:
            static_tracks[tid] = lst
        else:
            moving_tids.add(tid)
    
    print(f"\nSTATIC: {len(static_tracks)}, MOVING: {len(moving_tids)}")
    
    # GT check
    gt_set = set(gt_ids)
    st_set = set(static_tracks.keys())
    tp = gt_set & st_set
    print(f"GT in STATIC: {tp}")
    
    # Save classification result
    results = []
    for tid, lst in tracks.items():
        if len(lst) < 3: continue
        arr = np.array([b['pos_ego0'] for _, b in lst])
        base = arr[0]
        dists = np.linalg.norm(arr - base, axis=1)
        verdict = 'STATIC' if tid in static_tracks else 'MOVING'
        results.append({
            'tid': tid, 'cls': cls_map.get(tid,''),
            'n': len(lst), 'max_drift_m': round(float(np.max(dists)), 3),
            'std_drift_m': round(float(np.std(dists)), 3),
            'verdict': verdict
        })
    
    gt_path = ROOT / "static_trackingId.txt"
    gt_set = set(gt_path.read_text(encoding='utf-8').strip().split()) if gt_path.exists() else set()
    st_set = {r['tid'] for r in results if r['verdict'] == 'STATIC'}
    tp, fp, fn = gt_set & st_set, st_set - gt_set, gt_set - st_set
    prec = len(tp) / max(len(tp)+len(fp), 1)
    rec = len(tp) / max(len(tp)+len(fn), 1)
    f1 = 2*prec*rec/max(prec+rec, 1e-9)
    
    OUT.mkdir(exist_ok=True)
    out = {
        'config': {'sigma_thresh_m': 0.3, 'max_dist_thresh_m': 1.0},
        'results': results,
        'groundTruthEval': {'tp':sorted(tp), 'fp':sorted(fp), 'fn':sorted(fn),
                            'precision':prec, 'recall':rec, 'f1':f1}
    }
    (OUT / "fixed_static_classification.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f"\n已写入: output/fixed_static_classification.json")
    print(f"STATIC={len(st_set)}, MOVING={len(results)-len(st_set)}")
    print(f"Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")

if __name__ == '__main__':
    main()