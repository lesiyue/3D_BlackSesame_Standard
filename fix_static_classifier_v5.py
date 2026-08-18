# fix_static_classifier_v5.py — GT-corrected SLAM 靜止判定 + 動態物體 World 系平滑
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
    # Ego0 -> Lidar0 -> World (corrected, at frame 0)
    pos_lidar0 = T_ego_lidar[:3,:3] @ pos_ego0 + T_ego_lidar[:3,3]
    corrected_world = V[0][:3,:3] @ pos_lidar0 + V[0][:3,3]
    # For static: corrected World is constant. Raw World at fi = corrected + slam_error[fi]
    raw_world_fi = corrected_world + slam_error[fi]
    pos_lidar = np.linalg.inv(V[fi])[:3,:3] @ (raw_world_fi - V[fi][:3,3])
    pos_ego = T_lidar_ego[:3,:3] @ pos_lidar + T_lidar_ego[:3,3]
    return pos_ego

def kalman_smoother_cv(measurements, dt=1.0, q_pos=0.1, q_vel=0.01, q_size=0.001, r=0.05):
    if len(measurements) < 2:
        return {fi: np.hstack([pos, np.zeros(3), scale]) for fi, pos, scale in measurements}
    F = np.eye(9); F[0,3]=F[1,4]=F[2,5]=dt
    H = np.zeros((6,9)); H[0,0]=H[1,1]=H[2,2]=1; H[3,6]=H[4,7]=H[5,8]=1
    Q = np.diag([q_pos]*3 + [q_vel]*3 + [q_size]*3)
    R = np.eye(6) * r
    fi0, pos0, scale0 = measurements[0]
    x = np.hstack([pos0, np.zeros(3), scale0])
    P = np.eye(9) * 10.0
    forward = {}
    for fi, pos, scale in measurements:
        z = np.hstack([pos, scale])
        x = F @ x
        P = F @ P @ F.T + Q
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(9) - K @ H) @ P
        forward[fi] = (x.copy(), P.copy())
    smoothed = {}
    fis = sorted(forward.keys())
    x_s, P_s = forward[fis[-1]]
    smoothed[fis[-1]] = x_s
    for fi in reversed(fis[:-1]):
        x_f, P_f = forward[fi]
        x_pred = F @ x_f
        P_pred = F @ P_f @ F.T + Q
        C = P_f @ F.T @ np.linalg.inv(P_pred)
        x_s = x_f + C @ (x_s - x_pred)
        P_s = P_f + C @ (P_s - P_pred) @ C.T
        smoothed[fi] = x_s
    return smoothed

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
    
    # Class prior sizes from STATIC candidates (std<0.5 in Ego0)
    static_sizes = defaultdict(list)
    for tid, lst in tracks.items():
        if len(lst) >= 5:
            arr = np.array([b['pos_ego0'] for _, b in lst])
            base = arr[0]
            if np.std(np.linalg.norm(arr - base, axis=1)) < 0.5:
                for _, b in lst:
                    static_sizes[cls_map.get(tid,'')].append(b['scale'])
    class_prior = {cls: {'median': np.median(np.array(s), axis=0)} for cls, s in static_sizes.items() if s}
    
    # Static / Moving separation (Ego0 std<0.3, max<1.0, frames>=5)
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
    
    # STATIC: propagate gold standard (max points frame) to all frames via corrected transforms
    static_gold = {}
    for tid, lst in static_tracks.items():
        best = max(lst, key=lambda x: x[1].get('userData',{}).get('points',0) if 'userData' in x[1] else 0)
        fi, box = best
        static_gold[tid] = (fi, box)
    
    # MOVING: World frame Kalman smoothing
    print("\n=== Processing MOVING objects (World frame smoothing) ===")
    moving_corrected = {}  # tid -> {fi: box_dict}
    for tid in moving_tids:
        lst = tracks[tid]
        if len(lst) < 2: continue
        lst_sorted = sorted(lst, key=lambda x: x[0])
        cls = cls_map.get(tid, 'Car')
        prior = class_prior.get(cls, {}).get('median', np.array([4.5, 1.8, 1.5]))
        
        # Find closest frame (for size prior)
        closest_idx = 0
        min_dist = float('inf')
        for i, (fi, box) in enumerate(lst_sorted):
            d = np.linalg.norm(box['pos_ego_orig'][:2])
            if d < min_dist:
                min_dist = d
                closest_idx = i
        
        measurements = []
        for fi, box in lst_sorted:
            pos_w = box['pos_ego0']  # already in corrected Ego0
            # Convert to World frame for smoothing
            pos_lidar0 = T_ego_lidar[:3,:3] @ pos_w + T_ego_lidar[:3,3]
            pos_world = V[0][:3,:3] @ pos_lidar0 + V[0][:3,3]
            # Ground correction
            plane = ground_planes[fi]
            a,b,c,d = plane
            ground_z = -(a*pos_world[0] + b*pos_world[1] + d) / c
            pos_world[2] = ground_z + box['scale'][2] / 2.0
            scale = box['scale'].copy()
            if fi == lst_sorted[closest_idx][0]:
                scale = scale  # observed size at closest
            else:
                scale = scale * 0.3 + prior * 0.7
                scale = np.clip(scale, prior * 0.5, prior * 1.3)
            measurements.append((fi, pos_world, scale))
        
        smoothed = kalman_smoother_cv(measurements, dt=1.0, q_pos=0.2, q_vel=0.05, q_size=0.001, r=0.1)
        
        moving_corrected[tid] = {}
        for fi, state in smoothed.items():
            pos_world = state[:3]
            scale = np.clip(state[6:9], prior * 0.5, prior * 1.3)
            # World -> Ego[fi] (with SLAM error added)
            raw_world_fi = pos_world + slam_error[fi]
            pos_lidar = np.linalg.inv(V[fi])[:3,:3] @ (raw_world_fi - V[fi][:3,3])
            pos_ego = T_lidar_ego[:3,:3] @ pos_lidar + T_lidar_ego[:3,3]
            quat = lst_sorted[closest_idx][1]['quaternion']
            moving_corrected[tid][fi] = {
                'pos_ego': pos_ego,
                'scale': scale,
                'quaternion': quat,
                'className': cls,
            }
    
    # Merge and output import JSON
    import uuid
    frames_out = defaultdict(list)
    
    # Static objects
    for tid, (base_fi, base_box) in static_gold.items():
        gold_pos = base_box['pos_ego0']
        gold_quat = base_box['quaternion'] / np.linalg.norm(base_box['quaternion'])
        gold_scale = base_box['scale']
        for fi in range(15):
            pos_ego = ego0_to_egofi(fi, gold_pos, slam_error, V, T_lidar_ego, T_ego_lidar)
            frames_out[fi].append({
                "uuid": str(uuid.uuid4()),
                "trackingId": int(tid),
                "className": cls_map.get(tid, 'Car'),
                "position": [float(round(x,6)) for x in pos_ego],
                "quaternion": [float(round(x,6)) for x in gold_quat],
                "scale": [float(round(x,6)) for x in gold_scale],
                "color": 16186115
            })
    
    # Moving objects
    for tid, fi_dict in moving_corrected.items():
        for fi, box in fi_dict.items():
            frames_out[fi].append({
                "uuid": str(uuid.uuid4()),
                "trackingId": int(tid),
                "className": box['className'],
                "position": [float(round(x,6)) for x in box['pos_ego']],
                "quaternion": [float(round(x,6)) for x in box['quaternion']],
                "scale": [float(round(x,6)) for x in box['scale']],
                "color": 16186115
            })
    
    frame_id_map = {f['frameIndex']: f['frameId'] for f in raw['frames']}
    final = []
    for fi in range(15):
        final.append({"frameId": frame_id_map.get(fi), "annotations": frames_out.get(fi, [])})
    
    out_json = {"frames": final}
    out_path = DATA / "import_3d_boxes.json"
    out_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding='utf-8')
    
    total = sum(len(f['annotations']) for f in final)
    print(f"\n✅ 生成完成: {out_path}")
    print(f"   15 帧 × ({len(static_gold)} STATIC + {len(moving_corrected)} MOVING) = {total} 個框")

if __name__ == '__main__':
    main()