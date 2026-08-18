# process_dynamic_objects.py — 動態物體軌跡平滑 + 幾何修正
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.linalg import block_diag

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "output"

# ========== 1. 載入資料 ==========
# 靜止判定結果
fixed = json.load(open(OUT / "fixed_static_classification.json", encoding='utf-8'))
static_ids = {r['tid'] for r in fixed['static']}
moving_ids = {r['tid'] for r in fixed['moving']}
print(f"STATIC: {len(static_ids)}, MOVING: {len(moving_ids)}")

# 原始標註
raw = json.load(open(DATA / "task_export_with_annots.json", encoding='utf-8'))
lio_raw = raw['frames'][0]['lioJson']
lio = json.loads(lio_raw) if isinstance(lio_raw, str) else lio_raw

# select_frame mapping
select_frame = lio['select_frame']  # frameIndex -> pcd number

# Ground planes
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ground_estimator import estimate_ground_per_frame, get_ground_height_at
ground_planes = estimate_ground_per_frame(DATA / "pcd", frame_indices=list(range(15)), 
                                           select_frame=select_frame, method='percentile')

# ========== 2. SLAM 修正矩陣 (同 v5) ==========
def mat(m): return np.array(m, dtype=np.float64).reshape(4, 4)
V = [mat(m) for m in lio['velo_pose_lidar']]
T_lidar_ego = mat(lio['lidar2ego'])
T_ego_lidar = np.linalg.inv(T_lidar_ego)
inv_V0 = np.linalg.inv(V[0])

# GT IDs for SLAM error correction
gt_ids = {'222','376','428','441','960'}
gt_tracks_raw = defaultdict(list)
for f in raw['frames']:
    fi = f['frameIndex']
    for a in f.get('annotations', []) or []:
        tid = str(a.get('trackingId'))
        if tid in gt_ids and a.get('position'):
            gt_tracks_raw[tid].append((fi, np.array(a['position'])))

# Compute GT mean world pos
gt_mean_world = {}
for tid, lst in gt_tracks_raw.items():
    world_poss = []
    for fi, pos_ego in lst:
        pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
        pos_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
        world_poss.append(pos_world)
    gt_mean_world[tid] = np.mean(world_poss, axis=0)

# SLAM error per frame (world frame)
slam_error = {}
for fi in range(15):
    errors = []
    for tid in gt_ids:
        for f, pos_ego in gt_tracks_raw.get(tid, []):
            if f == fi:
                pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
                obs_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
                errors.append(obs_world - gt_mean_world[tid])
                break
    slam_error[fi] = np.mean(errors, axis=0) if errors else np.zeros(3)

# Correct box to Ego0 with SLAM error correction
def correct_box_to_ego0(fi, pos_ego):
    pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
    obs_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
    # Subtract SLAM error for this frame
    true_world = obs_world - slam_error[fi]
    pos_lidar0 = inv_V0[:3,:3] @ true_world + inv_V0[:3,3]
    pos_ego0 = T_lidar_ego[:3,:3] @ pos_lidar0 + T_lidar_ego[:3,3]
    return pos_ego0

# Transform Ego0 back to Ego[fi] (for output) - WITH SLAM ERROR CORRECTION
def ego0_to_egofi(fi, pos_ego0):
    # Ego0 -> Lidar0 -> World (corrected, at frame 0 reference)
    pos_lidar0 = T_ego_lidar[:3,:3] @ pos_ego0 + T_ego_lidar[:3,3]
    corrected_world = V[0][:3,:3] @ pos_lidar0 + V[0][:3,3]  # This is corrected World pos
    # For static objects: corrected World pos is constant across frames
    # Raw World at frame fi = corrected World + slam_error[fi]
    raw_world_fi = corrected_world + slam_error[fi]
    # Raw World -> Lidar[fi] -> Ego[fi]
    pos_lidar = np.linalg.inv(V[fi])[:3,:3] @ (raw_world_fi - V[fi][:3,3])
    pos_ego = T_lidar_ego[:3,:3] @ pos_lidar + T_lidar_ego[:3,3]
    return pos_ego

# ========== 3. 收集所有軌跡 (已 SLAM 修正到 Ego0) ==========
all_tracks = defaultdict(list)  # tid -> list of (fi, box_dict)
cls_map = {}
for f in raw['frames']:
    fi = f['frameIndex']
    for a in f.get('annotations', []) or []:
        tid = str(a.get('trackingId'))
        if tid and a.get('position'):
            pos_ego = np.array(a['position'])
            pos_ego0 = correct_box_to_ego0(fi, pos_ego)
            all_tracks[tid].append((fi, {
                'pos_ego0': pos_ego0,
                'pos_ego_orig': pos_ego,
                'quaternion': np.array(a['quaternion']),
                'scale': np.array(a['scale']),
                'className': a.get('className') or a.get('userData',{}).get('cls',''),
            }))
            cls_map[tid] = a.get('className') or a.get('userData',{}).get('cls','')

# ========== 4. 類別先驗尺寸 (從 STATIC 物體統計) ==========
static_sizes = defaultdict(list)
for tid in static_ids:
    if tid in all_tracks:
        for _, box in all_tracks[tid]:
            static_sizes[cls_map.get(tid,'')].append(box['scale'])

class_prior = {}
for cls, sizes in static_sizes.items():
    arr = np.array(sizes)
    class_prior[cls] = {'median': np.median(arr, axis=0)}
print("\n類別先驗尺寸 (median):")
for cls, v in class_prior.items():
    print(f"  {cls}: {v['median']}")

def kalman_smoother_cv(measurements, dt=1.0, q_pos=0.1, q_vel=0.01, q_size=0.001, r=0.05):
    """
    measurements: list of (fi, pos_world[3], scale[3]) sorted by fi
    狀態: [x, y, z, vx, vy, vz, l, w, h] (9 維)
    測量: [x, y, z, l, w, h] (6 維)
    返回: smoothed states per frame (dict fi -> state)
    """
    if len(measurements) < 2:
        return {fi: np.hstack([pos, np.zeros(3), scale]) for fi, pos, scale in measurements}
    
    F = np.eye(9)
    F[0,3] = F[1,4] = F[2,5] = dt
    
    H = np.zeros((6, 9))
    H[0,0] = H[1,1] = H[2,2] = 1.0
    H[3,6] = H[4,7] = H[5,8] = 1.0
    
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
    x_smooth, P_smooth = forward[fis[-1]]
    smoothed[fis[-1]] = x_smooth
    
    for fi in reversed(fis[:-1]):
        x_f, P_f = forward[fi]
        x_pred = F @ x_f
        P_pred = F @ P_f @ F.T + Q
        C = P_f @ F.T @ np.linalg.inv(P_pred)
        x_smooth = x_f + C @ (x_smooth - x_pred)
        P_smooth = P_f + C @ (P_smooth - P_pred) @ C.T
        smoothed[fi] = x_smooth
    
    return smoothed

# ========== 6. 處理每個 MOVING 軌跡 (在 WORLD 系平滑) ==========
print("\n=== 處理 MOVING 物體 (World 系平滑) ===")
corrected_moving = {}

for tid in moving_ids:
    if tid not in all_tracks:
        continue
    lst = all_tracks[tid]
    if len(lst) < 2:
        continue
    
    lst_sorted = sorted(lst, key=lambda x: x[0])
    fis = [x[0] for x in lst_sorted]
    
    # 轉到 World 系 (已含 SLAM error correction)
    measurements = []
    cls = cls_map.get(tid, 'Car')
    prior_size = class_prior.get(cls, {}).get('median', np.array([4.5, 1.8, 1.5]))
    
    # 找最近幀
    closest_idx = 0
    min_dist = float('inf')
    for i, (fi, box) in enumerate(lst_sorted):
        d = np.linalg.norm(box['pos_ego_orig'][:2])
        if d < min_dist:
            min_dist = d
            closest_idx = i
    closest_scale = lst_sorted[closest_idx][1]['scale']
    
    for fi, box in lst_sorted:
        pos_ego0 = box['pos_ego0']
        # Ego0 -> World (using corrected pose)
        pos_lidar0 = T_ego_lidar[:3,:3] @ pos_ego0 + T_ego_lidar[:3,3]
        pos_world = V[0][:3,:3] @ pos_lidar0 + V[0][:3,3]
        # Add SLAM error for frame 0
        true_world = pos_world + slam_error[0]
        
        # 地面修正 (在 World 系: 地面高度隨 frame 變)
        plane = ground_planes[fi]
        ground_z = get_ground_height_at(plane, true_world[0], true_world[1])
        height = box['scale'][2]
        true_world[2] = ground_z + height / 2.0
        
        # 尺寸策略
        if fi == fis[closest_idx]:
            scale = box['scale'].copy()
        else:
            scale = box['scale'] * 0.3 + prior_size * 0.7
            scale = np.minimum(scale, prior_size * 1.2)
            scale = np.maximum(scale, prior_size * 0.5)
        
        measurements.append((fi, true_world, scale))
    
    # Kalman Smoother in WORLD frame
    smoothed = kalman_smoother_cv(measurements, dt=1.0, q_pos=0.5, q_vel=0.1, q_size=0.001, r=0.2)
    
    # 輸出每幀修正後的 box (轉回 Ego[fi])
    corrected_moving[tid] = {}
    for fi in fis:
        state = smoothed[fi]
        pos_world_smooth = state[:3]
        scale_smooth = state[6:9]
        scale_smooth = np.clip(scale_smooth, prior_size * 0.5, prior_size * 1.3)
        
        # World -> Ego[fi] (with SLAM error correction for frame fi)
        # true_world = pos_world_smooth
        # World -> Lidar[fi]: inv(V[fi]) @ (true_world - slam_error[fi] - V[fi][:3,3])
        pos_lidar = np.linalg.inv(V[fi])[:3,:3] @ (pos_world_smooth - slam_error[fi] - V[fi][:3,3])
        pos_ego = T_lidar_ego[:3,:3] @ pos_lidar + T_lidar_ego[:3,3]
        
        # Quaternion: 朝向速度向量 (或保持最近幀朝向)
        orig_quat = lst_sorted[closest_idx][1]['quaternion']
        
        corrected_moving[tid][fi] = {
            'pos_ego': pos_ego,
            'scale': scale_smooth,
            'quaternion': orig_quat,
            'className': cls,
        }

print(f"已處理 {len(corrected_moving)} 個 MOVING 物體")

# ========== 7. 合併 STATIC + MOVING 生成導入包 ==========
# STATIC 物體: 用 gen_import_json.py 的邏輯 (已驗證)
# 這裡直接讀取 fixed_static_classification.json 的 STATIC 列表
# 並用 v5 邏輯生成

# 先收集 STATIC 物體的最佳幀
static_tracks = {}
for tid in static_ids:
    if tid not in all_tracks:
        continue
    lst = all_tracks[tid]
    # 選點數最多的幀
    best = max(lst, key=lambda x: x[1].get('userData',{}).get('points', 0) if 'userData' in x[1] else 0)
    fi, box = best
    static_tracks[tid] = (fi, box)

print(f"STATIC 物體: {len(static_tracks)}")

# FrameId mapping (move up before static processing)
frame_id_map = {f['frameIndex']: f['frameId'] for f in raw['frames']}

# 統一輸出格式：深拷貝原始標注，僅更新 position/quaternion/scale
frames_out = defaultdict(list)
import uuid

# Build original annotation lookup: tid -> {fi: original_ann}
orig_ann_lookup = defaultdict(dict)
for f in raw['frames']:
    fi = f['frameIndex']
    for a in f.get('annotations', []) or []:
        tid = str(a.get('trackingId'))
        if tid:
            orig_ann_lookup[tid][fi] = a

# STATIC 物體傳播
for tid, (base_fi, base_box) in static_tracks.items():
    gold_pos = base_box['pos_ego0']
    gold_quat = base_box['quaternion'] / np.linalg.norm(base_box['quaternion'])
    gold_scale = base_box['scale']
    
    for fi in range(15):
        pos_ego = ego0_to_egofi(fi, gold_pos)
        
        # 以該幀原始標注為底本（若無則用基準幀），僅更新位姿尺寸
        orig_ann_this_frame = orig_ann_lookup.get(tid, {}).get(fi)
        orig_ann = orig_ann_this_frame if orig_ann_this_frame is not None else orig_ann_lookup.get(tid, {}).get(base_fi, {})
        
        export_ann = json.loads(json.dumps(orig_ann)) if orig_ann else {
            "uuid": str(uuid.uuid4()),
            "trackingId": int(tid),
            "className": cls_map.get(tid, 'Car'),
            "readonly": False,
            "lastUpdateTs": 0,
            "editConfig": {"resize": True},
            "userData": {"cls": cls_map.get(tid, 'Car'), "trackingId": tid},
            "color": 16186115
        }
        
        export_ann["position"] = [float(round(x, 6)) for x in pos_ego]
        export_ann["quaternion"] = [float(round(x, 6)) for x in gold_quat]
        export_ann["scale"] = [float(round(x, 6)) for x in gold_scale]
        
        # 缺失帧：可见度设为空，frameId 修正为当前帧，points 清空
        if orig_ann_this_frame is None:
            if "userData" in export_ann:
                if "attributes" in export_ann["userData"]:
                    export_ann["userData"]["attributes"]["visible_1"] = ""
                export_ann["userData"]["frameId"] = frame_id_map.get(fi, "")
                export_ann["userData"]["points"] = 0
        
        if "uuid" not in export_ann or not export_ann["uuid"]:
            export_ann["uuid"] = str(uuid.uuid4())
        
        frames_out[fi].append(export_ann)

# MOVING 物體
for tid, fi_dict in corrected_moving.items():
    for fi, box in fi_dict.items():
        orig_ann = orig_ann_lookup.get(tid, {}).get(fi)
        if orig_ann is None:
            # fallback: 找最近有標注的幀
            available_fis = list(orig_ann_lookup.get(tid, {}).keys())
            if available_fis:
                closest_fi = min(available_fis, key=lambda x: abs(x - fi))
                orig_ann = orig_ann_lookup[tid][closest_fi]
        
        export_ann = json.loads(json.dumps(orig_ann)) if orig_ann else {
            "uuid": str(uuid.uuid4()),
            "trackingId": int(tid),
            "className": box['className'],
            "readonly": False,
            "lastUpdateTs": 0,
            "editConfig": {"resize": True},
            "userData": {"cls": box['className'], "trackingId": tid},
            "color": 16186115
        }
        
        export_ann["position"] = [float(round(x, 6)) for x in box['pos_ego']]
        export_ann["quaternion"] = [float(round(x, 6)) for x in box['quaternion']]
        export_ann["scale"] = [float(round(x, 6)) for x in box['scale']]
        
        # 缺失帧：可见度设为空，frameId 修正为当前帧，points 清空
        if orig_ann is None:
            if "userData" in export_ann:
                if "attributes" in export_ann["userData"]:
                    export_ann["userData"]["attributes"]["visible_1"] = ""
                export_ann["userData"]["frameId"] = frame_id_map.get(fi, "")
                export_ann["userData"]["points"] = 0
        
        if "uuid" not in export_ann or not export_ann["uuid"]:
            export_ann["uuid"] = str(uuid.uuid4())
        
        frames_out[fi].append(export_ann)

# FrameId mapping
frame_id_map = {f['frameIndex']: f['frameId'] for f in raw['frames']}
final_frames = []
for fi in range(15):
    final_frames.append({
        "frameId": frame_id_map.get(fi),
        "annotations": frames_out.get(fi, [])
    })

out_json = {"frames": final_frames}
out_path = DATA / "import_3d_boxes.json"
out_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding='utf-8')

total_boxes = sum(len(f['annotations']) for f in final_frames)
static_count = len(static_tracks)
moving_count = len(corrected_moving)
print(f"\n✅ 生成完成: {out_path}")
print(f"   15 帧 × ({static_count} STATIC + {moving_count} MOVING) = {total_boxes} 個框")
print(f"   直接拖入 Tampermonkey 面板 -> 導入 3D 框 (E)")