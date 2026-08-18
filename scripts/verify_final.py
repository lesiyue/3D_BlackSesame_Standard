import json
import numpy as np
from collections import defaultdict

# Load output JSON
data = json.load(open('data/import_3d_boxes.json', encoding='utf-8'))

# Load SLAM poses
raw = json.load(open('data/task_export_with_annots.json', encoding='utf-8'))
lio_raw = raw['frames'][0]['lioJson']
lio = json.loads(lio_raw) if isinstance(lio_raw, str) else lio_raw

def mat(m): return np.array(m, dtype=np.float64).reshape(4, 4)
V = [mat(m) for m in lio['velo_pose_lidar']]
T_lidar_ego = mat(lio['lidar2ego'])
T_ego_lidar = np.linalg.inv(T_lidar_ego)
inv_V0 = np.linalg.inv(V[0])

frame_id_to_idx = {f['frameId']: f['frameIndex'] for f in raw['frames']}

# Compute SLAM error from GT
gt_ids = {'222','376','428','441','960'}
gt_tracks = {}
for fr in raw['frames']:
    fi = fr['frameIndex']
    for a in fr.get('annotations', []) or []:
        tid = str(a.get('trackingId'))
        if tid in gt_ids and a.get('position'):
            if tid not in gt_tracks: gt_tracks[tid] = []
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
for fi in range(15):
    errors = []
    for tid in gt_ids:
        for f, pos_ego in gt_tracks.get(tid, []):
            if f == fi:
                pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
                obs_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
                errors.append(obs_world - gt_mean_world[tid])
                break
    slam_error[fi] = np.mean(errors, axis=0) if errors else np.zeros(3)

def output_ego_to_world_corrected(fi, pos_ego):
    """Output JSON Ego[fi] -> corrected World"""
    pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
    obs_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
    true_world = obs_world - slam_error[fi]
    return true_world

def output_ego_to_ego0_corrected(fi, pos_ego):
    """Output JSON Ego[fi] -> corrected Ego0"""
    pos_lidar = T_ego_lidar[:3,:3] @ pos_ego + T_ego_lidar[:3,3]
    obs_world = V[fi][:3,:3] @ pos_lidar + V[fi][:3,3]
    true_world = obs_world - slam_error[fi]
    pos_lidar0 = inv_V0[:3,:3] @ true_world + inv_V0[:3,3]
    pos_ego0 = T_lidar_ego[:3,:3] @ pos_lidar0 + T_lidar_ego[:3,3]
    return pos_ego0

# Check all tracks in output JSON
world_tracks = defaultdict(list)
ego0_tracks = defaultdict(list)

for f in data['frames']:
    fi = frame_id_to_idx[f['frameId']]
    for a in f['annotations']:
        tid = a['trackingId']
        pos_ego = np.array(a['position'])
        
        # Corrected World
        true_world = output_ego_to_world_corrected(fi, pos_ego)
        world_tracks[tid].append(true_world)
        
        # Corrected Ego0
        pos_ego0 = output_ego_to_ego0_corrected(fi, pos_ego)
        ego0_tracks[tid].append(pos_ego0)

print(f"Total unique tracking IDs in output: {len(world_tracks)}")

static_ids = []
moving_ids = []
for tid in world_tracks:
    poss_w = np.array(world_tracks[tid])
    poss_e0 = np.array(ego0_tracks[tid])
    if len(poss_w) < 5:
        continue
    base_w = poss_w[0]
    base_e0 = poss_e0[0]
    dists_w = np.linalg.norm(poss_w - base_w, axis=1)
    dists_e0 = np.linalg.norm(poss_e0 - base_e0, axis=1)
    max_w = np.max(dists_w)
    std_w = np.std(dists_w)
    max_e0 = np.max(dists_e0)
    std_e0 = np.std(dists_e0)
    if max_w < 1.0 and std_w < 0.3 and max_e0 < 1.0 and std_e0 < 0.3:
        static_ids.append((tid, len(poss_w), max_w, std_w, max_e0, std_e0))
    else:
        moving_ids.append((tid, len(poss_w), max_w, std_w, max_e0, std_e0))

print(f"\nCorrected World STATIC (max<1m, std<0.3m, frames>=5): {len(static_ids)}")
for tid, n, max_w, std_w, max_e0, std_e0 in sorted(static_ids, key=lambda x: x[2])[:30]:
    print(f"  #{tid}: n={n}, W_max={max_w:.3f}, W_std={std_w:.3f}, E0_max={max_e0:.3f}, E0_std={std_e0:.3f}")

print(f"\nCorrected World DYNAMIC: {len(moving_ids)}")
for tid, n, max_w, std_w, max_e0, std_e0 in sorted(moving_ids, key=lambda x: x[2], reverse=True)[:20]:
    print(f"  #{tid}: n={n}, W_max={max_w:.1f}, W_std={std_w:.2f}, E0_max={max_e0:.1f}, E0_std={std_e0:.2f}")

# GT check
print("\n=== GT objects ===")
for tid in [222, 376, 428, 441, 960]:
    if tid in world_tracks:
        arr = np.array(world_tracks[tid])
        base = arr[0]
        dists = np.linalg.norm(arr - base, axis=1)
        print(f"  #{tid}: n={len(arr)}, W_max={np.max(dists):.3f}, W_std={np.std(dists):.3f}")
        arr_e0 = np.array(ego0_tracks[tid])
        base_e0 = arr_e0[0]
        dists_e0 = np.linalg.norm(arr_e0 - base_e0, axis=1)
        print(f"        E0_max={np.max(dists_e0):.3f}, E0_std={np.std(dists_e0):.3f}")