# gen_import_json.py
# 生成 data/import_3d_boxes.json，供 Tampermonkey 命令 E 直接導入
import argparse
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict
import math
import uuid

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT  = ROOT / "output"

# 默認 classification 文件映射
DEFAULT_CLASSIFICATION = {
    "A": OUT / "fixed_static_classification.json",       # 流程 A
    "B": OUT / "dynamic_priority_classification.json",   # 流程 B
}

def mat(m): return np.array(m, dtype=np.float64).reshape(4, 4)

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

def q_mul(q1, q2):
    x1,y1,z1,w1 = q1; x2,y2,z2,w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ])

def load_all(classification_path: Path = None):
    # 1. 原始任务数据 (含 lioJson + 15帧标注)
    raw = json.load(open(DATA / "task_export_with_annots.json", encoding='utf-8'))
    lio_raw = raw['frames'][0]['lioJson']
    lio = json.loads(lio_raw) if isinstance(lio_raw, str) else lio_raw

    # 2. 讀 classification（流程 A 或 B）
    if classification_path is None:
        classification_path = DEFAULT_CLASSIFICATION["A"]
    if not classification_path.exists():
        print(f"找不到 {classification_path}，回退到 fixed_static_classification.json")
        classification_path = OUT / "fixed_static_classification.json"
    cls = json.load(open(classification_path, encoding='utf-8'))
    # 兼容兩種格式：
    #   流程 A（fixed_static_classification.json）: results[] 用 tid/verdict
    #   流程 B（dynamic_priority_classification.json）: ids[] 用 trackingId/label
    if 'ids' in cls:
        target_ids = {str(r['trackingId']) for r in cls['ids'] if r['label'] == 'STATIC'}
    elif 'results' in cls:
        target_ids = {str(r['tid']) for r in cls['results'] if r['verdict'] == 'STATIC'}
    else:
        raise ValueError(f"無法識別的 classification 格式：{list(cls.keys())}")
    print(f"使用 classification: {classification_path.name}")
    print(f"目標 IDs: {len(target_ids)} 個 (STATIC)")

    # 3. 构建矩阵链 T_ego_k_to_ego_0[k] : Ego[k] -> Ego[0]
    V = [mat(m) for m in lio['velo_pose_lidar']]          # World -> Lidar[k]
    T_lidar_ego = mat(lio['lidar2ego'])                   # Lidar -> Ego
    T_ego_lidar = np.linalg.inv(T_lidar_ego)              # Ego -> Lidar
    inv_V0 = np.linalg.inv(V[0])                          # Lidar[0] -> World

    T_ego_k_to_ego_0 = []
    for k in range(len(V)):
        # Ego[k] -> Lidar[k] -> World -> Lidar[0] -> Ego[0]
        T_lidar_k_to_lidar_0 = inv_V0 @ V[k]
        T = T_lidar_ego @ T_lidar_k_to_lidar_0 @ T_ego_lidar
        T_ego_k_to_ego_0.append(T)
    T_ego_k_to_ego_0[0] = np.eye(4)  # 确保单位阵

    # 4. 收集每个 target_id 在各帧的原始标注
    tracks = defaultdict(list)  # tid -> list of (frameIdx, annotation_dict)
    for f in raw['frames']:
        fi = f['frameIndex']
        for a in f.get('annotations', []) or []:
            tid = str(a.get('trackingId'))
            if tid in target_ids:
                tracks[tid].append((fi, a))

    return tracks, T_ego_k_to_ego_0

def transform_box(T, box):
    """box: dict with position[x,y,z], quaternion[x,y,z,w], scale[l,w,h]"""
    p = np.array(box['position'])
    p0 = T[:3,:3] @ p + T[:3,3]

    R = T[:3,:3]
    q_R = mat_to_quat(R)
    q_obj = np.array(box['quaternion'])
    q0 = q_mul(q_R, q_obj)
    q0 = q0 / np.linalg.norm(q0)  # 归一化

    return p0, q0

def load_size_refined():
    """讀取叠幀精算的 ID 尺寸（如果有）"""
    path = OUT / "id_size_refined.json"
    if not path.exists():
        return {}
    data = json.load(open(path, encoding='utf-8'))
    # 統一 key 為 str
    return {str(k): v.get("scale") for k, v in data.items() if v.get("scale")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=str, default="fixed_static_classification.json",
                        help="classification 文件名（或 A/B 簡寫）")
    args = parser.parse_args()

    # 解析 classification 路徑
    if args.classification in ("A", "B"):
        classification_path = DEFAULT_CLASSIFICATION[args.classification]
    else:
        classification_path = OUT / args.classification

    tracks, T_k_0 = load_all(classification_path)
    size_refined = load_size_refined()
    if size_refined:
        print(f"叠幀精算尺寸覆蓋：{sorted(size_refined.keys())}")

    # 預先讀取 frameId 映射，避免重複讀取檔案
    raw_data = json.load(open(DATA / "task_export_with_annots.json", encoding='utf-8'))
    frame_id_map = {f['frameIndex']: f['frameId'] for f in raw_data['frames']}

    frames_out = []  # 最終導出格式
    # 先按 frameIndex 聚合
    frame_map = defaultdict(list)  # frameIdx -> list of box_dict_for_export

    for tid, lst in tracks.items():
        if len(lst) < 1: continue

        # 选基准帧：points 最大
        base_fi, base_ann = max(lst, key=lambda x: x[1].get('userData',{}).get('points', 0))

        # 基准帧的“黄金标准” (在 Ego[base] 系下)
        gold_pos = np.array(base_ann['position'])
        gold_quat = np.array(base_ann['quaternion']) / np.linalg.norm(base_ann['quaternion'])
        gold_scale = np.array(base_ann['scale'])

        # 将黄金标准传播到所有 15 帧
        for fi in range(15):
            # 1. 黄金标准 Ego[base] -> Ego[0]
            T_base_0 = T_k_0[base_fi]
            p0 = T_base_0[:3,:3] @ gold_pos + T_base_0[:3,3]
            q0 = q_mul(mat_to_quat(T_base_0[:3,:3]), gold_quat)

            # 2. Ego[0] -> Ego[fi]
            T_0_fi = np.linalg.inv(T_k_0[fi])
            p_fi = T_0_fi[:3,:3] @ p0 + T_0_fi[:3,3]
            q_fi = q_mul(mat_to_quat(T_0_fi[:3,:3]), q0)
            q_fi = q_fi / np.linalg.norm(q_fi)

            # 3. 组装导出对象：基于原始标注深拷贝，仅更新 position/quaternion/scale
            # 找到该帧的原始标注（用于保留 readonly, lastUpdateTs, editConfig, userData 等）
            orig_ann = None
            for ffi, ann in lst:
                if ffi == fi:
                    orig_ann = ann
                    break
            
            if orig_ann is not None:
                # 深拷贝原始标注，保留所有字段
                export_ann = json.loads(json.dumps(orig_ann))
            else:
                # 该帧原本没有此 track，基于基准帧创建，保留关键字段
                export_ann = json.loads(json.dumps(base_ann))
                # 缺失帧：可见度设为空，frameId 修正为当前帧，points 清空
                if "userData" in export_ann:
                    if "attributes" in export_ann["userData"]:
                        export_ann["userData"]["attributes"]["visible_1"] = ""
                    export_ann["userData"]["frameId"] = frame_id_map.get(fi, "")
                    export_ann["userData"]["points"] = 0
            
            # 仅更新 position, quaternion, scale
            export_ann["position"] = [float(x) for x in p_fi]
            export_ann["quaternion"] = [float(x) for x in q_fi]
            # scale：優先用叠幀精算結果（保持原始 float64 精度），否則用原值
            if str(tid) in size_refined and size_refined[str(tid)] is not None:
                export_ann["scale"] = [float(x) for x in size_refined[str(tid)]]
            else:
                export_ann["scale"] = [float(x) for x in gold_scale]
            
            # uuid 保持原样（如果有），否则新建
            if "uuid" not in export_ann or not export_ann["uuid"]:
                export_ann["uuid"] = str(uuid.uuid4())
            
            frame_map[fi].append(export_ann)

    # 按 frameIndex 排序生成最終結構
    for fi in range(15):
        frame_id = frame_id_map.get(fi)
        frames_out.append({
            "frameId": frame_id,
            "annotations": frame_map.get(fi, [])
        })

    out_json = {"frames": frames_out}
    out_path = DATA / "import_3d_boxes.json"
    out_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding='utf-8')
    total_boxes = sum(len(f['annotations']) for f in frames_out)
    print(f"\n✅ 生成完成: {out_path}")
    print(f"   15 帧 × {len(tracks)} 个目标 = {total_boxes} 个框")
    print(f"   直接拖入 Tampermonkey 面板 -> 导入 3D 框 (E)")

if __name__ == '__main__':
    main()