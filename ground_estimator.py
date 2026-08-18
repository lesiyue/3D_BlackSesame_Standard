# ground_estimator.py — 從 PCD 擬合地面平面
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pcd_decoder import read_pcd

def fit_ground_plane_ransac(points, n_iter=100, dist_thresh=0.1, sample_size=3):
    """
    RANSAC 擬合地面平面: ax + by + cz + d = 0, 回傳 (a,b,c,d) 且 c > 0 (法向量朝上)
    points: Nx3 array
    """
    best_inliers = 0
    best_plane = None
    n = len(points)
    if n < sample_size:
        return None
    
    for _ in range(n_iter):
        idx = np.random.choice(n, sample_size, replace=False)
        pts = points[idx]
        # 三點定平面
        v1 = pts[1] - pts[0]
        v2 = pts[2] - pts[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        # 保證法向量朝上 (z 分量為正)
        if normal[2] < 0:
            normal = -normal
        d = -np.dot(normal, pts[0])
        # 計算內點
        dists = np.abs(points @ normal + d)
        inliers = np.sum(dists < dist_thresh)
        if inliers > best_inliers:
            best_inliers = inliers
            best_plane = (normal[0], normal[1], normal[2], d)
    return best_plane


def fit_ground_plane_percentile(points, z_percentile=5, xy_radius=30.0):
    """
    簡單且穩健: 取距離 ego 附近 (xy_radius) 最低 z_percentile 的點擬合平面
    適用於激光雷達地面點密集、平坦區域
    回傳 (a,b,c,d) 法向量歸一化, c > 0
    """
    # 只取車輛附近的點
    xy_dist = np.sqrt(points[:,0]**2 + points[:,1]**2)
    near = points[xy_dist < xy_radius]
    if len(near) < 100:
        near = points  # fallback
    
    # 取最低 z_percentile 的點作為地面候選
    z_thresh = np.percentile(near[:,2], z_percentile)
    ground_pts = near[near[:,2] < z_thresh + 0.2]  # 留一些厚度
    if len(ground_pts) < 50:
        ground_pts = near[near[:,2] < z_thresh + 0.5]
    
    # PCA 擬合平面
    centered = ground_pts - ground_pts.mean(axis=0)
    cov = centered.T @ centered / len(ground_pts)
    eigvals, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]  # 最小特徵值對應的特徵向量
    if normal[2] < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    d = -np.dot(normal, ground_pts.mean(axis=0))
    return (normal[0], normal[1], normal[2], d)


def get_ground_height_at(plane, x, y):
    """給定平面 (a,b,c,d) 和 x,y, 回傳地面 z"""
    a, b, c, d = plane
    if abs(c) < 1e-6:
        return 0.0
    return -(a*x + b*y + d) / c


def estimate_ground_per_frame(pcd_dir, frame_indices=None, select_frame=None, method='percentile', **kwargs):
    """
    批次估算每幀地面平面
    frame_indices: annotation 的 frameIndex (0-14)
    select_frame: 對應的 PCD 檔案編號 [171, 241, ...]
    回傳: {frame_idx: (a,b,c,d)}
    """
    pcd_dir = Path(pcd_dir)
    if select_frame is None:
        select_frame = list(range(len(frame_indices))) if frame_indices else []
    if frame_indices is None:
        frame_indices = list(range(len(select_frame)))
    
    planes = {}
    for fi, pcd_num in zip(frame_indices, select_frame):
        pcd_path = pcd_dir / f"{pcd_num:05d}.pcd"
        if not pcd_path.exists():
            print(f"Warning: {pcd_path.name} not found")
            planes[fi] = (0.0, 0.0, 1.0, 0.0)
            continue
        
        pts, info = read_pcd(pcd_path)
        if not pts:
            print(f"Warning: {pcd_path.name} read failed")
            planes[fi] = (0.0, 0.0, 1.0, 0.0)
            continue
        
        pts_np = np.array([(p['x'], p['y'], p['z']) for p in pts], dtype=np.float32)
        
        if method == 'percentile':
            plane = fit_ground_plane_percentile(pts_np, **kwargs)
        elif method == 'ransac':
            plane = fit_ground_plane_ransac(pts_np, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if plane is None:
            plane = (0.0, 0.0, 1.0, 0.0)
        planes[fi] = plane
        a,b,c,d = plane
        print(f"  f{fi} (pcd_{pcd_num}): plane=({a:.4f}, {b:.4f}, {c:.4f}, {d:.4f})  "
              f"height@ego={get_ground_height_at(plane, 0, 0):.3f}")
    
    return planes


if __name__ == '__main__':
    import sys
    ROOT = Path(__file__).resolve().parent.parent
    PCD_DIR = ROOT / "data" / "pcd"
    
    # 從 task_export 讀取 select_frame
    import json
    raw = json.load(open(ROOT / "data" / "task_export_with_annots.json", encoding='utf-8'))
    lio_raw = raw['frames'][0]['lioJson']
    lio = json.loads(lio_raw) if isinstance(lio_raw, str) else lio_raw
    select_frame = lio['select_frame']
    
    planes = estimate_ground_per_frame(PCD_DIR, frame_indices=list(range(15)), select_frame=select_frame, 
                                        method='percentile', z_percentile=5, xy_radius=30.0)