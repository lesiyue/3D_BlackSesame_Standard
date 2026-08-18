#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D 標注閉環工具 - Python CLI 包裝腳本
用法：
    python run_pipeline.py A           # 流程 A：靜態優先
    python run_pipeline.py B           # 流程 B：動態優先
    python run_pipeline.py --list     # 列出可用流程
"""

import sys
import os
import subprocess
from pathlib import Path
from datetime import datetime

# --- 專案根目錄自動偵測 ---
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

# 記錄進度
progress_log = []

def log_step(step_num, total, title):
    """記錄步驟進度"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{step_num}/{total}] {timestamp} | {title}"
    progress_log.append(entry)
    print(f"\n{entry}")
    print("-" * 60)

def run_script(script_name, *args):
    """運行腳本並返回是否成功"""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        print(f"❌ 錯誤：找不到腳本 {script_path}")
        print(f"   預期路徑：{script_path}")
        return False
    
    cmd = [sys.executable, "-X", "utf8", str(script_path)] + list(args)
    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=300  # 5分鐘超時
        )
        
        if result.returncode != 0:
            print(f"❌ {script_name} 執行失敗")
            print(f"   錯誤輸出: {result.stderr[-500:] if len(result.stderr) > 500 else result.stderr}")
            return False
        
        # 輸出主要日誌（過濾掉過多的調試資訊）
        stdout_lines = result.stdout.split('\n')
        for line in stdout_lines:
            if any(kw in line for kw in ['▶', '✅', '❌', '完成', '統計', 'PCD', 'ID', 'GT']):
                print(line)
        
        return True
    
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} 執行超時 (超過 5 分鐘)")
        return False
    except Exception as e:
        print(f"❌ 執行 {script_name} 時發生錯誤: {e}")
        return False

def check_dependencies():
    """檢查必要的文件是否存在"""
    print("🔍 檢查項目依賴...")
    
    checks = [
        ("task_export_with_annots.json", DATA_DIR / "task_export_with_annots.json"),
        ("static_trackingId.txt", DATA_DIR / "static_trackingId.txt"),
    ]
    
    all_ok = True
    for name, path in checks:
        if path.exists():
            print(f"  ✅ {name}")
        else:
            print(f"⚠️  {name} - 不存在，某些流程可能無法運行")
            all_ok = False
    
    return all_ok

def flow_A():
    """流程 A：靜態優先
    步驟：pcd_cluster_v3.py → fix_static_classifier_v4.py → gen_import_json.py
    """
    print("\n" + "="*60)
    print("🚀 執行流程 A：靜態優先")
    print("="*60)
    print("適合場景：靜態物多、動態物少（如停車場、工廠場景}")
    print()
    
    steps = [
        (1, "階段 1/3: PCD 點雲聚類"),
        (2, "階段 2/3: GT 校正生成 fixed_static_classification.json"),
        (3, "階段 3/3: 生成導入文件 import_3d_boxes.json"),
    ]
    
    for step_num, title in steps:
        progress = f"流程 A - {step_num}/{len(steps)}: {title}"
        print(f"\n{progress}")
        print("-" * 60)
    
    # --- 階段 1: pcd_cluster_v3.py ---
    print("[階段 1/3] 執行 pcd_cluster_v3.py (聚類)...")
    ok = run_script("pcd_cluster_v3.py", "0.05", "0.25", "50", "-1.5", "1.2")
    if not ok:
        print("❌ 階段 1 失敗，終止流程")
        return False
    print("  ✅ 階段 1 完成：點雲聚類")
    
    # --- 階段 2: fix_static_classifier_v4.py ---
    print("\n[階段 2/3] 執行 fix_static_classifier_v4.py (GT 校正)...")
    ok = run_script("fix_static_classifier_v4.py")
    if not ok:
        print("❌ 階段 2 失敗，終止流程")
        return False
    print("  ✅ 階段 2 完成：GT 校正生成 fixed_static_classification.json")
    
    # --- 階段 3: gen_import_json.py ---
    print("\n[階段 3/3] 執行 gen_import_json.py (生成導入文件)...")
    ok = run_script("gen_import_json.py", "--classification", "fixed_static_classification.json")
    if not ok:
        print("❌ 階段 3 失敗，終止流程")
        return False
    print("  ✅ 階段 3 完成：導入文件已生成 data/import_3d_boxes.json")
    
    # 結語
    print("\n" + "="*60)
    print("✅ 流程 A 執行完整！")
    print("="*60)
    print("\n📋 下一步（人工）：")
    print("  1. 到 3D 標注平台執行命令 E (導入 3D 框)")
    print("  2. 選擇 data/import_3d_boxes.json")
    print("  3. 人工校驗 200 個 ID 框位置和尺寸")
    print("="*60 + "\n")
    
    return True

def flow_B():
    """流程 B：動態優先
    步驟：identify_dynamic_ids.py → gen_import_json.py
    """
    print("\n" + "="*60)
    print("🚀 執行流程 B：動態優先")
    print("="*60)
    print("適合場景：動態物多、靜態物少的場景")
    print("⚠️ 注意：可能會產生大量「誤判靜態」的 ID，")
    print("   導入平台後需要人工批量刪除")
    print()
    
    steps = [
        (1, "階段 1/2: 動態 ID 判定"),
        (2, "階段 2/2: 生成導入文件"),
    ]
    
    for step_num, title in steps:
        progress = f"流程 B - {step_num}/{len(steps)}: {title}"
        print(f"\n{progress}")
        print("-" * 60)
    
    # --- 階段 1: identify_dynamic_ids.py ---
    print("[階段 1/2] 執行 identify_dynamic_ids.py (動態判定)...")
    ok = run_script("identify_dynamic_ids.py", "--threshold", "0.5")
    if not ok:
        print("❌ 階段 1 失敗，終止流程")
        return False
    print("  ✅ 階段 1 完成：動態判定結果寫入 output/dynamic_priority_classification.json")
    
    # --- 階段 2: gen_import_json.py ---
    print("\n[階段 2/2] 執行 gen_import_json.py (生成導入文件)...")
    ok = run_script("gen_import_json.py", "--classification", "dynamic_priority_classification.json")
    if not ok:
        print("❌ 階段 2 失敗，終止流程")
        return False
    print("  ✅ 階段 2 完成：導入文件已生成 data/import_3d_boxes.json")
    
    # 結語
    print("\n" + "="*60)
    print("✅ 流程 B 執行完整！")
    print("="*60)
    print("\n📋 下一步（人工）：")
    print("  1. 到 3D 標注平台執行命令 E (導入 3D 框)")
    print("  2. 選擇 data/import_3d_boxes.json")
    print("  2. 🔍 檢查 ID 數量：流程 B 可能產生較多 ID（約 137-200+個）")
    print("  3. 🗑️  人工刪除誤判的靜態標記（流程 B 的特性）")
    print("  4. 完成校驗")
    print("="*60 + "\n")
    
    return True

def main():
    """主入口"""
    print("""
  ╔═══════════════════════════════════════════════════════════════════
  ║ 3D 標注閉環工具 - Python CLI 包裝腳本                              ║
  ║ 適合：對方電腦環境不確定、需要臨時調整、接受「半自動」模式      ║
  ║ 使用方式：python run_pipeline.py A / B                         ║
  ╚═══════════════════════════════════════════════════════════════════
""")
    
    # 參數解析
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  python run_pipeline.py A     # 流程 A：靜態優先")
        print("  python run_pipeline.py B     # 流程 B：動態優先")
        print("  python run_pipeline.py --list  # 列出說明")
        print()
        print("說明：")
        print("  - 流程 A：適合靜態物多的場景（停車場等）")
        print("  - 流程 B：適合動態物多的場景，會產生需人工刪除的誤判 ID")
        print("  - 執行完成後，必須手動到平台執行命令 E 導入 import_3d_boxes.json")
        sys.exit(1)
    
    option = sys.argv[1].upper()
    
    # 檢查依賴
    check_dependencies()
    
    # 執行對應流程
    if option == "A":
        success = flow_A()
    elif option == "B":
        success = flow_B()
    elif option == "--list" or option == "-h" or option == "--help":
        print("\n可用流程:")
        print("  A: 流程 A - 靜態優先 (推薦新手、靜態物多場景)")
        print("  B: 流程 B - 動態優先 (動態物多、需人工刪除誤判)")
        print("  --list: 顯示此說明")
        sys.exit(0)
    else:
        print(f"❌ 未知選項: {sys.argv[1]}")
        print("使用方法: python run_pipeline.py A / B")
        sys.exit(1)
    
    if success:
        print("\n🎉 任務完成！")
    else:
        print("\n⚠️ 任務異常終止，請檢查上述錯誤訊息")
        sys.exit(1)

if __name__ == "__main__":
    main()