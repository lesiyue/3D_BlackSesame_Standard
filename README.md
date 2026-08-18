# 📚 3D 標注閉環工具 - 用戶指南

## 1. 工具是做什麼的

這個工具用於 3D 點雲標注的**閉環校驗**：

```
標注平台（人工） → 平台導出 → Python 處理 → 平台導入（人工校驗）
```

**核心場景**：採集車給出的 ID 框不準確，需要：
- 判定哪些 ID 框是**靜態物**（路燈、建築、交通標誌）
- 對**靜態物**做叠幀精算，得到更準確的尺寸
- 重新導入平台，人工做最後校驗

---

## 2. 安裝（僅首次需要）

### 2.1 安裝 Python

如果電腦上沒有 Python：
1. 下載：https://www.python.org/downloads/
2. 安裝時**勾選「Add Python to PATH」**
3. 推薦 Python 3.9 或以上

### 2.2 安裝依賴

雙擊 **`install.bat`**，它會自動安裝：
- fastapi（Web 框架）
- uvicorn（Web 服務器）
- pyyaml（配置文件讀取）
- numpy（數值計算）
- open3d（點雲處理）

---

## 3. 啟動

雙擊 **`start.bat`**，會自動：
1. 啟動 Web 服務（默認 http://127.0.0.1:8765）
2. 打開瀏覽器到主頁

如果沒有自動打開瀏覽器，手動訪問：http://127.0.0.1:8765

---

## 4. 使用流程

### 4.1 前期準備（純人工，在標注平台完成）

#### 步驟 1：平台導出

在 3D 標注平台任務頁：
- 點擊 Tampermonkey 面板 → **`导出 lioJson + 标注 汇总 (D)`**
- 保存文件為 `data/task_export_with_annots.json`

#### 步驟 2：人工挑選 GT ID

打開主頁的 **GT ID 列表** 區域：

**靜態 GT**（必須填）：
- 從平台識別結果中挑選**確認是靜態**的 ID 框
- 例如：路燈、建築、交通標誌、停著的車
- 填入 `static_trackingId.txt`

**動態 GT**（可空）：
- 從平台識別結果中挑選**確認是動態**的 ID 框
- 例如：行人、移動中的車、騎車的人
- 填入 `dynamic_trackingId.txt`

> 💡 **動態 GT 可以留空**。流程 B 只會用「靜態白名單 + 自動位移判定」。

### 4.3 執行 Python pipeline（CLI 方式）

如果不想使用 Web UI，可以直接使用命令列工具：

#### 選項 A：使用 `run_pipeline.py`（推薦，一個命令跑完）

```bash
# 流程 A（靜態優先，推薦新手）
python run_pipeline.py A

# 流程 B（動態優先）
python run_pipeline.py B
```

**流程 A 的執行結果**：
- 步驟 1/3：PCD 點雲聚類 (~80 秒)
- 步驟 2/3：GT 校正生成 fixed_static_classification.json
- 步驟 3/3：生成導入文件 import_3d_boxes.json (3000 個框)

**流程 B 的執行結果**：
- 步驟 1/2：動態判定（~30 秒）
- 步驟 2/2：生成導入文件 (2055 個框)
- ⚠️ 注意：流程 B 可能會產生大量誤判靜態 ID，導入平台後需要人工批量刪除

```bash
# 查看說明
python run_pipeline.py --list
```

```bash
# 運行後的輸出示例
▶ 步驟 1/3: 靜止判定
▶ 階段 1/3: 0.05m 高精度聚類
▶ 階段 2/3: 全 ID 自動判定
▶ 階段 3/3: GT 校正生成 fixed_static_classification.json
✅ 完成：137 STATIC + 361 DYNAMIC = 498 總計
📁 輸出：data/import_3d_boxes.json (15 帧 × 137 個目標 = 2055 個框)
 直接拖入 Tampermonkey 面板 -> 導入 3D 框 (E)
```

#### 步驟 3：平台導入

回到 3D 標注平台任務頁：
- 點擊 Tampermonkey 面板 → **`导入 3D 框 (E)`**
- 選擇 `data/import_3d_boxes.json`

#### 步驟 4：人工校驗

> ⚠️ **這一步是必須的**：所有流程導入後都需要人工校驗位置和尺寸。

- 查看框位置是否正確
- 檢查尺寸是否合理
- 流程 B 時：批量刪除「誤判靜態」的 ID
- 完成！

### 4.5 完整流程圖

```
前期人工
│
├─► 點平台按鈕 D 導出 → data/task_export_with_annots.json
│
├─► 填入 GT ID：static_trackingId.txt / dynamic_trackingId.txt
│
├─► 選擇流程（在 UI 或 CLI）
│   ├─ 流程 A：靜態優先（推薦，3步，約2分鐘）
│   │   ├─ pcd_cluster_v3.py (聚類)
│   │   └─ fix_static_classifier_v4.py (GT校正)
│   │
│   └─ 流程 B：動態優先（2步，約40秒）
│       ├─ identify_dynamic_ids.py (位移判定)
│       └─ gen_import_json.py (生成導入文件)
│
├─► 執行 Python CLI (run_pipeline.py A / B)
│   └─ 生成 data/import_3d_boxes.json
│
└─► 平台導入（命令 E）
    └─ ► 人工校驗位置和尺寸
```

---

## 5. 文件說明

### 5.1 核心文件說明

| 文件 | 作用 |
|---|---|
| `config.yaml` | 統一參數配置（核心），所有腳本啟動時讀取 |
| `server.py` | FastAPI 服務器（後端核心），提供 REST API + UI |
| `start.bat` | 雙擊啟動服務器 |
| `install.bat` | 自動安裝依賴 |
| `dynamic_trackingId.txt` | 動態 GT 列表（可空） |
| `static_trackingId.txt` | 靜態 GT 列表（固定：376 222 428 441 960） |
| `data\` | 原始數據（task_export + pcd 文件） |
| `output\` | Python 處理結果 |
| `scripts\` | Python 處理腳本 |
| `static\` | UI 樣式（main.css, main.js） |
| `templates\` | 網頁主頁面 index.html |
| `run_pipeline.py` | Python CLI 包裝（一個命令跑完） |

### 5.2 關鍵文件說明

#### `run_pipeline.py`
- **位置**：`D:\X\3D_BlackSesame_Standard\run_pipeline.py`
- **作用**：Python CLI 包裝腳本，一個命令跑完整個流程
- **使用方式**：
  - `python run_pipeline.py A` - 流程 A（靜態優先）
  - `python run_pipeline.py B` - 流程 B（動態優先）
- **優點**：一個命令跑完、簡單可靠、不用 GUI
- **缺點**：沒有視覺化、油猴端還是要手動點導入

#### `static/main.css`
- **位置**：`D:\X\3D_BlackSesame_Standard\static\main.css`
- **作用**：深色主題、卡片佈局、標籤芯、警告橫幅、響應式 grid

#### `static/main.js`
- **位置**：`D:\X\3D_BlackSesame_Standard\static\main.js`
- **作用**：GT ID 編輯、工作流單選、SSE 日誌訂閱、狀態輪詢、報告渲染

#### `docs/USER_GUIDE.md`
- **位置**：`D:\X\3D_BlackSesame_Standard\docs\USER_GUIDE.md`
- **作用**：圖文指南，包含安裝、啟動、流程、常見問題

### 5. 常見問題

| 問題 | 排查 |
|---|---|
| ** server 啟動失敗** | 確認 `install.bat` 已運行過 |
| **找不到 task_export_with_annots.json** | 確認平台導出的文件保存到了正確位置 |
| **報「找不到 pcd 文件」** | 確認 `data/pcd/` 下有 .pcd 文件 |
| **日誌持續報錯** | 把日誌複製出來分析 |
| **UI 點按鈕沒反應** | 開瀏覽器開發者工具 (F12) 查看 Console |

---

## 6. 跨電腦部署

把整個工程文件夾複製到目標電腦：

1. 確認目標電腦有 Python 3.9+
2. 雙擊 `install.bat`（自動裝依賴）
3. 雙擊 `start.bat`（啟動）
4. 把 `data/` 目錄下的平台導出文件複製過來

**注意**：所有路徑都是相對路徑，工程放在哪都能跑。

---

## 7. 故障排除

| 症狀 | 排查 |
|---|---|
| `start.bat` 一閃而過 | 看是否有 Python；用管理員身份運行 |
| 報「找不到 open3d」 | 重跑 `install.bat` |
| 報「找不到 pcd 文件」 | 確認 `data/pcd/` 下有 .pcd 文件 |
| 日誌持續報錯 | 把日誌複製出來分析 |
| UI 點按鈕沒反應 | 開瀏覽器開發者工具 (F12) 看 Console |

---

## 8. 更新日誌

| 版本 | 日期 | 變更內容 |
|------|------|----------|
| v1.0 | 2026-08-12 | 初始版本，包含流程 A/B、CLI 工具、UI 界面 |
