"""
3D 標注閉環工具 - Web 服務器
雙擊 start.bat 啟動，自動打開瀏覽器
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import yaml
    import uvicorn
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
except ImportError:
    print("=" * 60)
    print("缺少依賴！請先雙擊 install.bat 安裝")
    print("=" * 60)
    input("按 Enter 退出...")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

app = FastAPI(title="3D 標注閉環工具", version="1.0")

app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(ROOT / "templates"))

# ====================== 日誌系統 ======================
log_subscribers = []
log_history = []


def add_log(level, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = {"ts": ts, "level": level, "msg": msg}
    log_history.append(entry)
    if len(log_history) > 500:
        log_history.pop(0)
    for q in log_subscribers:
        try:
            q.put_nowait(entry)
        except asyncio.QueueFull:
            pass


# ====================== 狀態 ======================
state = {
    "running": False,
    "current_step": None,
    "last_result": None,
}


# ====================== GT ID 文件讀寫 ======================
def read_gt_file(path: Path) -> list:
    """讀 GT ID 文件，返回 list[str]；空文件返回 []"""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    return [x for x in text.split() if x]


def write_gt_file(path: Path, ids: list):
    """寫 GT ID 文件，一行空格分隔"""
    path.write_text(" ".join(ids), encoding="utf-8")


# ====================== 路由 ======================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"config": CONFIG})


@app.get("/api/status")
async def api_status():
    data_dir = ROOT / CONFIG["paths"]["data_dir"]
    output_dir = ROOT / CONFIG["paths"]["output_dir"]
    files_cfg = CONFIG["files"]
    return {
        "running": state["running"],
        "current_step": state["current_step"],
        "last_result": state["last_result"],
        "files": {
            "task_export_exists": (data_dir / files_cfg["task_export"]).exists(),
            "static_classification_exists": (output_dir / files_cfg["static_classification"]).exists(),
            "dynamic_classification_exists": (output_dir / files_cfg["dynamic_classification"]).exists(),
            "id_size_refined_exists": (output_dir / files_cfg["id_size_refined"]).exists(),
            "dynamic_output_exists": (data_dir / files_cfg["dynamic_output"]).exists(),
        },
        "log_count": len(log_history),
    }


@app.get("/api/logs/history")
async def api_log_history():
    return JSONResponse(log_history[-100:])


@app.get("/api/logs/stream")
async def api_log_stream():
    async def event_generator():
        q = asyncio.Queue(maxsize=200)
        log_subscribers.append(q)
        try:
            for entry in log_history[-50:]:
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
            while True:
                try:
                    entry = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            log_subscribers.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/config")
async def api_get_config():
    return CONFIG


@app.post("/api/config")
async def api_save_config(request: Request):
    new_config = await request.json()
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(new_config, f, allow_unicode=True, default_flow_style=False)
    global CONFIG
    CONFIG = new_config
    add_log("ok", "配置已保存（重啟後生效）")
    return {"ok": True}


@app.get("/api/gt/static")
async def api_get_static_gt():
    path = ROOT / CONFIG["paths"]["data_dir"] / CONFIG["files"]["static_gt"]
    return {"ids": read_gt_file(path)}


@app.post("/api/gt/static")
async def api_set_static_gt(request: Request):
    data = await request.json()
    ids = data.get("ids", [])
    path = ROOT / CONFIG["paths"]["data_dir"] / CONFIG["files"]["static_gt"]
    write_gt_file(path, [str(x) for x in ids])
    add_log("ok", f"靜態 GT 已保存 ({len(ids)} 個): {ids}")
    return {"ok": True, "count": len(ids)}


@app.get("/api/gt/dynamic")
async def api_get_dynamic_gt():
    path = ROOT / CONFIG["paths"]["data_dir"] / CONFIG["files"]["dynamic_gt"]
    return {"ids": read_gt_file(path)}


@app.post("/api/gt/dynamic")
async def api_set_dynamic_gt(request: Request):
    data = await request.json()
    ids = data.get("ids", [])
    path = ROOT / CONFIG["paths"]["data_dir"] / CONFIG["files"]["dynamic_gt"]
    write_gt_file(path, [str(x) for x in ids])
    add_log("ok", f"動態 GT 已保存 ({len(ids)} 個): {ids}")
    return {"ok": True, "count": len(ids)}


@app.get("/api/workflow")
async def api_get_workflow():
    return CONFIG["workflow"]


@app.post("/api/workflow")
async def api_set_workflow(request: Request):
    """保存工作流選擇到 config.yaml"""
    data = await request.json()
    CONFIG["workflow"].update(data)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(CONFIG, f, allow_unicode=True, default_flow_style=False)
    add_log("ok", f"工作流配置已保存: {data}")
    return {"ok": True}


@app.get("/api/report/static")
async def api_report_static():
    """讀取流程 A 的 classification"""
    output_dir = ROOT / CONFIG["paths"]["output_dir"]
    path = output_dir / CONFIG["files"]["static_classification"]
    if not path.exists():
        return {"error": "靜態判定尚未運行"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/report/dynamic")
async def api_report_dynamic():
    """讀取流程 B 的 classification"""
    output_dir = ROOT / CONFIG["paths"]["output_dir"]
    path = output_dir / CONFIG["files"]["dynamic_classification"]
    if not path.exists():
        return {"error": "動態判定尚未運行"}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ====================== 執行子進程 ======================
def run_subprocess(cmd, cwd=None):
    add_log("info", f"▶ 執行: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd or str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                level = "error" if any(k in line.lower() for k in ["error", "fail", "崩潰"]) else "info"
                if any(k in line for k in ["完成", "成功"]):
                    level = "ok"
                add_log(level, line)
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        add_log("error", f"執行失敗: {e}")
        return False


# ====================== Pipeline 步驟 ======================
async def run_workflow_a():
    """流程 A：靜態優先
    用 static_trackingId.txt 找其他靜態物
    """
    scripts_dir = ROOT / CONFIG["paths"]["scripts_dir"]
    output_dir = ROOT / CONFIG["paths"]["output_dir"]
    data_dir = ROOT / CONFIG["paths"]["data_dir"]

    cfg = CONFIG["static"]["pcd_cluster"]

    add_log("info", "▶ 階段 1/2: 0.05m 高精度聚類")
    ok = await asyncio.to_thread(
        run_subprocess,
        [sys.executable, "-X", "utf8", "pcd_cluster_v3.py",
         str(cfg["voxel_size"]), str(cfg["eps"]), str(cfg["min_samples"]),
         str(cfg["ground_thresh"]), str(cfg["max_cost"])],
        cwd=str(scripts_dir),
    )
    if not ok:
        add_log("error", "pcd_cluster_v3.py 失敗")
        return False

    add_log("info", "▶ 階段 2/2: GT 校正生成 fixed_static_classification.json")
    ok = await asyncio.to_thread(
        run_subprocess,
        [sys.executable, "-X", "utf8", "fix_static_classifier_v4.py"],
        cwd=str(data_dir),
    )
    if not ok:
        add_log("error", "fix_static_classifier_v4.py 失敗")
        return False

    add_log("ok", "流程 A 完成")
    return True


async def run_workflow_b():
    """流程 B：動態優先
    用 dynamic_trackingId.txt，剩下 ID 全標靜態
    """
    data_dir = ROOT / CONFIG["paths"]["data_dir"]
    output_dir = ROOT / CONFIG["paths"]["output_dir"]

    threshold = CONFIG["workflow"].get("displacement_threshold", 0.5)

    add_log("info", f"▶ 流程 B：動態優先（位移閾值 {threshold}m）")
    ok = await asyncio.to_thread(
        run_subprocess,
        [sys.executable, "-X", "utf8", "identify_dynamic_ids.py",
         "--threshold", str(threshold)],
        cwd=str(ROOT),
    )
    if not ok:
        add_log("error", "identify_dynamic_ids.py 失敗")
        return False

    add_log("ok", "流程 B 完成")
    return True


async def run_size_refine():
    """叠幀精算 ID 尺寸"""
    data_dir = ROOT / CONFIG["paths"]["data_dir"]
    frames = CONFIG["workflow"].get("size_frames", [7, 15])
    workflow_selected = CONFIG["workflow"].get("selected", "A")

    add_log("info", f"▶ 叠幀精算：幀段 {frames[0]}-{frames[1]}，基於流程 {workflow_selected} 的 GT")
    ok = await asyncio.to_thread(
        run_subprocess,
        [sys.executable, "-X", "utf8", "compute_size_from_anchors.py",
         "--start", str(frames[0]), "--end", str(frames[1])],
        cwd=str(ROOT),
    )
    if not ok:
        add_log("error", "compute_size_from_anchors.py 失敗")
        return False

    add_log("ok", "尺寸精算完成")
    return True


async def run_gen_import():
    """生成導入文件"""
    data_dir = ROOT / CONFIG["paths"]["data_dir"]
    output_dir = ROOT / CONFIG["paths"]["output_dir"]
    workflow_selected = CONFIG["workflow"].get("selected", "A")

    if workflow_selected == "A":
        classification_file = CONFIG["files"]["static_classification"]
    else:
        classification_file = CONFIG["files"]["dynamic_classification"]

    add_log("info", f"▶ 生成導入文件（使用 {classification_file}）")
    ok = await asyncio.to_thread(
        run_subprocess,
        [sys.executable, "-X", "utf8", "gen_import_json.py",
         "--classification", classification_file],
        cwd=str(data_dir),
    )
    if not ok:
        add_log("error", "gen_import_json.py 失敗")
        return False

    add_log("ok", "導入文件已生成")
    return True


# ====================== API 入口 ======================
@app.post("/api/run/classify")
async def api_run_classify(request: Request):
    """執行選中的流程"""
    if state["running"]:
        return {"error": "已有任務在跑"}

    data = await request.json()
    workflow = data.get("workflow", CONFIG["workflow"].get("selected", "A"))

    state["running"] = True
    state["current_step"] = "classify"

    async def _run():
        try:
            add_log("info", "=" * 50)
            if workflow == "A":
                await run_workflow_a()
            else:
                await run_workflow_b()
        except Exception as e:
            add_log("error", f"異常: {e}")
        finally:
            state["running"] = False
            state["current_step"] = None

    asyncio.create_task(_run())
    return {"ok": True, "started": True, "workflow": workflow}


@app.post("/api/run/size")
async def api_run_size():
    """執行尺寸精算"""
    if state["running"]:
        return {"error": "已有任務在跑"}

    state["running"] = True
    state["current_step"] = "size"

    async def _run():
        try:
            await run_size_refine()
        except Exception as e:
            add_log("error", f"異常: {e}")
        finally:
            state["running"] = False
            state["current_step"] = None

    asyncio.create_task(_run())
    return {"ok": True, "started": True}


@app.post("/api/run/gen-import")
async def api_run_gen_import():
    """生成導入文件"""
    if state["running"]:
        return {"error": "已有任務在跑"}

    state["running"] = True
    state["current_step"] = "gen-import"

    async def _run():
        try:
            await run_gen_import()
        except Exception as e:
            add_log("error", f"異常: {e}")
        finally:
            state["running"] = False
            state["current_step"] = None

    asyncio.create_task(_run())
    return {"ok": True, "started": True}


@app.post("/api/run/all")
async def api_run_all():
    """一鍵全跑：流程 + 尺寸精算 + 生成導入"""
    if state["running"]:
        return {"error": "已有任務在跑"}

    state["running"] = True
    state["current_step"] = "all"
    workflow = CONFIG["workflow"].get("selected", "A")
    enable_size = CONFIG["workflow"].get("enable_size_refine", True)

    async def _run():
        try:
            add_log("info", "=" * 50)
            add_log("info", f"🚀 一鍵全跑（流程 {workflow}，尺寸精算 {'開' if enable_size else '關'}）")

            if workflow == "A":
                ok = await run_workflow_a()
            else:
                ok = await run_workflow_b()
            if not ok:
                return

            if enable_size:
                ok = await run_size_refine()
                if not ok:
                    return

            ok = await run_gen_import()
            if not ok:
                return

            add_log("ok", "全部完成！請到平台導入 import_3d_boxes.json")
            state["last_result"] = {"step": "all", "ok": True, "ts": time.time()}
        except Exception as e:
            add_log("error", f"異常: {e}")
        finally:
            state["running"] = False
            state["current_step"] = None

    asyncio.create_task(_run())
    return {"ok": True, "started": True}


# ====================== 啟動 ======================
def open_browser(host, port):
    import webbrowser
    time.sleep(1)
    webbrowser.open(f"http://{host}:{port}")


if __name__ == "__main__":
    host = CONFIG["server"]["host"]
    port = CONFIG["server"]["port"]
    add_log("info", "服務啟動中...")
    add_log("info", f"工作目錄: {ROOT}")
    add_log("info", f"訪問: http://{host}:{port}")

    if CONFIG["server"].get("open_browser", True):
        import threading
        threading.Thread(target=open_browser, args=(host, port), daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="warning")
