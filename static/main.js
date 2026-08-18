// ====================== GT ID 管理 ======================
let staticGT = [];
let dynamicGT = [];
let workflowConfig = { selected: "A", enable_size_refine: true, size_frames: [7, 15], displacement_threshold: 0.5 };

async function loadGT() {
    const [s, d, w] = await Promise.all([
        fetch("/api/gt/static").then(r => r.json()),
        fetch("/api/gt/dynamic").then(r => r.json()),
        fetch("/api/workflow").then(r => r.json()),
    ]);
    staticGT = s.ids || [];
    dynamicGT = d.ids || [];
    workflowConfig = w;
    renderGT();
    renderWorkflow();
}

function renderGT() {
    renderTags("static-gt-tags", staticGT, "static");
    renderTags("dynamic-gt-tags", dynamicGT, "dynamic");
}

function renderTags(containerId, ids, type) {
    const c = document.getElementById(containerId);
    c.innerHTML = "";
    ids.forEach(id => {
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.innerHTML = `${id}<span class="remove" onclick="removeGT('${type}', '${id}')">×</span>`;
        c.appendChild(tag);
    });
}

function renderWorkflow() {
    document.querySelector(`input[name="workflow"][value="${workflowConfig.selected}"]`).checked = true;
    document.getElementById("enable-size-refine").checked = workflowConfig.enable_size_refine;
    document.getElementById("frame-start").value = workflowConfig.size_frames[0];
    document.getElementById("frame-end").value = workflowConfig.size_frames[1];
    document.getElementById("displacement-threshold").value = workflowConfig.displacement_threshold;
    updateThresholdVisibility();
}

function updateThresholdVisibility() {
    const selected = document.querySelector('input[name="workflow"]:checked').value;
    document.getElementById("threshold-row").style.display = selected === "B" ? "flex" : "none";
}

document.querySelectorAll('input[name="workflow"]').forEach(r => {
    r.addEventListener("change", updateThresholdVisibility);
});

async function addStaticGT() {
    const input = document.getElementById("static-gt-input");
    const id = input.value.trim();
    if (!id) return;
    if (!staticGT.includes(id)) staticGT.push(id);
    input.value = "";
    renderGT();
    await fetch("/api/gt/static", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: staticGT }) });
}

async function addDynamicGT() {
    const input = document.getElementById("dynamic-gt-input");
    const id = input.value.trim();
    if (!id) return;
    if (!dynamicGT.includes(id)) dynamicGT.push(id);
    input.value = "";
    renderGT();
    await fetch("/api/gt/dynamic", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: dynamicGT }) });
}

async function removeGT(type, id) {
    if (type === "static") staticGT = staticGT.filter(x => x !== id);
    else dynamicGT = dynamicGT.filter(x => x !== id);
    renderGT();
    await fetch(`/api/gt/${type}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ ids: type === "static" ? staticGT : dynamicGT }) });
}

document.getElementById("static-gt-input").addEventListener("keypress", e => { if (e.key === "Enter") addStaticGT(); });
document.getElementById("dynamic-gt-input").addEventListener("keypress", e => { if (e.key === "Enter") addDynamicGT(); });

async function saveWorkflow() {
    const selected = document.querySelector('input[name="workflow"]:checked').value;
    const enable_size_refine = document.getElementById("enable-size-refine").checked;
    const size_frames = [
        parseInt(document.getElementById("frame-start").value),
        parseInt(document.getElementById("frame-end").value),
    ];
    const displacement_threshold = parseFloat(document.getElementById("displacement-threshold").value);
    workflowConfig = { selected, enable_size_refine, size_frames, displacement_threshold };
    await fetch("/api/workflow", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(workflowConfig) });
}

// ====================== 執行按鈕 ======================
async function runClassify() {
    await saveWorkflow();
    const workflow = document.querySelector('input[name="workflow"]:checked').value;
    setRunning(true);
    await fetch("/api/run/classify", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ workflow }) });
}

async function runSize() {
    await saveWorkflow();
    setRunning(true);
    await fetch("/api/run/size", { method: "POST" });
}

async function runGenImport() {
    await saveWorkflow();
    setRunning(true);
    await fetch("/api/run/gen-import", { method: "POST" });
}

async function runAll() {
    await saveWorkflow();
    if (!confirm("一鍵全跑：執行流程 + 尺寸精算 + 生成導入。確認？")) return;
    setRunning(true);
    await fetch("/api/run/all", { method: "POST" });
}

function setRunning(running) {
    document.getElementById("btn-classify").disabled = running;
    document.getElementById("btn-all").disabled = running;
    document.getElementById("btn-gen-import").disabled = running;
    document.getElementById("running-indicator").className = running ? "dot dot-running" : "dot dot-idle";
    document.getElementById("running-text").textContent = running ? "運行中..." : "待命";
}

// ====================== 日誌 SSE ======================
function appendLog(entry) {
    const c = document.getElementById("log-container");
    const line = document.createElement("div");
    line.className = `log-line ${entry.level}`;
    line.innerHTML = `<span class="ts">${entry.ts}</span>${entry.msg}`;
    c.appendChild(line);
    c.scrollTop = c.scrollHeight;
}

function connectLogStream() {
    const es = new EventSource("/api/logs/stream");
    es.onmessage = e => appendLog(JSON.parse(e.data));
    es.onerror = () => setTimeout(connectLogStream, 2000);
}

function clearLog() { document.getElementById("log-container").innerHTML = ""; }

// ====================== 狀態輪詢 ======================
async function pollStatus() {
    const status = await fetch("/api/status").then(r => r.json());
    if (status.running) {
        setRunning(true);
    } else {
        setRunning(false);
        const lastResult = status.last_result;
        if (lastResult && lastResult.ok) {
            document.getElementById("file-classify").classList.add("ready");
            document.getElementById("file-classify").textContent = "已生成 classification";
            document.getElementById("file-import").classList.add("ready");
            document.getElementById("file-import").textContent = "已生成 import_3d_boxes.json";
            loadReport();
        }
    }
    // 文件檢測
    if (status.files.task_export_exists) {
        document.getElementById("file-export").classList.add("ready");
        document.getElementById("file-export").textContent = "已檢測到 task_export_with_annots.json";
    }
    setTimeout(pollStatus, 2000);
}

async function loadReport() {
    // 根據 workflow 選擇讀對應的 report
    const workflow = document.querySelector('input[name="workflow"]:checked').value;
    const endpoint = workflow === "A" ? "/api/report/static" : "/api/report/dynamic";
    const report = await fetch(endpoint).then(r => r.json());
    if (report.error) return;

    document.getElementById("stats-section").style.display = "block";
    document.getElementById("stat-static").textContent = report.static_count;
    document.getElementById("stat-moving").textContent = report.dynamic_count;
    document.getElementById("stat-total").textContent = report.total_count;

    const detail = document.getElementById("id-detail");
    detail.innerHTML = "";
    (report.ids || []).forEach(r => {
        const row = document.createElement("div");
        row.className = `id-row ${r.label}`;
        row.textContent = `${r.trackingId} (${r.className}) → ${r.label} | ${r.reason}`;
        detail.appendChild(row);
    });
}

// ====================== 啟動 ======================
window.addEventListener("load", () => {
    loadGT();
    connectLogStream();
    pollStatus();
});
