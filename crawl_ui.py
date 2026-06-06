#!/usr/bin/env python3
"""
Crawl4AI Studio — Web UI
Run: uv run crawl_ui.py
"""
import asyncio, json, uuid, httpx, re
from datetime import datetime
from typing import Dict, List, Optional, Any
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher, RateLimiter

app = FastAPI(title="Crawl4AI Studio")

# ── In-memory store ──────────────────────────────────────────
jobs: Dict[str, dict] = {}
ws_connections: Dict[str, List[WebSocket]] = {}

# ── Pydantic models ──────────────────────────────────────────
class CrawlRequest(BaseModel):
    urls: List[str]
    deep_crawl: Optional[str] = None
    max_pages: int = 20
    delay: float = 1.5
    content_filter: str = "none"
    wait_for: Optional[str] = None
    js_code: Optional[str] = None
    link_filter: Optional[str] = None   # pipeline mode: pattern to match extracted links
    pagination: int = 1                  # number of listing pages to crawl (auto-increments page= param)
    max_concurrent: int = 3              # max browser sessions in parallel for detail pages
    anti_bot: bool = False               # enable magic mode + simulate_user + override_navigator + random UA

class ExportRequest(BaseModel):
    job_id: str
    endpoint: str
    method: str = "POST"
    headers: Dict[str, str] = {}
    token: Optional[str] = None
    auth_type: str = "bearer"          # "bearer" | "cookie"
    rate_limit_ms: int = 300
    field_map: Optional[Dict[str, str]] = None
    static_fields: Optional[Dict[str, Any]] = None

# ── WebSocket broadcast ──────────────────────────────────────
async def broadcast(job_id: str, msg: dict):
    dead = []
    for ws in ws_connections.get(job_id, []):
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_connections[job_id].remove(ws)

# ── Pagination helper ────────────────────────────────────────
def _paginate_url(url: str, offset: int) -> Optional[str]:
    parsed = urlparse(url)
    params = parse_qs(parsed.query, keep_blank_values=True)
    for key in ("page", "p", "pg"):
        if key in params:
            try:
                current = int(params[key][0])
                params[key] = [str(current + offset)]
                return urlunparse(parsed._replace(query=urlencode(params, doseq=True)))
            except (ValueError, IndexError):
                pass
    return None

# ── Crawl worker ─────────────────────────────────────────────
def _extract_result_item(r, done: int, total: int) -> dict:
    md = str(r.markdown) if r.markdown else ""
    try:
        fit_md = str(r.markdown.fit_markdown) or md  # fallback to full markdown when empty
    except Exception:
        fit_md = md
    internal_links = [
        lk.get("href", "") for lk in (r.links or {}).get("internal", [])
        if lk.get("href", "").startswith("http")
    ]
    return {
        "url": r.url,
        "title": (r.metadata or {}).get("title", ""),
        "description": (r.metadata or {}).get("description", ""),
        "status_code": r.status_code,
        "markdown": md[:6000],
        "fit_markdown": fit_md[:4000],
        "links_internal": len(internal_links),
        "links_external": len((r.links or {}).get("external", [])),
        "internal_link_list": list(dict.fromkeys(internal_links)),
        "images": len((r.media or {}).get("images", [])),
        "crawled_at": datetime.now().isoformat(),
        "success": True,
    }

async def _stream_crawl(crawler, urls, run_cfg, dispatcher, job, job_id, done_offset, total):
    done = done_offset
    async for r in await crawler.arun_many(urls, config=run_cfg, dispatcher=dispatcher):
        done += 1
        if r.success:
            item = _extract_result_item(r, done, total)
            job["results"].append(item)
            await broadcast(job_id, {"type": "result", "item": item,
                "progress": {"done": done, "total": total}})
            await broadcast(job_id, {"type": "log", "level": "success",
                "message": f"✓ [{done}/{total}] {r.url} — \"{item['title'] or '(no title)'}\""
                           f"  links:{item['links_internal']+item['links_external']}  imgs:{item['images']}"})
        else:
            err = str(r.error_message or "Unknown error")
            job["results"].append({"url": r.url, "success": False,
                "error": err, "crawled_at": datetime.now().isoformat()})
            await broadcast(job_id, {"type": "log", "level": "error",
                "message": f"✗ [{done}/{total}] {r.url} — {err}"})
    return done

async def run_crawl(job_id: str, req: CrawlRequest):
    job = jobs[job_id]
    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat()
    await broadcast(job_id, {"type": "status", "status": "running"})

    try:
        browser_cfg = BrowserConfig(
            headless=True,
            verbose=False,
            user_agent_mode="random" if req.anti_bot else "default",
        )

        # Content filter
        md_generator = None
        if req.content_filter == "prune":
            from crawl4ai.content_filter_strategy import PruningContentFilter
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
            md_generator = DefaultMarkdownGenerator(
                content_filter=PruningContentFilter(threshold=0.45)
            )
        elif req.content_filter == "bm25":
            from crawl4ai.content_filter_strategy import BM25ContentFilter
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
            md_generator = DefaultMarkdownGenerator(
                content_filter=BM25ContentFilter()
            )

        # Deep crawl strategy
        deep_strategy = None
        if req.deep_crawl == "bfs":
            from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
            deep_strategy = BFSDeepCrawlStrategy(max_depth=2, max_pages=req.max_pages)
        elif req.deep_crawl == "dfs":
            from crawl4ai.deep_crawling import DFSDeepCrawlStrategy
            deep_strategy = DFSDeepCrawlStrategy(max_depth=2, max_pages=req.max_pages)
        elif req.deep_crawl == "best-first":
            from crawl4ai.deep_crawling import BestFirstCrawlingStrategy
            deep_strategy = BestFirstCrawlingStrategy(max_depth=2, max_pages=req.max_pages)

        detail_cfg = CrawlerRunConfig(
            delay_before_return_html=req.delay,
            page_timeout=45000,
            stream=True,
            markdown_generator=md_generator,
            deep_crawl_strategy=deep_strategy,
            wait_for=req.wait_for or None,
            js_code=req.js_code or None,
            verbose=False,
            magic=req.anti_bot,
            simulate_user=req.anti_bot,
            override_navigator=req.anti_bot,
        )
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=80,
            max_session_permit=req.max_concurrent,
            rate_limiter=RateLimiter(base_delay=(1.0, 2.0)),
        )

        async with AsyncWebCrawler(config=browser_cfg) as crawler:

            # ── Pipeline mode: listing → detail ──────────────────
            if req.link_filter:
                # Scroll script: 10 steps × 700ms + 1.5s final wait = ~8.5s
                # Ensures lazy-loaded items (VietnamWorks, Nhatot…) are all rendered
                _SCROLL = (
                    "await (async()=>{"
                    "let prev=0;"
                    "for(let i=1;i<=10;i++){"
                    "const h=document.body.scrollHeight;"
                    "window.scrollTo(0,(h/10)*i);"
                    "await new Promise(r=>setTimeout(r,700));"
                    "}"
                    "window.scrollTo(0,document.body.scrollHeight);"
                    "await new Promise(r=>setTimeout(r,1500));"
                    "})();"
                )
                listing_js = ((req.js_code + "\n") if req.js_code else "") + _SCROLL
                listing_cfg = CrawlerRunConfig(
                    delay_before_return_html=max(req.delay, 2.0),
                    page_timeout=60000,
                    stream=True,
                    js_code=listing_js,
                    verbose=False,
                    magic=req.anti_bot,
                    simulate_user=req.anti_bot,
                    override_navigator=req.anti_bot,
                )
                listing_dispatcher = MemoryAdaptiveDispatcher(
                    max_session_permit=3,
                    rate_limiter=RateLimiter(base_delay=(0.5, 1.0)),
                )
                # Expand listing URLs for pagination
                listing_urls: list[str] = []
                for url in req.urls:
                    listing_urls.append(url)
                    for offset in range(1, req.pagination):
                        nxt = _paginate_url(url, offset)
                        if nxt:
                            listing_urls.append(nxt)
                listing_urls = list(dict.fromkeys(listing_urls))

                await broadcast(job_id, {"type": "log", "level": "info",
                    "message": f"⬡ Phase 1/2 — crawling {len(listing_urls)} listing page(s)"
                               + (f" (pages {req.pagination})" if req.pagination > 1 else "") + "..."})

                extracted: list[str] = []
                async for r in await crawler.arun_many(
                    listing_urls, config=listing_cfg, dispatcher=listing_dispatcher
                ):
                    if r.success:
                        links = list(dict.fromkeys(
                            lk.get("href", "") for lk in (r.links or {}).get("internal", [])
                            if req.link_filter in lk.get("href", "")
                            and lk.get("href", "").startswith("http")
                        ))
                        extracted.extend(links)
                        await broadcast(job_id, {"type": "log", "level": "info",
                            "message": f"  ↳ {r.url}  →  {len(links)} links matched \"{req.link_filter}\""})
                    else:
                        await broadcast(job_id, {"type": "log", "level": "error",
                            "message": f"  ✗ listing failed: {r.url}"})

                detail_urls = list(dict.fromkeys(extracted))
                total = len(detail_urls)
                if not detail_urls:
                    await broadcast(job_id, {"type": "log", "level": "warn",
                        "message": f"⚠ No links matched filter \"{req.link_filter}\". Kiểm tra lại pattern."})
                else:
                    await broadcast(job_id, {"type": "log", "level": "info",
                        "message": f"⬡ Phase 2/2 — crawling {total} detail page(s)..."})
                    await _stream_crawl(crawler, detail_urls, detail_cfg,
                                        dispatcher, job, job_id, 0, total)

            # ── Normal mode ──────────────────────────────────────
            else:
                total = len(req.urls)
                await broadcast(job_id, {"type": "log", "level": "info",
                    "message": f"Starting crawl of {total} URL(s)..."})
                await _stream_crawl(crawler, req.urls, detail_cfg,
                                    dispatcher, job, job_id, 0, total)

        job["status"] = "done"
        job["finished_at"] = datetime.now().isoformat()
        success_count = sum(1 for r in job["results"] if r.get("success"))
        fail_count    = sum(1 for r in job["results"] if not r.get("success"))
        await broadcast(job_id, {
            "type": "status", "status": "done",
            "summary": {"total": total, "success": success_count, "failed": fail_count}
        })

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        await broadcast(job_id, {"type": "log", "level": "error", "message": f"Fatal: {e}"})
        await broadcast(job_id, {"type": "status", "status": "error"})

# ── API routes ───────────────────────────────────────────────
@app.post("/api/crawl")
async def start_crawl(req: CrawlRequest):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "pending", "results": [], "request": req.model_dump()}
    ws_connections[job_id] = []
    asyncio.create_task(run_crawl(job_id, req))
    return {"job_id": job_id}

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        return {"error": "Not found"}, 404
    return {
        "status": job["status"],
        "result_count": len(job["results"]),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
    }

@app.get("/api/jobs/{job_id}/results")
async def get_results(job_id: str):
    job = jobs.get(job_id, {})
    return {"results": job.get("results", [])}

@app.get("/api/jobs/{job_id}/links")
async def get_links(job_id: str, pattern: str = ""):
    job = jobs.get(job_id, {})
    all_links: list[str] = []
    for r in job.get("results", []):
        all_links.extend(r.get("internal_link_list", []))
    unique = list(dict.fromkeys(all_links))
    if pattern:
        unique = [u for u in unique if pattern in u]
    return {"total": len(unique), "links": unique}

@app.post("/api/export")
async def export_results(req: ExportRequest):
    job = jobs.get(req.job_id, {})
    results = [r for r in job.get("results", []) if r.get("success")]

    headers = dict(req.headers)
    if req.token:
        if req.auth_type == "cookie":
            headers["Cookie"] = f"auth_token={req.token}"
        else:
            headers["Authorization"] = f"Bearer {req.token}"
    headers.setdefault("Content-Type", "application/json")

    export_log = []
    async with httpx.AsyncClient(timeout=30) as client:
        for i, item in enumerate(results):
            if i > 0 and req.rate_limit_ms > 0:
                await asyncio.sleep(req.rate_limit_ms / 1000)
            payload = dict(item)
            if req.field_map:
                payload = {api_f: item.get(crawl_f, "") for api_f, crawl_f in req.field_map.items()}
            if req.static_fields:
                payload = {**payload, **req.static_fields}
            try:
                resp = await client.request(req.method, req.endpoint, json=payload, headers=headers)
                export_log.append({
                    "url": item.get("url"), "status": resp.status_code,
                    "ok": resp.is_success, "response": resp.text[:200]
                })
            except Exception as e:
                export_log.append({
                    "url": item.get("url"), "status": 0, "ok": False, "error": str(e)
                })

    return {"exported": len(export_log), "log": export_log}

@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    ws_connections.setdefault(job_id, []).append(websocket)

    # Replay existing state for late joiners
    if job_id in jobs:
        job = jobs[job_id]
        for item in job["results"]:
            await websocket.send_json({"type": "result", "item": item})
        await websocket.send_json({"type": "status", "status": job["status"]})

    try:
        while True:
            await asyncio.sleep(30)  # keep-alive
            await websocket.send_json({"type": "ping"})
    except (WebSocketDisconnect, Exception):
        if job_id in ws_connections and websocket in ws_connections[job_id]:
            ws_connections[job_id].remove(websocket)

# ── HTML UI ──────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Crawl4AI Studio</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0f1117;--bg2:#161924;--bg3:#1e2130;--border:#2d3348;
  --text:#e2e8f0;--muted:#64748b;--sub:#94a3b8;
  --blue:#3b82f6;--blue-l:#60a5fa;
  --green:#22c55e;--green-l:#4ade80;
  --red:#ef4444;--red-l:#f87171;
  --amber:#f59e0b;--amber-l:#fbbf24;
}
body{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--text);
  height:100vh;display:flex;flex-direction:column;overflow:hidden;font-size:13px}

/* ── Header ── */
.header{background:var(--bg2);border-bottom:1px solid var(--border);
  padding:10px 18px;display:flex;align-items:center;gap:12px;flex-shrink:0}
.header-brand{display:flex;align-items:center;gap:8px}
.header-brand .logo{font-size:22px}
.header-brand .name{font-weight:700;font-size:15px}
.header-brand .ver{font-size:10px;color:var(--muted);margin-top:1px}
.header-right{margin-left:auto;display:flex;align-items:center;gap:16px}

/* ── Status bar ── */
.statusbar{background:var(--bg2);border-bottom:1px solid var(--border);
  padding:6px 18px;display:flex;align-items:center;gap:16px;flex-shrink:0;font-size:12px}
.stat-item{color:var(--muted)}
.stat-item strong{color:var(--text)}
.prog-wrap{margin-left:auto;display:flex;align-items:center;gap:8px}
.prog-bar{width:180px;height:5px;background:var(--bg3);border-radius:3px;overflow:hidden}
.prog-fill{height:100%;background:var(--blue);border-radius:3px;transition:width .3s}
.prog-pct{font-size:11px;color:var(--muted);min-width:32px;text-align:right}

/* ── Dot indicator ── */
.dot{width:7px;height:7px;border-radius:50%;display:inline-block;flex-shrink:0}
.dot-gray{background:#475569}
.dot-blue{background:var(--blue);animation:pulse 1s infinite}
.dot-green{background:var(--green)}
.dot-red{background:var(--red)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ── Layout ── */
.main{display:flex;flex:1;overflow:hidden}

/* ── Sidebar ── */
.sidebar{width:300px;background:var(--bg2);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
.sidebar-inner{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:14px}
.sidebar-footer{padding:12px 14px;border-top:1px solid var(--border);display:flex;flex-direction:column;gap:8px}

/* ── Sections ── */
.sec-title{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:1.2px;
  color:var(--muted);margin-bottom:8px}
.divider{border:none;border-top:1px solid var(--border);margin:2px 0}

/* ── Form ── */
label{font-size:11px;color:var(--sub);margin-bottom:3px;display:block}
input,select,textarea{width:100%;background:var(--bg);border:1px solid var(--border);
  border-radius:6px;color:var(--text);padding:7px 9px;font-size:12px;outline:none;
  font-family:inherit}
input:focus,select:focus,textarea:focus{border-color:var(--blue);box-shadow:0 0 0 2px #3b82f620}
textarea{resize:vertical;min-height:52px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.form-group{margin-bottom:8px}

/* ── Buttons ── */
.btn{padding:7px 14px;border-radius:6px;border:none;cursor:pointer;
  font-size:12px;font-weight:600;display:inline-flex;align-items:center;gap:5px;
  justify-content:center;transition:all .15s;font-family:inherit}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn-primary{background:var(--blue);color:#fff}
.btn-primary:not(:disabled):hover{background:#2563eb}
.btn-success{background:#16a34a;color:#fff}
.btn-success:not(:disabled):hover{background:#15803d}
.btn-outline{background:var(--bg3);color:var(--sub);border:1px solid var(--border)}
.btn-outline:not(:disabled):hover{color:var(--text);border-color:var(--sub)}
.btn-danger{background:#7f1d1d;color:#fca5a5}
.btn-full{width:100%}
.btn-sm{padding:4px 10px;font-size:11px}

/* ── URL tags ── */
.url-input-row{display:flex;gap:6px}
.url-input-row input{flex:1}
.url-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.url-tag{background:var(--bg3);border:1px solid var(--border);border-radius:4px;
  padding:2px 6px 2px 8px;font-size:11px;color:var(--blue-l);
  display:flex;align-items:center;gap:4px;max-width:100%}
.url-tag span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:200px}
.url-tag-rm{background:none;border:none;color:var(--muted);cursor:pointer;
  font-size:13px;line-height:1;padding:0 1px}
.url-tag-rm:hover{color:var(--red-l)}

/* ── KV headers ── */
.kv-row{display:flex;gap:5px;margin-bottom:5px}
.kv-row input{flex:1}
.kv-rm{background:none;border:none;color:var(--muted);cursor:pointer;font-size:16px;padding:0 3px}
.kv-rm:hover{color:var(--red-l)}

/* ── Content area ── */
.content{flex:1;display:flex;flex-direction:column;overflow:hidden;min-width:0}

/* ── Tabs ── */
.tabs{background:var(--bg2);border-bottom:1px solid var(--border);
  display:flex;flex-shrink:0}
.tab{padding:9px 18px;cursor:pointer;font-size:12px;font-weight:500;
  color:var(--muted);border-bottom:2px solid transparent;user-select:none}
.tab:hover{color:var(--sub)}
.tab.active{color:var(--blue-l);border-bottom-color:var(--blue)}
.tab-badge{display:inline-block;background:var(--bg3);border-radius:10px;
  padding:1px 6px;font-size:10px;margin-left:4px;color:var(--muted)}

/* ── Panels ── */
.panel{flex:1;overflow:hidden;display:none;flex-direction:column}
.panel.active{display:flex}

/* ── Monitor ── */
.monitor-toolbar{padding:8px 14px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:8px;flex-shrink:0;background:var(--bg2)}
.log-area{flex:1;overflow-y:auto;padding:12px 14px;
  font-family:'Fira Code','Consolas',monospace;font-size:11.5px;line-height:1.6;
  background:var(--bg)}
.log-line{padding:1px 0;white-space:pre-wrap;word-break:break-all}
.log-success{color:var(--green-l)}
.log-error{color:var(--red-l)}
.log-info{color:var(--blue-l)}
.log-warn{color:var(--amber-l)}
.log-ts{color:var(--muted);margin-right:6px;user-select:none}

/* ── Results ── */
.results-toolbar{padding:8px 14px;display:flex;align-items:center;gap:10px;
  border-bottom:1px solid var(--border);flex-shrink:0;background:var(--bg2)}
.results-toolbar input{max-width:240px}
.table-wrap{flex:1;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:12px}
th{background:var(--bg3);padding:9px 12px;text-align:left;color:var(--sub);
  font-weight:600;border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:1;white-space:nowrap}
td{padding:9px 12px;border-bottom:1px solid var(--bg3);vertical-align:middle}
tr:hover td{background:#161924}
.cell-url{max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cell-title{max-width:180px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cell-url a{color:var(--blue-l);text-decoration:none}
.cell-url a:hover{text-decoration:underline}

/* ── Badge ── */
.badge{display:inline-block;padding:2px 8px;border-radius:10px;
  font-size:10px;font-weight:700}
.badge-ok{background:#0d2b1a;color:var(--green-l)}
.badge-err{background:#3b0e1e;color:var(--red-l)}
.badge-run{background:#1e3a5f;color:var(--blue-l)}
.badge-pend{background:var(--bg3);color:var(--muted)}

/* ── Export panel ── */
.export-panel-inner{flex:1;overflow-y:auto;padding:16px}
.export-row{display:flex;align-items:center;gap:10px;
  padding:8px 10px;border-bottom:1px solid var(--bg3);font-size:12px}
.export-url{flex:1;color:var(--sub);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}

/* ── Modal ── */
.overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);
  z-index:200;align-items:center;justify-content:center}
.overlay.open{display:flex}
.modal{background:var(--bg3);border:1px solid var(--border);border-radius:12px;
  padding:22px;width:min(720px,92vw);max-height:85vh;overflow:hidden;
  display:flex;flex-direction:column;gap:14px}
.modal-header{display:flex;justify-content:space-between;align-items:flex-start}
.modal-title{font-weight:700;font-size:14px;color:var(--text);max-width:580px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.modal-tabs{display:flex;gap:5px}
.modal-body{flex:1;overflow-y:auto}
.modal-body pre{background:var(--bg);border-radius:6px;padding:14px;
  font-size:11.5px;white-space:pre-wrap;word-break:break-all;line-height:1.6;
  max-height:calc(85vh - 130px);overflow-y:auto;color:var(--sub)}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
::-webkit-scrollbar-thumb:hover{background:var(--muted)}

/* ── Empty state ── */
.empty{display:flex;flex-direction:column;align-items:center;justify-content:center;
  height:100%;gap:10px;color:var(--muted);font-size:13px}
.empty-icon{font-size:36px;opacity:.4}

.text-muted{color:var(--muted)}
.text-sm{font-size:11px}
</style>
</head>
<body>

<!-- ═══════════ HEADER ═══════════ -->
<div class="header">
  <div class="header-brand">
    <div class="logo">🕷️</div>
    <div>
      <div class="name">Crawl4AI Studio</div>
      <div class="ver">v0.8.7</div>
    </div>
  </div>
  <div class="header-right">
    <div style="display:flex;align-items:center;gap:6px">
      <span class="dot dot-gray" id="dotStatus"></span>
      <span id="txtStatus" style="color:var(--muted);font-size:12px">Idle</span>
    </div>
  </div>
</div>

<!-- ═══════════ STATUS BAR ═══════════ -->
<div class="statusbar">
  <div class="stat-item">Job&nbsp;<strong id="sJobId">—</strong></div>
  <div class="stat-item">Total&nbsp;<strong id="sTotal">0</strong></div>
  <div class="stat-item" style="color:var(--green-l)">OK&nbsp;<strong id="sOk">0</strong></div>
  <div class="stat-item" style="color:var(--red-l)">Fail&nbsp;<strong id="sFail">0</strong></div>
  <div class="stat-item">Time&nbsp;<strong id="sTime">—</strong></div>
  <div class="prog-wrap">
    <div class="prog-bar"><div class="prog-fill" id="progFill" style="width:0"></div></div>
    <div class="prog-pct" id="progPct">0%</div>
  </div>
</div>

<!-- ═══════════ MAIN ═══════════ -->
<div class="main">

  <!-- ─── SIDEBAR ─── -->
  <div class="sidebar">
    <div class="sidebar-inner">

      <!-- URLs -->
      <div>
        <div class="sec-title">📌 URLs cần crawl</div>
        <div class="url-input-row">
          <input id="urlInput" placeholder="https://example.com" type="url"
                 onkeydown="if(event.key==='Enter'){addUrl();event.preventDefault()}">
          <button class="btn btn-outline btn-sm" onclick="addUrl()">＋ Add</button>
        </div>
        <div class="url-tags" id="urlTags"></div>
      </div>

      <hr class="divider">

      <!-- Options -->
      <div>
        <div class="sec-title">⚙️ Tuỳ chọn Crawl</div>

        <div class="form-group">
          <label>Chế độ Crawl</label>
          <select id="crawlMode" onchange="toggleDeep()">
            <option value="single">Single — từng trang</option>
            <option value="bfs">Deep BFS — breadth-first</option>
            <option value="dfs">Deep DFS — depth-first</option>
            <option value="best-first">Deep Best-First</option>
          </select>
        </div>

        <div class="form-group" id="optMaxPages" style="display:none">
          <label>Tối đa số trang</label>
          <input type="number" id="maxPages" value="20" min="1" max="500">
        </div>

        <div class="grid2">
          <div class="form-group">
            <label>Delay (giây)</label>
            <input type="number" id="delay" value="1.5" step="0.5" min="0">
          </div>
          <div class="form-group">
            <label>Content Filter</label>
            <select id="contentFilter">
              <option value="none">None</option>
              <option value="prune">Pruning</option>
              <option value="bm25">BM25</option>
            </select>
          </div>
        </div>

        <div class="form-group" style="display:flex;align-items:center;gap:8px;padding:6px 0">
          <input type="checkbox" id="antiBot"
                 style="accent-color:var(--blue);width:15px;height:15px;flex-shrink:0;cursor:pointer">
          <label for="antiBot"
                 style="font-size:12px;color:var(--text);margin:0;cursor:pointer;user-select:none">
            🛡 Anti-bot mode
            <span style="font-size:10px;color:var(--muted);font-weight:400">
              — magic + simulate_user + random UA
            </span>
          </label>
        </div>

        <div class="form-group">
          <label>Wait for (CSS selector)</label>
          <input id="waitFor" placeholder="css:.main-content">
        </div>

        <div class="form-group">
          <label>JS thực thi trước khi lấy HTML</label>
          <textarea id="jsCode" rows="2"
            placeholder="window.scrollTo(0, document.body.scrollHeight);"></textarea>
        </div>

        <div class="form-group" id="linkFilterGroup">
          <label style="display:flex;align-items:center;gap:6px">
            🔗 Link Filter
            <span style="font-size:10px;color:var(--muted);font-weight:400">
              — Pipeline mode: crawl listing → tự động crawl từng detail page
            </span>
          </label>
          <input id="linkFilter" placeholder='Vd: -jv (VietnamWorks)  |  .htm (Nhatot)'>
          <div style="display:flex;gap:12px;align-items:center;margin-top:6px;flex-wrap:wrap">
            <div style="display:flex;gap:6px;align-items:center">
              <label style="white-space:nowrap;font-size:11px;color:var(--sub);margin:0">Số trang:</label>
              <input type="number" id="pagination" value="1" min="1" max="50"
                     style="width:54px;background:var(--bg);border:1px solid var(--border);
                            border-radius:6px;color:var(--text);padding:5px 7px;font-size:12px">
            </div>
            <div style="display:flex;gap:6px;align-items:center">
              <label style="white-space:nowrap;font-size:11px;color:var(--sub);margin:0">Concurrent:</label>
              <input type="number" id="maxConcurrent" value="3" min="1" max="10"
                     style="width:54px;background:var(--bg);border:1px solid var(--border);
                            border-radius:6px;color:var(--text);padding:5px 7px;font-size:12px"
                     title="Số browser sessions song song. Giảm xuống 2-3 nếu bị timeout.">
            </div>
            <span style="font-size:10px;color:var(--muted)">page N, N+1…</span>
          </div>
          <div style="margin-top:4px;font-size:10px;color:var(--muted)">
            Để trống Link Filter = crawl từng URL thông thường. Có giá trị = tự extract links &amp; crawl detail.
          </div>
        </div>
      </div>

      <hr class="divider">

      <!-- API Export -->
      <div>
        <div class="sec-title">🔌 API Export Endpoint</div>

        <div class="form-group">
          <label>Endpoint URL</label>
          <input id="exportUrl" type="url" placeholder="https://api.yourapp.com/items">
        </div>

        <div class="grid2">
          <div class="form-group">
            <label>HTTP Method</label>
            <select id="exportMethod">
              <option>POST</option><option>PUT</option><option>PATCH</option>
            </select>
          </div>
          <div class="form-group">
            <label>Auth Type</label>
            <select id="exportAuthType">
              <option value="bearer">Bearer Token</option>
              <option value="cookie">Cookie (auth_token=…)</option>
            </select>
          </div>
        </div>

        <div class="form-group">
          <label>Token / JWT</label>
          <input id="exportToken" type="password" placeholder="eyJhbGci…">
        </div>

        <div class="grid2">
          <div class="form-group">
            <label>Rate Limit (ms/request)</label>
            <input type="number" id="exportRateLimit" value="300" min="0" max="5000">
          </div>
          <div class="form-group" style="display:flex;flex-direction:column;justify-content:flex-end">
            <button class="btn btn-outline btn-sm" onclick="addHeader()" style="margin-top:auto">
              ＋ Custom Header
            </button>
          </div>
        </div>

        <div id="headerRows"></div>

        <div class="form-group">
          <label>Field Mapping <span class="text-muted text-sm">(JSON — để trống = gửi tất cả fields)</span></label>
          <textarea id="fieldMap" rows="2"
            placeholder='{"title":"title","body":"fit_markdown","link":"url"}'></textarea>
        </div>

        <div class="form-group">
          <label>Static Fields <span class="text-muted text-sm">(JSON — giá trị cố định thêm vào mỗi payload)</span></label>
          <textarea id="staticFields" rows="2"
            placeholder='{"source":"vietnamworks","employmentType":"fulltime"}'></textarea>
        </div>
      </div>

    </div><!-- /sidebar-inner -->

    <div class="sidebar-footer">
      <button class="btn btn-primary btn-full" id="startBtn" onclick="startCrawl()">
        ▶&nbsp;Start Crawl
      </button>
      <div style="display:flex;gap:7px">
        <button class="btn btn-success btn-full" id="exportBtn" onclick="exportApi()" disabled>
          ↑&nbsp;Export to API
        </button>
        <button class="btn btn-outline" id="dlJsonBtn" onclick="dlJson()" disabled
                title="Download JSON" style="padding:7px 10px;flex-shrink:0">⬇ JSON</button>
        <button class="btn btn-outline" id="dlCsvBtn" onclick="dlCsv()" disabled
                title="Download CSV" style="padding:7px 10px;flex-shrink:0">⬇ CSV</button>
      </div>
    </div>
  </div><!-- /sidebar -->

  <!-- ─── CONTENT ─── -->
  <div class="content">
    <div class="tabs">
      <div class="tab active" data-tab="monitor" onclick="switchTab(this)">
        📡 Monitor
      </div>
      <div class="tab" data-tab="results" onclick="switchTab(this)">
        📋 Results<span class="tab-badge" id="badgeResults">0</span>
      </div>
      <div class="tab" data-tab="export" onclick="switchTab(this)">
        📤 Export Log<span class="tab-badge" id="badgeExport"></span>
      </div>
    </div>

    <!-- Monitor -->
    <div class="panel active" id="panel-monitor">
      <div class="monitor-toolbar">
        <div id="logMeta" class="text-muted text-sm">0 messages</div>
        <div style="flex:1"></div>
        <button class="btn btn-outline btn-sm" onclick="clearLog()">🗑 Clear</button>
      </div>
      <div class="log-area" id="logArea">
        <div class="text-muted text-sm" style="padding:4px">
          Thêm URL và nhấn "Start Crawl" để bắt đầu…
        </div>
      </div>
    </div>

    <!-- Results -->
    <div class="panel" id="panel-results">
      <div class="results-toolbar">
        <input id="filterInput" placeholder="🔍  Lọc theo URL hoặc tiêu đề…" oninput="filterRows()">
        <div class="text-muted text-sm" id="resultMeta"></div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>URL</th>
              <th>Tiêu đề</th>
              <th>Status</th>
              <th>Links (in/out)</th>
              <th>Ảnh</th>
              <th>Thời gian</th>
              <th></th>
            </tr>
          </thead>
          <tbody id="tBody"></tbody>
        </table>
        <div class="empty" id="emptyResults" style="display:none">
          <div class="empty-icon">📋</div>
          <div>Chưa có kết quả. Hãy chạy crawl trước.</div>
        </div>
      </div>
    </div>

    <!-- Export Log -->
    <div class="panel" id="panel-export">
      <div class="export-panel-inner" id="exportLog">
        <div class="empty">
          <div class="empty-icon">📤</div>
          <div>Chạy crawl xong và nhấn "Export to API" để xem kết quả ở đây.</div>
        </div>
      </div>
    </div>

  </div><!-- /content -->
</div><!-- /main -->

<!-- ═══════════ MODAL ═══════════ -->
<div class="overlay" id="overlay" onclick="overlayClick(event)">
  <div class="modal">
    <div class="modal-header">
      <div class="modal-title" id="modalTitle"></div>
      <button class="btn btn-outline btn-sm" onclick="closeModal()">✕ Đóng</button>
    </div>
    <div class="modal-tabs">
      <button class="btn btn-primary btn-sm" id="mTabMd" onclick="showMTab('md')">Markdown</button>
      <button class="btn btn-outline btn-sm" id="mTabFit" onclick="showMTab('fit')">Fit Markdown</button>
      <button class="btn btn-outline btn-sm" id="mTabJson" onclick="showMTab('json')">JSON đầy đủ</button>
    </div>
    <div class="modal-body"><pre id="modalPre"></pre></div>
  </div>
</div>

<!-- ═══════════ LINKS MODAL ═══════════ -->
<div class="overlay" id="linksOverlay" onclick="linksOverlayClick(event)">
  <div class="modal" style="width:min(860px,95vw)">
    <div class="modal-header">
      <div class="modal-title" id="linksModalTitle">Internal Links</div>
      <button class="btn btn-outline btn-sm" onclick="closeLinksModal()">✕ Đóng</button>
    </div>
    <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
      <input id="linksFilter" placeholder="🔍 Filter URL pattern (vd: /viec-lam-)"
             style="flex:1;min-width:200px;background:var(--bg);border:1px solid var(--border);
                    border-radius:6px;color:var(--text);padding:6px 9px;font-size:12px;outline:none"
             oninput="filterLinks()">
      <button class="btn btn-outline btn-sm" onclick="selectAllLinks()">✓ Chọn tất cả</button>
      <button class="btn btn-outline btn-sm" onclick="deselectAllLinks()">✗ Bỏ chọn</button>
      <span id="linksCount" style="color:var(--muted);font-size:11px"></span>
    </div>
    <div id="linksListWrap" class="modal-body"
         style="max-height:55vh;overflow-y:auto;border:1px solid var(--border);
                border-radius:6px;background:var(--bg)">
      <div id="linksList" style="padding:4px 0"></div>
    </div>
    <div style="display:flex;gap:8px;justify-content:flex-end">
      <button class="btn btn-primary" onclick="addSelectedLinks()">＋ Add to Crawl Queue</button>
      <button class="btn btn-outline" onclick="closeLinksModal()">Huỷ</button>
    </div>
  </div>
</div>

<script>
// ══════════════════════════════════════════════
// State
// ══════════════════════════════════════════════
let urls   = [];
let hdrs   = [];
let rows   = [];   // all result items
let jobId  = null;
let ws     = null;
let timer  = null;
let t0     = null;
let modalItem = null;

// ══════════════════════════════════════════════
// URL Management
// ══════════════════════════════════════════════
function addUrl() {
  const el = document.getElementById('urlInput');
  const v  = el.value.trim();
  if (!v) return;
  const list = v.split(/[\n,]+/).map(s=>s.trim()).filter(s=>s.startsWith('http'));
  list.forEach(u => { if (!urls.includes(u)) urls.push(u); });
  el.value = '';
  renderUrls();
}
function removeUrl(i){ urls.splice(i,1); renderUrls(); }
function renderUrls(){
  document.getElementById('urlTags').innerHTML =
    urls.map((u,i)=>`<div class="url-tag">
      <span title="${u}">${u}</span>
      <button class="url-tag-rm" onclick="removeUrl(${i})">✕</button>
    </div>`).join('');
}
document.getElementById('urlInput').addEventListener('paste', e=>{
  setTimeout(()=>{ if(e.target.value.includes('\n')){ addUrl(); } }, 20);
});

// ══════════════════════════════════════════════
// Header Management
// ══════════════════════════════════════════════
function addHeader(){ hdrs.push({k:'',v:''}); renderHdrs(); }
function removeHdr(i){ hdrs.splice(i,1); renderHdrs(); }
function renderHdrs(){
  document.getElementById('headerRows').innerHTML =
    hdrs.map((h,i)=>`<div class="kv-row">
      <input placeholder="Header" value="${h.k}" oninput="hdrs[${i}].k=this.value">
      <input placeholder="Value"  value="${h.v}" oninput="hdrs[${i}].v=this.value">
      <button class="kv-rm" onclick="removeHdr(${i})">×</button>
    </div>`).join('');
}

// ══════════════════════════════════════════════
// UI Helpers
// ══════════════════════════════════════════════
function toggleDeep(){
  const v = document.getElementById('crawlMode').value;
  document.getElementById('optMaxPages').style.display = v==='single'?'none':'block';
}
function switchTab(el){
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-'+el.dataset.tab).classList.add('active');
}
function setStatus(s, txt){
  const cls = {idle:'gray',running:'blue',done:'green',error:'red'}[s]||'gray';
  document.getElementById('dotStatus').className = `dot dot-${cls}`;
  document.getElementById('txtStatus').textContent = txt;
}
function setProgress(done, total){
  const pct = total>0 ? Math.round(done/total*100) : 0;
  document.getElementById('progFill').style.width  = pct+'%';
  document.getElementById('progPct').textContent   = pct+'%';
  document.getElementById('sOk').textContent = done;
  document.getElementById('sTotal').textContent = total;
}
function setActionBtns(enabled){
  ['exportBtn','dlJsonBtn','dlCsvBtn'].forEach(id=>{
    document.getElementById(id).disabled = !enabled;
  });
}
function startTimer(){
  t0 = Date.now();
  timer = setInterval(()=>{
    const s = Math.floor((Date.now()-t0)/1000);
    document.getElementById('sTime').textContent = `${Math.floor(s/60)}m ${s%60}s`;
  }, 1000);
}
function stopTimer(){ clearInterval(timer); }

// ══════════════════════════════════════════════
// Log
// ══════════════════════════════════════════════
function log(msg, lvl='info'){
  const area = document.getElementById('logArea');
  const ts   = new Date().toLocaleTimeString('vi-VN');
  const d    = document.createElement('div');
  d.className = `log-line log-${lvl}`;
  d.innerHTML = `<span class="log-ts">${ts}</span>${escHtml(msg)}`;
  area.appendChild(d);
  area.scrollTop = area.scrollHeight;
  document.getElementById('logMeta').textContent =
    area.querySelectorAll('.log-line').length + ' messages';
}
function clearLog(){
  document.getElementById('logArea').innerHTML='';
  document.getElementById('logMeta').textContent='0 messages';
}
function escHtml(s){ const d=document.createElement('div');d.textContent=s;return d.innerHTML; }

// ══════════════════════════════════════════════
// Results Table
// ══════════════════════════════════════════════
function addRow(item){
  rows.push(item);
  const tb   = document.getElementById('tBody');
  const idx  = rows.length;
  const tr   = document.createElement('tr');
  tr.dataset.url   = (item.url||'').toLowerCase();
  tr.dataset.title = (item.title||'').toLowerCase();

  if(item.success===false){
    tr.innerHTML = `
      <td style="color:var(--muted)">${idx}</td>
      <td colspan="5" class="cell-url" title="${item.url}">
        <a href="${item.url}" target="_blank">${item.url}</a></td>
      <td><span class="badge badge-err">Error</span></td>
      <td><span class="text-muted text-sm">${escHtml(item.error||'')}</span></td>`;
  } else {
    const t = item.crawled_at ? new Date(item.crawled_at).toLocaleTimeString('vi-VN') : '';
    tr.innerHTML = `
      <td style="color:var(--muted)">${idx}</td>
      <td class="cell-url" title="${item.url}">
        <a href="${item.url}" target="_blank">${item.url}</a></td>
      <td class="cell-title" title="${escHtml(item.title||'')}">
        ${escHtml(item.title)||'<span class="text-muted">—</span>'}</td>
      <td><span class="badge badge-ok">${item.status_code||200}</span></td>
      <td style="color:var(--sub)">${item.links_internal||0} / ${item.links_external||0}</td>
      <td style="color:var(--sub)">${item.images||0}</td>
      <td style="color:var(--muted)">${t}</td>
      <td style="display:flex;gap:4px">
        <button class="btn btn-outline btn-sm" onclick="viewItem(${idx-1})">🔍 Xem</button>
        ${item.links_internal>0?`<button class="btn btn-outline btn-sm" onclick="viewLinks(${idx-1})" title="${item.links_internal} internal links">🔗 ${item.links_internal}</button>`:''}
      </td>`;
  }
  tb.appendChild(tr);

  // Update counters
  const ok   = rows.filter(r=>r.success!==false).length;
  const fail = rows.filter(r=>r.success===false).length;
  document.getElementById('badgeResults').textContent = rows.length;
  document.getElementById('resultMeta').textContent   = `${ok} ok · ${fail} lỗi`;
  document.getElementById('sFail').textContent = fail;
  document.getElementById('emptyResults').style.display = rows.length?'none':'flex';
}

function filterRows(){
  const q = document.getElementById('filterInput').value.toLowerCase();
  document.querySelectorAll('#tBody tr').forEach(r=>{
    r.style.display = (!q||r.dataset.url.includes(q)||r.dataset.title.includes(q)) ? '' : 'none';
  });
}

// ══════════════════════════════════════════════
// Modal
// ══════════════════════════════════════════════
function viewItem(i){
  modalItem = rows[i];
  document.getElementById('modalTitle').textContent = modalItem.title||modalItem.url;
  document.getElementById('overlay').classList.add('open');
  showMTab('md');
}
function showMTab(t){
  ['md','fit','json'].forEach(k=>{
    const b = document.getElementById('mTab'+k[0].toUpperCase()+k.slice(1));
    if(b) b.className = t===k?'btn btn-primary btn-sm':'btn btn-outline btn-sm';
  });
  const pre = document.getElementById('modalPre');
  if(t==='md')   pre.textContent = modalItem.markdown||'(empty)';
  else if(t==='fit') pre.textContent = modalItem.fit_markdown||modalItem.markdown||'(empty)';
  else           pre.textContent = JSON.stringify(modalItem, null, 2);
}
function closeModal(){ document.getElementById('overlay').classList.remove('open'); modalItem=null; }
function overlayClick(e){ if(e.target===document.getElementById('overlay')) closeModal(); }

// ══════════════════════════════════════════════
// WebSocket
// ══════════════════════════════════════════════
function connectWs(jid){
  const proto = location.protocol==='https:'?'wss':'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/${jid}`);

  ws.onopen = ()=> log(`WebSocket connected — job ${jid}`, 'info');

  ws.onmessage = e=>{
    const m = JSON.parse(e.data);
    if(m.type==='ping') return;

    if(m.type==='log'){
      log(m.message, m.level||'info');
    }
    else if(m.type==='result'){
      addRow(m.item);
      if(m.progress) setProgress(m.progress.done, m.progress.total);
    }
    else if(m.type==='status'){
      if(m.status==='running'){
        setStatus('running','Crawling…');
      } else if(m.status==='done'){
        setStatus('done','Done');
        stopTimer();
        document.getElementById('startBtn').disabled=false;
        document.getElementById('startBtn').textContent='▶ Start Crawl';
        setActionBtns(true);
        if(m.summary) log(
          `✅ Hoàn thành! ${m.summary.success} thành công, ${m.summary.failed} thất bại`,
          'success'
        );
        setProgress(
          (m.summary?.success||0)+(m.summary?.failed||0),
          m.summary?.total||0
        );
      } else if(m.status==='error'){
        setStatus('error','Error');
        stopTimer();
        document.getElementById('startBtn').disabled=false;
        document.getElementById('startBtn').textContent='▶ Start Crawl';
      }
    }
  };
  ws.onerror  = ()=> log('WebSocket error','error');
  ws.onclose  = ()=> log('WebSocket closed','warn');
}

// ══════════════════════════════════════════════
// Start Crawl
// ══════════════════════════════════════════════
async function startCrawl(){
  if(!urls.length){ alert('Vui lòng thêm ít nhất 1 URL.'); return; }

  // Reset
  rows=[];
  document.getElementById('tBody').innerHTML='';
  document.getElementById('badgeResults').textContent='0';
  document.getElementById('resultMeta').textContent='';
  document.getElementById('emptyResults').style.display='flex';
  setProgress(0,0);
  document.getElementById('sOk').textContent='0';
  document.getElementById('sFail').textContent='0';
  document.getElementById('sTime').textContent='—';
  setActionBtns(false);
  clearLog();

  const mode = document.getElementById('crawlMode').value;
  const body = {
    urls,
    deep_crawl: mode==='single'?null:mode,
    max_pages:  parseInt(document.getElementById('maxPages').value)||20,
    delay:      parseFloat(document.getElementById('delay').value)||1.5,
    content_filter: document.getElementById('contentFilter').value,
    wait_for:   document.getElementById('waitFor').value.trim()||null,
    js_code:    document.getElementById('jsCode').value.trim()||null,
    link_filter: document.getElementById('linkFilter').value.trim()||null,
    pagination:     parseInt(document.getElementById('pagination').value)||1,
    max_concurrent: parseInt(document.getElementById('maxConcurrent').value)||3,
    anti_bot:       document.getElementById('antiBot').checked,
  };

  document.getElementById('startBtn').disabled=true;
  document.getElementById('startBtn').textContent='⏳ Đang khởi động…';
  setStatus('running','Starting…');

  try {
    const res = await fetch('/api/crawl',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const d = await res.json();
    jobId = d.job_id;
    document.getElementById('sJobId').textContent = jobId;
    startTimer();
    connectWs(jobId);
    log(`Job ${jobId} started — crawling ${urls.length} URL(s) [mode: ${mode}]`,'info');

    // Switch to monitor
    document.querySelector('.tab[data-tab="monitor"]').click();
    document.getElementById('startBtn').textContent='⏳ Running…';
  } catch(err){
    log('Lỗi khi start: '+err.message,'error');
    setStatus('error','Error');
    document.getElementById('startBtn').disabled=false;
    document.getElementById('startBtn').textContent='▶ Start Crawl';
  }
}

// ══════════════════════════════════════════════
// Export to API
// ══════════════════════════════════════════════
async function exportApi(){
  const ep = document.getElementById('exportUrl').value.trim();
  if(!ep){ alert('Nhập API endpoint URL.'); return; }
  if(!jobId){ alert('Chưa có dữ liệu crawl.'); return; }

  const headers={};
  hdrs.forEach(h=>{ if(h.k) headers[h.k]=h.v; });

  let fieldMap=null;
  const fm = document.getElementById('fieldMap').value.trim();
  if(fm){ try{fieldMap=JSON.parse(fm)}catch{alert('Field mapping JSON không hợp lệ.');return;} }

  let staticFields=null;
  const sf = document.getElementById('staticFields').value.trim();
  if(sf){ try{staticFields=JSON.parse(sf)}catch{alert('Static fields JSON không hợp lệ.');return;} }

  const body={
    job_id:jobId, endpoint:ep,
    method: document.getElementById('exportMethod').value,
    auth_type: document.getElementById('exportAuthType').value,
    rate_limit_ms: parseInt(document.getElementById('exportRateLimit').value)||300,
    headers,
    token: document.getElementById('exportToken').value.trim()||null,
    field_map: fieldMap,
    static_fields: staticFields
  };

  document.getElementById('exportBtn').disabled=true;
  document.getElementById('exportBtn').textContent='⏳ Exporting…';
  document.querySelector('.tab[data-tab="export"]').click();
  document.getElementById('exportLog').innerHTML =
    '<div class="text-muted" style="padding:16px">Đang export…</div>';

  try {
    const res  = await fetch('/api/export',{method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const data = await res.json();
    if(!res.ok || !Array.isArray(data.log)){
      const msg = data.detail
        ? JSON.stringify(data.detail)
        : (data.message || JSON.stringify(data));
      throw new Error(`Server ${res.status}: ${msg}`);
    }
    if(data.exported === 0){
      document.getElementById('exportLog').innerHTML =
        `<div style="color:var(--amber-l);padding:16px">
          ⚠ Không có item nào để export.<br>
          <span style="color:var(--muted);font-size:11px">
            Có thể job_id không còn tồn tại (server vừa restart) — hãy chạy crawl mới.
          </span>
        </div>`;
      return;
    }
    const ok   = data.log.filter(r=>r.ok).length;
    const fail = data.log.filter(r=>!r.ok).length;
    document.getElementById('badgeExport').textContent = data.exported;

    let html = `<div style="padding:12px 0 16px;font-size:13px;color:var(--sub)">
      Đã export <strong style="color:var(--green-l)">${data.exported}</strong> items&nbsp;→&nbsp;
      <code style="color:var(--blue-l)">${escHtml(ep)}</code>
      &nbsp;·&nbsp; <span style="color:var(--green-l)">${ok} ok</span>
      &nbsp;·&nbsp; <span style="color:var(--red-l)">${fail} lỗi</span>
    </div>`;
    for(const r of data.log){
      const c = r.ok?'var(--green-l)':'var(--red-l)';
      const ic = r.ok?'✓':'✗';
      html+=`<div class="export-row">
        <span style="color:${c};font-weight:700;width:14px">${ic}</span>
        <span class="badge ${r.ok?'badge-ok':'badge-err'}">${r.status}</span>
        <span class="export-url" title="${r.url||''}">${escHtml(r.url||'')}</span>
        ${r.response?`<span class="text-muted text-sm" style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${escHtml(r.response)}</span>`:''}
        ${r.error?`<span style="color:var(--red-l);font-size:11px">${escHtml(r.error)}</span>`:''}
      </div>`;
    }
    document.getElementById('exportLog').innerHTML = html;
  } catch(err){
    document.getElementById('exportLog').innerHTML =
      `<div style="color:var(--red-l);padding:16px">Export thất bại: ${escHtml(err.message)}</div>`;
  } finally {
    document.getElementById('exportBtn').disabled=false;
    document.getElementById('exportBtn').textContent='↑ Export to API';
  }
}

// ══════════════════════════════════════════════
// Download
// ══════════════════════════════════════════════
function dlJson(){
  const b = new Blob([JSON.stringify(rows,null,2)],{type:'application/json'});
  dlBlob(b,`crawl_${jobId||'data'}.json`);
}
function dlCsv(){
  const cols=['url','title','status_code','links_internal','links_external',
               'images','crawled_at','description','markdown'];
  const hdr = cols.join(',');
  const body= rows.map(r=>
    cols.map(c=>`"${(r[c]||'').toString().replace(/"/g,'""').slice(0,2000)}"`).join(',')
  );
  const b = new Blob(['﻿'+[hdr,...body].join('\r\n')],
    {type:'text/csv;charset=utf-8'});
  dlBlob(b,`crawl_${jobId||'data'}.csv`);
}
function dlBlob(blob, name){
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download=name; a.click();
}

// ══════════════════════════════════════════════
// Links Modal
// ══════════════════════════════════════════════
let allLinks = [];

async function viewLinks(i){
  const item = rows[i];
  const links = item.internal_link_list || [];
  allLinks = links;
  document.getElementById('linksModalTitle').textContent =
    `🔗 Internal Links — ${escHtml(item.url)} (${links.length} links)`;
  document.getElementById('linksFilter').value = '';
  document.getElementById('linksOverlay').classList.add('open');
  renderLinks(links);
}

function renderLinks(links){
  const wrap = document.getElementById('linksList');
  document.getElementById('linksCount').textContent = `${links.length} / ${allLinks.length} hiển thị`;
  wrap.innerHTML = links.map((u,i)=>`
    <label style="display:flex;align-items:center;gap:8px;padding:5px 10px;
                  border-bottom:1px solid var(--bg3);cursor:pointer;font-size:11.5px">
      <input type="checkbox" class="link-chk" value="${u}" checked
             style="accent-color:var(--blue);width:14px;height:14px;flex-shrink:0"
             onchange="updateCheckedCount()">
      <span style="color:var(--blue-l);word-break:break-all" title="${u}">${escHtml(u)}</span>
    </label>`).join('');
  updateCheckedCount();
}

function filterLinks(){
  const q = document.getElementById('linksFilter').value.toLowerCase().trim();
  const filtered = q ? allLinks.filter(u=>u.toLowerCase().includes(q)) : allLinks;
  renderLinks(filtered);
}

function updateCheckedCount(){
  const total = document.querySelectorAll('.link-chk').length;
  const checked = document.querySelectorAll('.link-chk:checked').length;
  document.getElementById('linksCount').textContent =
    `${document.querySelectorAll('.link-chk').length} hiển thị · ${checked} đã chọn`;
}

function selectAllLinks(){
  document.querySelectorAll('.link-chk').forEach(c=>c.checked=true);
  updateCheckedCount();
}
function deselectAllLinks(){
  document.querySelectorAll('.link-chk').forEach(c=>c.checked=false);
  updateCheckedCount();
}

function addSelectedLinks(){
  const selected = [...document.querySelectorAll('.link-chk:checked')].map(c=>c.value);
  if(!selected.length){ alert('Chưa chọn link nào.'); return; }
  let added = 0;
  selected.forEach(u=>{ if(!urls.includes(u)){ urls.push(u); added++; } });
  renderUrls();
  closeLinksModal();
  log(`✚ Đã thêm ${added} link vào crawl queue (${selected.length-added} đã có sẵn)`, 'info');
  // Switch to monitor tab so user can see the queue
  document.querySelector('.tab[data-tab="monitor"]').click();
}

function closeLinksModal(){
  document.getElementById('linksOverlay').classList.remove('open');
}
function linksOverlayClick(e){
  if(e.target===document.getElementById('linksOverlay')) closeLinksModal();
}
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML

if __name__ == "__main__":
    uvicorn.run("crawl_ui:app", host="0.0.0.0", port=8888, reload=False, log_level="warning")
