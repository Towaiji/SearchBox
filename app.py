#!/usr/bin/env python3
# searchbox.py — One-file Local Document Search Engine (BM25) with Web UI
# Zero dependencies. Indexes .md .txt .html, serves a fast UI + JSON API.
# Usage: python3 searchbox.py /path/to/folder --port 8000
import os, re, sys, json, math, time, html, mimetypes, urllib.parse, argparse, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

###############################################################################
#                                INDEXER                                      #
###############################################################################

TOKEN_RE = re.compile(r"[A-Za-z0-9]+", re.UNICODE)

def read_text(path):
    # Read text with safe fallback encodings
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with open(path, "r", encoding=enc, errors="ignore") as f:
                return f.read()
        except Exception:
            continue
    return ""

def strip_html(text):
    # crude but effective stripping for .html
    text = re.sub(r"(?is)<script.*?>.*?</script>", " ", text)
    text = re.sub(r"(?is)<style.*?>.*?</style>", " ", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return html.unescape(text)

def tokenize(text):
    return [t.lower() for t in TOKEN_RE.findall(text)]

def file_to_text(path):
    ext = os.path.splitext(path)[1].lower()
    raw = read_text(path)
    if ext == ".html" or ext == ".htm":
        return strip_html(raw)
    return raw

def walk_docs(root):
    exts = {".md", ".txt", ".html", ".htm"}
    for base, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts:
                p = os.path.join(base, fn)
                yield p

class BM25Index:
    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.lock = threading.Lock()
        self.reindex()

    def reindex(self):
        with self.lock:
            self.docs = []          # list of dicts: {id, path, title, text, tokens, mtime, len}
            self.df = {}            # term -> doc freq
            self.tf = []            # list of dict: term -> frequency in that doc
            self.avgdl = 0.0
            self.idf = {}           # computed later
            self.last_build = time.time()

            for i, path in enumerate(walk_docs(self.root)):
                try:
                    text = file_to_text(path)
                    title = os.path.basename(path)
                    toks = tokenize(text)
                    freq = {}
                    for t in toks:
                        freq[t] = freq.get(t, 0) + 1
                    for term in freq.keys():
                        self.df[term] = self.df.get(term, 0) + 1
                    self.docs.append({
                        "id": i, "path": path, "title": title, "text": text,
                        "mtime": os.path.getmtime(path), "len": len(toks)
                    })
                    self.tf.append(freq)
                except Exception:
                    continue

            N = len(self.docs) or 1
            self.avgdl = sum(d["len"] for d in self.docs)/N
            # BM25 IDF
            for term, df in self.df.items():
                # BM25+ like IDF with 0.5 add-one
                self.idf[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def maybe_reindex(self):
        # quick poll: if any mtime changed or new files exist, rebuild
        with self.lock:
            try:
                snapshot = [(p, os.path.getmtime(p)) for p in walk_docs(self.root)]
            except Exception:
                snapshot = []
            have = {(d["path"], d["mtime"]) for d in self.docs}
            now = set(snapshot)
        if now != have:
            self.reindex()

    def score(self, q_terms, doc_idx, k1=1.5, b=0.75):
        with self.lock:
            d = self.docs[doc_idx]
            freq = self.tf[doc_idx]
            score = 0.0
            for t in q_terms:
                if t not in freq:
                    continue
                idf = self.idf.get(t, 0.0)
                tf = freq[t]
                dl = d["len"] or 1
                denom = tf + k1 * (1 - b + b * dl/self.avgdl)
                score += idf * (tf * (k1 + 1)) / denom
            return score

    def search(self, query, limit=20):
        self.maybe_reindex()
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores = []
        with self.lock:
            for i, d in enumerate(self.docs):
                s = self.score(q_terms, i)
                if s > 0:
                    scores.append((s, i))
        scores.sort(reverse=True, key=lambda x: x[0])
        results = []
        for s, i in scores[:limit]:
            d = self.docs[i]
            snippet = make_snippet(d["text"], q_terms, 240)
            results.append({
                "title": d["title"],
                "path": os.path.relpath(d["path"], self.root),
                "score": round(float(s), 4),
                "snippet": snippet,
                "mtime": int(d["mtime"])
            })
        return results

###############################################################################
#                                SNIPPETS                                     #
###############################################################################

def first_hit_span(text, terms):
    low = text.lower()
    positions = []
    for t in terms:
        p = low.find(t)
        if p != -1:
            positions.append(p)
    return min(positions) if positions else 0

def make_snippet(text, terms, width=220):
    # pick window around first match
    pos = first_hit_span(text, terms)
    start = max(0, pos - width // 4)
    end = min(len(text), start + width)
    chunk = text[start:end]

    # escape + highlight
    esc = html.escape(chunk)
    for t in sorted(set(terms), key=len, reverse=True):
        esc = re.sub(fr"(?i)\b({re.escape(t)})\b", r"<mark>\1</mark>", esc)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + esc + suffix

###############################################################################
#                                WEB SERVER                                   #
###############################################################################

INDEX_HTML = """<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SearchBox — Local Search</title>
<style>
:root{
  --bg:#0f1115;--panel:#141822;--ink:#e8e9ee;--muted:#9aa4b2;--acc:#CA3EA7;--ring:0 0 0 2px #ffffff26
}
html,body{height:100%}
body{margin:0;background:radial-gradient(1200px 600px at 70% -10%, #CA3EA720, transparent),var(--bg);color:var(--ink);font:14px/1.3 system-ui,Segoe UI,Inter,Roboto,Helvetica,Arial}
.container{max-width:980px;margin:0 auto;padding:18px}
header{position:sticky;top:0;background:color-mix(in oklab, var(--bg) 82%, transparent);backdrop-filter:blur(8px);border-bottom:1px solid #ffffff14;z-index:5}
h1{font:900 22px/1 Inter,system-ui;letter-spacing:.3px}
.brand{background:linear-gradient(90deg,#E4454A,#CA3EA7);-webkit-background-clip:text;background-clip:text;color:transparent}
.search{display:flex;gap:10px;margin:12px 0 16px}
.input{flex:1;padding:13px 14px;border-radius:10px;border:1px solid #ffffff24;background:#ffffff0f;color:var(--ink);outline:none}
.input:focus{box-shadow:var(--ring)}
.btn{padding:12px 14px;border-radius:10px;border:1px solid #ffffff24;background:#ffffff10;color:#fff;cursor:pointer}
.btn:hover{filter:brightness(1.1)}
small{color:var(--muted)}
.card{background:var(--panel);border:1px solid #ffffff14;border-radius:14px;padding:14px;margin:10px 0;box-shadow:0 8px 30px rgba(0,0,0,.25)}
.title{font-weight:800;margin-bottom:6px}
.meta{display:flex;gap:10px;color:#9aa4b2;font-size:12px}
.path{color:#cfd6e4}
.grid{display:grid;grid-template-columns:1fr;gap:10px}
footer{opacity:.75;text-align:center;margin:28px 0 8px;color:#9aa4b2}
kbd{background:#000;border:1px solid #333;border-bottom-color:#111;border-radius:6px;padding:0 6px;box-shadow: inset 0 -1px 0 #000}
.progress{position:fixed;left:0;top:0;height:3px;background:linear-gradient(90deg,#E4454A,#CA3EA7);width:0%;z-index:10}
</style>
<div class="progress" id="bar"></div>
<header>
  <div class="container" style="display:flex;align-items:center;justify-content:space-between;gap:12px">
    <h1>Search<span class="brand">Box</span></h1>
    <small>Press <kbd>/</kbd> to focus • <kbd>Enter</kbd> to search</small>
  </div>
</header>
<main class="container">
  <div class="search">
    <input class="input" id="q" placeholder="Search your docs (md, txt, html)..." autofocus>
    <button class="btn" id="go">Search</button>
  </div>
  <div id="stats"><small>Ready.</small></div>
  <div class="grid" id="out"></div>
</main>
<footer>Local, offline, private. Powered by BM25. </footer>
<script>
const $=sel=>document.querySelector(sel);
const bar=$('#bar');
function progress(p){ bar.style.width=(p*100).toFixed(1)+'%'; if(p>=1) setTimeout(()=>bar.style.width='0%',500); }

async function search(q){
  progress(.2);
  const res = await fetch('/search?q='+encodeURIComponent(q));
  progress(.6);
  const data = await res.json();
  progress(1);
  return data;
}
function render(results, took, q){
  const out=$('#out'); out.innerHTML='';
  $('#stats').innerHTML = '<small>'+results.length+' results in '+took.toFixed(1)+' ms</small>';
  for(const r of results){
    const el=document.createElement('article');
    el.className='card';
    el.innerHTML = `
      <div class="title">${escapeHtml(r.title)} <small class="path">/ ${escapeHtml(r.path)}</small></div>
      <div class="meta">
        <div>Score: ${r.score}</div>
        <div>Modified: ${new Date(r.mtime*1000).toLocaleString()}</div>
      </div>
      <div class="snippet" style="margin-top:8px;color:#dfe6f3">${r.snippet}</div>
      <div style="margin-top:10px">
        <a class="btn" href="/raw?path=${encodeURIComponent(r.path)}" target="_blank">View Raw</a>
        <a class="btn" href="/open?path=${encodeURIComponent(r.path)}">Open in Folder</a>
      </div>`;
    out.appendChild(el);
  }
}
function escapeHtml(s){ return s.replace(/[&<>\"]/g, c=>({\"&\":\"&amp;\",\"<\":\"&lt;\",\">\":\"&gt;\",\"\\\"\":\"&quot;\"}[c])); }

const q=$('#q'), go=$('#go');
go.onclick=run; q.addEventListener('keydown',e=>{ if(e.key==='Enter') run(); });
document.addEventListener('keydown',e=>{ if(e.key==='/'){ e.preventDefault(); q.focus(); } });
async function run(){
  const t0=performance.now(); const data=await search(q.value.trim()); const t1=performance.now();
  render(data, t1-t0, q.value.trim());
}
// Initial hash search
if (location.hash.length>1){ q.value=decodeURIComponent(location.hash.slice(1)); run(); }
q.addEventListener('input', ()=>{ history.replaceState(null,'','#'+encodeURIComponent(q.value.trim())); });
</script>
"""

class App(BaseHTTPRequestHandler):
    index: BM25Index = None  # attached later

    def _send(self, code=200, ctype="text/html; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if path == "/":
            self._send()
            self.wfile.write(INDEX_HTML.encode("utf-8"))
            return

        if path == "/search":
            query = (qs.get("q", [""])[0] or "").strip()
            limit = int((qs.get("limit", ["20"])[0]))
            t0 = time.time()
            results = self.index.search(query, limit=limit)
            took = (time.time() - t0) * 1000
            self._send(ctype="application/json; charset=utf-8")
            self.wfile.write(json.dumps(results).encode("utf-8"))
            return

        if path == "/reindex":
            self.index.reindex()
            self._send(ctype="application/json; charset=utf-8")
            self.wfile.write(json.dumps({"status": "ok", "docs": len(self.index.docs)}).encode("utf-8"))
            return

        if path == "/raw":
            # serve raw file for quick view
            rel = (qs.get("path", [""])[0]).replace("\\", "/")
            abs_path = os.path.abspath(os.path.join(self.index.root, rel))
            if not abs_path.startswith(self.index.root) or not os.path.exists(abs_path):
                self._send(404); self.wfile.write(b"Not found"); return
            ctype = mimetypes.guess_type(abs_path)[0] or "text/plain; charset=utf-8"
            self._send(ctype=ctype)
            try:
                with open(abs_path, "rb") as f:
                    self.wfile.write(f.read())
            except Exception:
                self.wfile.write(b"")
            return

        if path == "/open":
            # open in system file browser where possible
            rel = (qs.get("path", [""])[0]).replace("\\", "/")
            abs_path = os.path.abspath(os.path.join(self.index.root, rel))
            if not abs_path.startswith(self.index.root) or not os.path.exists(abs_path):
                self._send(404); self.wfile.write(b"Not found"); return
            self._send(ctype="application/json; charset=utf-8")
            threading.Thread(target=open_in_folder, args=(abs_path,), daemon=True).start()
            self.wfile.write(json.dumps({"status":"opening"}).encode("utf-8"))
            return

        # 404
        self._send(404); self.wfile.write(b"Not found")

def open_in_folder(path):
    try:
        if sys.platform.startswith("darwin"):
            os.system(f"open -R {shquote(path)}")
        elif os.name == "nt":
            os.system(f'explorer /select,"{path}"')
        else:
            # Linux: try xdg-open on the directory
            os.system(f"xdg-open {shquote(os.path.dirname(path))}")
    except Exception:
        pass

def shquote(s):
    return "'" + s.replace("'", "'\\''") + "'"

###############################################################################
#                                MAIN                                         #
###############################################################################

def main():
    p = argparse.ArgumentParser(description="SearchBox — one-file local search engine (BM25).")
    p.add_argument("folder", help="Folder to index (.md .txt .html)")
    p.add_argument("--port", type=int, default=8000, help="Port (default 8000)")
    args = p.parse_args()

    if not os.path.isdir(args.folder):
        print("Folder not found:", args.folder)
        sys.exit(1)

    idx = BM25Index(args.folder)
    App.index = idx

    print(f"Indexing '{idx.root}' — {len(idx.docs)} docs")
    print(f"Serving  http://localhost:{args.port}")
    print("API:     /search?q=hello   •  Reindex: /reindex   •  Raw file: /raw?path=...")

    httpd = HTTPServer(("0.0.0.0", args.port), App)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nBye!")

if __name__ == "__main__":
    main()
