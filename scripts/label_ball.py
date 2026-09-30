"""A local click-to-label tool for the ball batch (data/ball_labelling), in the browser.

    python scripts/label_ball.py            # then open http://127.0.0.1:8765

The ball is a 6-15 px dot, so dragging boxes in a general tool is slow and imprecise: here a
click on the ball places a box of the batch's typical ball size, with a magnifier under the
cursor. Boxes pre-filled by the current model are shown; fix them and move on.

  click            ball here (replaces this frame's ball)
  drag             draw a box of the current class (1 ball, 2 stump)
  right-click      delete the box under the cursor
  N / Space / ->   save, next          <-  save, previous
  X                no ball in this frame (the ball is hidden / not yet released)
  S                skip frame (wrong view, replay): left out of training
A frame shows a warning while it has more than one ball box (only one ball is in play);
--multi-ball opens only those frames.
Writes Pascal VOC XML back to annotations/ and progress to review.json. Nothing leaves this machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_ball_labelling import voc_xml  # noqa: E402

CLASSES = {"ball": 0, "stump": 1}


def read_voc(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    boxes = []
    for obj in ET.parse(path).getroot().iter("object"):
        bb = obj.find("bndbox")
        boxes.append({"cls": obj.findtext("name"), "box": [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]})
    return boxes


class Handler(BaseHTTPRequestHandler):
    base: Path
    multi_ball_only = False

    def log_message(self, *_):  # quiet
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _review(self) -> dict:
        p = self.base / "review.json"
        return json.loads(p.read_text()) if p.is_file() else {}

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/list":
            stems = sorted(p.stem for p in (self.base / "images").glob("*.jpg"))
            if self.multi_ball_only:
                stems = [s for s in stems if sum(b["cls"] == "ball" for b in read_voc(self.base / "annotations" / f"{s}.xml")) > 1]
            sizes = [b["box"][2] - b["box"][0] for s in stems for b in read_voc(self.base / "annotations" / f"{s}.xml") if b["cls"] == "ball"]
            size = sorted(sizes)[len(sizes) // 2] if sizes else 12
            self._send(json.dumps({"stems": stems, "review": self._review(), "ball_size": size}).encode(), "application/json")
        elif self.path.startswith("/api/ann/"):
            stem = Path(self.path.rsplit("/", 1)[1]).name
            self._send(json.dumps(read_voc(self.base / "annotations" / f"{stem}.xml")).encode(), "application/json")
        elif self.path.startswith("/img/"):
            p = self.base / "images" / Path(self.path.rsplit("/", 1)[1]).name
            if p.is_file():
                self._send(p.read_bytes(), "image/jpeg")
            else:
                self._send(b"not found", "text/plain", 404)
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self) -> None:
        if not self.path.startswith("/api/ann/"):
            return self._send(b"not found", "text/plain", 404)
        stem = Path(self.path.rsplit("/", 1)[1]).name
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        w, h = data["width"], data["height"]
        boxes = [(CLASSES[b["cls"]], b["box"]) for b in data["boxes"] if b["cls"] in CLASSES]
        (self.base / "annotations" / f"{stem}.xml").write_text(voc_xml(f"{stem}.jpg", w, h, boxes))
        review = self._review()
        review[stem] = data["status"]  # "done" | "skip"
        (self.base / "review.json").write_text(json.dumps(review, indent=1, sort_keys=True))
        self._send(b"{}", "application/json")


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ball Labeller</title>
<style>
:root { --bg:#111418; --panel:#1b2027; --text:#e6e9ee; --muted:#8a93a0; --ball:#39e27a; --stump:#4aa8ff; --warn:#ffb020; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:14px/1.4 system-ui, sans-serif; }
header { display:flex; gap:16px; align-items:center; padding:8px 14px; background:var(--panel); flex-wrap:wrap; }
header b { font-size:15px; }
.pill { padding:2px 8px; border-radius:10px; background:#2a313b; color:var(--muted); }
.pill.on { background:var(--ball); color:#062; } .pill.stump.on { background:var(--stump); color:#012; }
.pill.done { background:#1f5130; color:#bff5cf; } .pill.skip { background:#5a4210; color:#ffe2a8; }
#wrap { position:relative; margin:8px auto; width:fit-content; }
canvas { display:block; cursor:crosshair; max-width:calc(100vw - 16px); }
#loupe { position:fixed; pointer-events:none; width:180px; height:180px; border:2px solid #fff; border-radius:50%; overflow:hidden; display:none; box-shadow:0 2px 12px #000a; }
#help { color:var(--muted); padding:0 14px 12px; }
kbd { background:#2a313b; border-radius:3px; padding:0 5px; color:var(--text); }
#bar { height:4px; background:#2a313b; } #bar i { display:block; height:100%; background:var(--ball); width:0; }
</style></head><body>
<header><b>Ball labeller</b><span id="name"></span><span id="state" class="pill"></span>
<span class="pill on" id="c1">1 ball</span><span class="pill stump" id="c2">2 stump</span><span id="prog" class="pill"></span><span id="warn" class="pill" style="display:none;background:#ffb020;color:#221500"></span></header>
<div id="bar"><i></i></div>
<div id="wrap"><canvas id="cv"></canvas></div><canvas id="loupe" width="180" height="180"></canvas>
<div id="help"><kbd>click</kbd> ball here &nbsp; <kbd>drag</kbd> draw box &nbsp; <kbd>right-click</kbd> delete box &nbsp;
<kbd>N</kbd>/<kbd>Space</kbd>/<kbd>&rarr;</kbd> save + next &nbsp; <kbd>&larr;</kbd> previous &nbsp; <kbd>X</kbd> no ball &nbsp; <kbd>S</kbd> skip frame &nbsp; <kbd>1</kbd>/<kbd>2</kbd> class</div>
<script>
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const loupe = document.getElementById('loupe'), lctx = loupe.getContext('2d');
let stems = [], review = {}, ballSize = 12, i = 0, img = new Image(), boxes = [], cls = 'ball', drag = null, scale = 1, mouse = null;
const $ = id => document.getElementById(id);

async function init() {
  const r = await (await fetch('/api/list')).json();
  stems = r.stems; review = r.review; ballSize = Math.max(8, Math.round(r.ball_size));
  i = Math.max(0, stems.findIndex(s => !review[s]));
  if (!stems.length) { document.body.insertAdjacentHTML('beforeend', '<p style="padding:14px">Nothing to review.</p>'); return; }
  load();
}
async function load() {
  const s = stems[i];
  boxes = await (await fetch('/api/ann/' + s)).json();
  img = new Image(); img.onload = () => { fit(); draw(); }; img.src = '/img/' + s + '.jpg';
  $('name').textContent = `${i + 1} / ${stems.length}  ${s}`;
  const st = review[s]; $('state').textContent = st ? (st === 'skip' ? 'skipped' : 'reviewed') : 'new';
  $('state').className = 'pill ' + (st || '');
  const n = Object.keys(review).length; $('prog').textContent = `${n} reviewed`;
  document.querySelector('#bar i').style.width = (100 * n / stems.length) + '%';
}
function fit() {
  const maxW = window.innerWidth - 16, maxH = window.innerHeight - 110;
  scale = Math.min(maxW / img.width, maxH / img.height, 1);
  cv.width = Math.round(img.width * scale); cv.height = Math.round(img.height * scale);
}
function draw() {
  ctx.drawImage(img, 0, 0, cv.width, cv.height);
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.box.map(v => v * scale), pad = b.cls === 'ball' ? 6 : 2;
    ctx.strokeStyle = b.cls === 'ball' ? '#39e27a' : '#4aa8ff'; ctx.lineWidth = 2;
    ctx.strokeRect(x1 - pad, y1 - pad, x2 - x1 + 2 * pad, y2 - y1 + 2 * pad);
  }
  const nBall = boxes.filter(b => b.cls === 'ball').length;
  $('warn').style.display = nBall > 1 ? 'inline' : 'none';
  $('warn').textContent = `${nBall} balls - keep only the real one (right-click the others, or click on the real ball)`;
  if (drag && drag.moved) { ctx.strokeStyle = '#fff'; ctx.setLineDash([4, 3]);
    ctx.strokeRect(drag.x0 * scale, drag.y0 * scale, (drag.x - drag.x0) * scale, (drag.y - drag.y0) * scale); ctx.setLineDash([]); }
  drawLoupe();
}
function drawLoupe() {
  if (!mouse) { loupe.style.display = 'none'; return; }
  loupe.style.display = 'block';
  loupe.style.left = (mouse.cx + 24) + 'px'; loupe.style.top = (mouse.cy - 200 < 0 ? mouse.cy + 24 : mouse.cy - 200) + 'px';
  const Z = 5, src = 180 / Z;
  lctx.imageSmoothingEnabled = false;
  lctx.fillStyle = '#000'; lctx.fillRect(0, 0, 180, 180);
  lctx.drawImage(img, mouse.x - src / 2, mouse.y - src / 2, src, src, 0, 0, 180, 180);
  for (const b of boxes) {
    const [x1, y1, x2, y2] = b.box;
    lctx.strokeStyle = b.cls === 'ball' ? '#39e27a' : '#4aa8ff'; lctx.lineWidth = 1.5;
    lctx.strokeRect((x1 - mouse.x + src / 2) * Z, (y1 - mouse.y + src / 2) * Z, (x2 - x1) * Z, (y2 - y1) * Z);
  }
  lctx.strokeStyle = '#fff8'; lctx.beginPath(); lctx.moveTo(90, 80); lctx.lineTo(90, 100); lctx.moveTo(80, 90); lctx.lineTo(100, 90); lctx.stroke();
}
function pos(e) { const r = cv.getBoundingClientRect(); return { x: (e.clientX - r.left) / scale, y: (e.clientY - r.top) / scale, cx: e.clientX, cy: e.clientY }; }
cv.addEventListener('mousedown', e => { if (e.button !== 0) return; const p = pos(e); drag = { x0: p.x, y0: p.y, x: p.x, y: p.y, moved: false }; });
cv.addEventListener('mousemove', e => { mouse = pos(e); if (drag) { drag.x = mouse.x; drag.y = mouse.y;
  drag.moved = drag.moved || Math.hypot(drag.x - drag.x0, drag.y - drag.y0) * scale > 5; } draw(); });
cv.addEventListener('mouseleave', () => { mouse = null; draw(); });
window.addEventListener('mouseup', e => {
  if (!drag || e.button !== 0) return;
  const d = drag; drag = null;
  if (d.moved) {
    const box = [Math.min(d.x0, d.x), Math.min(d.y0, d.y), Math.max(d.x0, d.x), Math.max(d.y0, d.y)];
    if (cls === 'ball') boxes = boxes.filter(b => b.cls !== 'ball');
    boxes.push({ cls, box });
  } else {
    const h = ballSize / 2;
    boxes = boxes.filter(b => b.cls !== 'ball');
    boxes.push({ cls: 'ball', box: [d.x0 - h, d.y0 - h, d.x0 + h, d.y0 + h] });
  }
  draw();
});
cv.addEventListener('contextmenu', e => {
  e.preventDefault(); const p = pos(e);
  const hit = boxes.map((b, k) => [k, b]).filter(([, b]) => p.x >= b.box[0] - 8 && p.x <= b.box[2] + 8 && p.y >= b.box[1] - 8 && p.y <= b.box[3] + 8)
    .sort((a, b) => (a[1].box[2] - a[1].box[0]) - (b[1].box[2] - b[1].box[0]));
  if (hit.length) { boxes.splice(hit[0][0], 1); draw(); }
});
async function save(status) {
  const s = stems[i];
  await fetch('/api/ann/' + s, { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ width: img.width, height: img.height, boxes, status }) });
  review[s] = status;
}
async function go(step, status = 'done') { await save(status); i = Math.min(stems.length - 1, Math.max(0, i + step)); load(); }
function setCls(c) { cls = c; $('c1').classList.toggle('on', c === 'ball'); $('c2').classList.toggle('on', c === 'stump'); }
window.addEventListener('keydown', e => {
  const k = e.key.toLowerCase();
  if (k === 'n' || k === ' ' || k === 'arrowright') { e.preventDefault(); go(1); }
  else if (k === 'arrowleft') go(-1);
  else if (k === 'x') { boxes = boxes.filter(b => b.cls !== 'ball'); draw(); go(1); }
  else if (k === 's') go(1, 'skip');
  else if (k === '1') setCls('ball'); else if (k === '2') setCls('stump');
});
window.addEventListener('resize', () => { if (img.width) { fit(); draw(); } });
init();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=ROOT / "data" / "ball_labelling")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--multi-ball", action="store_true", help="only frames with more than one ball box")
    args = parser.parse_args(argv)
    Handler.base = args.dir
    Handler.multi_ball_only = args.multi_ball
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    n = len(list((args.dir / "images").glob("*.jpg")))
    print(f"{n} images in {args.dir}\nopen http://127.0.0.1:{args.port}   (Ctrl+C to stop; progress is saved as you go)")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
