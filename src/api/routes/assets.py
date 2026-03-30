"""
Module: src.api.routes.assets
==============================

Protected asset serving Blueprint for FixProtoGPT.

Serves JS as a single minified bundle through Flask (not as static files)
so the raw source isn't directly accessible via the browser.
Adds anti-devtools / anti-view-source deterrents for the client.

Coding Standards
----------------
- PEP 8  : Python Style Guide
- PEP 257 : Docstring Conventions — Google-style docstrings
- PEP 484 : Type Hints
- Google Python Style Guide

Author : FixProtoGPT Team
"""

import hashlib
from pathlib import Path
from flask import Blueprint, Response, request

try:
    import rjsmin
    _minify = rjsmin.jsmin
except ImportError:
    _minify = lambda s, **kw: s          # fallback: no minification

assets_bp = Blueprint("assets", __name__)

# ── JS source directory (NOT under static/) ──────────────────────
_JS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "ui" / "js_src"

# Load order must match the original script tag order
_JS_FILES = [
    "api.js",
    "ui.js",
    "tokens.js",
    "versions.js",
    "generate.js",
    "conversations.js",
    "file-io.js",
    "app.js",
]

# ── Cache ─────────────────────────────────────────────────────────
_bundle_cache: dict = {}          # {"js": minified_str, "etag": hash}
_bundle_mtime: float = 0.0       # track source changes


def _build_bundle() -> str:
    """Concatenate and minify all JS source files into one bundle.

    Returns:
        Minified JavaScript string.
    """
    parts: list[str] = []
    for name in _JS_FILES:
        path = _JS_DIR / name
        if path.exists():
            parts.append(f"/* {name} */")
            parts.append(path.read_text(encoding="utf-8"))
    raw = "\n".join(parts)
    return _minify(raw)


def _get_bundle() -> tuple[str, str]:
    """Return ``(minified_js, etag)``.  Rebuilds when sources change.

    Returns:
        Tuple of ``(js_text, etag_string)``.
    """
    global _bundle_mtime
    current_mtime = max(
        ((_JS_DIR / f).stat().st_mtime for f in _JS_FILES if (_JS_DIR / f).exists()),
        default=0.0,
    )
    if not _bundle_cache or current_mtime > _bundle_mtime:
        js = _build_bundle()
        etag = hashlib.md5(js.encode()).hexdigest()[:12]
        _bundle_cache["js"] = js
        _bundle_cache["etag"] = etag
        _bundle_mtime = current_mtime
    return _bundle_cache["js"], _bundle_cache["etag"]


@assets_bp.route("/assets/bundle.js")
def serve_bundle():
    """Serve the combined, minified JS bundle with cache headers."""
    js, etag = _get_bundle()

    # ETag-based 304
    if request.headers.get("If-None-Match") == etag:
        return Response(status=304)

    resp = Response(js, mimetype="application/javascript")
    resp.headers["ETag"] = etag
    resp.headers["Cache-Control"] = "no-cache, must-revalidate"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp


# ── Inline anti-devtools snippet (injected into templates) ────────

SOURCE_PROTECTION_JS = r"""
<script>
(function(){
    // ── Disable right-click context menu ──
    document.addEventListener('contextmenu',function(e){e.preventDefault();return false;});

    // ── Block common DevTools keyboard shortcuts ──
    document.addEventListener('keydown',function(e){
        // F12
        if(e.key==='F12'){e.preventDefault();return false;}
        // Ctrl+Shift+I / Cmd+Opt+I (Inspector)
        if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key==='I'){e.preventDefault();return false;}
        // Ctrl+Shift+J / Cmd+Opt+J (Console)
        if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key==='J'){e.preventDefault();return false;}
        // Ctrl+Shift+C / Cmd+Opt+C (Element picker)
        if((e.ctrlKey||e.metaKey)&&e.shiftKey&&e.key==='C'){e.preventDefault();return false;}
        // Ctrl+U / Cmd+U (View Source)
        if((e.ctrlKey||e.metaKey)&&e.key==='u'){e.preventDefault();return false;}
        if((e.ctrlKey||e.metaKey)&&e.key==='U'){e.preventDefault();return false;}
        // Ctrl+S / Cmd+S (Save page)
        if((e.ctrlKey||e.metaKey)&&e.key==='s'){e.preventDefault();return false;}
        if((e.ctrlKey||e.metaKey)&&e.key==='S'){e.preventDefault();return false;}
    });

    // ── Detect DevTools via debugger-timing trick ──
    var _dtOpen=false;
    setInterval(function(){
        var t0=performance.now();
        debugger;
        if(performance.now()-t0>100){
            if(!_dtOpen){
                _dtOpen=true;
                document.body.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100vh;background:#0f0f23;color:#ef4444;font-family:Inter,sans-serif;font-size:1.4rem;text-align:center;padding:2rem;"><div><i class="fas fa-shield-alt" style="font-size:3rem;margin-bottom:1rem;display:block;"></i>Developer tools are not permitted.<br><small style="color:#888;font-size:0.9rem;">Please close DevTools and refresh the page.</small></div></div>';
            }
        }
    },2000);

    // ── Disable text selection on body ──
    document.addEventListener('selectstart',function(e){
        if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
        e.preventDefault();
    });

    // ── Disable drag ──
    document.addEventListener('dragstart',function(e){e.preventDefault();});
})();
</script>
"""
