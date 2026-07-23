#!/usr/bin/env python3
"""
The Vision Suite design system — one source of truth for every page.

The whole suite wears the Annotation Studio's look: a small set of CSS custom
properties with a dark and a light variant, switched by ``data-theme`` on
<html> and remembered in localStorage. Because all three tools are now served
from the same origin, that one localStorage key keeps the theme in sync as you
move between them.

Two things live here:

1. ``TOKENS`` — the palette itself, plus ``LEGACY_ALIASES``, which re-expresses
   the inference apps' original variable names (``--panel``, ``--hivis``, …) in
   terms of the tokens. That is what lets ~4 000 lines of existing CSS in
   web_app.py / crop_balancer_app.py adopt the new theme (including light mode)
   without touching a single rule.
2. ``THEME_SCRIPT`` / ``THEME_JS`` / ``theme_button()`` — the pre-paint theme
   applier, the toggle, and the button markup, shared verbatim by all pages.

Add a colour here and every tool gets it.
"""

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
# Dark is the base; the light block overrides. Every value the suite paints with
# comes from this list — no page should hardcode a colour.
TOKENS = """
  :root{
    --bg:#0a0a0a; --surface:#171717; --surface-2:#222; --surface-3:#1c1c1c;
    --border:#333; --border-2:#444;
    --text:#f5f5f5; --text-muted:#a3a3a3; --text-dim:#6f6f6f;
    --accent:#6ea8fe; --accent-soft:rgba(110,168,254,.14); --accent-fg:#0a0a0a; --accent-hover:#86b8ff;
    --ok:#34d399; --ok-soft:rgba(52,211,153,.14); --ok-deep:#10b981; --ok-fg:#05291d;
    --danger:#f87171; --danger-soft:rgba(248,113,113,.12); --danger-border:rgba(248,113,113,.32);
    --danger-fg:#fecaca;
    --warn:#fbbf24; --warn-soft:rgba(251,191,36,.14); --canvas-bg:#000;
    /* Chrome that floats *over* the live video, where the backdrop is the frame
       itself rather than the page — hence translucent rather than --surface. */
    --overlay:rgba(16,16,16,.72); --overlay-strong:rgba(10,10,10,.92); --overlay-solid:#1c1c1c;
    --r:8px; --r-lg:12px;
    --sh-sm:0 1px 2px rgba(0,0,0,.5);
    --sh-md:0 6px 18px rgba(0,0,0,.5);
    --sh-lg:0 18px 44px rgba(0,0,0,.6);
    --ring:0 0 0 3px rgba(110,168,254,.22);
    --mono:ui-monospace,"SF Mono","JetBrains Mono","Cascadia Code",Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  :root[data-theme="light"]{
    --bg:#f7f7f8; --surface:#ffffff; --surface-2:#eef0f2; --surface-3:#f2f3f5;
    --border:#e2e4e8; --border-2:#cfd2d8;
    --text:#171717; --text-muted:#5b6470; --text-dim:#9aa0aa;
    --accent:#2563eb; --accent-soft:rgba(37,99,235,.10); --accent-fg:#ffffff; --accent-hover:#1d4ed8;
    --ok:#059669; --ok-soft:rgba(5,150,105,.12); --ok-deep:#047857; --ok-fg:#ffffff;
    --danger:#dc2626; --danger-soft:rgba(220,38,38,.08); --danger-border:rgba(220,38,38,.30);
    --danger-fg:#b91c1c;
    --warn:#b45309; --warn-soft:rgba(180,83,9,.12); --canvas-bg:#d7d9dd;
    --overlay:rgba(255,255,255,.84); --overlay-strong:rgba(255,255,255,.96); --overlay-solid:#ffffff;
    --sh-sm:0 1px 2px rgba(16,24,40,.06);
    --sh-md:0 6px 18px rgba(16,24,40,.10);
    --sh-lg:0 18px 44px rgba(16,24,40,.16);
    --ring:0 0 0 3px rgba(37,99,235,.20);
  }
"""

# The inference web app and the crop tools were built against a different set of
# names (a dark "hi-vis amber on navy" palette). Rather than rewrite thousands
# of `var(--panel)` references, we alias the old names onto the tokens above:
# their CSS keeps working, but it now reads the suite palette and follows the
# light/dark switch for free.
LEGACY_ALIASES = """
  :root{
    --panel:var(--surface); --panel-2:var(--surface-2); --panel-3:var(--surface-3);
    --line:var(--border); --line-2:var(--border-2);
    --muted:var(--text-muted); --muted-2:var(--text-dim);
    --stage:var(--canvas-bg);
    --hivis:var(--accent); --hivis-2:var(--accent-hover); --hivis-soft:var(--accent-soft);
    --hivis-fg:var(--accent-fg);
    --go:var(--ok); --go-deep:var(--ok-deep); --go-fg:var(--ok-fg);
    --bad:var(--danger); --bad-soft:var(--danger-soft); --bad-fg:var(--danger-fg);
  }
"""

# Page chrome every tool shares: box model, base type, and the thin scrollbars.
BASE_CSS = """
  *{box-sizing:border-box;}
  html,body{background:var(--bg);color:var(--text);font-family:var(--sans);
            -webkit-font-smoothing:antialiased;}
  *{scrollbar-width:thin;scrollbar-color:var(--border-2) transparent;}
  *::-webkit-scrollbar{width:9px;height:9px;}
  *::-webkit-scrollbar-track{background:transparent;}
  *::-webkit-scrollbar-thumb{background:var(--border-2);border-radius:9px;border:2px solid transparent;background-clip:padding-box;}
  *::-webkit-scrollbar-thumb:hover{background:var(--text-dim);background-clip:padding-box;}
  .mono{font-family:var(--mono);font-variant-numeric:tabular-nums;}
"""

# The toggle button + the "back to the suite" home pill, styled the same on
# every page so the two ways out of a tool are always in the same place.
CHROME_CSS = """
  .theme-btn{display:inline-flex;align-items:center;justify-content:center;
    width:38px;height:38px;flex:0 0 auto;padding:0;
    background:var(--surface);border:1px solid var(--border);color:var(--text-muted);
    border-radius:var(--r);cursor:pointer;transition:background .15s,color .15s,border-color .15s;}
  .theme-btn:hover{background:var(--surface-2);color:var(--text);border-color:var(--border-2);}
  .theme-btn svg{width:16px;height:16px;}
  .home-btn{display:inline-flex;align-items:center;gap:8px;font-family:var(--sans);
    font-size:12.5px;font-weight:600;letter-spacing:.2px;text-decoration:none;
    color:var(--accent);cursor:pointer;flex:0 0 auto;min-height:0;width:auto;
    background:var(--accent-soft);border:1px solid transparent;border-radius:var(--r);
    padding:9px 15px;transition:background .15s,color .15s,border-color .15s,box-shadow .15s;}
  .home-btn svg{width:15px;height:15px;flex:0 0 auto;}
  .home-btn:hover{background:var(--accent);color:var(--accent-fg);border-color:var(--accent);
    box-shadow:var(--ring);filter:none;}
"""

# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #
# Runs before first paint so a dark-mode user never sees a white flash. Light is
# the default for a first-time visitor, matching the Annotation Studio.
THEME_SCRIPT = """<script>
  (function(){ try{ document.documentElement.setAttribute('data-theme',
    localStorage.getItem('theme')||'light'); }catch(e){ document.documentElement.setAttribute('data-theme','light'); } })();
</script>"""

# toggleTheme() is what the button calls; updateThemeIcons() repaints every
# .theme-btn on the page. Identical to the Annotation Studio's implementation so
# the two never drift.
THEME_JS = """
const _SUN='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
const _MOON='<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
function currentTheme(){ return document.documentElement.getAttribute('data-theme')||'light'; }
function updateThemeIcons(){
  const dark=currentTheme()==='dark';
  document.querySelectorAll('.theme-btn').forEach(b=>{
    b.innerHTML = dark?_SUN:_MOON;      // moon in light (-> go dark), sun in dark (-> go light)
    b.title = dark?'switch to light mode':'switch to dark mode';
  });
}
function toggleTheme(){
  const next = currentTheme()==='light' ? 'dark' : 'light';
  document.documentElement.setAttribute('data-theme', next);
  try{ localStorage.setItem('theme', next); }catch(e){}   // remember the last choice
  updateThemeIcons();
}
updateThemeIcons();
"""

_HOME_ICON = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
              'stroke-linecap="round" stroke-linejoin="round"><path d="M3 10.5 12 3l9 7.5"/>'
              '<path d="M5 9.5V20a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V9.5"/>'
              '<path d="M9.5 21v-6h5v6"/></svg>')

THEME_BUTTON = '<button class="theme-btn" onclick="toggleTheme()" title="toggle theme"></button>'
# "/home" is served by the suite shell. Sub-apps link here rather than "/" because
# run.py rewrites their root-absolute URLs to their own mount point.
HOME_BUTTON = f'<a class="home-btn" href="/home" title="Back to the suite">{_HOME_ICON}Suite</a>'


def head(title, extra_css=""):
    """The <head> contents every suite page shares: title, pre-paint theme
    script, palette, aliases, base chrome, then the page's own CSS."""
    return (
        f'<meta charset="utf-8">\n'
        f'<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<title>{title}</title>\n'
        f'{THEME_SCRIPT}\n'
        f'<style>{TOKENS}{LEGACY_ALIASES}{BASE_CSS}{CHROME_CSS}{extra_css}</style>'
    )


def stylesheet(extra_css=""):
    """Just the CSS text (palette + aliases + chrome + page rules), for pages
    that build their own <style> block."""
    return f"{TOKENS}{LEGACY_ALIASES}{BASE_CSS}{CHROME_CSS}{extra_css}"
