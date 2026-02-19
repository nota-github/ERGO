"""
UI components, formatting, and layout for ERGO Demo.
"""

import os
import time
import base64
from queue import Queue, Empty
from threading import Thread
from PIL import Image
from transformers.image_utils import load_image

from .config import CUR_DIR
from .inference import StreamingState, ergo_inference
from .utils import load_examples

# ============== Assets ==============
def get_base64_image(path):
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
    except Exception:
        pass
    return ""

LOGO_WHITE_PATH = "/workspace/ERGO/data/demo/NotaAI_Logo_White_RGB.png"
LOGO_NAVY_PATH = "/workspace/ERGO/data/demo/NotaAI_Logo_Basic_Navy_RGB.png"
ICLR_LOGO_PATH = "/workspace/ERGO/data/demo/ICLR-logo.svg"

LOGO_WHITE_B64 = get_base64_image(LOGO_WHITE_PATH)
LOGO_NAVY_B64 = get_base64_image(LOGO_NAVY_PATH)

def get_svg_content(path):
    """Read SVG file content."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
    except Exception:
        pass
    return ""

ICLR_LOGO_SVG = get_svg_content(ICLR_LOGO_PATH)

# ============== Example Data ==============
EXAMPLES = load_examples(example_dir="/workspace/ERGO/Vstar_Bench")

# ============== Custom CSS ==============
CUSTOM_CSS = """
/* Theme variables */
:root {
    --bg-color: #0a0a0f;
    --container-bg: rgba(15, 23, 42, 0.6);
    --border-color: rgba(51, 65, 85, 0.5);
    --text-primary: #e2e8f0;
    --text-secondary: #64748b;
    --accent-color: #06b6d4;
    --accent-green: #10b981;
    --header-bg: linear-gradient(180deg, rgba(6, 182, 212, 0.08) 0%, transparent 100%);
    --header-border: 1px solid rgba(6, 182, 212, 0.15);
    --section-header-bg: rgba(6, 182, 212, 0.08);
    --vision-header-bg: rgba(16, 185, 129, 0.08);
    --link-bg: rgba(15, 23, 42, 0.3);
    --link-border: rgba(148, 163, 184, 0.2);
}

/* Light mode overrides - applied on body */
body.light-mode {
    --bg-color: #f8fafc;
    --container-bg: rgba(255, 255, 255, 0.9);
    --border-color: rgba(203, 213, 225, 0.8);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --accent-color: #0891b2;
    --accent-green: #059669;
    --header-bg: linear-gradient(180deg, rgba(6, 182, 212, 0.12) 0%, transparent 100%);
    --header-border: 1px solid rgba(6, 182, 212, 0.2);
    --section-header-bg: rgba(6, 182, 212, 0.08);
    --vision-header-bg: rgba(16, 185, 129, 0.08);
    --link-bg: rgba(255, 255, 255, 0.8);
    --link-border: rgba(148, 163, 184, 0.4);
}

/* ===== Force ALL Gradio elements to follow light mode ===== */

/* Page background */
body.light-mode,
body.light-mode .gradio-container,
body.light-mode .main,
body.light-mode .app {
    background: #f8fafc !important;
    color: #0f172a !important;
}

/* All blocks / panels / groups */
body.light-mode .block,
body.light-mode .form,
body.light-mode .panel,
body.light-mode .group {
    background: transparent !important;
    border-color: rgba(203, 213, 225, 0.8) !important;
}

/* Section containers (Select Example, Model Output, Vision Token) */
body.light-mode .section-container {
    background: rgba(255, 255, 255, 0.95) !important;
    border-color: rgba(203, 213, 225, 0.8) !important;
}

body.light-mode .section-header {
    background: rgba(6, 182, 212, 0.06) !important;
    border-color: rgba(203, 213, 225, 0.8) !important;
}

body.light-mode .section-header h3 {
    color: #0f172a !important;
}

body.light-mode .vision-section .section-header,
body.light-mode .vision-header {
    background: rgba(16, 185, 129, 0.06) !important;
}

/* Inputs, dropdowns, textareas */
body.light-mode input,
body.light-mode textarea,
body.light-mode select,
body.light-mode .wrap,
body.light-mode .secondary-wrap,
body.light-mode .dropdown-container {
    background: #ffffff !important;
    color: #0f172a !important;
    border-color: rgba(203, 213, 225, 0.8) !important;
}

/* All text elements */
body.light-mode label,
body.light-mode .label-wrap,
body.light-mode span,
body.light-mode p,
body.light-mode h1,
body.light-mode h2,
body.light-mode h3,
body.light-mode h4,
body.light-mode h5,
body.light-mode div {
    color: #0f172a;
}

/* Keep ERGO title gradient blue */
body.light-mode .header-container h1 span {
    background: linear-gradient(135deg, #06b6d4, #0891b2, #0e7490);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Keep subtitle secondary color */
body.light-mode .header-container .subtitle {
    color: #475569 !important;
}

body.light-mode .header-container .subtitle span {
    color: #0891b2 !important;
}

/* Question display */
body.light-mode .question-display {
    background: rgba(6, 182, 212, 0.04) !important;
}

body.light-mode .question-display p {
    color: #0f172a !important;
}

/* Chatbot messages */
body.light-mode .message,
body.light-mode .message p,
body.light-mode .message span,
body.light-mode .message div {
    color: #0f172a !important;
}

body.light-mode .bot,
body.light-mode .bot .message {
    background: rgba(6, 182, 212, 0.05) !important;
}

body.light-mode .user,
body.light-mode .user .message {
    background: rgba(99, 102, 241, 0.08) !important;
}

/* Info section */
body.light-mode .info-section {
    background: rgba(255, 255, 255, 0.9) !important;
    border-color: rgba(203, 213, 225, 0.8) !important;
}

body.light-mode .info-section h4 {
    color: #475569 !important;
}

body.light-mode .info-section p {
    color: #475569 !important;
}

/* Image component */
body.light-mode .image-container,
body.light-mode .image-frame {
    background: #ffffff !important;
    border-color: rgba(203, 213, 225, 0.8) !important;
}

/* ===== Hide ALL Gradio share / download / copy action buttons globally ===== */
/* Attribute-based selectors (aria-label, title, data-testid) */
button[aria-label*="download" i],
button[aria-label*="share" i],
button[title*="download" i],
button[title*="share" i],
[data-testid*="share" i],
[data-testid*="download" i],
/* Class-based selectors */
.share-button,
.download-button,
.copy-button,
/* Gradio icon-button toolbar rows that contain share/download */
.icon-button-wrapper,
.icon-buttons,
/* Image component specific */
[data-testid="image"] button[aria-label*="fullscreen" i],
[data-testid="image"] button[aria-label*="full screen" i],
[data-testid="image"] button[aria-label*="open" i],
[data-testid="image"] button[title*="fullscreen" i],
.image-container button[aria-label*="fullscreen" i],
.image-frame button[aria-label*="fullscreen" i],
/* Chatbot specific copy */
.chatbot-container button[aria-label*="copy" i],
.chatbot-container button[title*="copy" i],
.chatbot-container [data-testid*="copy" i],
.prose .copy-button {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}

/* Image modal/lightbox styles */
.image-modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.9);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10000;
    cursor: zoom-out;
    animation: fadeIn 0.2s ease;
}

.image-modal-content {
    max-width: 95vw;
    max-height: 95vh;
    object-fit: contain;
    border-radius: 8px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
}

.image-modal-close {
    position: absolute;
    top: 20px;
    right: 20px;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    cursor: pointer;
    font-size: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}

.image-modal-close:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: scale(1.1);
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Custom image enlargement button */
.image-enlarge-btn {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(6, 182, 212, 0.9) !important;
    border: none !important;
    color: white !important;
    width: 32px;
    height: 32px;
    border-radius: 6px !important;
    cursor: pointer !important;
    font-size: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 10;
    opacity: 0;
    transition: opacity 0.2s ease, transform 0.2s ease !important;
}

.image-container:hover .image-enlarge-btn,
[data-testid="image"]:hover .image-enlarge-btn {
    opacity: 1;
}

.image-enlarge-btn:hover {
    transform: scale(1.1);
    background: rgba(6, 182, 212, 1) !important;
}

/* Links */
body.light-mode .header-link {
    color: #475569 !important;
    background: rgba(255, 255, 255, 0.8) !important;
    border-color: rgba(148, 163, 184, 0.4) !important;
}

/* Theme toggle in light mode */
body.light-mode .theme-toggle-btn {
    background: rgba(255, 255, 255, 0.8) !important;
    border-color: rgba(148, 163, 184, 0.4) !important;
    color: #475569 !important;
}

/* Markdown text */
body.light-mode .prose,
body.light-mode .prose * {
    color: #0f172a !important;
}

/* Keep accent colors for specific elements */
body.light-mode .stat-value-accent,
body.light-mode [style*="color: #06b6d4"] {
    color: #0891b2 !important;
}

body.light-mode .stat-value-green,
body.light-mode [style*="color: #10b981"] {
    color: #059669 !important;
}

/* Base styling using variables */
.gradio-container {
    background: var(--bg-color) !important;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    transition: background 0.3s ease;
}

/* Main header styling */
.header-container {
    padding: 1.5rem 2rem;
    background: var(--header-bg);
    border-bottom: var(--header-border);
    margin-bottom: 2rem;
    transition: all 0.3s ease;
}

.header-content {
    display: grid;
    grid-template-columns: 180px minmax(0, 1fr) 180px;
    align-items: center;
    max-width: 1600px;
    margin: 0 auto;
    gap: 1.5rem;
}

.logo-container {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 0.5rem;
    grid-column: 1;
}

.logo-img {
    height: 48px;
    max-height: 48px;
    min-height: 48px;
    width: 140px;
    max-width: 140px;
    object-fit: contain;
    object-position: left center;
}

.iclr-logo {
    height: 44px;
    width: 180px;
    display: flex;
    align-items: center;
    justify-content: flex-start;
}

.iclr-logo svg {
    height: 100%;
    width: auto;
    filter: none;
}

/* Night mode color-map tuning for ICLR logo while keeping no box. */
body:not(.light-mode) .iclr-logo svg {
    filter: hue-rotate(170deg) saturate(1.5) brightness(1.22) contrast(1.08);
}

.title-container {
    grid-column: 2;
    text-align: center;
    min-width: 0;
}

.links-container {
    display: flex;
    justify-content: center;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
    margin-top: 0.85rem;
}

.header-link {
    color: var(--text-secondary) !important;
    text-decoration: none !important;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.2s;
    border: 1px solid var(--link-border);
    padding: 0.5rem 1rem;
    border-radius: 8px;
    background: var(--link-bg);
}

.header-link:hover {
    color: var(--accent-color) !important;
    border-color: var(--accent-color);
    background: rgba(6, 182, 212, 0.1);
    transform: translateY(-1px);
}

/* Theme toggle button */
.theme-toggle {
    background: transparent;
    border: 1px solid var(--link-border);
    color: var(--text-secondary);
    padding: 0.5rem;
    border-radius: 8px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
}

.theme-toggle:hover {
    color: var(--accent-color);
    border-color: var(--accent-color);
    background: rgba(6, 182, 212, 0.1);
}

.header-container h1 {
    font-size: 2.5rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0 0 0.25rem 0;
}

.header-container h1 span {
    background: linear-gradient(135deg, #06b6d4, #0891b2, #0e7490);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.header-container .subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 400;
    margin: 0;
}

.header-container .subtitle span {
    color: var(--accent-color);
    font-weight: 500;
}

/* Unified section container */
.section-container {
    background: var(--container-bg) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    padding: 0 !important;
    margin-bottom: 1rem !important;
    overflow: hidden;
    transition: all 0.3s ease;
}

.section-header {
    padding: 1rem 1.25rem;
    background: var(--section-header-bg);
    border-bottom: 1px solid var(--border-color);
}

.section-header h3 {
    color: var(--text-primary);
    font-size: 1.12rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0;
}

/* Vision section green accent */
.vision-section .section-header,
.vision-header {
    background: var(--vision-header-bg) !important;
}

.vision-section .section-header h3,
.vision-header h3 {
    color: var(--text-primary) !important;
}

/* Content inside sections */
.section-container > div {
    padding: 1rem 1.25rem;
}

.section-container > div:first-child {
    padding: 0;
}

/* Question display */
.question-display {
    background: rgba(6, 182, 212, 0.05);
    border-left: 3px solid var(--accent-color);
    padding: 1rem 1.25rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
}

.question-display p {
    color: var(--text-primary);
    font-size: 1rem;
    line-height: 1.6;
    margin: 0;
    white-space: pre-wrap;
}

/* Run button */
.run-btn {
    background: linear-gradient(135deg, #06b6d4, #0891b2) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.875rem 2.5rem !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(6, 182, 212, 0.25) !important;
    margin-top: 0.5rem !important;
}

.run-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.35) !important;
}

/* Info section */
.info-section {
    padding: 1.5rem;
    background: var(--container-bg);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    margin-top: 2rem;
}

.info-section h4 {
    color: var(--text-secondary);
    font-size: 0.8rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 1rem 0;
}

.info-section p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.7;
    margin: 0;
}

/* Chatbot styling - auto-expand without scroll */
.chatbot-container {
    border: none !important;
    background: transparent !important;
    max-height: none !important;
    overflow: visible !important;
}

.chatbot-container > div {
    max-height: none !important;
    overflow: visible !important;
}

/* Vision stats text colors */
.stat-label { color: var(--text-secondary); }
.stat-value { color: var(--text-primary); }
.stat-value-accent { color: var(--accent-color); }
.stat-value-green { color: var(--accent-green); }

/* Theme toggle button - positioned inside header area */
.theme-toggle-btn {
    position: absolute !important;
    top: 1.5rem !important;
    right: 2rem !important;
    z-index: 100 !important;
    background: var(--link-bg) !important;
    border: 1px solid var(--link-border) !important;
    color: var(--text-secondary) !important;
    padding: 0.4rem 0.9rem !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    min-width: auto !important;
    width: auto !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.theme-toggle-btn:hover {
    border-color: var(--accent-color) !important;
    color: var(--accent-color) !important;
    background: rgba(6, 182, 212, 0.1) !important;
}

/* Header wrapper for positioning toggle button */
.header-wrapper {
    position: relative !important;
    padding: 0 !important;
    gap: 0 !important;
}

.header-wrapper > div:last-child {
    position: absolute;
    top: 1.5rem;
    right: 2rem;
    z-index: 100;
}
"""

# ============== HTML Components ==============
HEADER_HTML = f"""
<div class="header-container">
    <div class="header-content">
        <div class="logo-container">
            <img id="theme-logo" src="data:image/png;base64,{LOGO_WHITE_B64}" alt="Nota AI" class="logo-img" />
            <div class="iclr-logo">{ICLR_LOGO_SVG}</div>
        </div>
        <div class="title-container">
            <h1><span>ERGO</span>: Efficient High-Resolution Visual Understanding for Vision-Language Models</h1>
            <p class="subtitle">Two-Stage <span>Reasoning-Driven Perception</span> for Vision-Language Models</p>
            <div class="links-container">
                <a href="https://github.com/nota-ai/ERGO" target="_blank" class="header-link">GitHub</a>
                <a href="https://arxiv.org/abs/2509.21991" target="_blank" class="header-link">arXiv Paper</a>
                <a href="https://arxiv.org/abs/2509.21991" target="_blank" class="header-link">Project Page</a>
            </div>
        </div>
    </div>
</div>
"""

# JavaScript for theme toggle (executed via Gradio's js parameter)
# Directly applies inline styles to bypass CSS specificity issues
THEME_TOGGLE_JS = f"""
() => {{
    const root = document.documentElement;
    const body = document.body;
    const isCurrentlyLight = root.getAttribute('data-theme') === 'light';
    const isLight = !isCurrentlyLight;

    root.setAttribute('data-theme', isLight ? 'light' : 'dark');
    body.classList.toggle('light-mode', isLight);

    const vars = isLight
        ? {{
            '--bg-color': '#f1f5f9',
            '--container-bg': '#ffffff',
            '--border-color': 'rgba(203, 213, 225, 0.9)',
            '--text-primary': '#0f172a',
            '--text-secondary': '#475569',
            '--accent-color': '#0891b2',
            '--accent-green': '#059669',
            '--header-bg': 'linear-gradient(180deg, rgba(6, 182, 212, 0.10) 0%, transparent 100%)',
            '--header-border': '1px solid rgba(6, 182, 212, 0.2)',
            '--section-header-bg': 'rgba(6, 182, 212, 0.06)',
            '--vision-header-bg': 'rgba(16, 185, 129, 0.06)',
            '--link-bg': 'rgba(255, 255, 255, 0.85)',
            '--link-border': 'rgba(148, 163, 184, 0.45)',
            // Gradio internal vars
            '--body-background-fill': '#f1f5f9',
            '--background-fill-primary': '#f1f5f9',
            '--background-fill-secondary': '#ffffff',
            '--block-background-fill': '#ffffff',
            '--block-border-color': 'rgba(203, 213, 225, 0.9)',
            '--panel-background-fill': '#ffffff',
            '--input-background-fill': '#ffffff',
            '--input-border-color': 'rgba(203, 213, 225, 0.9)',
            '--input-text-color': '#0f172a',
            '--body-text-color': '#0f172a',
            '--body-text-color-subdued': '#475569',
        }}
        : {{
            '--bg-color': '#0a0a0f',
            '--container-bg': 'rgba(15, 23, 42, 0.6)',
            '--border-color': 'rgba(51, 65, 85, 0.5)',
            '--text-primary': '#e2e8f0',
            '--text-secondary': '#64748b',
            '--accent-color': '#06b6d4',
            '--accent-green': '#10b981',
            '--header-bg': 'linear-gradient(180deg, rgba(6, 182, 212, 0.08) 0%, transparent 100%)',
            '--header-border': '1px solid rgba(6, 182, 212, 0.15)',
            '--section-header-bg': 'rgba(6, 182, 212, 0.08)',
            '--vision-header-bg': 'rgba(16, 185, 129, 0.08)',
            '--link-bg': 'rgba(15, 23, 42, 0.3)',
            '--link-border': 'rgba(148, 163, 184, 0.2)',
            // Gradio internal vars
            '--body-background-fill': '#0a0a0f',
            '--background-fill-primary': '#0a0a0f',
            '--background-fill-secondary': '#111827',
            '--block-background-fill': '#111827',
            '--block-border-color': 'rgba(51, 65, 85, 0.5)',
            '--panel-background-fill': '#111827',
            '--input-background-fill': '#111827',
            '--input-border-color': 'rgba(51, 65, 85, 0.5)',
            '--input-text-color': '#e2e8f0',
            '--body-text-color': '#e2e8f0',
            '--body-text-color-subdued': '#64748b',
        }};

    Object.entries(vars).forEach(([k, v]) => {{
        root.style.setProperty(k, v);
        body.style.setProperty(k, v);
    }});

    const bg = vars['--bg-color'];
    const containerBg = vars['--container-bg'];
    const borderCol = vars['--border-color'];
    const textCol = vars['--text-primary'];
    const textSecCol = vars['--text-secondary'];

    const gc = document.querySelector('.gradio-container');
    if (gc) gc.style.setProperty('background', bg, 'important');

    document.querySelectorAll('.section-container, .info-section').forEach(el => {{
        el.style.setProperty('background', containerBg, 'important');
        el.style.setProperty('border-color', borderCol, 'important');
        el.style.setProperty('color', textCol, 'important');
    }});

    document.querySelectorAll('.section-container *').forEach(el => {{
        if (el.classList.contains('run-btn')) return;
        const tag = (el.tagName || '').toLowerCase();
        if (['p', 'span', 'label', 'h1', 'h2', 'h3', 'h4', 'h5', 'div'].includes(tag)) {{
            el.style.setProperty('color', textCol, 'important');
        }}
    }});

    document.querySelectorAll('input, textarea, select, .wrap, .secondary-wrap, .dropdown-container').forEach(el => {{
        el.style.setProperty('background', isLight ? '#ffffff' : 'rgba(15, 23, 42, 0.8)', 'important');
        el.style.setProperty('color', textCol, 'important');
        el.style.setProperty('border-color', borderCol, 'important');
    }});

    document.querySelectorAll('.header-link, .theme-toggle-btn').forEach(el => {{
        el.style.setProperty('color', textSecCol, 'important');
        el.style.setProperty('background', vars['--link-bg'], 'important');
        el.style.setProperty('border-color', vars['--link-border'], 'important');
    }});

    const logo = document.getElementById('theme-logo');
    if (logo) {{
        logo.src = isLight
            ? "data:image/png;base64,{LOGO_NAVY_B64}"
            : "data:image/png;base64,{LOGO_WHITE_B64}";
    }}

    document.querySelectorAll('.theme-toggle-btn').forEach(el => {{
        el.textContent = isLight ? '🌙 Night' : '☀️ Day';
    }});
}}
"""

INFO_HTML = """
<div class="info-section">
    <h4>About ERGO</h4>
    <p>
        ERGO employs a two-stage zoom reasoning approach: first identifying the most relevant region 
        in the image, then analyzing the zoomed view for improved accuracy on fine-grained visual tasks.
    </p>
</div>
"""

# JavaScript for image enlargement functionality
IMAGE_ENLARGE_JS = """
() => {
    const MODAL_ID = "ergo-image-modal";
    const BTN_CLASS = "ergo-fullscreen-btn";

    function removeModal() {
        const existing = document.getElementById(MODAL_ID);
        if (existing) existing.remove();
    }

    function createModal(imgSrc) {
        if (!imgSrc) return;
        removeModal();

        const overlay = document.createElement("div");
        overlay.id = MODAL_ID;
        overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.95);display:flex;align-items:center;justify-content:center;z-index:10000;cursor:zoom-out;";
        overlay.addEventListener("click", removeModal);

        const img = document.createElement("img");
        img.src = imgSrc;
        img.style.cssText = "max-width:95vw;max-height:95vh;object-fit:contain;border-radius:8px;box-shadow:0 20px 60px rgba(0,0,0,0.5);";
        img.addEventListener("click", (e) => e.stopPropagation());
        overlay.appendChild(img);

        const closeBtn = document.createElement("button");
        closeBtn.type = "button";
        closeBtn.textContent = "×";
        closeBtn.style.cssText = "position:absolute;top:20px;right:20px;background:rgba(255,255,255,0.12);border:1px solid rgba(255,255,255,0.35);color:#fff;width:40px;height:40px;border-radius:999px;cursor:pointer;font-size:24px;line-height:1;";
        closeBtn.addEventListener("click", (e) => {
            e.stopPropagation();
            removeModal();
        });
        overlay.appendChild(closeBtn);
        document.body.appendChild(overlay);
    }

    function bindImage(img) {
        if (!img || img.dataset.ergoFullscreenBound === "1") return;
        if (!img.src || img.src.startsWith("data:image/svg")) return;

        const container =
            img.closest("[data-testid='image']") ||
            img.closest(".image-container") ||
            img.closest(".image-frame") ||
            img.parentElement;
        if (!container) return;

        img.dataset.ergoFullscreenBound = "1";
        img.style.cursor = "zoom-in";
        img.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            createModal(img.src);
        });

        if (container.querySelector("." + BTN_CLASS)) return;
        container.style.position = "relative";

        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = BTN_CLASS;
        btn.textContent = "⛶";
        btn.title = "View fullscreen";
        btn.style.cssText = "position:absolute;top:16px;right:8px;width:34px;height:34px;border:none;border-radius:8px;background:rgba(6,182,212,0.92);color:#fff;font-size:18px;line-height:1;display:flex;align-items:center;justify-content:center;cursor:pointer;z-index:50;opacity:0.92;";
        btn.addEventListener("click", (e) => {
            e.preventDefault();
            e.stopPropagation();
            createModal(img.src);
        });
        container.appendChild(btn);
    }

    function setupImageEnlargement() {
        document
            .querySelectorAll("[data-testid='image'] img, .image-container img, .image-frame img")
            .forEach(bindImage);
    }

    setupImageEnlargement();
    setTimeout(setupImageEnlargement, 300);
    setTimeout(setupImageEnlargement, 1200);

    const observer = new MutationObserver(() => setupImageEnlargement());
    observer.observe(document.body, { childList: true, subtree: true });

    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") removeModal();
    });
}
"""


# ============== Formatting Functions ==============
def build_metrics_header(
    total_time: float,
    is_done: bool,
    ttft: float = None,
    tpot: float = None,
    tokens: int = 0,
) -> str:
    """Build a formatted metrics header."""
    status = '✓ Complete' if is_done else 'Processing...'
    header = f"**{status}** — {total_time:.1f}s\n\n"
    return header


def format_vision_stats(stats: dict) -> str:
    """Format vision token stats as concise HTML."""
    if not stats:
        return ""
    
    reduction = 100.0 - stats["reduction_pct"]
    
    return f"""
<div style="display: flex; gap: 1.5rem; align-items: center; padding: 1rem; flex-wrap: wrap;">
    <div style="text-align: center;">
        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase;">Original</div>
        <div style="color: #e2e8f0; font-size: 1.5rem; font-weight: 600;">{stats['original']:.0f} <span style="font-size: 0.8rem; color: #64748b; font-weight: 400;">vision tokens</span></div>
    </div>
    <div style="color: #64748b; font-size: 1.5rem;">→</div>
    <div style="text-align: center;">
        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase;">Stage 1</div>
        <div style="color: #06b6d4; font-size: 1.5rem; font-weight: 600;">{stats['stage1']:.0f} <span style="font-size: 0.8rem; color: #64748b; font-weight: 400;">vision tokens</span></div>
    </div>
    <div style="color: #64748b; font-size: 1rem;">+</div>
    <div style="text-align: center;">
        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase;">Stage 2</div>
        <div style="color: #06b6d4; font-size: 1.5rem; font-weight: 600;">{stats['stage2']:.0f} <span style="font-size: 0.8rem; color: #64748b; font-weight: 400;">vision tokens</span></div>
    </div>
    <div style="color: #64748b; font-size: 1rem;">=</div>
    <div style="text-align: center;">
        <div style="color: #64748b; font-size: 0.75rem; text-transform: uppercase;">ERGO Total</div>
        <div style="color: #10b981; font-size: 1.5rem; font-weight: 600;">{stats['total']:.0f} <span style="font-size: 0.8rem; color: #64748b; font-weight: 400;">vision tokens</span></div>
    </div>
    <div style="margin-left: auto; text-align: center; background: rgba(16, 185, 129, 0.1); padding: 0.5rem 1rem; border-radius: 8px;">
        <div style="color: #10b981; font-size: 1.75rem; font-weight: 700;">{reduction:.0f}%</div>
        <div style="color: #10b981; font-size: 0.7rem; text-transform: uppercase;">Saved</div>
    </div>
</div>
"""


def format_for_chatbot(output_list):
    """Convert output list to Gradio chatbot messages format."""
    if not output_list:
        return []
    
    messages = []
    current_text = ""
    
    for item in output_list:
        if isinstance(item, str):
            current_text += item
        elif isinstance(item, dict) and "files" in item:
            if current_text:
                messages.append({"role": "assistant", "content": current_text})
                current_text = ""
            for f in item.get("files", []):
                messages.append({"role": "assistant", "content": {"path": f}})
        elif isinstance(item, dict) and "role" in item:
            if current_text:
                messages.append({"role": "assistant", "content": current_text})
                current_text = ""
            messages.append(item)
    
    if current_text:
        messages.append({"role": "assistant", "content": current_text})
    
    return messages


# ============== Runner Functions ==============
def run_ergo_example(example_idx: int):
    """Run ERGO model on selected example with streaming."""
    if example_idx is None or example_idx < 0 or example_idx >= len(EXAMPLES):
        yield [], ""
        return
    
    example = EXAMPLES[example_idx]
    image_path = example["image"]
    question = example["question"]
    
    # Load and prepare image
    img = load_image(image_path)
    if hasattr(img, 'convert'):
        img = img.convert("RGB")
    
    # Save input image
    input_img_path = os.path.join(CUR_DIR, "tmp_input_ergo.png")
    img.save(input_img_path)
    
    # Create queue and state
    ergo_queue = Queue()
    ergo_state = StreamingState(ergo_queue)
    
    # Start inference thread
    ergo_thread = Thread(target=ergo_inference, args=(img.copy(), question, ergo_state))
    ergo_thread.start()
    
    # Stream updates
    while not ergo_state.done:
        try:
            while True:
                ergo_queue.get_nowait()
        except Empty:
            pass
        
        ergo_output = list(ergo_state.output)
        ergo_time = (ergo_state.end_time or time.time()) - ergo_state.start_time if ergo_state.start_time else 0
        ergo_ttft = ergo_state.get_ttft()
        ergo_tpot = ergo_state.get_tpot() if ergo_state.done else None
        ergo_tokens = ergo_state.token_count
        
        header = build_metrics_header(ergo_time, ergo_state.done, ergo_ttft, ergo_tpot, ergo_tokens)
        vision_html = format_vision_stats(ergo_state.vision_stats) if ergo_state.vision_stats else ""
        yield format_for_chatbot([header] + ergo_output), vision_html
        time.sleep(0.05)
    
    ergo_thread.join()
    
    # Final output
    ergo_output = list(ergo_state.output)
    ergo_time = ergo_state.end_time - ergo_state.start_time if ergo_state.start_time else 0
    ergo_ttft = ergo_state.get_ttft()
    ergo_tpot = ergo_state.get_tpot()
    ergo_tokens = ergo_state.token_count
    
    header = build_metrics_header(ergo_time, True, ergo_ttft, ergo_tpot, ergo_tokens)
    vision_html = format_vision_stats(ergo_state.vision_stats) if ergo_state.vision_stats else ""
    yield format_for_chatbot([header] + ergo_output), vision_html


def get_example_choices():
    """Get example choices for dropdown."""
    return [(f"{ex['title']}", ex['id']) for ex in EXAMPLES]


def load_example_display(example_idx: int):
    """Load example image and question for display."""
    if example_idx is None or example_idx < 0 or example_idx >= len(EXAMPLES):
        return None, ""
    
    example = EXAMPLES[example_idx]
    return example["image"], example["question"]
