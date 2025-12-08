import os
import time
import base64
import json
import re
import html
import uuid
import secrets
import hashlib
import hmac
import binascii
from functools import lru_cache
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager
from urllib.parse import urlencode

import requests
import streamlit as st
import yaml
import streamlit.components.v1 as components

# -------------------------------
# Config
# -------------------------------
st.set_page_config(
    page_title="Interactive Fault Tree + Proof-to-Order",
    layout="centered",
    initial_sidebar_state="collapsed",
)

ACCESS_PIN = os.getenv("ACCESS_PIN")
P2O_ENDPOINT = os.getenv("P2O_ENDPOINT")

# If True → log every step to Google Sheet
# If False → only log on finalization (Gate Token step) → much faster UI
LOG_EVERY_STEP = False

DEFAULT_LANG = "en"
SESSION_HISTORY_KEY = "fault_history"
SESSION_TIMER_PREFIX = "timer_start_"
INPUT_VALUE_PREFIX = "ctrl_value_"
INPUT_LABEL_PREFIX = "ctrl_label_"
BASE_DIR = Path(__file__).resolve().parent
CAMERA_CAPTURE_ENABLED = (
    os.getenv("ENABLE_CAMERA_CAPTURE", "true").strip().lower() == "true"
)

SPINNER_COLOR = os.getenv("SPINNER_COLOR", "#dc0d3a")

EVIDENCE_COLLECTION_ENABLED = (
    os.getenv("ENABLE_EVIDENCE_UPLOAD", "false").strip().lower() == "true"
)
RECOMMENDED_PARTS_KEY = "recommended_parts"
ISSUE_LABEL_COLOR = os.getenv("ISSUE_LABEL_COLOR", "#f00c0c")
PRODUCT_IMAGES_DIR = BASE_DIR / "product_images"
PRODUCT_CATEGORIES = [
    {"id": "TV", "label": "TV", "image": "TV.png", "available": False},
    {"id": "WM", "label": "Washing Machine", "image": "WM.png", "available": True},
    {"id": "AC", "label": "AC", "image": "AC.png", "available": False},
    {"id": "REF", "label": "REF", "image": "REF.png", "available": False},
    {"id": "MWO", "label": "MWO", "image": "MWO.png", "available": False},
    {"id": "CHIMNEY", "label": "Chimney", "image": "Chimney.png", "available": False},
]
PRODUCT_CATEGORY_LABELS = {entry["id"]: entry["label"] for entry in PRODUCT_CATEGORIES}
AVAILABLE_FLOW_CATEGORIES = {entry["id"] for entry in PRODUCT_CATEGORIES if entry["available"]}
CATEGORY_FLOW_PATTERNS = {
    "WM": ("wm", "washingmachine", "washing_machine"),
}
YAML_FALLBACK_ENCODINGS = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
ACCESS_TOKEN_PARAM = "access_token"
ACCESS_TOKEN_SECRET = os.getenv("ACCESS_TOKEN_SECRET")
ACCESS_TOKEN_TTL_SECONDS = max(int(os.getenv("ACCESS_TOKEN_TTL_SECONDS", "3600")), 60)
BACK_BUTTON_LABEL = "⬅️ Back to Previous Step"
BACK_BUTTON_CLASS = "back-step-button"
DARK_THEME_STYLE = """
<style id="dark-theme-override">
body,
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at 10% 20%, rgba(15,23,42,0.75), rgba(2,6,23,0.98) 55%) !important;
    color: #f8fafc !important;
}
[data-testid="stHeader"] {
    background: rgba(2, 6, 23, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
}
[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, 0.95) !important;
    color: #f8fafc !important;
}
[data-testid="stSidebar"] * {
    color: #f8fafc !important;
}
[data-testid="stToolbar"],
[data-testid="stBottomBlockContainer"] {
    background: transparent !important;
}
</style>
"""


@contextmanager
def jeeves_spinner(
    text: str = "🚀 Syncing your step with Jeeves Cloud...",
    color: str = "#e6d81e",
    size_px: int = 22,
    thickness_px: int = 3,
    speed_sec: float = 0.75,
) -> None:
    """
    Render a custom spinner so the tech always sees progress feedback during long calls.
    """
    html = f"""
    <div class="jeeves-spinner-wrap"
         style="--spin-color:{color};
                --spin-size:{size_px}px;
                --spin-thickness:{thickness_px}px;
                --spin-speed:{speed_sec}s;
                display:flex;align-items:center;gap:.5rem;margin-bottom:0.4rem;">
      <span class="jeeves-spinner"
            style="width:var(--spin-size);height:var(--spin-size);
                   border:var(--spin-thickness) solid rgba(255,255,255,.25);
                   border-top-color:var(--spin-color);
                   border-radius:50%;
                   display:inline-block;
                   animation:jeeves-spin var(--spin-speed) linear infinite;"></span>
      <span>{text}</span>
    </div>
    <style>
      @keyframes jeeves-spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
    """
    placeholder = st.empty()
    placeholder.markdown(html, unsafe_allow_html=True)
    try:
        yield
    finally:
        placeholder.empty()


# -----------------------------
# Helpers
# -----------------------------
def init_session_state() -> None:
    """Ensure all keys exist in Streamlit session state."""
    st.session_state.setdefault("tree", None)
    st.session_state.setdefault("node_id", None)
    st.session_state.setdefault("passed", {})
    st.session_state.setdefault("answers", {})
    st.session_state.setdefault("lang", DEFAULT_LANG)
    st.session_state.setdefault("meta", {})
    st.session_state.setdefault("case", {"case_id": "", "st_id": ""})
    st.session_state.setdefault(SESSION_HISTORY_KEY, [])
    st.session_state.setdefault("path_total_steps", 0)
    st.session_state.setdefault("flow_status", None)
    st.session_state.setdefault("parts_used", set())
    st.session_state.setdefault("part_photos", {})
    st.session_state.setdefault("final_token", None)
    st.session_state.setdefault("visited_stack", [])
    st.session_state.setdefault("second_visit_mode", False)
    st.session_state.setdefault("p2o_step_buffer", [])
    st.session_state.setdefault("pending_resolution", None)
    st.session_state.setdefault("selected_flow_path", None)
    st.session_state.setdefault("selected_product", None)
    st.session_state.setdefault("product_notice", None)
    st.session_state.setdefault("_product_selector_css_loaded", False)
    st.session_state.setdefault("access_granted", False)
    st.session_state.setdefault("access_token", None)
    st.session_state.setdefault("recommended_parts_dynamic", set())


def _access_token_secret() -> Optional[str]:
    secret = ACCESS_TOKEN_SECRET or ACCESS_PIN
    return secret


def _sign_access_payload(payload: str) -> Optional[str]:
    secret = _access_token_secret()
    if not secret:
        return None
    return hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def _encode_access_token(payload: str, signature: str) -> str:
    token_bytes = f"{payload}:{signature}".encode("utf-8")
    return base64.urlsafe_b64encode(token_bytes).decode("ascii")


def _decode_access_token(token: str) -> Optional[Tuple[str, str, str]]:
    try:
        decoded = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError, binascii.Error):
        return None
    parts = decoded.split(":")
    if len(parts) != 3:
        return None
    ts_str, nonce, signature = parts
    return ts_str, nonce, signature


def _issue_access_token() -> Optional[str]:
    if not _access_token_secret():
        return None
    payload = f"{int(time.time())}:{secrets.token_hex(8)}"
    signature = _sign_access_payload(payload)
    if not signature:
        return None
    token = _encode_access_token(payload, signature)
    st.session_state.access_token = token
    st.session_state.access_granted = True
    return token


def _validate_access_token(token: Optional[str]) -> Optional[str]:
    if not token or not _access_token_secret():
        return None
    decoded = _decode_access_token(token)
    if not decoded:
        return None
    ts_str, nonce, signature = decoded
    payload = f"{ts_str}:{nonce}"
    expected = _sign_access_payload(payload)
    if not expected or not hmac.compare_digest(signature, expected):
        return None
    try:
        ts_val = int(ts_str)
    except ValueError:
        return None
    if time.time() - ts_val > ACCESS_TOKEN_TTL_SECONDS:
        return None
    return _encode_access_token(payload, signature)


def ensure_session_access_token() -> Optional[str]:
    if not ACCESS_PIN:
        return None
    token = _validate_access_token(st.session_state.get("access_token"))
    if token:
        st.session_state.access_token = token
        return token
    return _issue_access_token()


def current_access_token() -> Optional[str]:
    if not ACCESS_PIN:
        return None
    token = _validate_access_token(st.session_state.get("access_token"))
    st.session_state.access_token = token
    return token


def restore_access_from_token_query() -> None:
    if not ACCESS_PIN:
        return
    token_vals = st.query_params.get(ACCESS_TOKEN_PARAM)
    if not token_vals:
        return
    token = token_vals if isinstance(token_vals, str) else token_vals[0]
    token = _validate_access_token(token)
    if token:
        st.session_state.access_granted = True
        st.session_state.access_token = token
        st.query_params[ACCESS_TOKEN_PARAM] = token


def persist_access_token_query_param() -> None:
    token = current_access_token()
    if token:
        st.query_params[ACCESS_TOKEN_PARAM] = token


def clear_query_params_preserving_access_token() -> None:
    token = current_access_token()
    st.query_params.clear()
    if token:
        st.query_params[ACCESS_TOKEN_PARAM] = token


def normalize_tree(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Support both 'steps' and 'nodes' schema; ensure a start id exists."""
    if "nodes" not in tree and "steps" in tree:
        tree["nodes"] = {step["id"]: step for step in tree["steps"]}
    nodes = tree.get("nodes") or {}
    if isinstance(nodes, list):
        tree["nodes"] = {step["id"]: step for step in nodes}

    if "start" not in tree:
        if "steps" in tree and tree["steps"]:
            tree["start"] = tree["steps"][0]["id"]
        elif tree["nodes"]:
            tree["start"] = next(iter(tree["nodes"]))
        else:
            raise ValueError("YAML must define either steps or nodes.")
    return tree


def count_progress_steps(nodes: Dict[str, Any], second_visit: bool = False) -> int:
    if not nodes:
        return 1
    # Count all actionable nodes (anything that isn't a terminal end state)
    actionable = [
        nid
        for nid, node in nodes.items()
        if node.get("type") not in {"end"}
    ]
    return max(len(actionable), 1)


def reset_tree_progress(tree: Dict[str, Any]) -> None:
    """Reset per-run state so a technician can restart without reloading."""
    start_id = tree.get("start")
    st.session_state.node_id = start_id
    st.session_state.passed = {}
    st.session_state.answers = {}
    st.session_state[SESSION_HISTORY_KEY] = []
    st.session_state.path_total_steps = count_progress_steps(
        tree.get("nodes") or {}, st.session_state.second_visit_mode
    )
    st.session_state.flow_status = None
    st.session_state.parts_used = set()
    st.session_state.part_photos = {}
    st.session_state.final_token = None
    st.session_state.visited_stack = [start_id] if start_id else []
    st.session_state.second_visit_mode = False
    st.session_state.pending_resolution = None
    st.session_state.recommended_parts_dynamic = set()
    keys_to_clear = [
        key
        for key in list(st.session_state.keys())
        if str(key).startswith(SESSION_TIMER_PREFIX)
        or str(key).startswith(INPUT_VALUE_PREFIX)
        or str(key).startswith(INPUT_LABEL_PREFIX)
        or str(key).startswith("confirm_")
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def fault_code_from_meta(meta_id: Optional[str]) -> str:
    """Derive a compact fault code from meta.id; fallback to meta.id itself."""
    if not meta_id:
        return "GEN"
    parts = meta_id.split("_")
    if len(parts) >= 4:
        code = "_".join(parts[2:-1])
        return code or meta_id
    return meta_id


def b64_of_uploaded(file) -> Tuple[Optional[str], Optional[str]]:
    if file is None:
        return None, None
    content = file.getvalue()
    mime = getattr(file, "type", None) or "image/jpeg"
    return base64.b64encode(content).decode("ascii"), mime


def post_step_log(endpoint: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    if not endpoint:
        return {"ok": True, "skipped": True}
    try:
        response = requests.post(endpoint, json=payload, timeout=60)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            return {
                "ok": False,
                "error": "Invalid JSON response",
                "status_code": response.status_code,
                "text": response.text,
            }
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}
    except ValueError:
        return {"ok": False, "error": "Invalid JSON response from endpoint."}


def eval_branches(
    node: Dict[str, Any], selection: Optional[str], selected_list: List[str]
) -> Optional[str]:
    """Evaluate simple expressions used in YAML branches."""
    for branch in node.get("branches") or []:
        expr = (branch.get("when") or "").strip()
        if not expr:
            return branch.get("next")
        tokens = [token.strip() for token in re.split(r"\s+or\s+", expr)]
        for token in tokens:
            if token.startswith("selection =="):
                _, _, raw_val = token.partition("==")
                raw_val = raw_val.strip().strip("'\"")
                if selection == raw_val:
                    return branch.get("next")
            elif token.startswith("selected_contains(") and token.endswith(")"):
                expected = token[len("selected_contains(") : -1].strip().strip("'\"")
                if expected in selected_list:
                    return branch.get("next")
    return None


def validate_node(
    node: Dict[str, Any],
    value: Any,
    elapsed_sec: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str]:
    """Return (True/False, message)."""
    extra = extra or {}
    validation = node.get("validation") or {}
    ui = node.get("ui") or {}
    ctrl = ui.get("control")

    if validation.get("confirm_required"):
        if ctrl == "confirm":
            if value not in {"Done", "Done as instructed", True}:
                return False, "Please confirm to proceed."
        else:
            if not extra.get("confirm_ack"):
                return False, "Please confirm to proceed."

    if validation.get("one_of"):
        allowed = set(validation["one_of"])
        if isinstance(value, list):
            if not (set(value) & allowed):
                return False, "Select a valid option."
        elif value not in allowed:
            return False, "Select a valid option."

    if "min_selected" in validation:
        min_required = int(validation.get("min_selected", 1))
        if not isinstance(value, list) or len(value) < min_required:
            return False, f"Select at least {min_required} item(s)."

    if ctrl == "numeric":
        try:
            measured = float(value)
        except (TypeError, ValueError):
            return False, "Enter a numeric value."
        if "min" in validation and measured < validation["min"]:
            return False, f"Minimum is {validation['min']}."
        if "max" in validation and measured > validation["max"]:
            return False, f"Maximum is {validation['max']}."
        if "within_percent_of_label" in validation and "label_value" in extra:
            label = float(extra["label_value"])
            pct = float(validation["within_percent_of_label"])
            allowed_delta = label * (pct / 100.0)
            if not (label - allowed_delta <= measured <= label + allowed_delta):
                return (
                    False,
                    f"Measured {measured} outside +/-{pct}% of label value {label}.",
                )

    if "min_seconds" in validation and elapsed_sec is not None:
        if elapsed_sec < int(validation["min_seconds"]):
            return False, f"Please complete at least {validation['min_seconds']} seconds."

    required_selection = validation.get("requires_selection")
    if required_selection:
        if isinstance(value, list):
            if not any(choice in required_selection for choice in value):
                return False, "Please select the required outcome."
        elif value not in required_selection:
            return False, "Please select the required outcome."

    return True, ""


def format_answer_value(
    ctrl: Optional[str],
    raw_value: Any,
    elapsed_sec: Optional[int] = None,
    ui: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Convert raw widget values into human-friendly strings for logging/ValueJSON.
    Keeps lists/numerics intact; normalizes booleans and timers to readable text.
    """
    ui = ui or {}
    if ctrl == "timer":
        if elapsed_sec is not None:
            return f"{elapsed_sec}s elapsed"
        target = ui.get("seconds")
        return f"Timer in progress ({target}s target)" if target else "Timer in progress"
    if isinstance(raw_value, bool):
        return "Done" if raw_value else "Not done"
    return raw_value


def get_prompt(node: Dict[str, Any], lang: str) -> str:
    prompt = node.get("prompt") or {}
    if isinstance(prompt, dict):
        return prompt.get(lang) or prompt.get("en") or next(iter(prompt.values()), "")
    return str(prompt)


def require_evidence(node: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    evidence = node.get("evidence") or {}
    if node.get("require_evidence") is False:
        return False, "photo", evidence
    return bool(evidence.get("required")), evidence.get("capture", "photo"), evidence


def step_label(node: Dict[str, Any], lang: str) -> str:
    text = get_prompt(node, lang)
    return (text[:140] + "...") if len(text) > 140 else text


def reset_timer_for(node_id: str) -> None:
    st.session_state[f"{SESSION_TIMER_PREFIX}{node_id}"] = None
    st.session_state.pop(f"timer_refresh_{node_id}", None)


def timer_elapsed_for(node_id: str) -> int:
    start_ts = st.session_state.get(f"{SESSION_TIMER_PREFIX}{node_id}")
    if not start_ts:
        return 0
    return int(time.time() - start_ts)


def language_options(meta: Dict[str, Any]) -> List[str]:
    langs = meta.get("language") or [DEFAULT_LANG]
    if isinstance(langs, str):
        langs = [langs]
    return langs


def log_local_step(entry: Dict[str, Any]) -> None:
    history = st.session_state[SESSION_HISTORY_KEY]
    history.append(entry)
    # keep the list from growing indefinitely
    if len(history) > 500:
        del history[0]


def update_recommended_parts(node: Optional[Dict[str, Any]]) -> None:
    """Check the current node for part recommendations and add them to session state."""
    if not node:
        return

    parts_set = st.session_state.get("recommended_parts_dynamic", set())

    # Handle single part recommendation
    single_part = node.get("recommends_part")
    if single_part and isinstance(single_part, str):
        parts_set.add(single_part)

    # Handle multiple parts recommendation
    multiple_parts = node.get("recommends_parts")
    if multiple_parts and isinstance(multiple_parts, list):
        for part in multiple_parts:
            if isinstance(part, str):
                parts_set.add(part)

    st.session_state.recommended_parts_dynamic = parts_set


def discover_flow_files() -> List[Path]:
    files = []
    seen = set()
    for pattern in ("*.yaml", "*.yml"):
        for path in BASE_DIR.glob(pattern):
            if path.name.lower() == "p2o_flow_schema_v1.yaml":
                continue
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                files.append(resolved)
    files.sort(key=lambda p: p.name.lower())
    return files


def apply_loaded_tree(tree: Dict[str, Any], source_label: Optional[str]) -> None:
    st.session_state.tree = tree
    st.session_state.meta = tree.get("meta") or {}
    st.session_state.selected_flow_path = source_label
    reset_tree_progress(tree)


def decode_yaml_bytes(data: bytes, source_label: str) -> str:
    errors = []
    for encoding in YAML_FALLBACK_ENCODINGS:
        try:
            return data.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    detail = "; ".join(errors) if errors else "Unknown decoding error"
    raise ValueError(
        f"Unable to decode {source_label}. Tried encodings: "
        f"{', '.join(YAML_FALLBACK_ENCODINGS)}. Details: {detail}"
    )


def load_flow_from_path(flow_path: Path) -> None:
    try:
        raw_bytes = flow_path.read_bytes()
        yaml_text = decode_yaml_bytes(raw_bytes, str(flow_path))
        yaml_data = yaml.safe_load(yaml_text)
        tree = normalize_tree(yaml_data)
        apply_loaded_tree(tree, str(flow_path))
    except Exception as exc:
        st.error(f"Failed to load flow '{flow_path.name}': {exc}")


def load_flow_from_upload(upload) -> None:
    try:
        raw_bytes = upload.read()
        yaml_text = decode_yaml_bytes(raw_bytes, upload.name or "uploaded YAML")
        yaml_data = yaml.safe_load(yaml_text)
        tree = normalize_tree(yaml_data)
        apply_loaded_tree(tree, None)
        st.success("Custom YAML loaded.")
    except Exception as exc:
        st.error(f"Failed to load uploaded YAML: {exc}")


def ensure_widget_state(key: str, default: Any) -> Any:
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def reset_full_session() -> None:
    st.session_state.clear()
    st.rerun()


def key_from_part(part: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", part).strip("_")
    return slug.lower() or "part"


def wants_resolution_prompt(node: Dict[str, Any]) -> bool:
    return bool(node.get("resolution_prompt", True))


def prettify_flow_label(path: Path) -> str:
    """Generate a reader-friendly name without prefixes like Ftree/P2O."""
    raw = path.stem.replace("_", " ")
    raw = re.sub(r"\b(ftree|p2o)\b", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+", " ", raw).strip()
    if not raw:
        raw = path.stem
    return raw.title()


def render_token_copy(token: str) -> None:
    safe_token = html.escape(token)
    token_js = json.dumps(token)
    token_block_id = f"token-copy-{uuid.uuid4().hex}"
    button_id = f"{token_block_id}-btn"
    components.html(
        f"""
        <div id="{token_block_id}" style="margin:0.8rem 0 1.2rem 0;">
            <div style="font-weight:700;color:#ffffff;margin-bottom:0.35rem;">
                Gate Token
            </div>
            <div style="
                background:#fff3cd;
                border:2px solid #f4c430;
                border-radius:12px;
                padding:0.9rem 1rem;
                font-weight:800;
                font-size:1.2rem;
                letter-spacing:0.05em;
                color:#1b1b1b;
                text-align:center;
                box-shadow:0 4px 12px rgba(0,0,0,0.15);
            ">
                {safe_token}
            </div>
            <button id="{button_id}" style="
                margin-top:0.55rem;
                width:100%;
                padding:0.55rem 1rem;
                border:none;
                border-radius:10px;
                background:#118ab2;
                color:#ffffff;
                font-weight:700;
                font-size:0.95rem;
                cursor:pointer;
                transition:background 0.2s ease, transform 0.2s ease;
            ">
                Copy Token Number
            </button>
        </div>
        <script>
        (function(){{
            const btn = document.getElementById("{button_id}");
            if(!btn) return;

            const tokenValue = {token_js};
            const setCopied = (success) => {{
                const original = 'Copy Token Number';
                btn.innerText = success ? 'Token Copied!' : 'Copy not available';
                btn.style.background = success ? '#06d6a0' : '#ef476f';
                setTimeout(() => {{
                    btn.innerText = original;
                    btn.style.background = '#118ab2';
                }}, 1800);
            }};

            const copyFallback = () => {{
                try {{
                    const txt = document.createElement('textarea');
                    txt.value = tokenValue;
                    document.body.appendChild(txt);
                    txt.select();
                    document.execCommand('copy');
                    document.body.removeChild(txt);
                    setCopied(true);
                }} catch (err) {{
                    setCopied(false);
                }}
            }};

            btn.addEventListener('click', async () => {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    try {{
                        await navigator.clipboard.writeText(tokenValue);
                        setCopied(true);
                    }} catch (err) {{
                        copyFallback();
                    }}
                }} else {{
                    copyFallback();
                }}
            }});
        }})();
        </script>
        """,
        height=210,
    )


def apply_button_style_by_label(label: str, css_class: str) -> None:
    """
    Attach a CSS class to a Streamlit button by matching its label in the
    rendered DOM so we can style it without affecting other controls.
    """
    if not label or not css_class:
        return
    components.html(
        f"""
        <script>
        (function() {{
            const doc = window.parent.document;
            if (!doc) return;
            const target = Array.from(doc.querySelectorAll('button')).find(
                (btn) => btn.textContent.trim().startsWith({json.dumps(label)})
            );
            if (target) {{
                target.classList.add({json.dumps(css_class)});
            }}
        }})();
        </script>
        """,
        height=0,
    )


def inject_product_selector_styles() -> None:
    if st.session_state.get("_product_selector_css_loaded"):
        return
    st.session_state["_product_selector_css_loaded"] = True
    st.markdown(
        """
        <style>
        .magic-grid-row {
            margin-top: 1rem;
        }
        .magic-product-card {
            position: relative;
            border-radius: 30px;
            padding: 1.5px;
            margin: 0.8rem auto 1rem;
            max-width: 360px;
            background: linear-gradient(135deg, rgba(147,51,234,0.8), rgba(59,130,246,0.8));
            box-shadow: 0 25px 60px rgba(8, 12, 40, 0.6);
            overflow: hidden;
            display: block;
            text-decoration: none;
            color: inherit;
            perspective: 1000px;
        }
        .magic-product-card::after {
            content: '';
            position: absolute;
            inset: 0;
            background:
                linear-gradient(to top, rgba(10, 15, 40, 0.7) 0%, transparent 50%),
                linear-gradient(to bottom, rgba(10, 15, 40, 0.7) 0%, transparent 40%);
            background-size: 100% 100%;
            animation: smokeScreen 7s ease-in-out infinite;
            pointer-events: none;
            z-index: 3;
        }
        .magic-product-card.magic-clickable .card-core {
            transition: transform 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94),
                        box-shadow 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94);
            animation: cinematicGlow 4s ease-in-out infinite;
        }
        .magic-product-card.magic-clickable:hover .card-core {
            animation: glitch 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) both infinite;
        }
        .magic-product-card.magic-clickable:active .card-core {
            transform: translateY(0px) scale(0.97);
            transition-duration: 0.15s;
        }
        .magic-product-card.magic-disabled {
            pointer-events: none;
            filter: grayscale(0.1);
        }
        .magic-product-card::before {
            content: "";
            position: absolute;
            inset: -60% -20%;
            background: conic-gradient(from 120deg, rgba(255,255,255,0.2), rgba(255,255,255,0));
            animation: auroraDrift 12s linear infinite;
            pointer-events: none;
        }
        .magic-product-card .card-core {
            position: relative;
            z-index: 2;
            background: rgba(7,11,35,0.92);
            border-radius: 29px;
            padding: 1.05rem;
            overflow: hidden;
        }
        .magic-product-card .card-core::before {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at 20% 10%, rgba(255,215,0,0.2), transparent 50%),
                        radial-gradient(circle at 80% 20%, rgba(59,130,246,0.25), transparent 60%);
            pointer-events: none;
            opacity: 0.9;
        }
        .magic-product-card .card-core::after {
            content: '';
            position: absolute;
            top: 0;
            left: -200%;
            width: 200%;
            height: 100%;
            transform: skewX(-20deg);
            background-image: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.12), transparent);
            pointer-events: none;
            animation: cardShine 4s ease-in-out infinite;
            animation-delay: 1s;
        }
        .magic-product-card.magic-clickable:hover .card-core::after {
            animation: cardShine 1.2s ease-in-out 0s 1; /* Faster shine on hover */
        }
        .magic-product-card .card-title {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 0.5rem;
            position: relative;
            z-index: 3;
        }
        .magic-product-card .card-chip {
            font-size: 1.1rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            color: #f8fbff;
            text-transform: uppercase;
        }
        .magic-product-card .card-badge {
            font-size: 0.85rem;
            font-weight: 700;
            color: #0b132b;
            background: linear-gradient(120deg, #f9d423, #ff4e50);
            padding: 0.2rem 0.8rem;
            border-radius: 999px;
            box-shadow: 0 8px 25px rgba(249, 212, 35, 0.4);
        }
        .magic-product-card .card-image {
            height: 140px;
            border-radius: 20px;
            margin: 1.1rem 0;
            background-size: cover;
            background-position: center;
            box-shadow: inset 0 0 20px rgba(0,0,0,0.35);
            animation: imagePulseZoom 8s ease-in-out infinite;
            position: relative;
            overflow: hidden;
        }
        .magic-product-card .card-image::before {
            content: '';
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle, rgba(255, 255, 255, 0.1) 10%, transparent 10.5%) 0 0,
                radial-gradient(circle, rgba(255, 255, 255, 0.1) 10%, transparent 10.5%) 10px 10px,
                linear-gradient(rgba(255, 255, 255, 0.05) 1px, transparent 1.5px),
                linear-gradient(90deg, rgba(255, 255, 255, 0.05) 1px, transparent 1.5px);
            background-size: 20px 20px, 20px 20px, 10px 10px, 10px 10px;
            animation: flowingEnergy 15s linear infinite;
            opacity: 0.5;
        }
        .magic-product-card.magic-clickable:hover .card-image::before {
            animation-duration: 5s; /* Faster energy flow on hover */
        }
        .magic-product-card .card-meta {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 0.75rem;
            position: relative;
            z-index: 3;
        }
        .magic-product-card .card-desc {
            font-size: 0.88rem;
            color: #cdd4ff;
            line-height: 1.4;
            flex: 1;
        }
        .magic-product-card .card-status {
            font-weight: 700;
            font-size: 0.85rem;
            padding: 0.35rem 0.8rem;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.2);
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .magic-product-card .card-status.available {
            color: #04f6c5;
            border-color: rgba(4,246,197,0.35);
        }
        .magic-product-card .card-status.soon {
            color: #ff99c8;
            border-color: rgba(255,153,200,0.35);
        }
        .magic-product-card .card-cta {
            margin-top: 0.8rem;
            padding: 0.75rem 1.2rem;
            border-radius: 14px;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            display: inline-flex;
            gap: 0.35rem;
            align-items: center;
            justify-content: center;
            box-shadow: 0 15px 40px rgba(255, 0, 128, 0.35);
            border: 1px solid rgba(255,255,255,0.18);
        }
    .magic-product-card .card-cta.available {
        background: linear-gradient(120deg, #ff8b5f, #f72585);
        color: #fff8f9;
        animation: launchPulse 3.5s ease-in-out infinite;
    }
    .magic-product-card.magic-clickable:hover .card-cta.available {
        box-shadow: 0 25px 60px rgba(255, 142, 95, 0.45);
        animation-duration: 2.5s;
    }
        .magic-product-card .card-cta.soon {
            background: linear-gradient(120deg, #3f3d56, #1d1b2f);
            color: #b0b7de;
            box-shadow: none;
        }
        @keyframes cinematicGlow {
            0% { box-shadow: 0 0 15px rgba(59, 130, 246, 0.4), 0 0 25px rgba(147, 51, 234, 0.3); }
            50% { box-shadow: 0 0 25px rgba(59, 130, 246, 0.6), 0 0 40px rgba(147, 51, 234, 0.5); }
            100% { box-shadow: 0 0 15px rgba(59, 130, 246, 0.4), 0 0 25px rgba(147, 51, 234, 0.3); }
        }
        @keyframes auroraDrift {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    @keyframes imagePulseZoom {
        0% { transform: scale(1.0); }
        40% { transform: scale(1.08); }
        60% { transform: scale(1.05); }
        80% { transform: scale(1.10); }
        100% { transform: scale(1.0); }
    }
    @keyframes launchPulse {
        0%, 100% { transform: translateY(0) scale(1); box-shadow: 0 15px 40px rgba(255, 0, 128, 0.35); }
        30% { transform: translateY(-4px) scale(1.04); box-shadow: 0 18px 50px rgba(255, 142, 95, 0.45); }
        60% { transform: translateY(0) scale(1); box-shadow: 0 15px 40px rgba(255, 0, 128, 0.35); }
    }
        @keyframes smokeScreen {
            0% { background-position: 0 100%, 0 -100%; opacity: 0; }
            20% { opacity: 0.1; }
            80% { opacity: 0.1; }
            100% { background-position: 0 -100%, 0 100%; opacity: 0; }
        }
        @keyframes cardShine {
            100% { left: 150%; }
        }
        @keyframes flowingEnergy {
            0% { background-position: 0 0, 10px 10px, 0 0, 0 0; }
            100% { background-position: -40px -40px, -30px -30px, -20px -20px, -10px -10px; }
        }
        @keyframes glitch {
            2%, 64% { transform: translate(2px, 0) skew(0deg); }
            4%, 60% { transform: translate(-2px, 0) skew(0deg); }
            62% { transform: translate(0, 0) skew(5deg); }
        }
        .magic-product-card.magic-clickable:hover .card-chip {
            animation: glitch-text 0.8s cubic-bezier(0.25, 0.46, 0.45, 0.94) both infinite;
        }
        @keyframes glitch-text {
            0%, 100% { text-shadow: none; }
            25% { text-shadow: 2px 2px 0 #ff00ff, -2px -2px 0 #00ffff; }
            50% { text-shadow: 2px -2px 0 #ff00ff, -2px 2px 0 #00ffff; }
            75% { text-shadow: -2px 2px 0 #ff00ff, 2px -2px 0 #00ffff; }
        }
        @media (max-width: 900px) {
            .magic-product-card {
                border-radius: 22px;
                max-width: 100%;
            }
            .magic-product-card .card-core {
                padding: 0.85rem;
            }
            .magic-product-card .card-chip {
                font-size: 0.95rem;
            }
            .magic-product-card .card-image {
                height: 110px;
                border-radius: 16px;
            }
            .magic-product-card .card-desc {
                font-size: 0.82rem;
            }
            .magic-product-card .card-cta {
                font-size: 0.78rem;
                padding: 0.6rem 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_product_selector() -> None:
    inject_product_selector_styles()
    st.markdown(
        "<div class='product-grid-headline'>Choose the product you are troubleshooting</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='product-grid-subtitle'>Tap a product card to continue the AI-powered flow.</div>",
        unsafe_allow_html=True,
    )
    pending_notice = st.session_state.pop("product_notice", None)
    if pending_notice:
        st.warning(
            f"Troubleshooting paths for **{pending_notice}** are coming soon. "
            "Please select Washing Machine (WM) for now."
        )
    active_token = current_access_token()

    cards_per_row = 2
    for start in range(0, len(PRODUCT_CATEGORIES), cards_per_row):
        cols = st.columns(cards_per_row)
        for offset, col in enumerate(cols):
            idx = start + offset
            if idx >= len(PRODUCT_CATEGORIES):
                break
            category = PRODUCT_CATEGORIES[idx]
            available_flag = category["available"]
            desc_text = (
                "Beam into full AI troubleshooting with holo-guided steps."
                if available_flag
                else "Holograms charging - this flow unlocks soon."
            )
            status_label = "AI Ready" if available_flag else "Calibrating"
            cta_label = "✨ Launch AI Flow" if available_flag else "🚧 Coming Soon"
            wrapper_tag = "a" if available_flag else "div"
            wrapper_classes = (
                "magic-product-card magic-clickable"
                if available_flag
                else "magic-product-card magic-disabled"
            )
            wrapper_attrs = f'class="{wrapper_classes}"'
            if available_flag:
                query_bits = {"product": category["id"]}
                if active_token:
                    query_bits[ACCESS_TOKEN_PARAM] = active_token
                wrapper_attrs += f' href="?{urlencode(query_bits)}" target="_self"'
            else:
                wrapper_attrs += ' role="button" aria-disabled="true"'
            image_bytes = load_product_image(category["image"])
            image_b64 = (
                base64.b64encode(image_bytes).decode("ascii") if image_bytes else ""
            )
            bg_image = (
                f"linear-gradient(135deg, rgba(15,23,42,0.6), rgba(56,189,248,0.15)), url('data:image/png;base64,{image_b64}')"
                if image_b64
                else "linear-gradient(135deg,#1f0a39,#3f2b96)"
            )
            card_html = f"""
            <{wrapper_tag} {wrapper_attrs}>
                <div class="card-core">
                    <div class="card-title">
                        <span class="card-chip">{html.escape(category['label'])}</span>
                        <span class="card-badge">{status_label}</span>
                    </div>
                    <div class="card-image" style="background-image:{bg_image};"></div>
                    <div class="card-meta">
                        <div class="card-desc">{html.escape(desc_text)}</div>
                        <div class="card-status {'available' if available_flag else 'soon'}">
                            {cta_label}
                        </div>
                    </div>
                    <div class="card-cta {'available' if available_flag else 'soon'}">
                        {cta_label}
                    </div>
                </div>
            </{wrapper_tag}>
            """
            with col:
                st.markdown(card_html, unsafe_allow_html=True)

def product_label(product_id: Optional[str]) -> str:
    if not product_id:
        return ""
    return PRODUCT_CATEGORY_LABELS.get(product_id, product_id)


@lru_cache(maxsize=32)
def load_product_image(image_name: str) -> Optional[bytes]:
    if not image_name:
        return None
    path = PRODUCT_IMAGES_DIR / image_name
    if not path.exists():
        return None
    try:
        return path.read_bytes()
    except OSError:
        return None


def set_selected_product(product_id: Optional[str]) -> None:
    access_ok = st.session_state.get("access_granted", False)
    st.session_state.selected_product = product_id
    st.session_state.product_notice = None
    st.session_state["_product_selector_css_loaded"] = False
    st.session_state.selected_flow_path = None
    st.session_state.tree = None
    st.session_state.meta = {}
    st.session_state.node_id = None
    st.session_state.visited_stack = []
    st.session_state.passed = {}
    st.session_state.answers = {}
    st.session_state.parts_used = set()
    st.session_state.part_photos = {}
    st.session_state.flow_status = None
    st.session_state.final_token = None
    st.session_state.pending_resolution = None
    st.session_state.path_total_steps = 0
    st.session_state.second_visit_mode = False
    st.session_state[SESSION_HISTORY_KEY] = []
    st.session_state.recommended_parts_dynamic = set()
    if access_ok:
        st.session_state.access_granted = True


def filter_flows_for_category(files: List[Path], category_id: Optional[str]) -> List[Path]:
    if not category_id:
        return []
    if category_id not in CATEGORY_FLOW_PATTERNS:
        return files if category_id in AVAILABLE_FLOW_CATEGORIES else []
    patterns = CATEGORY_FLOW_PATTERNS[category_id]
    filtered: List[Path] = []
    for path in files:
        name = path.stem.lower()
        if any(pattern in name for pattern in patterns):
            filtered.append(path)
    return filtered


def handle_product_query_param() -> None:
    params = st.query_params
    product_vals = params.get("product")
    if not product_vals:
        return
    product_choice = product_vals if isinstance(product_vals, str) else product_vals[0]
    product_choice = (product_choice or "").strip().upper()
    clear_query_params_preserving_access_token()
    if product_choice in AVAILABLE_FLOW_CATEGORIES:
        set_selected_product(product_choice)
        st.session_state["_scroll_target"] = "top"
        st.rerun()
    elif product_choice:
        st.session_state.product_notice = product_label(product_choice)


def render_resolution_prompt(tree: Dict[str, Any], lang: str) -> bool:
    pending = st.session_state.get("pending_resolution")
    if not pending:
        return False

    prev_id = pending.get("prev_node")
    force_restart = pending.get("force_restart", False)
    node = (tree.get("nodes") or {}).get(prev_id, {})
    st.success(f"Step logged: {step_label(node, lang)}")
    choice_key = f"resolution_choice_{prev_id}"
    ensure_widget_state(choice_key, "No")
    choice = "No"
    if not force_restart:
        st.markdown(
            "<div style='font-size:1.2rem;font-weight:700;color:#ffffff;'>Was the issue resolved after the previous step?</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='resolution-radio'>", unsafe_allow_html=True)
        st.radio(
            "",
            ["Yes", "No"],
            key=choice_key,
            horizontal=True,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        choice = st.session_state[choice_key]
    else:
        st.info("This branch requires re-running the full troubleshooting flow if the issue persists.")
        st.session_state[choice_key] = "No"

    if st.button("Submit Resolution Decision", type="primary"):
        st.session_state.pending_resolution = None
        if choice == "Yes":
            st.session_state.flow_status = {
                "type": "resolved",
                "node_id": prev_id,
                "all_valid": True,
                "resolved_without_parts": not st.session_state.parts_used,
            }
            st.session_state["_scroll_target"] = "top"
            st.rerun()
        elif force_restart or (
            pending.get("second_visit_mode") and choice == "No"
        ):
            st.session_state.flow_status = {
                "type": "restart",
                "node_id": prev_id,
            }
            st.session_state.second_visit_mode = False
            st.session_state.path_total_steps = count_progress_steps(
                tree.get("nodes") or {}, False
            )
            st.session_state["_scroll_target"] = "top"
            st.rerun()
        else:
            next_id = pending.get("next_node")
            if next_id:
                st.session_state.visited_stack.append(next_id)
                st.session_state.node_id = next_id
                st.session_state["_scroll_anchor"] = f"node-{next_id}"
                st.session_state["_scroll_target"] = "top"
                st.rerun()
            else:
                st.session_state.flow_status = {
                    "type": "completed",
                    "node_id": prev_id,
                    "all_valid": pending.get("final_all_valid", True),
                }
                st.session_state["_scroll_target"] = "top"
                st.rerun()
    return True


def render_completion_panel(tree: Dict[str, Any], meta: Dict[str, Any], lang: str) -> bool:
    """
    Show the completion / finalize panel once the flow_status is set.
    Handles:
      - resolved / restart / completed states
      - required part-photos before finalization
      - calling the P2O endpoint and showing a gate token
      - robust error handling (no UnboundLocalError on resp)
    """
    flow_status = st.session_state.get("flow_status")
    if not flow_status:
        return False

    if st.session_state.final_token is None:
        if st.button(BACK_BUTTON_LABEL, key="back_from_completion", use_container_width=True):
            st.session_state.flow_status = None
            st.rerun()
        apply_button_style_by_label(BACK_BUTTON_LABEL, BACK_BUTTON_CLASS)

    status = flow_status.get("type", "completed")
    node_id = flow_status.get("node_id") or st.session_state.node_id
    nodes = tree.get("nodes") or {}
    node = nodes.get(node_id, {})

    # High-level status messaging
    if status == "resolved":
        st.success(
            "Great job! Issue resolved. Submit your final log to close the case."
        )
    elif status == "restart":
        st.warning(
            "Issue persists. Please re-run the full troubleshooting flow from the beginning."
        )
        if st.button("Restart Troubleshooting"):
            reset_tree_progress(tree)
            st.session_state["_scroll_target"] = "node"
            st.rerun()
        return True
    else:
        st.info("All mandatory checks completed. Finalize to generate the gate token.")

    # -----------------------------
    # Part photos (for recommended parts)
    # -----------------------------
    recommended_parts = sorted(
        list(st.session_state.get("recommended_parts_dynamic", set()))
    )
    if "part_photos" not in st.session_state:
        st.session_state.part_photos = {}
    part_photos = st.session_state.part_photos

    photos_required = recommended_parts if (recommended_parts and status != "resolved") else []
    all_photos_ready = True

    if photos_required:
        parts_text = ", ".join(recommended_parts)
        st.markdown(f'''
        <style>
        @keyframes pulse-glow-animation {{
            0%, 100% {{
                transform: scale(1);
                box-shadow: 0 0 15px rgba(59, 130, 246, 0.5), 0 0 25px rgba(147, 51, 234, 0.4), inset 0 0 8px rgba(59, 130, 246, 0.3);
            }}
            50% {{
                transform: scale(1.02);
                box-shadow: 0 0 25px rgba(59, 130, 246, 0.9), 0 0 45px rgba(147, 51, 234, 0.8), inset 0 0 12px rgba(147, 51, 234, 0.5);
            }}
        }}

        .pulsing-glow-box {{
            background: linear-gradient(145deg, #101018, #181820);
            color: #e0e7ff;
            border: 2px solid;
            border-image-slice: 1;
            border-image-source: linear-gradient(to right bottom, #3b82f6, #a855f7, #ec4899);
            border-radius: 18px;
            padding: 1.2rem;
            margin: 1.2rem 0;
            animation: pulse-glow-animation 2s ease-in-out infinite;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            text-align: center;
        }}
        .pulsing-glow-title {{
            font-size: 1.15rem;
            font-weight: 800;
            margin-bottom: 0.6rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: #ffffff;
            text-shadow: 0 0 6px #ec4899, 0 0 12px #a855f7, 0 0 18px #3b82f6;
        }}
        .pulsing-glow-parts {{
            font-size: 1.35rem;
            font-weight: 700;
            color: #f0f9ff;
            text-shadow: 0 1px 2px rgba(0,0,0,0.5);
            word-wrap: break-word;
        }}
        </style>
        <div class="pulsing-glow-box">
            <div class="pulsing-glow-title">✨ Recommended Parts to Order ✨</div>
            <div class="pulsing-glow-parts">{parts_text}</div>
        </div>
        ''', unsafe_allow_html=True)
        st.info("Capture clear photos of each part you plan to order.")
        for part in photos_required:
            slug = key_from_part(part)
            stored_photo = part_photos.get(part)
            with st.container():
                st.markdown(f"**{part}**")
                if stored_photo:
                    st.success("Photo captured.")
                    if st.button(f"Retake photo - {part}", key=f"retake_{slug}"):
                        part_photos.pop(part, None)
                        # Clean up camera visibility state on retake
                        st.session_state.pop(f"show_camera_{slug}", None)
                        st.rerun()
                else:
                    show_camera_key = f"show_camera_{slug}"
                    upload = None
                    cam = None

                    # Check if we should be showing the camera for this part
                    if st.session_state.get(show_camera_key):
                        st.caption("Tip: switch to the rear camera in the picker for part photos.")
                        cam = st.camera_input(f"Capture photo of {part}", key=f"cam_{slug}")
                        if st.button(f"Cancel Capture", key=f"cancelcam_{slug}"):
                            st.session_state[show_camera_key] = False
                            st.rerun()
                    else:
                        # Default state: Show a button to open the camera, or an uploader
                        if CAMERA_CAPTURE_ENABLED:
                            if st.button(f"📸 Capture photo of {part}", key=f"opencam_{slug}"):
                                st.session_state[show_camera_key] = True
                                st.rerun()

                        upload = st.file_uploader(
                            f"Or upload a photo of {part}",
                            type=["jpg", "jpeg", "png"],
                            key=f"upload_{slug}",
                        )

                    # Process whichever input has data
                    upload = cam or upload
                    if upload:
                        encoded, mime = b64_of_uploaded(upload)
                        if encoded:
                            part_photos[part] = {"photo_b64": encoded, "photo_mime": mime}
                            st.session_state.pop(show_camera_key, None)  # Clean up state
                            st.rerun()
                    else:
                        all_photos_ready = False

        if photos_required:
            all_photos_ready = all(part in part_photos for part in photos_required)  

    # -----------------------------
    # Finalize / Gate token
    # -----------------------------
    finalized = flow_status.get("finalized", False)
    token = st.session_state.final_token
    ready_to_finalize = (not photos_required) or all_photos_ready

    if not finalized:
        if not ready_to_finalize:
            st.info("Upload required part photos to enable finalization.")

        finalize_clicked = st.button(
            "Finalize & Generate Gate Token",
            type="primary",
            disabled=not ready_to_finalize,
        )

        if finalize_clicked:
            # keep viewport sane after rerun
            st.session_state["_scroll_target"] = "top"

            sku_value = st.session_state.case.get("sku", "") or "NA"
            answers_state = st.session_state.answers.get(node_id, {})
            final_ctrl = (node or {}).get("ui", {}).get("control") if node else None
            raw_final_value = answers_state.get("raw_value", answers_state.get("value"))
            final_answer_value = format_answer_value(
                final_ctrl,
                raw_final_value,
                elapsed_sec=answers_state.get("elapsed_sec"),
                ui=(node or {}).get("ui"),
            )
            if answers_state:
                st.session_state.answers[node_id] = {
                    **answers_state,
                    "value": final_answer_value,
                }
            answers_for_payload = {
                "value": final_answer_value,
                "label_value": answers_state.get("label_value"),
                "elapsed_sec": answers_state.get("elapsed_sec"),
            }
            parts_used_list = sorted(st.session_state.parts_used)
            payload = {
                "flow_id": meta.get("id", ""),
                "case_id": st.session_state.case.get("case_id", ""),
                "sku": sku_value,
                "st_id": st.session_state.case.get("st_id", ""),
                "step_id": node_id or "",
                "step_label": step_label(node, lang) if node else "",
                "answers": answers_for_payload,
                "answer_value": answers_for_payload.get("value"),
                "label_value": answers_for_payload.get("label_value"),
                "elapsed_sec": answers_for_payload.get("elapsed_sec"),
                "pass": True,
                "photo_b64": None,
                "photo_mime": None,
                "finalize": True,
                "all_steps_valid": flow_status.get("all_valid", False),
                "token_pattern": (meta.get("gating") or {}).get(
                    "token_pattern", "{FAULT}-{SKU}-{RAND5}"
                ),
                "fault_code": fault_code_from_meta(meta.get("id", "")),
                "parts_used": parts_used_list or None,
                "resolution": status,
                "part_photos": part_photos if photos_required else {},
                "visit_selfie_b64": st.session_state.get("visit_selfie"),
                "visit_selfie_mime": st.session_state.get("visit_selfie_mime"),
            }

            resp = None
            spinner_tip = st.empty()
            try:
                with jeeves_spinner("🚀 Generating your Token Number, please wait...", SPINNER_COLOR):
                    spinner_tip.markdown(
                        "<div class='spinner-tip'>✨ Uploading evidence, updating logs, and loading the next action...</div>",
                        unsafe_allow_html=True,
                    )

                    # Prepare progress bar for visual feedback
                    step_buffer = st.session_state.get("p2o_step_buffer", [])
                    total_to_log = (len(step_buffer) or 0) + 1  # +1 for final Gate Token payload
                    
                    progress_placeholder = st.empty()
                    completed = 0

                    def render_custom_progress(pct: int):
                        # Ensure percentage is between 0 and 100
                        pct = max(0, min(100, pct))
                        progress_html = f"""
                        <style>
                        .custom-progress-container {{
                            width: 100%;
                            background-color: #4A5568; /* gray-700 */
                            border-radius: 12px;
                            overflow: hidden;
                            border: 2px solid #2D3748; /* gray-800 */
                            box-shadow: inset 0 2px 4px rgba(0,0,0,0.4);
                            margin: 8px 0;
                        }}

                        .custom-progress-bar {{
                            height: 28px;
                            border-radius: 10px;
                            background: linear-gradient(120deg, #22c55e, #84cc16, #fde047);
                            background-size: 200% 100%;
                            box-shadow: 0 0 12px rgba(132, 204, 22, 0.8);
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-weight: 800;
                            font-size: 0.9rem;
                            color: #1a202c; /* gray-900 */
                            text-shadow: 0 1px 1px rgba(255,255,255,0.3);
                            transition: width 0.4s ease-in-out;
                            animation: gradient-flow 2.5s linear infinite;
                        }}

                        @keyframes gradient-flow {{
                            0% {{ background-position: 200% 0; }}
                            100% {{ background-position: -200% 0; }}
                        }}
                        </style>
                        <div class="custom-progress-container">
                            <div class="custom-progress-bar" style="width: {pct}%;">
                                {pct}%
                            </div>
                        </div>
                        <div style='text-align:center; font-size:0.9rem; margin-top:4px; font-weight: 700; color: #E2E8F0;'>
                            Generating Gate Token...
                        </div>
                        """
                        progress_placeholder.markdown(progress_html, unsafe_allow_html=True)

                    # Initial 0% render
                    render_custom_progress(0)

                    # 1) Flush all buffered non-final steps (if any)
                    if P2O_ENDPOINT and step_buffer:
                        for step_payload in step_buffer:
                            step_resp = post_step_log(P2O_ENDPOINT, step_payload)
                            completed += 1
                            
                            # Update progress bar
                            frac = min(completed / total_to_log, 1.0)
                            pct = int(frac * 100)
                            render_custom_progress(pct)
                            
                            # Small delay to make the animation visible between steps
                            time.sleep(0.1) 

                            if not step_resp.get("ok", True):
                                # Non-fatal: warn but continue flushing
                                st.warning(
                                    f"Step log failed: {step_resp.get('error', 'Unknown error')}"
                                )

                        # Clear buffer after flushing
                        st.session_state.p2o_step_buffer = []

                    # 2) Now send the final payload to generate Gate Token
                    final_resp = post_step_log(P2O_ENDPOINT, payload)
                    completed += 1

                    # Force 100% when final payload is done
                    render_custom_progress(100)

                    resp = final_resp                     

                if resp.get("ok", True):
                    token = resp.get("token", "(no token — endpoint not set)")
                    st.session_state.final_token = token
                    st.session_state.flow_status["finalized"] = True
                    st.success("Gate token generated successfully.")
                    render_token_copy(token)
                    st.caption("Paste this token in Strider Notes until API integration.")
                else:
                    base_err = resp.get("error") or "Finalize call failed."
                    detail = resp.get("text") or resp.get("status_code")
                    detail_msg = f"{base_err} ({detail})" if detail else base_err
                    st.error(f"Finalize failed: {detail_msg}")

            except Exception as e:
                base_err = str(e)
                detail = None
                if isinstance(resp, dict):
                    detail = resp.get("text") or resp.get("status_code")
                detail_msg = f"{base_err} ({detail})" if detail else base_err
                st.error(f"Finalize failed: {detail_msg}")
            finally:
                spinner_tip.empty()

    else:
        token = token or "(no token — endpoint not set)"
        st.success("Gate token generated successfully.")
        render_token_copy(token)
        st.caption("Paste this token in Strider Notes until API integration.")

    # -----------------------------
    # Restart option
    # -----------------------------
    if st.button("Restart Session"):
        reset_full_session()

    return True


# -----------------------------
# UI - Header & PIN
# -----------------------------
init_session_state()
restore_access_from_token_query()
handle_product_query_param()

st.markdown(
    """
    <style>
    .main-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        font-size: clamp(1.2rem, 2.4vw, 2rem);
        font-weight: 800;
        letter-spacing: 0.04em;
        color: #f8fbff;
        width: min(960px, 96%);
        margin: 1.4rem auto 1.8rem auto;
        padding: 1.8rem 3rem;
        position: relative;
        overflow: hidden;
        border-radius: 32px;
        border: 1px solid rgba(71, 85, 105, 0.8);
        background:
            radial-gradient(circle at 15% 35%, rgba(56, 189, 248, 0.18), transparent 55%),
            radial-gradient(circle at 80% 20%, rgba(244, 114, 182, 0.2), transparent 50%),
            linear-gradient(135deg, #050711, #0a1428 45%, #111f3d 100%);
        box-shadow:
            0 30px 70px rgba(2, 6, 23, 0.92),
            inset 0 0 40px rgba(59, 130, 246, 0.25),
            inset 0 0 70px rgba(2, 132, 199, 0.18);
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
        isolation: isolate;
        transform-style: preserve-3d;
        backdrop-filter: blur(4px);
        animation: titleFloat 10s ease-in-out infinite;
    }
    .main-title::before {
        content: "";
        position: absolute;
        inset: -40%;
        background: conic-gradient(
            from 0deg,
            rgba(59, 130, 246, 0.22),
            rgba(236, 72, 153, 0.18),
            rgba(14, 165, 233, 0.25),
            rgba(59, 130, 246, 0.22)
        );
        filter: blur(42px);
        opacity: 0.5;
        animation: auroraBreathe 18s linear infinite;
        z-index: 0;
    }
    .main-title::after {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(120deg, transparent 30%, rgba(255,255,255,0.15), transparent 70%);
        transform: translateX(-140%) rotate(5deg);
        animation: scanlineSweep 8s ease-in-out infinite;
        z-index: 2;
        opacity: 0.45;
    }

    .title-neon-ring {
        position: absolute;
        width: 140%;
        height: 260%;
        border-radius: 50%;
        border: 1px solid rgba(148, 163, 184, 0.35);
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(12deg);
        filter: blur(0.8px);
        animation: ringPulse 14s linear infinite;
        opacity: 0.4;
        z-index: 1;
        pointer-events: none;
    }
    .title-neon-ring::before,
    .title-neon-ring::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 1px solid rgba(99, 102, 241, 0.3);
        transform-origin: center;
    }
    .title-neon-ring::after {
        transform: scale(0.82) rotate(35deg);
        border-color: rgba(59, 130, 246, 0.4);
    }

    .title-particles {
        position: absolute;
        inset: 0;
        z-index: 1;
        pointer-events: none;
    }
    .title-particles span {
        position: absolute;
        width: 5px;
        height: 5px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(255,255,255,0.9), transparent 70%);
        filter: blur(0.6px);
        opacity: 0.6;
        animation: particleDrift linear infinite;
    }
    .title-particles span:nth-child(1) {
        top: 15%;
        left: 10%;
        animation-duration: 12s;
        animation-delay: -2s;
    }
    .title-particles span:nth-child(2) {
        top: 65%;
        left: 20%;
        animation-duration: 9s;
        animation-delay: -4s;
    }
    .title-particles span:nth-child(3) {
        top: 30%;
        right: 12%;
        animation-duration: 11s;
    }
    .title-particles span:nth-child(4) {
        top: 70%;
        right: 25%;
        animation-duration: 13s;
        animation-delay: -6s;
    }
    .title-particles span:nth-child(5) {
        top: 50%;
        left: 45%;
        animation-duration: 10s;
        animation-delay: -1s;
    }

    .title-comet {
        position: absolute;
        top: 28%;
        left: -25%;
        width: 170px;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(240, 171, 252, 0.65), rgba(125, 211, 252, 0.9));
        filter: drop-shadow(0 0 12px rgba(125, 211, 252, 0.7));
        animation: cometTrail 9s ease-in-out infinite;
        z-index: 1;
        opacity: 0.55;
    }
    .title-comet::after {
        content: "";
        position: absolute;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        top: -16px;
        right: -12px;
        background: radial-gradient(circle, rgba(125,211,252,0.6), transparent 60%);
        filter: blur(0.8px);
    }

    .title-core {
        position: relative;
        z-index: 3;
        font-weight: 800;
        text-transform: uppercase;
        text-align: center;
        line-height: 1.35;
        text-shadow:
            0 6px 16px rgba(4, 10, 28, 0.65),
            0 0 12px rgba(148, 163, 184, 0.45);
        animation: textGlow 8s ease-in-out infinite;
    }
    .title-core span {
        position: relative;
        background: linear-gradient(120deg, #fff7ae, #ffbbf4, #c084fc, #7dd3fc, #34f5c6, #f472b6);
        -webkit-background-clip: text;
        color: transparent;
        display: inline-block;
        background-size: 400% 400%;
        -webkit-text-stroke: 0.35px rgba(255, 255, 255, 0.35);
        text-shadow:
            0 0 12px rgba(59, 130, 246, 0.4),
            0 0 25px rgba(236, 72, 153, 0.35);
        animation: textAurora 9s ease-in-out infinite,
                   neonBlink 2.8s steps(2) infinite;
    }
    .title-core::before,
    .title-core::after {
        content: attr(data-text);
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        pointer-events: none;
        mix-blend-mode: normal;
        opacity: 0.18;
        filter: blur(0.4px);
    }
    .title-core::before {
        color: transparent;
        -webkit-text-stroke: 1.4px rgba(94, 234, 212, 0.75);
        filter: drop-shadow(0 0 16px rgba(45, 212, 191, 0.65));
        animation: outlinePulse 6s ease-in-out infinite,
                   hueShift 9s linear infinite;
    }
    .title-core::after {
        background: linear-gradient(100deg, transparent 15%, rgba(255,255,255,0.55), transparent 85%);
        -webkit-background-clip: text;
        color: transparent;
        animation: highlightSweep 5s ease-in-out infinite,
                   colorWarp 14s linear infinite;
    }

    .title-symbol {
        font-size: clamp(2rem, 4vw, 2.6rem);
        line-height: 1;
        z-index: 3;
        text-shadow:
            0 0 10px rgba(253, 224, 71, 0.8),
            0 0 24px rgba(255, 255, 255, 0.6);
        animation: symbolPulse 5s ease-in-out infinite;
    }
    .selected-product-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.9rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        color: #ffffff;
        font-weight: 600;
        border: 1px solid rgba(255,255,255,0.22);
        margin: 0.3rem 0 0.8rem 0;
        box-shadow: 0 4px 18px rgba(15,23,42,0.4);
    }
    .sub-caption {
        font-size: 1rem;
        font-weight: 600;
        text-align: center;
        margin: 0 auto 1.2rem;
        letter-spacing: 0.04em;
        background: linear-gradient(120deg, #7dd3fc, #c084fc, #f97316, #34f5c6);
        -webkit-background-clip: text;
        color: transparent;
        display: inline-block;
        padding: 0.35rem 1.15rem;
        border-radius: 999px;
        position: relative;
        filter: drop-shadow(0 0 8px rgba(59, 130, 246, 0.25));
        background-size: 250% 250%;
        animation: taglineGradient 11s linear infinite;
    }
    .sub-caption::after {
        content: "";
        position: absolute;
        inset: -1px;
        border-radius: inherit;
        border: 1px solid rgba(59, 130, 246, 0.35);
        opacity: 0.4;
        filter: blur(4px);
        animation: taglinePulse 6s ease-in-out infinite;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f3057;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
    }
    .product-grid-headline {
        font-size: 1.4rem;
        font-weight: 800;
        color: #f4f5ff;
        text-align: center;
        margin-top: 0.5rem;
    }
    .product-grid-subtitle {
        text-align: center;
        color: #bfd2ff;
        margin-bottom: 0.8rem;
    }
    .dark-mode-card {
        background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(15,118,110,0.45));
        border: 1px solid rgba(59,130,246,0.35);
        border-radius: 18px;
        padding: 0.9rem 1.2rem;
        color: #e2e8f0;
        box-shadow: 0 15px 30px rgba(2,6,23,0.55);
        margin-bottom: 0.8rem;
    }
    .dark-mode-card.active {
        border-color: rgba(94,234,212,0.55);
        box-shadow: 0 20px 40px rgba(15,118,110,0.4);
    }
    .product-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
        gap: 1.1rem;
        margin: 1rem 0 1.3rem 0;
    }
    .product-card-link {
        text-decoration: none;
        border-radius: 26px;
        overflow: hidden;
        display: block;
        min-height: 235px;
        position: relative;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.55);
        transition: transform 0.4s ease, box-shadow 0.4s ease;
    }
    .product-card-link:hover {
        transform: translateY(-10px) scale(1.02);
        box-shadow: 0 28px 60px rgba(99, 102, 241, 0.55);
    }
    .product-card-link[data-available="false"] {
        cursor: not-allowed;
    }
    .product-card-visual {
        position: relative;
        width: 100%;
        height: 100%;
        min-height: 235px;
        background-size: cover;
        background-position: center;
        border-radius: inherit;
        overflow: hidden;
    }
    .product-card-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(7, 11, 41, 0.15) 0%, rgba(7,11,41,0.75) 75%);
        mix-blend-mode: multiply;
    }
    .product-card-top-badge {
        position: absolute;
        top: 12px;
        left: 14px;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.25);
        font-weight: 700;
        color: #fdfdfd;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        z-index: 2;
        backdrop-filter: blur(6px);
    }
    .product-card-bottom-cta {
        position: absolute;
        bottom: 18px;
        left: 50%;
        transform: translateX(-50%);
        padding: 0.35rem 1.4rem;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.65);
        color: #ffffff;
        font-weight: 700;
        background: rgba(255,255,255,0.12);
        text-transform: uppercase;
        letter-spacing: 0.1em;
        z-index: 2;
        box-shadow: 0 12px 30px rgba(5, 10, 35, 0.65);
    }
    .product-card-link[data-available="false"] .product-card-bottom-cta {
        background: rgba(0,0,0,0.35);
        border-color: rgba(255,255,255,0.25);
        letter-spacing: 0.08em;
    }
    .progress-pill {
        background: linear-gradient(90deg, #00a8cc, #407088);
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
        margin: 0.6rem 0;
    }
    @keyframes glowing-blue-grey {
        0% { box-shadow: 0 2px 3px rgba(38, 50, 56, 0.3), 0 0 3px rgba(120, 144, 156, 0.4), 0 0 5px rgba(120, 144, 156, 0.4); }
        50% { box-shadow: 0 3px 6px rgba(38, 50, 56, 0.4), 0 0 7px rgba(144, 164, 174, 0.6), 0 0 11px rgba(144, 164, 174, 0.6); }
        100% { box-shadow: 0 2px 3px rgba(38, 50, 56, 0.3), 0 0 3px rgba(120, 144, 156, 0.4), 0 0 5px rgba(120, 144, 156, 0.4); }
    }
    .back-step-button {
        background-color: #546E7A !important; /* Blue-Grey, Text 2 */
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        border: 1px solid #263238 !important;
        box-shadow: 0 2px 4px rgba(38, 50, 56, 0.3); /* 3D Effect */
        text-transform: uppercase;
        letter-spacing: 0.06em;
        transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
        animation: glowing-blue-grey 2.2s ease-in-out infinite;
        position: relative;
    }
    .back-step-button:hover {
        background-color: #607D8B !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(38, 50, 56, 0.4), 0 0 16px rgba(144, 164, 174, 0.9), 0 0 24px rgba(144, 164, 174, 0.9) !important;
        animation-play-state: paused;
    }
    .back-step-button:active {
        transform: translateY(1px);
        box-shadow: 0 1px 2px rgba(38, 50, 56, 0.3), inset 0 1px 3px rgba(0,0,0,0.2) !important;
    }
    .resolution-radio div[data-baseweb="radiogroup"] {
        display: flex;
        gap: 0.6rem;
    }
    .resolution-radio label[data-baseweb="radio"] {
        flex: 1;
        border: 2px solid #1d6fa5;
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
        background: #f0f4ff;
        color: #0f3057;
        font-weight: 600;
        justify-content: center;
    }
    .resolution-radio label[data-baseweb="radio"] input {
        display: none;
    }
    .resolution-radio label[data-baseweb="radio"]:has(input:checked) {
        background: #00b894;
        color: #ffffff;
        border-color: #00b894;
    }
    .resolution-radio label[data-baseweb="radio"]:has(input:checked):last-of-type {
        background: #d63031;
        border-color: #d63031;
    }
    .spinner-tip {
        margin-top: 0.6rem;
        padding: 0.55rem 0.95rem;
        border-radius: 12px;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(147, 197, 253, 0.4);
        color: #e2e8f0;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.35);
        animation: pulseGlow 2s ease-in-out infinite;
    }
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 6px rgba(15, 23, 42, 0.5); }
        50% { box-shadow: 0 0 18px rgba(96, 165, 250, 0.9); }
        100% { box-shadow: 0 0 6px rgba(15, 23, 42, .5); }
    }
    .stSpinner > div > div {
        border-top-color: #f59e0b !important;
        border-right-color: #f97316 !important;
        border-bottom-color: #f59e0b !important;
        border-left-color: rgba(249, 115, 22, 0.4) !important;
    }
    @keyframes titleFloat {
        0%, 100% { transform: translateY(0) scale(1); }
        50% { transform: translateY(-10px) scale(1.01); }
    }
    @keyframes auroraBreathe {
        0% { transform: rotate(0deg); opacity: 0.7; }
        50% { opacity: 0.95; }
        100% { transform: rotate(360deg); opacity: 0.7; }
    }
    @keyframes scanlineSweep {
        0% { transform: translateX(-140%) rotate(5deg); opacity: 0; }
        30% { opacity: 0.8; }
        70% { opacity: 0.8; }
        100% { transform: translateX(160%) rotate(5deg); opacity: 0; }
    }
    @keyframes ringPulse {
        0% { transform: translate(-50%, -50%) rotate(0deg) scale(1); opacity: 0.5; }
        50% { transform: translate(-50%, -50%) rotate(180deg) scale(1.05); opacity: 0.8; }
        100% { transform: translate(-50%, -50%) rotate(360deg) scale(1); opacity: 0.5; }
    }
    @keyframes particleDrift {
        0% { transform: translateY(0) scale(0.8); opacity: 0; }
        15% { opacity: 1; }
        100% { transform: translateY(-120px) scale(1.3); opacity: 0; }
    }
    @keyframes cometTrail {
        0% { transform: translateX(0); opacity: 0; }
        20% { opacity: 0.9; }
        60% { opacity: 0.7; }
        100% { transform: translateX(220%); opacity: 0; }
    }
    @keyframes textGlow {
        0%, 100% { text-shadow: 0 4px 12px rgba(4, 10, 28, 0.6), 0 0 10px rgba(148, 163, 184, 0.5); }
        50% { text-shadow: 0 6px 18px rgba(4, 10, 28, 0.75), 0 0 16px rgba(59, 130, 246, 0.55); }
    }
    @keyframes textAurora {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes taglineGradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes taglinePulse {
        0%, 100% { opacity: 0.25; filter: blur(4px); }
        50% { opacity: 0.6; filter: blur(6px); }
    }
    @keyframes neonBlink {
        0%, 75%, 100% { text-shadow: 0 0 12px rgba(59, 130, 246, 0.45), 0 0 25px rgba(236, 72, 153, 0.38); }
        78% { text-shadow: 0 0 20px rgba(255, 255, 255, 0.9), 0 0 38px rgba(251, 191, 36, 0.75); }
        85% { text-shadow: 0 0 8px rgba(148, 163, 184, 0.3), 0 0 16px rgba(14, 165, 233, 0.35); }
    }
    @keyframes outlinePulse {
        0%, 100% { opacity: 0.4; filter: drop-shadow(0 0 8px rgba(45, 212, 191, 0.35)); }
        50% { opacity: 0.8; filter: drop-shadow(0 0 22px rgba(94, 234, 212, 0.65)); }
    }
    @keyframes hueShift {
        0% { -webkit-text-stroke-color: rgba(94, 234, 212, 0.75); }
        33% { -webkit-text-stroke-color: rgba(14, 165, 233, 0.75); }
        66% { -webkit-text-stroke-color: rgba(236, 72, 153, 0.75); }
        100% { -webkit-text-stroke-color: rgba(94, 234, 212, 0.75); }
    }
    @keyframes highlightSweep {
        0% { background-position: -120% 0; opacity: 0; }
        20% { opacity: 0.6; }
        80% { opacity: 0.6; }
        100% { background-position: 220% 0; opacity: 0; }
    }
    @keyframes colorWarp {
        0% { filter: hue-rotate(0deg); opacity: 0.25; }
        50% { filter: hue-rotate(160deg); opacity: 0.55; }
        100% { filter: hue-rotate(320deg); opacity: 0.3; }
    }
    @keyframes textGlitch {
        0% { clip-path: inset(0 0 0 0); }
        20% { clip-path: inset(4% 0 15% 0); }
        40% { clip-path: inset(15% 0 4% 0); }
        60% { clip-path: inset(0 0 0 0); }
        80% { clip-path: inset(8% 0 12% 0); }
        100% { clip-path: inset(0 0 0 0); }
    }
    @keyframes symbolPulse {
        0%, 100% { transform: scale(1) rotate(0deg); }
        50% { transform: scale(1.08) rotate(-6deg); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def render_dark_mode_guidelines() -> None:
    st.markdown(DARK_THEME_STYLE, unsafe_allow_html=True)
    expander_label = "Enable dark mode for the best viewing experience — click here for guidelines"
    with st.expander(expander_label, expanded=False):
        st.markdown(
            """
            <div class='dark-mode-card'>
                <strong>Dark mode recommended</strong><br/>
                Change your browser or device theme to dark for the best neon/glow effect and longer battery life on OLED devices.<br/><br/>
                <strong>How to change your device/system theme</strong>
                <ul style="margin:0.3rem 0 0.1rem 1rem;">
                    <li>Windows: Settings &gt; Personalization &gt; Colors &gt; Choose your mode → Dark</li>
                    <li>macOS: System Settings &gt; Appearance &gt; Dark</li>
                    <li>Android: Quick Settings tray (Dark theme) or Settings &gt; Display &gt; Dark theme</li>
                    <li>iOS/iPadOS: Settings &gt; Display & Brightness &gt; Appearance → Dark</li>
                </ul>
                <strong>How to change your browser theme</strong>
                <ul style="margin:0.3rem 0 0.1rem 1rem;">
                    <li>Chrome: chrome://settings/appearance → “Mode” dropdown → Dark</li>
                    <li>Edge: edge://settings/appearance → “Theme” dropdown → Dark</li>
                    <li>Firefox: about:addons → Themes → Dark</li>
                    <li>Safari: inherits the macOS appearance set above</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


render_dark_mode_guidelines()

title_colors = {
    "yellow": "#ffd166",
    "green": "#06d6a0",
    "blue": "#118ab2",
    "red": "#ef476f",
}
title_color = "linear-gradient(135deg, #00c9ff, #92fe9d)"
title_text = "✨ AI driven Interactive Troubleshooting - Automated Flow"
title_block = f"""
<div class='main-title'>
  <div class='title-neon-ring'></div>
  <div class='title-particles'>
    <span></span>
    <span></span>
    <span></span>
    <span></span>
    <span></span>
  </div>
  <div class='title-comet'></div>
  <div class='title-symbol'>⚡</div>
  <div class='title-core' data-text="{title_text}"><span>{title_text}</span></div>
</div>
"""
st.markdown(title_block, unsafe_allow_html=True)
st.markdown(
    "<div class='sub-caption'>Shape troubleshooting decisions at every step to reduce RR and MPD and deliver a more reliable repair experience for customers.</div>",
    unsafe_allow_html=True,
)

if ACCESS_PIN and not st.session_state.get("access_granted"):
    pin_in = st.text_input("Access PIN (after entering the password, press Enter)", type="password")
    if pin_in:
        components.html(
            """
            <script>
            const active = window.parent.document.activeElement;
            if (active && active.tagName === 'INPUT') {
                active.blur();
            }
            </script>
            """,
            height=0,
        )
    if not pin_in:
        st.stop()
    if pin_in == ACCESS_PIN:
        st.session_state.access_granted = True
        ensure_session_access_token()
        persist_access_token_query_param()
        st.session_state["_scroll_target"] = "top"
        st.session_state["_scroll_anchor"] = "product-selector"
    else:
        st.error("Incorrect PIN. Please try again.")
        st.stop()

selected_product = st.session_state.get("selected_product")
if not selected_product:
    st.markdown("<div id='product-selector'></div>", unsafe_allow_html=True)
    render_product_selector()
    st.stop()
else:
    pill_col, action_col = st.columns([3, 1])
    with pill_col:
        st.markdown(
            f"<div class='selected-product-pill'>Selected Product · {product_label(selected_product)}</div>",
            unsafe_allow_html=True,
        )
    with action_col:
        if st.button("Change product", use_container_width=True):
            clear_query_params_preserving_access_token()
            set_selected_product(None)
            st.rerun()


# -----------------------------
# Load YAML / Flow selection
# -----------------------------
all_flows = discover_flow_files()
available_flows = filter_flows_for_category(all_flows, selected_product)
available_flow_paths = {str(path) for path in available_flows}
selected_flow_path = st.session_state.get("selected_flow_path")
if selected_flow_path and selected_flow_path not in available_flow_paths:
    st.session_state.selected_flow_path = None
    selected_flow_path = None

if not available_flows:
    st.info(
        f"Troubleshooting flows for **{product_label(selected_product)}** are not available yet."
    )
    if st.button("Choose a different product", type="secondary"):
        set_selected_product(None)
        st.rerun()
    st.stop()

if available_flows:
    labels: List[str] = []
    label_to_path: Dict[str, Path] = {}
    seen_labels: Dict[str, int] = {}
    default_index: Optional[int] = None
    for idx, path in enumerate(available_flows):
        friendly = prettify_flow_label(path)
        label = friendly
        if label in seen_labels:
            seen_labels[label] += 1
            label = f"{friendly} ({seen_labels[label]})"
        else:
            seen_labels[label] = 1
        labels.append(label)
        label_to_path[label] = path
        if selected_flow_path and str(path) == selected_flow_path:
            default_index = idx
    # Conditionally add a flashing class if no default is selected (i.e., placeholder will be shown)
    header_class = ""
    if default_index is None:
        header_class = "flashing-issue-header"
        st.markdown('''
        <style>
        @keyframes header-pulse {
            0% { box-shadow: 0 3px 10px rgba(0,0,0,0.15); }
            50% { box-shadow: 0 3px 25px rgba(236, 72, 153, 0.7); }
            100% { box-shadow: 0 3px 10px rgba(0,0,0,0.15); }
        }
        .flashing-issue-header {
            animation: header-pulse 2s ease-in-out infinite;
        }
        </style>
        ''', unsafe_allow_html=True)

    issue_label_color = ISSUE_LABEL_COLOR or "#ef476f"
    issue_label_text_color = "#ffffff"
    st.markdown(
        f"<div class='{header_class}' style='font-size:1.2rem;font-weight:800;color:{issue_label_text_color};background:{issue_label_color};padding:0.5rem 0.9rem;border-radius:10px;margin-top:1.2rem;text-align:center;box-shadow:0 3px 10px rgba(0,0,0,0.15);'>Select troubleshooting issue</div>",
        unsafe_allow_html=True,
    )
    selected_label = None
    if default_index is not None:
        selected_label = st.selectbox(
            "",
            options=labels,
            index=default_index,
            label_visibility="collapsed",
        )
    else:
        selected_label = st.selectbox(
            "",
            options=labels,
            index=None,
            placeholder="Choose an issue to begin",
            label_visibility="collapsed",
        )
    if selected_label:
        chosen_path = label_to_path[selected_label]
        if selected_flow_path != str(chosen_path) or st.session_state.get("tree") is None:
            load_flow_from_path(chosen_path)
else:
    st.warning("No YAML flows found in the workspace. Upload one to begin.")

tree = st.session_state.tree
if not tree:
    st.info("Choose a troubleshooting issue from the list to get started.")
    st.stop()

nodes = tree.get("nodes") or {}
if not nodes:
    st.error("Loaded YAML has no nodes to render.")
    st.stop()

if not st.session_state.node_id or st.session_state.node_id not in nodes:
    st.session_state.node_id = tree.get("start")
    st.session_state.path_total_steps = count_progress_steps(
        nodes, st.session_state.second_visit_mode
    )
    st.session_state.visited_stack = [st.session_state.node_id]

if not st.session_state.visited_stack:
    st.session_state.visited_stack = [st.session_state.node_id]
elif st.session_state.visited_stack[-1] != st.session_state.node_id:
    st.session_state.visited_stack.append(st.session_state.node_id)

current_node_id = st.session_state.get("node_id")
if current_node_id and nodes:
    node = nodes.get(current_node_id)
    if node:
        update_recommended_parts(node)

meta = st.session_state.meta
lang_options = language_options(meta)
lang_index = (
    lang_options.index(st.session_state.lang)
    if st.session_state.lang in lang_options
    else 0
)
lang = st.selectbox("Language", options=lang_options, index=lang_index)
st.session_state.lang = lang
st.markdown(
    f"**Training:** {meta.get('title', 'N/A')} · Version: {meta.get('version', '1.0')} · Tree ID: {meta.get('id', '')}"
)

if render_completion_panel(tree, meta, lang):
    st.stop()

if render_resolution_prompt(tree, lang):
    st.stop()

# Case info
with st.expander("Technician / Case Info", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.case["case_id"] = st.text_input(
            "Case ID", value=st.session_state.case.get("case_id", "")
        )
    with c2:
        st.session_state.case["st_id"] = st.text_input(
            "ST ID", value=st.session_state.case.get("st_id", "")
        )

# Technician visit proof (selfie)
st.markdown(
    "<div class='section-title' style='color:#ffffff;font-weight:700;'>Take your Selfie with product</div>",
    unsafe_allow_html=True,
)
selfie_key = "visit_selfie_capture"
existing_selfie = st.session_state.get("visit_selfie")
if existing_selfie:
    st.success("Selfie captured for this visit.")
    if st.button("Retake selfie"):
        st.session_state.pop("visit_selfie", None)
        st.session_state.pop("visit_selfie_mime", None)
        st.rerun()
else:
    selfie_file = st.camera_input("Capture selfie with product", key=selfie_key)
    if selfie_file:
        encoded_selfie, mime_selfie = b64_of_uploaded(selfie_file)
        st.session_state["visit_selfie"] = encoded_selfie
        st.session_state["visit_selfie_mime"] = mime_selfie
    else:
        st.stop()


# -----------------------------
# Sidebar Progress & Utilities
# -----------------------------
required_nodes = [
    nid for nid, node in nodes.items() if node.get("require_pass", False)
]
required_total = len(required_nodes) or len(nodes)
completed_required = sum(
    1 for nid in required_nodes if st.session_state.passed.get(nid)
)
completed_total = len(st.session_state.passed)
progress_ratio = completed_required / required_total if required_total else 0.0

with st.sidebar:
    st.header("Session Progress")
    st.metric(
        "Steps passed",
        f"{completed_required}/{required_total}",
        delta=f"{completed_total} total",
    )
    st.progress(max(0.0, min(progress_ratio, 1.0)))
    if st.button("Restart Flow"):
        reset_tree_progress(tree)
        st.rerun()

    history_data = st.session_state[SESSION_HISTORY_KEY]
    if history_data:
        st.caption("Recent steps")
        for entry in reversed(history_data[-5:]):
            st.write(f"{entry['timestamp'][11:16]} · {entry['step_label']}")
        download_bytes = json.dumps(
            {"case": st.session_state.case, "history": history_data}, indent=2
        ).encode("utf-8")
        st.download_button(
            "Download session log",
            data=download_bytes,
            file_name=f"fault_tree_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )


# -----------------------------
# Node UI
# -----------------------------
node_id = st.session_state.node_id
node = nodes.get(node_id)
if not node:
    st.error(f"Node '{node_id}' not found in tree.")
    st.stop()

# Anchor for this node so we can scroll to it reliably on mobile
st.markdown(f'<div id="node-{node_id}"></div>', unsafe_allow_html=True)

scroll_target = st.session_state.pop("_scroll_target", "node")
scroll_anchor = st.session_state.pop("_scroll_anchor", None)

selector_map = {
    "node": "section.main",
    "completion": "section.main div[data-testid='stVerticalBlock']:last-child",
    "top": "body",
}
target_selector = selector_map.get(scroll_target, selector_map["node"])

components.html(
    f"""
    <script>
    const anchorId = {json.dumps(scroll_anchor)};
    const selector = "{target_selector}";

    const scrollToAnchor = () => {{
        if (!anchorId) return false;
        const el = window.parent.document.getElementById(anchorId);
        if (!el) return false;
        // Scroll the anchor near the top (mobile friendly)
        el.scrollIntoView({{ block: 'start', behavior: 'instant' }});
        // Nudge a tiny bit to avoid being hidden under any fixed headers
        window.parent.scrollBy(0, -10);
        return true;
    }};

    const scrollFallback = () => {{
        if (selector === "body" || selector === "html") {{
            window.parent.scrollTo({{ top: 0, behavior: 'smooth' }});
            return;
        }}
        const el = window.parent.document.querySelector(selector) || window.parent.document.querySelector("section.main");
        if (el) {{
            const rect = el.getBoundingClientRect();
            const top = Math.max(window.parent.scrollY + rect.top - 10, 0);
            window.parent.scrollTo({{ top, behavior: 'smooth' }});
        }} else {{
            window.parent.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}
    }};

    const doScroll = () => {{
        if (!scrollToAnchor()) {{
            scrollFallback();
        }}
    }};

    window.parent.requestAnimationFrame(doScroll);
    setTimeout(doScroll, 250);
    </script>
    """,
    height=0,
)

path_total_steps = count_progress_steps(nodes, st.session_state.second_visit_mode)
st.session_state.path_total_steps = path_total_steps
current_step_number = len(st.session_state.passed) + (
    0 if st.session_state.passed.get(node_id) else 1
)
current_step_number = max(current_step_number, 1)
st.markdown(
    f"<div class='progress-pill'>Steps: {current_step_number} of {path_total_steps}</div>",
    unsafe_allow_html=True,
)

# Back button
if len(st.session_state.visited_stack) > 1:
    back_clicked = st.button(
        BACK_BUTTON_LABEL, key="back_to_previous_step", use_container_width=True
    )
    apply_button_style_by_label(BACK_BUTTON_LABEL, BACK_BUTTON_CLASS)
    if back_clicked:
        st.session_state.visited_stack.pop()
        st.session_state.node_id = st.session_state.visited_stack[-1]
        st.rerun()
st.subheader(get_prompt(node, lang))
input_col, action_col = st.columns([3, 1], vertical_alignment="top")

ui = node.get("ui") or {}
ctrl = ui.get("control")
val: Any = None
step_extra: Dict[str, Any] = {}
stored_answer = st.session_state.answers.get(node_id, {})
stored_value = stored_answer.get("raw_value", stored_answer.get("value"))
control_key = f"{INPUT_VALUE_PREFIX}{node_id}"

ev_required, ev_capture, ev_meta = require_evidence(node)

with input_col:
    if ctrl == "confirm":
        ensure_widget_state(
            control_key, stored_value if stored_value is not None else "Done"
        )
        val = st.radio("Confirm", options=["Done"], horizontal=True, key=control_key)
    elif ctrl == "numeric":
        rng = ui.get("range") or [None, None]
        min_value = float(rng[0]) if rng[0] is not None else None
        max_value = float(rng[1]) if rng[1] is not None else None
        decimals = ui.get("decimals") or 0
        step = 1.0 if decimals else 1
        default_value = stored_value if stored_value is not None else ui.get("default")
        if default_value is None:
            default_value = min_value if min_value is not None else 0.0
        try:
            default_value = float(default_value)
        except (TypeError, ValueError):
            default_value = 0.0
        ensure_widget_state(control_key, default_value)
        val = st.number_input(
            f"Enter value ({ui.get('units', '')})",
            min_value=min_value,
            max_value=max_value,
            step=step,
            format=f"%.{decimals}f",
            key=control_key,
        )
        if (node.get("validation") or {}).get("within_percent_of_label"):
            label_key = f"{INPUT_LABEL_PREFIX}{node_id}"
            label_default = stored_answer.get("label_value", 0.0)
            ensure_widget_state(label_key, label_default)
            step_extra["label_value"] = st.number_input(
                f"Enter label/spec value ({ui.get('units', '')})",
                min_value=0.0,
                step=step,
                format=f"%.{decimals}f",
                key=label_key,
            )
    elif ctrl == "radio":
        options = ui.get("options") or node.get("options") or []
        if options:
            default_option = stored_value if stored_value in options else options[0]
            ensure_widget_state(control_key, default_option)
            val = st.radio("Select one", options=options, key=control_key)
        else:
            st.warning("No options configured for this step; please enter a note.")
            ensure_widget_state(control_key, stored_value if stored_value is not None else "")
            val = st.text_input("Manual response", key=control_key)
    elif ctrl == "chips":
        options = ui.get("options") or []
        default_multi = stored_value if stored_value is not None else []
        if not isinstance(default_multi, list):
            default_multi = [default_multi]
        ensure_widget_state(control_key, default_multi)
        val = st.multiselect("Select all that apply", options=options, key=control_key)
    elif ctrl == "timer":
        secs = int(ui.get("seconds", 60))
        timer_key = f"{SESSION_TIMER_PREFIX}{node_id}"
        if st.session_state.get(timer_key) is None:
            if st.button(f"Start {secs}s timer"):
                st.session_state[timer_key] = int(time.time())
                st.session_state["_scroll_target"] = "node"
                st.rerun()
            elapsed = 0
            remaining = secs
            st.progress(0.0)
            st.write(f"Elapsed: 0s / {secs}s")
            st.write(f"Remaining: {remaining}s")
        else:
            elapsed = timer_elapsed_for(node_id)
            st.progress(min(1.0, elapsed / max(secs, 1)))
            st.write(f"Elapsed: {elapsed}s / {secs}s")
            remaining = max(0, secs - elapsed)
            st.write(f"Remaining: {remaining}s")
            cols = st.columns(2)
            if cols[0].button("Reset timer"):
                st.session_state[timer_key] = None
                st.session_state["_scroll_target"] = "completion"
                st.rerun()
            if cols[1].button("Complete timer"):
                elapsed = secs
                st.session_state[timer_key] = int(time.time()) - secs
                st.rerun()
            if remaining > 0:
                time.sleep(1)
                st.rerun()
        val = "TimerDone" if st.session_state.get(timer_key) and elapsed >= secs else None
    else:
        st.info("No control configured; mark this step complete when ready.")
        default_checkbox = (
            stored_value
            if isinstance(stored_value, bool)
            else (bool(stored_value) if stored_value is not None else False)
        )
        ensure_widget_state(control_key, default_checkbox)
        val = st.checkbox("Step completed?", key=control_key)

    validation_cfg = node.get("validation") or {}
    if validation_cfg.get("confirm_required") and ctrl != "confirm":
        confirm_key = f"confirm_{node_id}"
        ensure_widget_state(confirm_key, False)
        step_extra["confirm_ack"] = st.checkbox(
            "Confirm step completed", key=confirm_key
        )

    photo_b64 = None
    photo_mime = None
    evidence_required_now = ev_required and EVIDENCE_COLLECTION_ENABLED
    if ev_required and not EVIDENCE_COLLECTION_ENABLED:
        pass
    elif evidence_required_now:
        st.markdown("**Evidence required:**")
        if ev_capture == "video":
            st.caption("Video requested; capture photo for now.")
        prompt = (ev_meta.get("instructions") or {}).get(lang) or "Capture evidence"
        camera_file = None
        if CAMERA_CAPTURE_ENABLED:
            camera_file = st.camera_input(prompt)
        if camera_file is None:
            uploader = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"])
            camera_file = uploader
        if camera_file:
            photo_b64, photo_mime = b64_of_uploaded(camera_file)

with action_col:
    go_next = st.button(
        "Submit Step", type="primary", use_container_width=True, key=f"submit_{node_id}"
    )

if go_next:
    tip_placeholder = st.empty()
    with jeeves_spinner("🚀 Syncing your step with Jeeves Cloud...", SPINNER_COLOR):
        tip_placeholder.markdown(
            "<div class='spinner-tip'>✨ Uploading evidence, updating logs, and loading the next action...</div>",
            unsafe_allow_html=True,
        )
        elapsed = None
        if ctrl == "timer":
            elapsed = timer_elapsed_for(node_id)

        ok, message = validate_node(node, val, elapsed_sec=elapsed, extra=step_extra)
        if not ok:
            st.error(message)
        elif evidence_required_now and not photo_b64:
            st.error("Evidence required - please capture or upload a photo.")
        else:
            logged_value = format_answer_value(ctrl, val, elapsed_sec=elapsed, ui=ui)
            answers_payload = {
                "value": logged_value,
                "raw_value": val,
                "label_value": step_extra.get("label_value"),
                "elapsed_sec": elapsed,
            }
            st.session_state.passed[node_id] = True
            st.session_state.answers[node_id] = answers_payload
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "step_id": node_id,
                "step_label": step_label(node, lang),
                "value": logged_value,
                "elapsed_sec": elapsed,
                "photo_attached": bool(photo_b64),
            }
            log_local_step(log_entry)

            sku_value = st.session_state.case.get("sku", "") or "NA"
            answers_for_payload = {
                key: value for key, value in answers_payload.items() if key != "raw_value"
            }
            payload = {
                "flow_id": meta.get("id", ""),
                "case_id": st.session_state.case.get("case_id", ""),
                "sku": sku_value,
                "st_id": st.session_state.case.get("st_id", ""),
                "step_id": node_id,
                "step_label": step_label(node, lang),
                "answers": answers_for_payload,
                "answer_value": answers_for_payload.get("value"),
                "label_value": answers_for_payload.get("label_value"),
                "elapsed_sec": answers_for_payload.get("elapsed_sec"),
                "pass": True,
                "photo_b64": photo_b64,
                "photo_mime": photo_mime,
                "finalize": False,
                "all_steps_valid": False,
                "token_pattern": (meta.get("gating") or {}).get(
                    "token_pattern", "{FAULT}-{SKU}-{RAND5}"
                ),
                "fault_code": fault_code_from_meta(meta.get("id", "")),
            }
            # Buffer this step payload to be logged later at finalization
            buffer = st.session_state.get("p2o_step_buffer", [])
            buffer.append(payload)
            st.session_state.p2o_step_buffer = buffer

            # Capture any recommended parts from this node into parts_used
            single_part = node.get("recommends_part")
            if isinstance(single_part, str) and single_part:
                st.session_state.parts_used.add(single_part)

            multiple_parts = node.get("recommends_parts")
            if isinstance(multiple_parts, list):
                for part in multiple_parts:
                    if isinstance(part, str) and part:
                        st.session_state.parts_used.add(part)

            prompt_allowed = wants_resolution_prompt(node)
            if node_id == tree.get("start") and isinstance(val, str):
                st.session_state.second_visit_mode = (
                    "Second visit - Replace ordered part" in val
                )
                st.session_state.path_total_steps = count_progress_steps(
                    nodes, st.session_state.second_visit_mode
                )

            selection = val if isinstance(val, str) else None
            selected_list = val if isinstance(val, list) else []
            next_id = eval_branches(node, selection, selected_list) or node.get("next")
            reset_timer_for(node_id)
            final_entry = None
            if not next_id:
                gating = meta.get("gating") or {}
                require_all = bool(gating.get("require_all_steps", True))
                all_valid = True
                if require_all:
                    for nid, n in nodes.items():
                        if n.get("require_pass") and not st.session_state.passed.get(nid):
                            all_valid = False
                            break
                st.session_state.path_total_steps = max(
                    st.session_state.path_total_steps, current_step_number
                )
                final_entry = {"final_all_valid": all_valid}
            if node.get("force_restart"):
                st.session_state.flow_status = {
                    "type": "restart",
                    "node_id": node_id,
                }
                st.rerun()
            if prompt_allowed:
                st.session_state.pending_resolution = {
                    "prev_node": node_id,
                    "next_node": next_id,
                    "force_restart": bool(node.get("force_restart")),
                    "second_visit_mode": bool(st.session_state.second_visit_mode),
                    **(final_entry or {}),
                }
                st.rerun()
            else:
                if next_id:
                    st.session_state.visited_stack.append(next_id)
                    st.session_state.node_id = next_id
                    st.session_state["_scroll_anchor"] = f"node-{next_id}"
                    st.session_state["_scroll_target"] = "top"
                    st.rerun()
                else:
                    st.session_state.flow_status = {
                        "type": "completed",
                        "node_id": node_id,
                        "all_valid": (final_entry or {}).get("final_all_valid", True),
                    }
                    st.session_state["_scroll_target"] = "top"
                    st.rerun()
    tip_placeholder.empty()
