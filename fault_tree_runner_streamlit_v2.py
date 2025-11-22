import os
import time
import base64
import json
import re
import html
import uuid
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_LANG = "en"
SESSION_HISTORY_KEY = "fault_history"
SESSION_TIMER_PREFIX = "timer_start_"
INPUT_VALUE_PREFIX = "ctrl_value_"
INPUT_LABEL_PREFIX = "ctrl_label_"
BASE_DIR = Path(__file__).resolve().parent
CAMERA_CAPTURE_ENABLED = (
    os.getenv("ENABLE_CAMERA_CAPTURE", "false").strip().lower() == "true"
)
EVIDENCE_COLLECTION_ENABLED = (
    os.getenv("ENABLE_EVIDENCE_UPLOAD", "false").strip().lower() == "true"
)
RECOMMENDED_PARTS_KEY = "recommended_parts"
ISSUE_LABEL_COLOR = os.getenv("ISSUE_LABEL_COLOR", "#ef476f")


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
    st.session_state.setdefault("pending_resolution", None)
    st.session_state.setdefault("selected_flow_path", None)


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


def load_flow_from_path(flow_path: Path) -> None:
    try:
        yaml_data = yaml.safe_load(flow_path.read_text(encoding="utf-8"))
        tree = normalize_tree(yaml_data)
        apply_loaded_tree(tree, str(flow_path))
    except Exception as exc:
        st.error(f"Failed to load flow '{flow_path.name}': {exc}")


def load_flow_from_upload(upload) -> None:
    try:
        yaml_data = yaml.safe_load(upload.read().decode("utf-8"))
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
    recommended_parts = meta.get(RECOMMENDED_PARTS_KEY) or []
    if "part_photos" not in st.session_state:
        st.session_state.part_photos = {}
    part_photos = st.session_state.part_photos

    photos_required = recommended_parts if (recommended_parts and status != "resolved") else []
    all_photos_ready = True

    if photos_required:
        st.warning(
            "Recommended parts to order if symptoms persist: "
            + ", ".join(recommended_parts)
        )
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
                        st.session_state.pop(f"cam_pref_{slug}", None)
                        st.rerun()
                else:
                    upload = None
                    cam = None
                    cam_pref_key = f"cam_pref_{slug}"
                    use_camera = st.session_state.get(cam_pref_key, False)
                    if CAMERA_CAPTURE_ENABLED:
                        if use_camera:
                            cam = st.camera_input(f"Capture {part}", key=f"cam_{slug}")
                            if st.button(
                                f"Use gallery/upload for {part}", key=f"closecam_{slug}"
                            ):
                                st.session_state[cam_pref_key] = False
                                st.rerun()
                        else:
                            st.caption(
                                "Need a live photo? Tap below to open your device camera."
                            )
                            if st.button(
                                f"Use camera for {part}", key=f"opencam_{slug}"
                            ):
                                st.session_state[cam_pref_key] = True
                                st.rerun()
                    upload = cam
                    if upload is None:
                        upload = st.file_uploader(
                            f"Upload photo for {part}",
                            type=["jpg", "jpeg", "png"],
                            key=f"upload_{slug}",
                        )
                    if upload:
                        encoded, mime = b64_of_uploaded(upload)
                        if encoded:
                            part_photos[part] = {
                                "photo_b64": encoded,
                                "photo_mime": mime,
                            }
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
            sku_value = st.session_state.case.get("sku", "") or "NA"
            payload = {
                "flow_id": meta.get("id", ""),
                "case_id": st.session_state.case.get("case_id", ""),
                "sku": sku_value,
                "st_id": st.session_state.case.get("st_id", ""),
                "step_id": node_id or "",
                "step_label": step_label(node, lang) if node else "",
                "answers": st.session_state.answers.get(node_id, {}),
                "pass": True,
                "photo_b64": None,
                "photo_mime": None,
                "finalize": True,
                "all_steps_valid": flow_status.get("all_valid", False),
                "token_pattern": (meta.get("gating") or {}).get(
                    "token_pattern", "{FAULT}-{SKU}-{RAND5}"
                ),
                "fault_code": fault_code_from_meta(meta.get("id", "")),
                "parts_used": sorted(st.session_state.parts_used),
                "resolution": status,
                "part_photos": part_photos if photos_required else {},
            }

            resp = None
            try:
                resp = post_step_log(P2O_ENDPOINT, payload)
                if resp.get("ok", True):
                    token = resp.get("token", "(no token — endpoint not set)")
                    st.session_state.final_token = token
                    st.session_state.flow_status["finalized"] = True
                    st.success("Gate token generated successfully.")
                    render_token_copy(token)
                    st.caption("Paste this token in Strider Notes until API integration.")
                else:
                    # non-OK from endpoint
                    base_err = resp.get("error") or "Finalize call failed."
                    detail = resp.get("text") or resp.get("status_code")
                    detail_msg = f"{base_err} ({detail})" if detail else base_err
                    st.error(f"Finalize failed: {detail_msg}")
            except Exception as e:
                # hard failure (network, JSON, etc.)
                base_err = str(e)
                detail = None
                if isinstance(resp, dict):
                    detail = resp.get("text") or resp.get("status_code")
                detail_msg = f"{base_err} ({detail})" if detail else base_err
                st.error(f"Finalize failed: {detail_msg}")
    else:
        token = token or "(no token ? endpoint not set)"
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

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #0a437c;
        margin-bottom: 0.1rem;
    }
    .sub-caption {
        font-size: 0.95rem;
        color: #1d6fa5;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #0f3057;
        margin-top: 1.2rem;
        margin-bottom: 0.4rem;
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
        100% { box-shadow: 0 0 6px rgba(15, 23, 42, 0.5); }
    }
    .stSpinner > div > div {
        border-top-color: #f59e0b !important;
        border-right-color: #f97316 !important;
        border-bottom-color: #f59e0b !important;
        border-left-color: rgba(249, 115, 22, 0.4) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

title_colors = {
    "yellow": "#ffd166",
    "green": "#06d6a0",
    "blue": "#118ab2",
    "red": "#ef476f",
}
title_color = title_colors.get(os.getenv("FLOW_SELECT_COLOR", "yellow").lower(), "#ffd166")
st.markdown(
    f"<div class='main-title' style='background:{title_color};padding:0.6rem 1.2rem;border-radius:10px;text-align:center;margin:0.2rem auto 0.6rem auto;max-width:760px;font-size:1.4rem;letter-spacing:0.03em;'>AI based Interactive Troubleshooting - Automated Flow</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='sub-caption' style='color:#ffffff;font-weight:700;text-align:center;'>Train decision-making, not just steps. Logs MPD/RR risk proxies for coaching.</div>",
    unsafe_allow_html=True,
)

if ACCESS_PIN:
    pin_in = st.text_input("Access PIN", type="password")
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
    if pin_in != ACCESS_PIN:
        st.stop()


# -----------------------------
# Load YAML / Flow selection
# -----------------------------
available_flows = discover_flow_files()
selected_flow_path = st.session_state.get("selected_flow_path")

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
    issue_label_color = ISSUE_LABEL_COLOR or "#ef476f"
    issue_label_text_color = "#ffffff"
    st.markdown(
        f"<div style='font-size:1.2rem;font-weight:800;color:{issue_label_text_color};background:{issue_label_color};padding:0.5rem 0.9rem;border-radius:10px;margin-top:1.2rem;text-align:center;box-shadow:0 3px 10px rgba(0,0,0,0.15);'>Select troubleshooting issue</div>",
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
st.subheader(get_prompt(node, lang))
input_col, action_col = st.columns([3, 1], vertical_alignment="top")

ui = node.get("ui") or {}
ctrl = ui.get("control")
val: Any = None
step_extra: Dict[str, Any] = {}
stored_answer = st.session_state.answers.get(node_id, {})
control_key = f"{INPUT_VALUE_PREFIX}{node_id}"

ev_required, ev_capture, ev_meta = require_evidence(node)

with input_col:
    if ctrl == "confirm":
        ensure_widget_state(control_key, stored_answer.get("value", "Done"))
        val = st.radio("Confirm", options=["Done"], horizontal=True, key=control_key)
    elif ctrl == "numeric":
        rng = ui.get("range") or [None, None]
        min_value = float(rng[0]) if rng[0] is not None else None
        max_value = float(rng[1]) if rng[1] is not None else None
        decimals = ui.get("decimals") or 0
        step = 1.0 if decimals else 1
        default_value = stored_answer.get("value", ui.get("default"))
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
            default_option = (
                stored_answer.get("value") if stored_answer.get("value") in options else options[0]
            )
            ensure_widget_state(control_key, default_option)
            val = st.radio("Select one", options=options, key=control_key)
        else:
            st.warning("No options configured for this step; please enter a note.")
            ensure_widget_state(control_key, stored_answer.get("value", ""))
            val = st.text_input("Manual response", key=control_key)
    elif ctrl == "chips":
        options = ui.get("options") or []
        default_multi = stored_answer.get("value", [])
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
        ensure_widget_state(control_key, stored_answer.get("value", False))
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
    go_back = st.button(
        "Previous Step",
        disabled=len(st.session_state.visited_stack) <= 1,
        use_container_width=True,
        key=f"back_{node_id}",
    )
    go_next = st.button(
        "Submit Step", type="primary", use_container_width=True, key=f"submit_{node_id}"
    )

if go_back and len(st.session_state.visited_stack) > 1:
    st.session_state.visited_stack.pop()
    st.session_state.node_id = st.session_state.visited_stack[-1]
    st.rerun()

from contextlib import contextmanager

@contextmanager
def jeeves_spinner(text="Syncing...", color="#e45614"):
    html = f"""
    <style>
    .jeeves-spin::before {{
      content: "";
      width: 22px; height: 22px; margin-right: .5rem;
      border: 3px solid rgba(255,255,255,0.25);
      border-top-color: {color};
      border-radius: 50%; display: inline-block; vertical-align: middle;
      animation: jeeves-spin 0.75s linear infinite;
    }}
    @keyframes jeeves-spin {{ to {{ transform: rotate(360deg); }} }}
    </style>
    <div class="jeeves-spin"></div><span>{text}</span>
    """
    ph = st.empty()
    ph.markdown(html, unsafe_allow_html=True)
    try: yield
    finally: ph.empty()

if go_next:
    
    with jeeves_spinner("🚀 Syncing your step with Jeeves Cloud...", "#00c8ff"):
        st.markdown(
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
            st.session_state.passed[node_id] = True
            st.session_state.answers[node_id] = {
                "value": val,
                "label_value": step_extra.get("label_value"),
                "elapsed_sec": elapsed,
            }
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "step_id": node_id,
                "step_label": step_label(node, lang),
                "value": val,
                "elapsed_sec": elapsed,
                "photo_attached": bool(photo_b64),
            }
            log_local_step(log_entry)

            sku_value = st.session_state.case.get("sku", "") or "NA"
            payload = {
                "flow_id": meta.get("id", ""),
                "case_id": st.session_state.case.get("case_id", ""),
                "sku": sku_value,
                "st_id": st.session_state.case.get("st_id", ""),
                "step_id": node_id,
                "step_label": step_label(node, lang),
                "answers": st.session_state.answers.get(node_id, {}),
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
            resp = post_step_log(P2O_ENDPOINT, payload)
            if not resp.get("ok", True):
                st.warning(f"Log post failed: {resp.get('error')}")

            part_tag = node.get("part_tag")
            if part_tag:
                st.session_state.parts_used.add(part_tag)

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
