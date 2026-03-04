"""
DocExtract UI

Flow:
  1. User uploads a PDF/image
  2. App detects supplier + doc type via /v1/suppliers/detect
  3a. Template EXISTS  → show available fields as checkboxes → user picks → extract
  3b. Template MISSING → warn user → let them upload a YAML template →
                         save it → show fields → extract same PDF
  4. Poll job status → display results
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL_SEC", "1.0"))
POLL_TIMEOUT = float(os.getenv("POLL_TIMEOUT_SEC", "120.0"))


# ── API helpers ───────────────────────────────────────────────────────────────


def _detect(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    r = requests.post(
        f"{API_BASE_URL}/v1/suppliers/detect",
        files={"file": (filename, file_bytes)},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _upload_template(
    party_id: str, yaml_bytes: bytes, yaml_name: str
) -> Dict[str, Any]:
    r = requests.post(
        f"{API_BASE_URL}/v1/suppliers/{party_id}/template",
        files={"file": (yaml_name, yaml_bytes, "application/x-yaml")},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _extract(
    file_bytes: bytes,
    filename: str,
    document_type: str,
    supplier_id: Optional[str],
    requested_fields: List[str],
) -> Dict[str, Any]:
    data: Dict[str, Any] = {"document_type": document_type}
    if supplier_id:
        data["supplier_id"] = supplier_id
    if requested_fields:
        data["requested_fields"] = json.dumps(requested_fields)
    r = requests.post(
        f"{API_BASE_URL}/v1/extract",
        files={"file": (filename, file_bytes)},
        data=data,
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def _get_job(job_id: str) -> Dict[str, Any]:
    r = requests.get(f"{API_BASE_URL}/v1/jobs/{job_id}", timeout=30)
    r.raise_for_status()
    return r.json()


def _get_artifacts(document_id: str) -> List[Dict]:
    r = requests.get(f"{API_BASE_URL}/v1/documents/{document_id}/artifacts", timeout=30)
    r.raise_for_status()
    return r.json().get("artifacts", [])


def _format_fields(result: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, fr in (result.get("fields") or {}).items():
        rows.append(
            {
                "field": name,
                "value": fr.get("value"),
                "confidence": fr.get("confidence"),
                "status": fr.get("status"),
                "method": fr.get("method"),
                "page": fr.get("page"),
                "evidence": fr.get("evidence"),
                "bbox": fr.get("bbox"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["status", "confidence"], ascending=[True, False])
    return df


def _pipeline_steps(status: str) -> str:
    steps = {
        "pending": "✅ Uploaded → ⏳ Queued → ⬜ Processing → ⬜ Complete",
        "processing": "✅ Uploaded → ✅ Queued → ⏳ Processing → ⬜ Complete",
        "completed": "✅ Uploaded → ✅ Queued → ✅ Processing → ✅ Complete",
        "failed": "✅ Uploaded → ✅ Queued → ✅ Processing → ❌ Failed",
    }
    return steps.get(status, status)


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="DocExtract", layout="wide")
st.title("DocExtract")
st.caption(f"API: {API_BASE_URL}")

# Session state keys
for key in ["detect_result", "last_submit", "selected_fields", "template_saved"]:
    if key not in st.session_state:
        st.session_state[key] = None

show_raw = st.sidebar.checkbox("Show raw API responses", value=False)
st.sidebar.divider()
poll_interval = st.sidebar.slider("Poll interval (s)", 0.3, 5.0, POLL_INTERVAL, 0.1)
poll_timeout = st.sidebar.slider("Timeout (s)", 10, 600, int(POLL_TIMEOUT), 10)

# ── System status panel ───────────────────────────────────────────────────────
st.sidebar.divider()
with st.sidebar.expander("🖥 System status", expanded=False):
    if "system_status" not in st.session_state:
        st.session_state["system_status"] = None

    if st.button("Refresh status", key="refresh_status"):
        try:
            r = requests.get(f"{API_BASE_URL}/v1/status", timeout=4)
            r.raise_for_status()
            st.session_state["system_status"] = r.json()
        except Exception as e:
            st.session_state["system_status"] = {"_error": str(e)}

    # Auto-fetch only once on first load (when state is None)
    if st.session_state["system_status"] is None:
        try:
            r = requests.get(f"{API_BASE_URL}/v1/status", timeout=4)
            r.raise_for_status()
            st.session_state["system_status"] = r.json()
        except Exception as e:
            st.session_state["system_status"] = {"_error": str(e)}

    status_data = st.session_state.get("system_status") or {}

    if "_error" in status_data:
        st.warning(f"⚠️ API unreachable\n\n`{status_data['_error']}`")
        st.caption(f"Expects API at: `{API_BASE_URL}`")
        st.caption(
            "If running via Docker, set `API_BASE_URL=http://api:8000` in the Streamlit service env."
        )
    elif not status_data:
        st.info("Click **Refresh status** to check services.")
    else:
        overall = status_data.get("overall", "unknown")
        st.markdown(f"**Overall: {'🟢 OK' if overall == 'ok' else '🔴 Degraded'}**")

        def _badge(svc_status: str) -> str:
            return (
                "🟢"
                if svc_status == "ok"
                else ("🟡" if svc_status == "no_workers" else "🔴")
            )

        redis_s = status_data.get("redis", {})
        st.markdown(
            f"{_badge(redis_s.get('status', '?'))} **Redis** — {redis_s.get('status', '?')}"
        )
        if redis_s.get("detail"):
            st.caption(redis_s["detail"])

        cel = status_data.get("celery", {})
        cel_status = cel.get("status", "?")
        workers = cel.get("workers", [])
        st.markdown(
            f"{_badge(cel_status)} **Celery** — {cel_status} ({len(workers)} worker{'s' if len(workers) != 1 else ''})"
        )
        for w in workers:
            st.caption(
                f"  • {w['name']}  |  active: {w['active_tasks']}  |  concurrency: {w['concurrency']}"
            )
        if cel.get("detail"):
            st.caption(cel["detail"])

        pg = status_data.get("postgres", {})
        st.markdown(
            f"{_badge(pg.get('status', '?'))} **PostgreSQL** — {pg.get('status', '?')}"
        )
        if pg.get("detail"):
            st.caption(pg["detail"])

        minio = status_data.get("minio", {})
        st.markdown(
            f"{_badge(minio.get('status', '?'))} **MinIO** — {minio.get('status', '?')}"
        )
        if minio.get("detail"):
            st.caption(minio["detail"])


# ── Step 1: Upload PDF ────────────────────────────────────────────────────────

st.subheader("① Upload document")
uploaded = st.file_uploader(
    "Choose a PDF or image",
    type=["pdf", "png", "jpg", "jpeg"],
    key="doc_upload",
)

if uploaded:
    col_detect, _ = st.columns([1, 3])
    with col_detect:
        detect_btn = st.button("Detect supplier", type="primary")

    if detect_btn:
        with st.spinner("Detecting supplier and document type…"):
            try:
                result = _detect(uploaded.getvalue(), uploaded.name)
                st.session_state["detect_result"] = result
                st.session_state["last_submit"] = None
                st.session_state["selected_fields"] = None
                st.session_state["template_saved"] = None
            except requests.RequestException as e:
                st.error(f"Detection failed: {e}")


# ── Step 2: Template check ────────────────────────────────────────────────────

det = st.session_state["detect_result"]

if det:
    st.divider()
    party_id = det.get("party_id")
    doc_type = det.get("document_type", "invoice")
    has_tmpl = det.get("has_template", False)
    avail = det.get("available_fields", [])

    st.subheader("② Supplier detection")

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Supplier ID",
        party_id or "Unknown",
        delta=f"confidence {det.get('party_confidence', 0):.0%}",
    )
    c2.metric(
        "Document type",
        doc_type,
        delta=f"confidence {det.get('type_confidence', 0):.0%}",
    )
    c3.metric("Template", "✅ Found" if has_tmpl else "❌ Missing")

    if show_raw:
        st.json(det)

    # ── No template → ask user to upload one ─────────────────────────────────
    if not has_tmpl or st.session_state["template_saved"] is False:
        st.divider()
        st.subheader("③ No template found — upload one")

        if not party_id:
            party_id = st.text_input(
                "Supplier ID (used as filename, e.g. `grainger`)",
                placeholder="grainger",
            )

        st.info(
            f"No extraction template exists for **{party_id or 'this supplier'}**. "
            "Upload a YAML template that describes this supplier's document structure. "
            "Once saved, the pipeline will use it to extract fields from this document."
        )

        tmpl_file = st.file_uploader(
            "Upload supplier YAML template",
            type=["yaml", "yml"],
            key="tmpl_upload",
        )

        if tmpl_file and party_id:
            if st.button("Save template", type="primary"):
                with st.spinner("Saving template…"):
                    try:
                        resp = _upload_template(
                            party_id,
                            tmpl_file.getvalue(),
                            tmpl_file.name,
                        )
                        st.success(f"Template saved for **{party_id}**.")
                        avail = resp.get("available_fields", [])
                        # Update detect result in session so we proceed
                        det["has_template"] = True
                        det["available_fields"] = avail
                        det["party_id"] = party_id
                        st.session_state["detect_result"] = det
                        st.session_state["template_saved"] = True
                        st.rerun()
                    except requests.RequestException as e:
                        st.error(f"Failed to save template: {e}")

    # ── Template exists → pick fields ─────────────────────────────────────────
    if det.get("has_template") and avail:
        st.divider()
        st.subheader("③ Select fields to extract")
        st.caption(
            f"These fields are defined in the **{party_id}** template. "
            "Pick the ones you want extracted from this document."
        )

        # Default: all selected
        if st.session_state["selected_fields"] is None:
            st.session_state["selected_fields"] = avail[:]

        cols = st.columns(3)
        selected = []
        for i, fname in enumerate(avail):
            checked = fname in (st.session_state["selected_fields"] or [])
            if cols[i % 3].checkbox(fname, value=checked, key=f"field_{fname}"):
                selected.append(fname)

        st.session_state["selected_fields"] = selected

        col_sel, col_all, col_none = st.columns([2, 1, 1])
        if col_all.button("Select all"):
            st.session_state["selected_fields"] = avail[:]
            st.rerun()
        if col_none.button("Clear all"):
            st.session_state["selected_fields"] = []
            st.rerun()

        st.divider()

        # ── Step 3: Extract ───────────────────────────────────────────────────
        st.subheader("④ Run extraction")

        if not selected:
            st.warning("Select at least one field to extract.")
        else:
            st.write(f"Will extract: `{'`, `'.join(selected)}`")
            if st.button("Start extraction", type="primary", disabled=not uploaded):
                with st.spinner("Submitting job…"):
                    try:
                        payload = _extract(
                            file_bytes=uploaded.getvalue(),
                            filename=uploaded.name,
                            document_type=doc_type,
                            supplier_id=party_id,
                            requested_fields=selected,
                        )
                        st.session_state["last_submit"] = {
                            "submitted_at": datetime.utcnow().isoformat(),
                            "upload": payload,
                            "requested_fields": selected,
                        }
                        st.success("Job submitted!")
                        if show_raw:
                            st.json(payload)
                    except requests.RequestException as e:
                        st.error(f"Submission failed: {e}")


# ── Step 4: Poll + results ────────────────────────────────────────────────────

last = st.session_state.get("last_submit")
if last:
    st.divider()
    st.subheader("⑤ Extraction results")

    job_id = last["upload"].get("job_id")
    document_id = last["upload"].get("document_id")
    req_fields = last.get("requested_fields", [])

    st.caption(
        f"job `{job_id}` · doc `{document_id}` · fields: `{'`, `'.join(req_fields)}`"
    )

    col_auto, col_manual = st.columns([1, 1])
    auto_poll = col_auto.checkbox("Auto-poll", value=True)
    manual_poll = col_manual.button("Refresh now")

    status_box = st.empty()
    steps_box = st.empty()
    error_box = st.empty()

    job_data = None
    start_t = time.time()

    if auto_poll or manual_poll:
        while True:
            try:
                job_data = _get_job(job_id)
            except requests.RequestException as e:
                st.error(f"Failed to fetch job: {e}")
                break

            status = job_data.get("status", "unknown")
            status_box.metric("Status", status)
            steps_box.write(_pipeline_steps(status))

            if status == "failed":
                error_box.error(job_data.get("error") or "Job failed")
                break

            if status == "completed":
                break

            if not auto_poll:
                break

            if time.time() - start_t > poll_timeout:
                st.warning("Polling timed out — click Refresh to check again.")
                break

            time.sleep(poll_interval)

    # Display results
    if job_data and job_data.get("status") == "completed":
        result = job_data.get("result") or {}
        df = _format_fields(result)

        if df.empty:
            st.warning("No fields returned.")
        else:
            st.dataframe(
                df[["field", "value", "confidence", "status", "method", "page"]],
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Field details")
            for _, row in df.iterrows():
                with st.expander(
                    f"{row['field']}  ·  {row['status']}  ·  conf={row['confidence']}"
                ):
                    st.write(
                        {
                            "value": row["value"],
                            "method": row["method"],
                            "page": row["page"],
                        }
                    )
                    if row["evidence"]:
                        st.markdown("**Evidence**")
                        st.code(str(row["evidence"]))
                    if row["bbox"]:
                        st.markdown("**BBox**")
                        st.json(row["bbox"])

        # Artifacts
        st.divider()
        st.markdown("#### Pipeline artifacts")
        try:
            artifacts = _get_artifacts(document_id)
            if artifacts:
                st.dataframe(
                    pd.DataFrame(artifacts), use_container_width=True, hide_index=True
                )
            else:
                st.info("No artifacts stored for this job.")
        except requests.RequestException:
            st.warning("Could not load artifacts.")

        if show_raw:
            st.json(result)

        # ── Pipeline inspector ────────────────────────────────────────────────
        st.divider()
        with st.expander("🔬 Pipeline inspector — step-by-step trace", expanded=False):
            try:
                tr = requests.get(f"{API_BASE_URL}/v1/jobs/{job_id}/trace", timeout=15)
                tr.raise_for_status()
                trace = tr.json()
            except Exception as e:
                st.error(f"Could not load trace: {e}")
                trace = None

            if trace:
                stages = trace.get("stages", {})
                fields_trace = trace.get("fields", {})

                # ── Stage 1: Normalization ────────────────────────────────────
                with st.expander("① Normalization", expanded=False):
                    s1 = stages.get("1_normalization", {})
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Pages", s1.get("pages", "?"))
                    c2.metric("OCR used", "Yes" if s1.get("ocr_used") else "No")
                    c3.metric(
                        "Method", "PaddleOCR" if s1.get("ocr_used") else "pdfminer"
                    )

                    for pg in s1.get("page_detail", []):
                        with st.expander(
                            f"Page {pg['page_no']}  —  {pg['token_count']} tokens  |  {pg['layout_lines']} lines"
                        ):
                            st.markdown("**Full text extracted:**")
                            st.code(pg.get("full_text", ""), language=None)
                            st.markdown(
                                f"Layout columns detected: `{pg.get('layout_columns', 0)}`"
                            )
                            st.markdown(
                                "**First 100 tokens (text · bbox · ocr_conf):**"
                            )
                            tok_df = pd.DataFrame(pg.get("tokens_sample", []))
                            if not tok_df.empty:
                                st.dataframe(
                                    tok_df, use_container_width=True, hide_index=True
                                )

                # ── Stage 2: Detection ────────────────────────────────────────
                with st.expander("② Supplier & doc-type detection", expanded=False):
                    s2 = stages.get("2_detection", {})
                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Supplier detected",
                        s2.get("supplier_id", "None"),
                        delta=f"conf {s2.get('party_confidence', 0):.0%}",
                    )
                    c2.metric(
                        "Doc type detected",
                        s2.get("document_type", "?"),
                        delta=f"conf {s2.get('type_confidence', 0):.0%}",
                    )
                    st.markdown("**First-page text used for detection:**")
                    st.code(s2.get("first_page_text", ""), language=None)

                # ── Stage 3: Template ─────────────────────────────────────────
                with st.expander("③ Template loaded", expanded=False):
                    s3 = stages.get("3_template", {})
                    st.write(f"**Profile:** `{s3.get('profile_id', 'unknown')}`")
                    st.write(
                        f"**All blueprint fields:** `{'`, `'.join(s3.get('all_blueprint_fields', []))}`"
                    )
                    st.write(
                        f"**Requested by user:** `{'`, `'.join(s3.get('requested_fields') or ['(all)'])}`"
                    )
                    st.write(
                        f"**Actually extracted:** `{'`, `'.join(s3.get('fields_to_extract', []))}`"
                    )
                    st.markdown(
                        "**Field configs (anchors · patterns · search window):**"
                    )
                    for fname, fc in (s3.get("field_configs") or {}).items():
                        st.markdown(
                            f"- **{fname}**: anchors=`{fc.get('anchors')}` · window=`{fc.get('search_window')}` · direction=`{fc.get('direction')}`"
                        )

                # ── Stage 4: Per-field extraction ─────────────────────────────
                with st.expander("④ Per-field extraction detail", expanded=True):
                    if not fields_trace:
                        st.info("No per-field trace available.")
                    for fname, ft in fields_trace.items():
                        final = ft.get("final", {})
                        status_icon = {
                            "verified": "✅",
                            "needs_review": "⚠️",
                            "missing": "❌",
                        }.get(final.get("status", ""), "❓")
                        with st.expander(
                            f"{status_icon} **{fname}**  →  `{final.get('value', 'MISSING')}`"
                            f"  ·  conf={final.get('confidence', 0):.3f}"
                            f"  ·  method={final.get('method', '?')}",
                            expanded=False,
                        ):
                            # Anchor candidates
                            st.markdown(
                                f"**Anchor candidates** ({len(ft.get('anchor_candidates', []))} found):"
                            )
                            for i, c in enumerate(ft.get("anchor_candidates", [])):
                                st.markdown(
                                    f"  `{i + 1}`. value=`{c.get('raw_value')}` · "
                                    f"conf=`{c.get('confidence')}` · page=`{c.get('page')}`"
                                )
                                st.caption(f"     snippet: {c.get('snippet', '')}")

                            # Regex candidates
                            st.markdown(
                                f"**Regex candidates** ({len(ft.get('regex_candidates', []))} found):"
                            )
                            for i, c in enumerate(ft.get("regex_candidates", [])):
                                st.markdown(
                                    f"  `{i + 1}`. value=`{c.get('raw_value')}` · "
                                    f"conf=`{c.get('confidence')}` · page=`{c.get('page')}`"
                                )
                                st.caption(f"     snippet: {c.get('snippet', '')}")

                            # Scored ranking
                            st.markdown("**Scored ranking (top 5):**")
                            sc_df = pd.DataFrame(ft.get("scored_candidates", []))
                            if not sc_df.empty:
                                st.dataframe(
                                    sc_df, use_container_width=True, hide_index=True
                                )

                            # Final result
                            st.markdown("**Final result:**")
                            fin = ft.get("final", {})
                            st.json(fin)

                            if fin.get("validation_errors"):
                                st.error(
                                    f"Validation errors: {fin['validation_errors']}"
                                )

                            # LayoutLM
                            if ft.get("layoutlm"):
                                lm = ft["layoutlm"]
                                if lm.get("upgraded"):
                                    st.success(
                                        f"LayoutLM upgraded: {lm['before_conf']} → {lm['after_conf']}  new value=`{lm.get('new_value')}`"
                                    )
                                else:
                                    st.info(
                                        f"LayoutLM ran but did not improve (before={lm.get('before_conf')})"
                                    )

                            # LLM
                            if ft.get("llm") and ft["llm"].get("triggered"):
                                st.warning(
                                    f"LLM fallback used → value=`{ft['llm'].get('value')}` conf={ft['llm'].get('confidence')}"
                                )

                # ── Stage 5: LayoutLM summary ─────────────────────────────────
                with st.expander("⑤ LayoutLM fallback summary", expanded=False):
                    s4 = stages.get("4_layoutlm", {})
                    st.write(f"Used: `{s4.get('used', False)}`")
                    for fname, lf in (s4.get("fields") or {}).items():
                        icon = (
                            "🔼"
                            if lf.get("upgraded")
                            else ("🔁" if lf.get("triggered") else "⏭")
                        )
                        st.markdown(
                            f"{icon} **{fname}**: triggered=`{lf.get('triggered')}` upgraded=`{lf.get('upgraded', False)}`"
                        )

                # ── Stage 6: Validation ───────────────────────────────────────
                with st.expander("⑥ Validation summary", expanded=False):
                    s6 = stages.get("6_validation", {})
                    c1, c2 = st.columns(2)
                    c1.metric(
                        "Required fields present",
                        "✅ Yes" if s6.get("required_present") else "❌ No",
                    )
                    c2.metric(
                        "Overall confidence", f"{s6.get('overall_confidence', 0):.3f}"
                    )
                    if s6.get("required_missing"):
                        st.error(f"Missing required: {s6['required_missing']}")
                    if s6.get("cross_field_errors"):
                        st.warning(f"Cross-field errors: {s6['cross_field_errors']}")
