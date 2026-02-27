# app_streamlit.py
# Streamlit UI for docextract:
# - Upload PDF/image
# - Submit to FastAPI (/v1/extract)
# - Poll job status (/v1/jobs/{job_id})
# - Display extracted fields + evidence + bbox
# - Show a "pipeline" view (status timeline + artifacts + debug panels)

import os
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
POLL_INTERVAL_SEC = float(os.getenv("POLL_INTERVAL_SEC", "1.0"))
POLL_TIMEOUT_SEC = float(os.getenv("POLL_TIMEOUT_SEC", "120.0"))

SUPPORTED_DOC_TYPES = ["invoice", "receipt", "purchase_order"]


def _post_extract(file_bytes: bytes, filename: str, document_type: str, supplier_id: Optional[str]) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/v1/extract"
    files = {"file": (filename, file_bytes)}
    data = {"document_type": document_type}
    if supplier_id:
        data["supplier_id"] = supplier_id
    r = requests.post(url, files=files, data=data, timeout=60)
    r.raise_for_status()
    return r.json()


def _get_job(job_id: str) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/v1/jobs/{job_id}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _get_artifacts(document_id: str) -> Dict[str, Any]:
    url = f"{API_BASE_URL}/v1/documents/{document_id}/artifacts"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def _format_fields(result_obj: Dict[str, Any]) -> pd.DataFrame:
    """
    result_obj is what your API returns in JobResponse.result:
    {"job_id":..., "document_id":..., "fields": {field_name: {...}}}
    """
    fields = (result_obj or {}).get("fields") or {}
    rows = []
    for name, fr in fields.items():
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


def _pipeline_steps(status: str) -> Tuple[str, ...]:
    """
    A simple pipeline view based on your system’s lifecycle.
    (We can’t see internal steps unless you expose logs/trace endpoints,
    but this is a good UX approximation.)
    """
    # Your statuses: pending/processing/completed/failed
    base = ["Queued", "Processing", "Extraction Complete"]
    if status == "pending":
        return ("✅ Uploaded", "⏳ Queued", "⬜ Processing", "⬜ Complete")
    if status == "processing":
        return ("✅ Uploaded", "✅ Queued", "⏳ Processing", "⬜ Complete")
    if status == "completed":
        return ("✅ Uploaded", "✅ Queued", "✅ Processing", "✅ Complete")
    if status == "failed":
        return ("✅ Uploaded", "✅ Queued", "✅ Processing", "❌ Failed")
    return tuple(base)


st.set_page_config(page_title="DocExtract UI", layout="wide")
st.title("DocExtract UI")
st.caption(f"API: {API_BASE_URL}")

with st.sidebar:
    st.header("Submit")
    doc_type = st.selectbox("Document type", SUPPORTED_DOC_TYPES, index=0)
    supplier_id = st.text_input("Supplier ID (optional)", value="")
    st.divider()
    st.header("Polling")
    poll_interval = st.slider("Poll interval (sec)", min_value=0.3, max_value=5.0, value=float(POLL_INTERVAL_SEC), step=0.1)
    poll_timeout = st.slider("Timeout (sec)", min_value=10, max_value=600, value=int(POLL_TIMEOUT_SEC), step=10)
    st.divider()
    st.header("Debug")
    show_raw = st.checkbox("Show raw API responses", value=False)


col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Upload")
    uploaded = st.file_uploader("Choose a PDF (or image)", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=False)

    submit = st.button("Start extraction", type="primary", disabled=(uploaded is None))

    if submit and uploaded is not None:
        try:
            payload = _post_extract(
                file_bytes=uploaded.getvalue(),
                filename=uploaded.name,
                document_type=doc_type,
                supplier_id=(supplier_id.strip() or None),
            )
            st.session_state["last_submit"] = {
                "submitted_at": datetime.utcnow().isoformat(),
                "upload": payload,
            }
            st.success("Submitted!")
        except requests.RequestException as e:
            st.error(f"Submit failed: {e}")
            st.stop()

    st.divider()
    st.subheader("Last submission")

    last = st.session_state.get("last_submit")
    if not last:
        st.info("Upload a file and click **Start extraction**.")
    else:
        upload = last["upload"]
        job_id = upload.get("job_id")
        document_id = upload.get("document_id")
        st.write(
            {
                "job_id": job_id,
                "document_id": document_id,
                "submitted_at_utc": last.get("submitted_at"),
            }
        )

        # Poll controls
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            auto_poll = st.checkbox("Auto-poll", value=True)
        with c2:
            manual_refresh = st.button("Refresh now")
        with c3:
            st.caption("Auto-poll updates status until completed/failed (or timeout).")

        # Poll / fetch job
        status_box = st.empty()
        steps_box = st.empty()
        error_box = st.empty()

        def render_job(job: Dict[str, Any]):
            status = job.get("status", "unknown")
            status_box.metric("Job status", status)
            steps_box.write(" → ".join(_pipeline_steps(status)))
            if status == "failed":
                error_box.error(job.get("error") or "Unknown error")

        job_data = None
        start_t = time.time()

        if manual_refresh or auto_poll:
            while True:
                try:
                    job_data = _get_job(job_id)
                except requests.RequestException as e:
                    st.error(f"Failed to fetch job: {e}")
                    break

                render_job(job_data)

                if show_raw:
                    st.json(job_data)

                if job_data.get("status") in ("completed", "failed"):
                    break

                if not auto_poll:
                    break

                if time.time() - start_t > poll_timeout:
                    st.warning("Polling timed out. Refresh manually.")
                    break

                time.sleep(poll_interval)

with col_right:
    st.subheader("Results")

    last = st.session_state.get("last_submit")
    if not last:
        st.info("No results yet.")
    else:
        upload = last["upload"]
        job_id = upload.get("job_id")
        document_id = upload.get("document_id")

        # Fetch current status once for rendering results
        try:
            job = _get_job(job_id)
        except requests.RequestException as e:
            st.error(f"Failed to fetch job: {e}")
            st.stop()

        status = job.get("status")
        if status not in ("completed", "failed"):
            st.info(f"Job is **{status}**. Wait for completion to see extracted fields.")
        elif status == "failed":
            st.error(job.get("error") or "Job failed.")
        else:
            result = job.get("result") or {}
            df = _format_fields(result)

            if df.empty:
                st.warning("No fields returned.")
            else:
                # Main table
                st.dataframe(
                    df[["field", "value", "confidence", "status", "method", "page"]],
                    use_container_width=True,
                    hide_index=True,
                )

                # Details: evidence + bbox per field
                st.markdown("### Field details")
                for _, row in df.iterrows():
                    with st.expander(f"{row['field']}  ·  {row['status']}  ·  conf={row['confidence']}"):
                        st.write({"value": row["value"], "method": row["method"], "page": row["page"]})
                        if row["evidence"]:
                            st.markdown("**Evidence**")
                            st.code(str(row["evidence"]))
                        if row["bbox"]:
                            st.markdown("**BBox**")
                            st.json(row["bbox"])

            st.divider()
            st.subheader("Artifacts / Pipeline visibility")

            # Artifacts can help "see the pipeline" (original, page_image, ocr_json, debug_trace, etc.)
            # If you later store debug traces as artifacts, they’ll appear here automatically.
            try:
                artifacts_resp = _get_artifacts(document_id)
                artifacts = artifacts_resp.get("artifacts") or []
            except requests.RequestException as e:
                st.warning(f"Could not load artifacts: {e}")
                artifacts = []

            if not artifacts:
                st.info(
                    "No artifacts found. If you want deeper 'pipeline view', store step-by-step debug traces "
                    "as artifacts (e.g., kind='debug_trace') and expose their URLs."
                )
            else:
                a_df = pd.DataFrame(artifacts)
                st.dataframe(a_df, use_container_width=True, hide_index=True)

                # Quick grouped view
                kinds = sorted(set([a.get("kind") for a in artifacts if a.get("kind")]))
                st.markdown("#### Artifact links")
                for k in kinds:
                    with st.expander(f"{k}"):
                        for a in [x for x in artifacts if x.get("kind") == k]:
                            st.write({"url": a.get("url"), "meta": a.get("meta")})

            if show_raw:
                st.markdown("### Raw result")
                st.json(result)

st.caption(
    "Tip: For a richer pipeline view, add a debug trace artifact per step "
    "(upload → parse → OCR → field extract → validate → save)."
)