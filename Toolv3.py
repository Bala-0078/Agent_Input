"""
Tool.py
=======
CrewAI tool that:
  1. Downloads two .saz files from a GitHub repository (via raw URL).
  2. Runs the full SAZ → JSON → Grouping → JMeter Data → Correlation → Summary pipeline.
  3. Creates a new folder in the same GitHub repo and uploads every generated
     *_summary.txt file there.

GitHub token is provided directly by the user at runtime.
"""

from __future__ import annotations  # makes all type hints lazy strings – works on Python 3.7+

import base64
import concurrent.futures
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import textwrap
import time
import zipfile
import zlib
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Type
from urllib.parse import (
    parse_qsl, urlencode, urlparse, urlsplit, urlunsplit
)

import requests
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


# ══════════════════════════════════════════════════════════════════════════════
# SCHEMA & TOOL CLASS  (must come first so the platform can read the schema)
# ══════════════════════════════════════════════════════════════════════════════

class SazSummaryToolSchema(BaseModel):
    """Input schema for SazSummaryTool."""
    saz_url_1: str = Field(
        ...,
        description=(
            "GitHub URL (raw or blob) of the FIRST .saz file, e.g. "
            "https://github.com/owner/repo/blob/main/captures/run1.saz"
        ),
    )
    saz_url_2: str = Field(
        ...,
        description=(
            "GitHub URL (raw or blob) of the SECOND .saz file, e.g. "
            "https://github.com/owner/repo/blob/main/captures/run2.saz"
        ),
    )
    github_token: str = Field(
        ...,
        description=(
            "GitHub Personal Access Token with repo read and write scope, e.g. ghp_xxxxxxxxxxxx"
        ),
    )
    output_folder_name: str = Field(
        default="",
        description=(
            "Name of the new folder to create in the repo for the summary files. "
            "Leave empty to auto-generate a timestamped name like 'saz_summary_20260416_153000'."
        ),
    )



class SazSummaryTool(BaseTool):
    """
    SazSummaryTool – Downloads two .saz Fiddler capture files from a GitHub
    repository, runs the full SAZ-to-summary pipeline, then uploads the
    generated *_summary.txt files into a new folder in the same repository.
    """

    name: str = "SAZ Summary Tool"
    description: str = (
        "Use this tool when you have two GitHub URLs pointing to .saz Fiddler capture files. "
        "It requires three inputs: "
        "'saz_url_1' (GitHub URL of the FIRST .saz file), "
        "'saz_url_2' (GitHub URL of the SECOND .saz file), and "
        "'github_token' (GitHub Personal Access Token with repo read/write scope). "
        "Optionally provide 'output_folder_name' for the GitHub folder where summaries will be saved. "
        "The tool downloads the .saz files, runs the full SAZ → JSON → Grouping → "
        "JMeter Data → Correlation → Summary pipeline, and uploads the resulting "
        "*_summary.txt files to a new folder in the same GitHub repository."
    )
    args_schema: Type[BaseModel] = SazSummaryToolSchema

    def _run(
        self,
        saz_url_1: str,
        saz_url_2: str,
        github_token: str,
        output_folder_name: str = "",
    ) -> str:
        try:
            token = github_token.strip()
            if not token:
                return "Error: github_token is required and cannot be empty."

            info     = _parse_github_url(saz_url_1)
            api_base = info["api_base"]
            branch   = info["branch"]

            folder_name = output_folder_name.strip() or (
                "saz_summary_" + time.strftime("%Y%m%d_%H%M%S")
            )

            print(f"[SazSummaryTool] Repo   : {info['owner']}/{info['repo']}")
            print(f"[SazSummaryTool] Branch : {branch}")
            print(f"[SazSummaryTool] Output : {folder_name}/")

            raw_url_1  = _to_raw_url(saz_url_1)
            raw_url_2  = _to_raw_url(saz_url_2)
            saz_name_1 = Path(urlparse(raw_url_1).path).name
            saz_name_2 = Path(urlparse(raw_url_2).path).name

            if saz_name_1 == saz_name_2:
                saz_name_2 = "run2_" + saz_name_2

            print(f"[SazSummaryTool] Downloading {saz_name_1} and {saz_name_2} in parallel …")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fut1 = pool.submit(_download_bytes, raw_url_1, token)
                fut2 = pool.submit(_download_bytes, raw_url_2, token)
                saz_bytes_1 = fut1.result()
                saz_bytes_2 = fut2.result()
            print(f"  {saz_name_1}: {len(saz_bytes_1):,} bytes")
            print(f"  {saz_name_2}: {len(saz_bytes_2):,} bytes")

            print("[SazSummaryTool] Running in-memory pipeline …")
            summary_files = _run_pipeline_in_memory(
                saz_bytes_1, saz_name_1,
                saz_bytes_2, saz_name_2,
            )

            if not summary_files:
                return (
                    "Pipeline completed but no *_summary.txt files were generated. "
                    "Check that the .saz files contain valid Fiddler sessions with comments."
                )

            print(f"[SazSummaryTool] Generated {len(summary_files)} summary file(s).")

            uploaded_urls = []
            for rel_path, content_bytes in sorted(summary_files.items()):
                rp  = f"{folder_name}/{rel_path}"
                msg = f"Add SAZ summary: {Path(rel_path).name}"
                print(f"[SazSummaryTool] Uploading → {rp}")
                url = _upload_file_to_github(
                    api_base=api_base, branch=branch, repo_path=rp,
                    content_bytes=content_bytes, token=token, commit_message=msg,
                )
                uploaded_urls.append(url)

            result_lines = [
                f"Successfully processed {len(uploaded_urls)} summary file(s).",
                f"Uploaded to: {info['owner']}/{info['repo']} → {folder_name}/",
                "",
                "Files:",
            ] + [f"  • {u}" for u in uploaded_urls]
            return "\n".join(result_lines)

        except requests.HTTPError as e:
            return f"HTTP error during GitHub API call: {e.response.status_code} – {e.response.text[:300]}"
        except Exception as e:
            import traceback
            return f"Error running SAZ Summary Tool: {str(e)}\n{traceback.format_exc()}"



# ══════════════════════════════════════════════════════════════════════════════
# TOOL HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _to_raw_url(github_url: str) -> str:
    """Convert a GitHub blob URL to the equivalent raw.githubusercontent.com URL."""
    if "raw.githubusercontent.com" in github_url:
        return github_url
    m = re.match(
        r"https://github\.com/([^/]+/[^/]+)/blob/([^/]+)/(.+)",
        github_url,
    )
    if m:
        return f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3)}"
    return github_url


def _parse_github_url(github_url: str) -> dict:
    raw = _to_raw_url(github_url)
    m = re.match(
        r"https://raw\.githubusercontent\.com/([^/]+)/([^/]+)/([^/]+)/(.+)",
        raw,
    )
    if not m:
        raise ValueError(f"Cannot parse GitHub URL: {github_url}")
    return {
        "owner":    m.group(1),
        "repo":     m.group(2),
        "branch":   m.group(3),
        "path":     m.group(4),
        "api_base": f"https://api.github.com/repos/{m.group(1)}/{m.group(2)}",
    }


def _download_bytes(url: str, token: str) -> bytes:
    """Download a URL into memory and return the raw bytes."""
    headers = {"Authorization": f"token {token}"}
    resp = requests.get(url, headers=headers, timeout=120)
    resp.raise_for_status()
    return resp.content


def _download_file(url: str, dest: Path, token: str) -> None:
    headers = {"Authorization": f"token {token}"}
    resp = requests.get(url, headers=headers, stream=True, timeout=120)
    resp.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)


def _upload_file_to_github(
    api_base: str, branch: str, repo_path: str,
    content_bytes: bytes, token: str, commit_message: str,
) -> str:
    content_b64 = base64.b64encode(content_bytes).decode("ascii")
    url     = f"{api_base}/contents/{repo_path}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    # Fetch existing file SHA (needed when updating an existing file)
    sha = None
    get_resp = requests.get(url, params={"ref": branch}, headers=headers, timeout=30)
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")

    body = {"message": commit_message, "branch": branch, "content": content_b64}
    if sha:
        body["sha"] = sha

    resp = requests.put(url, json=body, headers=headers, timeout=60)
    resp.raise_for_status()
    return resp.json().get("content", {}).get("html_url", repo_path)


# ── In-memory pipeline helpers ────────────────────────────────────────────────

def _parse_saz_bytes(saz_bytes: bytes, name: str, url_filter=None) -> dict:
    """Parse a .saz file from raw bytes into a sessions dict (no disk I/O)."""
    normalized_filters = normalize_url_filters(url_filter)
    sessions = []
    with zipfile.ZipFile(io.BytesIO(saz_bytes), "r") as saz:
        all_files = saz.namelist()
        session_ids: set = set()
        for f in all_files:
            m = re.match(r"raw/(\d+)_c\.txt", f)
            if m:
                session_ids.add(m.group(1))

        for sid in sorted(session_ids, key=lambda x: int(x)):
            request_file  = f"raw/{sid}_c.txt"
            response_file = f"raw/{sid}_s.txt"
            metadata_file = f"raw/{sid}_m.xml"

            request_data  = parse_raw_request(saz.read(request_file)) \
                            if request_file in all_files else {}
            response_data = parse_raw_response(saz.read(response_file)) \
                            if response_file in all_files else {}
            metadata      = parse_metadata_xml(saz.read(metadata_file)) \
                            if metadata_file in all_files else {"comment": "", "flags": {}}

            session_url = request_data.get("url", "")
            if normalized_filters and not any(
                fv in session_url.lower() for fv in normalized_filters
            ):
                continue

            sessions.append({
                "session_id": int(sid),
                "comment":    metadata.get("comment", ""),
                "flags":      metadata.get("flags", {}),
                "request":    request_data,
                "response":   response_data,
            })

    return {"source_file": name, "total_sessions": len(sessions), "sessions": sessions}


def _process_group_data_in_memory(
    group_sessions: list, group_name: str, scenario_name: str,
    skip_options: bool = False,
) -> dict:
    """In-memory equivalent of process_group_file – no file I/O."""
    n_connect = n_options = n_no_url = 0
    samplers: list = []

    for session in group_sessions:
        method = session.get("request", {}).get("method", "").upper()
        url    = session.get("request", {}).get("url", "")
        if method == "CONNECT":
            n_connect += 1; continue
        if skip_options and method == "OPTIONS":
            n_options += 1; continue
        if not url.startswith("http"):
            n_no_url += 1; continue
        result = extract_session(session, skip_options)
        if result:
            samplers.append(result)

    attach_think_times(samplers)
    unique_servers = sorted({
        f"{s['http_sampler']['protocol']}://"
        f"{s['http_sampler']['server_name']}:"
        f"{s['http_sampler']['port']}"
        for s in samplers
    })
    header_keys  = sorted({k for s in samplers for k in s["header_manager"].keys()})
    methods_used = sorted({s["http_sampler"]["method"] for s in samplers})

    return {
        "scenario":    scenario_name,
        "group":       group_name,
        "source_file": f"{scenario_name}/{group_name}",
        "statistics": {
            "total_sessions":      len(group_sessions),
            "actionable_requests": len(samplers),
            "skipped_connect":     n_connect,
            "skipped_options":     n_options,
            "skipped_no_url":      n_no_url,
        },
        "jmeter_hints": {
            "unique_servers":        unique_servers,
            "http_request_defaults": _suggest_defaults(unique_servers),
            "header_keys_used":      header_keys,
            "methods_used":          methods_used,
            "has_auth_headers":      any(s["authorization"] for s in samplers),
            "has_cookies":           any(s["cookies"] for s in samplers),
            "has_post_body":         any(s["http_sampler"]["body"] for s in samplers),
        },
        "http_samplers": samplers,
    }


def _correlate_in_memory(sessions1: list, sessions2: list) -> list:
    """
    In-memory correlation. Returns group_results:
    list of (group_name, dynamic_fields, correlations).
    """
    cfg     = Config()
    groups1 = group_sessions_by_comment(sessions1)
    groups2 = group_sessions_by_comment(sessions2)

    seen: set = set()
    all_groups: list = []
    for grp in list(groups1.keys()) + list(groups2.keys()):
        if grp not in seen:
            seen.add(grp)
            all_groups.append(grp)

    group_results = []
    for grp in all_groups:
        g1 = groups1.get(grp, [])
        g2 = groups2.get(grp, [])
        if not g1 or not g2:
            continue
        pairs          = match_sessions(g1, g2, cfg)
        dynamic_fields = find_dynamic_values(pairs, cfg)
        correlations   = find_correlations(g1, dynamic_fields, cfg)
        structural     = find_structural_correlations(g1, correlations, cfg)
        group_results.append((grp, dynamic_fields, correlations + structural))

    return group_results


def _build_correlations_map(group_results: list) -> dict:
    """
    Build {group_name: corr_dict} suitable for passing to build_summary.
    Mirrors the JSON structure written by save_group_correlations.
    """
    corr_map: dict = {}
    for _idx, (grp_name, _dyn, corrs) in enumerate(group_results, 1):
        real = [c for c in corrs if c["correlated"]]
        if real:
            corr_map[grp_name] = {
                "group":             grp_name,
                "dynamic_variables": _build_dynamic_variables(real),
                "correlations":      real,
            }
    return corr_map


def _run_pipeline_in_memory(
    saz_bytes_1: bytes, saz_name_1: str,
    saz_bytes_2: bytes, saz_name_2: str,
) -> dict:
    """
    Run the full SAZ pipeline entirely in memory.
    Returns {relative_path_str: content_bytes} for all *_summary.txt files.
    """
    # Stage 1: Parse SAZ bytes → session dicts (parallel)
    print("[Pipeline] Stage 1 – Parsing SAZ bytes …")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut1 = pool.submit(_parse_saz_bytes, saz_bytes_1, saz_name_1)
        fut2 = pool.submit(_parse_saz_bytes, saz_bytes_2, saz_name_2)
        data1 = fut1.result()
        data2 = fut2.result()
    sessions1 = data1.get("sessions", [])
    sessions2 = data2.get("sessions", [])
    stem1     = Path(saz_name_1).stem
    stem2     = Path(saz_name_2).stem
    print(f"  {saz_name_1}: {len(sessions1)} sessions")
    print(f"  {saz_name_2}: {len(sessions2)} sessions")

    # Stage 2: Group by comment (in memory)
    print("[Pipeline] Stage 2 – Grouping sessions …")
    groups1 = group_sessions_by_comment(sessions1)
    groups2 = group_sessions_by_comment(sessions2)

    # Stage 3: Correlation (in memory)
    print("[Pipeline] Stage 3 – Correlating …")
    group_results = _correlate_in_memory(sessions1, sessions2)
    corr_map      = _build_correlations_map(group_results)

    # Stage 4: JMeter data extraction + summary generation (parallel per group)
    print("[Pipeline] Stage 4 – Building summaries …")
    output_files: dict = {}   # rel_path_str → bytes

    def _build_one(stem, idx, group_name, grp_sessions):
        jmeter_data  = _process_group_data_in_memory(grp_sessions, group_name, stem)
        corr         = corr_map.get(group_name, {})
        summary_text = build_summary(
            jmeter_data, body_limit=0,
            correlations=corr, skip_static=True,
        )
        stem_prefix = f"{idx:03d}_{group_name}"
        rel_path    = f"{stem}/{stem_prefix}/{stem_prefix}_summary.txt"
        return rel_path, summary_text.encode("utf-8")

    tasks = []
    for stem, groups in [(stem1, groups1), (stem2, groups2)]:
        for idx, (group_name, grp_sessions) in enumerate(groups.items(), 1):
            tasks.append((stem, idx, group_name, grp_sessions))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(tasks) or 1)) as pool:
        futs = [pool.submit(_build_one, *t) for t in tasks]
        for fut in concurrent.futures.as_completed(futs):
            rel_path, content = fut.result()
            output_files[rel_path] = content
            print(f"  ✓ {rel_path}  ({len(content):,} bytes)")

    return output_files


def _run_pipeline(saz_file_1: Path, saz_file_2: Path, work_dir: Path) -> list[Path]:
    """Run the 5-stage pipeline and return all generated *_summary.txt files."""
    saz_input_dir    = work_dir / ".saz_Files"
    json_output_dir  = work_dir / "json_files"
    grouped_dir      = json_output_dir / "grouped"
    jmeter_dir       = json_output_dir / "JMeter Specific Data"
    agent_input_dir  = work_dir / "agent_input"
    correlations_dir = work_dir / "correlations"
    report_path      = work_dir / "report.json"

    saz_input_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(saz_file_1, saz_input_dir / saz_file_1.name)
    shutil.copy2(saz_file_2, saz_input_dir / saz_file_2.name)

    stats = batch_convert_saz_files(
        input_folder=str(saz_input_dir),
        output_folder=str(json_output_dir),
        verbose=False, url_filter=None,
    )
    generated_files = [
        Path(f["output"]) for f in stats.get("files", [])
        if f.get("status") == "success" and f.get("output")
    ]
    if len(generated_files) < 2:
        generated_files = list(json_output_dir.glob("*.json"))
    if len(generated_files) < 2:
        raise RuntimeError(f"Stage 1 produced only {len(generated_files)} JSON file(s); need 2.")

    file_a, file_b = generated_files[0], generated_files[1]

    group_only(file_a=file_a, file_b=file_b, grouped_root=grouped_dir, verbose=False)
    main(grouped_root=grouped_dir, output_root=jmeter_dir, skip_options=False)
    correlate_only(
        file_a=file_a, file_b=file_b, grouped_root=grouped_dir,
        output_report=report_path, correlations_dir=correlations_dir, verbose=False,
    )

    agent_input_dir.mkdir(parents=True, exist_ok=True)
    process_folder(
        input_dir=jmeter_dir, output_dir=agent_input_dir,
        mode="summary", chunk_size=10, body_limit=0,
        correlations_dir=correlations_dir,
    )

    return list(agent_input_dir.rglob("*_summary.txt"))


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE LOGIC  (hardcoded from all source modules)
# ══════════════════════════════════════════════════════════════════════════════

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_INPUT_DIR        = Path("json(.saz)_Files/JMeter Specific Data")
DEFAULT_OUTPUT_DIR       = Path("agent_input")
DEFAULT_CORRELATIONS_DIR = Path("correlations")

# ── Static asset filtering ────────────────────────────────────────────────────
# These requests are the same across every run – no parameterization needed.
STATIC_EXTENSIONS = {
    ".js", ".css", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".map", ".html", ".htm",
}
STATIC_PATH_PATTERNS = re.compile(
    r'/(?:chunk-|src-|polyfills-|scripts-|styles-|main-|media/|images/|'
    r'static/|pkgs/|node_modules/|css/|js/|_stylesheets/|_skin/|content/)',
    re.IGNORECASE
)

# ── Per-sampler warning rules ─────────────────────────────────────────────────

# Query params that are runtime-generated timestamps / cache busters.
# Their values MUST be generated at test execution time, not hardcoded.
_TIMESTAMP_PARAMS = {"_", "once", "ts", "nonce", "rand", "t", "nocache", "v"}

# WebSocket-specific headers that JMeter manages automatically.
# They must NOT appear in the JMeter Header Manager for an HTTP sampler.
_WS_HEADERS = {
    "sec-websocket-key", "sec-websocket-version",
    "sec-websocket-extensions", "upgrade",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jmeter_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _short_body(body: str, limit: int = 300) -> str:
    """
    Truncate a body string for compact display.
    limit <= 0 means NO truncation – return the full body.
    """
    if not body:
        return ""
    body = body.strip()
    if limit <= 0:
        return body                          # full body, no cut
    return body[:limit] + (" …[truncated]" if len(body) > limit else "")


# ── Correlation helpers ───────────────────────────────────────────────────────

def _load_correlations_for_group(correlations_dir: Path, group_name: str) -> dict:
    """
    Find and load the correlations JSON file that matches *group_name*.
    Files are named like  001_Login.json  –  strip the numeric prefix to match.
    Returns an empty dict if nothing is found.
    """
    if not correlations_dir or not correlations_dir.exists():
        return {}
    for f in correlations_dir.glob("*.json"):
        stem = re.sub(r'^\d+_', '', f.stem)   # "001_Login" → "Login"
        if stem == group_name:
            try:
                with open(f, encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception:
                pass
    return {}


def _build_corr_maps(corr_data: dict) -> tuple:
    """
    Build two lookup dicts from a group's correlations file.

    source_map  {session_id: [{"variable_name", "token_value",
                               "source_field", "extraction_note"}, …]}
    dest_map    {session_id: [{"variable_name", "token_value", "field"}, …]}

    HTML body attribute correlations (a[@href], html.* paths) and full-URL
    path tokens are filtered out – they are noise, not real JMeter parameters.
    """
    source_map: dict = {}
    dest_map:   dict = {}
    if not corr_data:
        return source_map, dest_map

    # Build a quick token→variable_name lookup from dynamic_variables
    # (already filtered + correctly named by _build_dynamic_variables)
    token_to_var: dict = {
        dv["token_value"]: dv["variable_name"]
        for dv in corr_data.get("dynamic_variables", [])
        if dv.get("token_value")
    }

    for c in corr_data.get("correlations", []):
        token           = c.get("token_value", "")
        extraction_note = c.get("extraction_note", "")

        # ── Skip noisy HTML / full-URL correlations ───────────────────────────
        if "html." in extraction_note or "[@" in extraction_note:
            continue
        if token.startswith("/") and ("?" in token or "&" in token):
            continue

        # Use the variable name from dynamic_variables (correctly derived)
        var_name = token_to_var.get(token)
        if not var_name:
            # Fallback: derive from extraction_note
            m = re.search(r'cookie\[([^\]]+)\]', extraction_note)
            if m:
                var_name = re.sub(r'\W+', '_', m.group(1)).strip("_")
            else:
                m = re.search(r'redirect_param\[([^\]]+)\]', extraction_note)
                if m:
                    var_name = re.sub(r'\W+', '_', m.group(1)).strip("_")
                elif extraction_note == "bearer_token":
                    var_name = "bearer_token"
                else:
                    var_name = re.sub(
                        r'\W+', '_',
                        c.get("dynamic_field_name", "var").split(".")[-1]
                    ).strip("_") or "dynamic_var"

        src     = c.get("source", {})
        src_sid = src.get("session_id")
        if src_sid is not None:
            source_map.setdefault(src_sid, []).append({
                "variable_name":   var_name,
                "token_value":     token,
                "source_field":    src.get("field", ""),
                "extraction_note": extraction_note,
            })

        for dest in c.get("destinations", []):
            dsid = dest.get("session_id")
            if dsid is not None:
                dest_map.setdefault(dsid, []).append({
                    "variable_name": var_name,
                    "token_value":   token,
                    "field":         dest.get("field", ""),
                })

    return source_map, dest_map


def _extractor_hint(extraction_note: str, source_field: str, var_name: str) -> str:
    """Return a one-line JMeter extractor hint for the agent."""
    note  = extraction_note.lower()
    field = source_field.lower()

    if "cookie[" in note:
        # If the value originates in the response BODY (e.g. response.body.response.user.userid),
        # use a JSON Path Extractor – more reliable than parsing Set-Cookie.
        if "response.body." in field:
            body_path = re.split(r'response\.body\.', field, maxsplit=1)[-1]
            jp = "$." + body_path
            return f"JSON Path Extractor (response body): {jp}"
        # Otherwise the only reliable source IS the Set-Cookie header.
        m     = re.search(r'cookie\[([^\]]+)\]', extraction_note)
        cname = m.group(1) if m else var_name
        return (f"Regex Extractor (response header Set-Cookie): "
                f"{re.escape(cname)}=([^;]+)")

    if "redirect_param[" in note:
        m     = re.search(r'redirect_param\[([^\]]+)\]', extraction_note)
        pname = m.group(1) if m else var_name
        return (f"Boundary/Regex Extractor (response header Location): "
                f"{pname}=([^& ]+)")

    if "bearer_token" in note:
        # Token comes from the response BODY, not from a response Authorization header.
        # Use JSON Path if the source_field points to the body (most reliable).
        if "response.body." in field:
            body_path = re.split(r'response\.body\.', field, maxsplit=1)[-1]
            jp = "$." + body_path
            return f"JSON Path Extractor (response body): {jp}"
        return "JSON Path Extractor (response body): $.response.user.token"

    if "body_field[" in note:
        m    = re.search(r'body_field\[([^\]]+)\]', extraction_note)
        path = m.group(1) if m else ""
        # Build JSONPath preserving the FULL dotted path.
        # DO NOT strip "response." — the server wraps all responses in
        # {"response": {...}} so $.response.user.token is the correct root.
        jp = "$." + path
        return f"JSON Path Extractor (response body): {jp}"

    if "response.headers." in field:
        hname = field.split("response.headers.")[-1]
        return f"Regex Extractor (response header {hname})"

    if "response.body" in field:
        # Derive JSONPath from source_field directly (most reliable)
        body_path = re.split(r'response\.body\.', field, maxsplit=1)
        if len(body_path) > 1:
            jp = "$." + body_path[-1]
            return f"JSON Path Extractor (response body): {jp}"
        return "JSON Path Extractor (response body): [derive from response structure]"

    return f"Extractor needed (note: {extraction_note})"


def _parameterize(text: str, dest_vars: list) -> str:
    """Replace each known dynamic token value with its ${variable_name}."""
    for dv in dest_vars:
        tok = dv.get("token_value", "")
        var = dv.get("variable_name", "")
        if tok and var and tok in text:
            text = text.replace(tok, f"${{{var}}}")
    return text


def is_static_sampler(sampler: dict) -> bool:
    """
    Return True if this sampler is a static asset (JS/CSS/image/font/etc.)
    that does not change between test runs and needs no parameterization.

    Never treats a sampler as static when it is a correlation source
    (i.e. the agent must extract a value from its response).
    """
    hs = sampler.get("http_sampler", {})
    method = hs.get("method", "").upper()

    # POST / PUT / PATCH always carry dynamic data
    if method in {"POST", "PUT", "PATCH"}:
        return False

    # If it has a request body it is dynamic
    if hs.get("body"):
        return False

    path = hs.get("path", "")
    ext  = Path(path.split("?")[0]).suffix.lower()

    if ext in STATIC_EXTENSIONS:
        return True
    if STATIC_PATH_PATTERNS.search(path):
        return True

    return False


# ── Mode 1: Compact Summary ───────────────────────────────────────────────────

# Max characters shown for a single cookie value inline in the summary.
# Values longer than this are truncated and stored in full in _reference.txt.
_COOKIE_VALUE_LIMIT = 60


def build_reference(data: dict, correlations: dict = None) -> str:
    """
    Build a human-readable reference file with all captured cookie values
    and correlation variables for MANUAL JMeter test-plan setup.
    This file is NOT sent to the agent – it supplements the summary for
    the test-plan author who needs the actual captured values.
    """
    # Build token→variable lookup from correlations
    token_to_var: dict = {}
    if correlations:
        for dv in correlations.get("dynamic_variables", []):
            if dv.get("token_value"):
                token_to_var[dv["token_value"]] = dv["variable_name"]

    # Collect all unique cookies (name → first example value)
    all_cookies: dict = {}
    for s in data.get("http_samplers", []):
        for name, value in (s.get("cookies") or {}).items():
            if name not in all_cookies:
                all_cookies[name] = value

    lines = [
        f"=== REFERENCE : {data.get('scenario', '?')} / {data.get('group', '?')} ===",
        "  Manual JMeter setup guide – this file is NOT sent to the agent.",
        "  Cookie values here are captured from the SAZ recording.",
        "",
        "=" * 68,
        "  COOKIES  →  Configure in JMeter HTTP Cookie Manager",
        "=" * 68,
        "  Correlated cookies are extracted at runtime via Regex Extractor.",
        "  Non-correlated cookies are server-managed (load-balancer stickiness,",
        "  CSRF tokens, etc.) – JMeter's Cookie Manager handles them automatically",
        "  once the first response sets them via Set-Cookie.",
        "",
    ]

    if all_cookies:
        for name in sorted(all_cookies.keys()):
            raw_val  = all_cookies[name]
            jmeter_var = token_to_var.get(raw_val, "")
            lines.append(f"  Name    : {name}")
            if jmeter_var:
                lines.append(f"  JMeter  : ${{{jmeter_var}}}  (correlated – extracted at runtime)")
            else:
                lines.append(f"  JMeter  : (auto-managed by Cookie Manager)")
            lines.append(f"  Example : {raw_val[:200]}" +
                         (" …[truncated]" if len(raw_val) > 200 else ""))
            lines.append("")
    else:
        lines += ["  (no cookies found)", ""]

    # Correlation variables quick-reference table
    dyn_vars = (correlations or {}).get("dynamic_variables", [])
    if dyn_vars:
        lines += [
            "=" * 68,
            "  CORRELATION VARIABLES  →  Add Extractors in JMeter",
            "=" * 68,
            "",
        ]
        for dv in dyn_vars:
            hint = _extractor_hint(
                dv.get("extraction_note", ""),
                dv.get("source_field", ""),
                dv.get("variable_name", ""),
            )
            lines += [
                f"  Variable  : ${{{dv['variable_name']}}}",
                f"  Source    : Session {dv.get('source_session')} | "
                f"{dv.get('source_method')} {dv.get('source_url', '')[:80]}",
                f"  Example   : {str(dv.get('token_value', ''))[:120]}",
                f"  Extractor : {hint}",
                "",
            ]

    return "\n".join(lines)


def write_reference(data: dict, out_dir: Path, stem: str,
                    correlations: dict = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_reference.txt"
    out_path.write_text(
        build_reference(data, correlations),
        encoding="utf-8",
    )
    sz = out_path.stat().st_size
    print(f"  ✓ Reference → {out_path}  ({sz:,} bytes)")
    return out_path

def build_summary(data: dict, body_limit: int = 300,
                  correlations: dict = None,
                  skip_static: bool = True) -> str:
    """
    Produce a compact, human-readable text block that an LLM can easily
    consume.  Each sampler becomes ~10 lines instead of ~30 JSON lines.

    When *skip_static* is True (default), static asset requests (JS, CSS,
    images, fonts, …) are omitted – they are identical across every run and
    carry no dynamic values, so the agent doesn't need to see them.

    When *correlations* is provided (loaded from the per-group correlations
    file), the summary is enriched with:
      • A "CORRELATIONS NEEDED" section listing every dynamic variable,
        where to extract it, and a suggested JMeter extractor expression.
      • Per-sampler  🔑 EXTRACT  lines on the response source session.
      • Per-sampler  ⚡ PARAM    lines on every destination session, with
        token values replaced by  ${variable_name}  in QParams / Headers /
        Body so the agent can copy-paste directly into JMeter XML.
    """
    source_map, dest_map = _build_corr_maps(correlations) if correlations else ({}, {})

    all_samplers = data.get("http_samplers", [])

    # Name of the bearer token variable (from dynamic_variables if present)
    bearer_token_var = "bearer_token"
    if correlations:
        for dv in correlations.get("dynamic_variables", []):
            if dv.get("extraction_note") == "bearer_token":
                bearer_token_var = dv.get("variable_name", "bearer_token")
                break

    # ── Static filtering ─────────────────────────────────────────────────────
    # A sampler that is a correlation SOURCE must never be skipped even if its
    # path looks static (we need the agent to know it must add an extractor).
    if skip_static:
        source_sids = set(source_map.keys())
        active_samplers = [
            s for s in all_samplers
            if not is_static_sampler(s) or s.get("session_id") in source_sids
        ]
        n_skipped_static = len(all_samplers) - len(active_samplers)
    else:
        active_samplers    = all_samplers
        n_skipped_static   = 0

    lines = []

    # ── File-level metadata ──────────────────────────────────────────────────
    stats  = data.get("statistics", {})
    hints  = data.get("jmeter_hints", {})
    lines += [
        f"=== SCENARIO : {data.get('scenario', '?')} / {data.get('group', '?')} ===",
        f"Actionable requests : {stats.get('actionable_requests', '?')}"
        + (f"  (showing {len(active_samplers)}, skipped {n_skipped_static} static)"
           if n_skipped_static else ""),
        f"Methods used        : {', '.join(hints.get('methods_used', []))}",
        f"Has auth headers    : {hints.get('has_auth_headers', False)}",
        f"Has cookies         : {hints.get('has_cookies', False)}",
        f"Has POST body       : {hints.get('has_post_body', False)}",
        f"Servers             :",
    ]
    for s in hints.get("unique_servers", []):
        lines.append(f"  - {s}")
    lines.append(f"Header keys used    : {', '.join(hints.get('header_keys_used', []))}")

    # Flag browser-generated headers that should be EXCLUDED from JMeter
    _BROWSER_HEADERS = {
        "newrelic", "traceparent", "tracestate",
        "x-newrelic-id", "x-newrelic-transaction",
    }
    browser_hdrs = [
        h for h in hints.get("header_keys_used", [])
        if h.lower() in _BROWSER_HEADERS
    ]
    if browser_hdrs:
        lines.append(
            f"⚠ EXCLUDE FROM JMETER (browser-auto-generated, do NOT add to Header Manager): "
            f"{', '.join(browser_hdrs)}"
        )
    lines.append("")

    # ── Correlations section ─────────────────────────────────────────────────
    dyn_vars = (correlations or {}).get("dynamic_variables", [])
    if dyn_vars:
        lines.append("=" * 68)
        lines.append("  CORRELATIONS NEEDED – extract these values and parameterize requests")
        lines.append("=" * 68)
        for idx, dv in enumerate(dyn_vars, 1):
            hint = _extractor_hint(
                dv.get("extraction_note", ""),
                dv.get("source_field", ""),
                dv.get("variable_name", ""),
            )
            lines += [
                f"  [{idx}] Variable     : ${{{dv['variable_name']}}}",
                f"      Captured value: {dv.get('token_value', '')[:80]}",
                f"      Extract from  : Session {dv.get('source_session')} "
                f"| {dv.get('source_method')} {dv.get('source_url', '')[:70]}",
                f"      Response field: {dv.get('source_field', '')}",
                f"      JMeter hint   : {hint}",
                "",
            ]
        lines.append("=" * 68)
        lines.append("")

    # ── Global JMeter guidance notes ─────────────────────────────────────────
    # Detect _token suffix variants used across samplers and warn the agent.
    all_token_suffixes: set = set()
    for s in active_samplers:
        hs  = s.get("http_sampler", {})
        qps = hs.get("query_params", {}) or {}
        bdy = hs.get("body", "") or ""
        for src in [json.dumps(qps), bdy]:
            for m in re.finditer(r'\$\{bearer_token\}(~[^\s"&,]+)', src):
                all_token_suffixes.add(m.group(1))
            for m in re.finditer(r'~0\.[^~"&\s,]{10,}(~[^\s"&,]+)', src):
                all_token_suffixes.add(m.group(1))
    if all_token_suffixes:
        lines.append("=" * 68)
        lines.append("  ⚠ TOKEN SUFFIX VARIANTS – important for correct parameterization")
        lines.append("=" * 68)
        lines.append("  The bearer_token is used with different application suffixes.")
        lines.append("  Each suffix targets a different sub-application. Use EXACTLY as shown:")
        for sfx in sorted(all_token_suffixes):
            lines.append(f"    ${{{bearer_token_var}}}{sfx}")
        lines.append("  Do NOT mix suffixes between requests.")
        lines.append("")

    # ── Per-sampler compact block ────────────────────────────────────────────
    for i, s in enumerate(active_samplers, 1):
        hs     = s.get("http_sampler", {})
        auth   = s.get("authorization") or {}
        hdrs   = s.get("header_manager", {})
        ck     = s.get("cookies", {})
        asrt   = s.get("assertion", {})
        sid    = s.get("session_id")

        src_vars  = source_map.get(sid, [])
        dest_vars = dest_map.get(sid, [])

        lines.append(f"--- [{i:03d}] {s.get('label', '?')} ---")
        lines.append(f"  Method   : {hs.get('method')}  "
                     f"Protocol : {hs.get('protocol')}  "
                     f"Server   : {hs.get('server_name')}:{hs.get('port')}")

        # Path – substitute dynamic values
        path_str = _parameterize(hs.get('path', ''), dest_vars)
        lines.append(f"  Path     : {path_str}")

        # Query params – substitute dynamic values
        qp = hs.get("query_params", {})
        if qp:
            qp_str = _parameterize(json.dumps(qp), dest_vars)
            lines.append(f"  QParams  : {qp_str}")

        # Body – substitute dynamic values
        if hs.get("body"):
            body_str = _parameterize(
                _short_body(hs['body'], body_limit), dest_vars
            )
            lines.append(f"  Body({hs.get('body_format')}) : {body_str}")

        # Headers – substitute dynamic values
        if hdrs:
            hdrs_str = _parameterize(json.dumps(hdrs), dest_vars)
            lines.append(f"  Headers  : {hdrs_str}")

        # Cookies – truncate long values; full values are in _reference.txt
        if ck:
            ck_str = _parameterize(json.dumps(ck), dest_vars)
            ck_obj = json.loads(ck_str)
            ck_display = {
                k: (v if len(v) <= _COOKIE_VALUE_LIMIT
                    else v[:_COOKIE_VALUE_LIMIT] + "… [→ _reference.txt]")
                for k, v in ck_obj.items()
            }
            lines.append(f"  Cookies  : {json.dumps(ck_display)}")

        if auth:
            auth_str = _parameterize(
                f"{auth.get('type')} {auth.get('token', '')[:40]}…", dest_vars
            )
            lines.append(f"  Auth     : {auth_str}")

        hint_str  = asrt.get("response_hint", "")
        hint_part = f"  →  Body contains: {hint_str}" if hint_str else ""
        lines.append(f"  Assert   : {asrt.get('response_code')} {asrt.get('response_message')}{hint_part}")

        # ── Per-sampler warnings ──────────────────────────────────────────────

        # 1. Timestamp / cache-buster query params – must be runtime-generated
        ts_params = [k for k in qp if k.lower() in _TIMESTAMP_PARAMS
                     and "${" not in str(qp.get(k, ""))]
        for tp in ts_params:
            lines.append(
                f"  ⚠ TIMESTAMP PARAM '{tp}': value changes every run. "
                f"In JMeter use: ${{__time()}} (ms) or ${{__Random(1,9999999999,)}}"
            )

        # 2. SignalR connectionToken – not auto-correlated, must be extracted
        if "connectionToken" in qp and "${" not in str(qp.get("connectionToken", "")):
            lines.append(
                f"  ⚠ MISSING CORRELATION 'connectionToken': hardcoded but changes "
                f"every run. Add a JSON Extractor on the preceding /signalr/negotiate "
                f"response → JSON Path: $.ConnectionToken  → Variable: connectionToken"
            )

        # 3. Uncorrelated _token param (not yet substituted with a variable)
        raw_token = qp.get("_token", "") or ""
        if raw_token and "${" not in raw_token:
            lines.append(
                f"  ⚠ UNCORRELATED '_token': '{raw_token[:40]}…' is hardcoded. "
                f"This is likely a scoped token from POST /cmd/createscopedtoken. "
                f"Extract it with a JSON Path Extractor: $.response.token  "
                f"→ Variable: scoped_token, then use here."
            )

        # 4. WebSocket headers – must NOT be added to JMeter Header Manager
        ws_hdrs = [h for h in hdrs if h.lower() in _WS_HEADERS]
        if ws_hdrs:
            lines.append(
                f"  ⚠ WEBSOCKET HEADERS {ws_hdrs}: do NOT add to JMeter Header Manager. "
                f"These are browser-managed. If using HTTP Sampler for WebSocket upgrade, "
                f"remove them; if using WebSocket Sampler plugin, it handles them automatically."
            )

        tt = s.get("think_time_ms")
        if tt is not None:
            lines.append(f"  ThinkTime: {tt} ms")

        # Source annotations – tell agent what to extract from this response
        for sv in src_vars:
            hint = _extractor_hint(
                sv["extraction_note"], sv["source_field"], sv["variable_name"]
            )
            lines.append(
                f"  🔑 EXTRACT: ${{{sv['variable_name']}}} "
                f"← {sv['source_field']}  [{sv['extraction_note']}]"
            )
            lines.append(f"             {hint}")

        # Destination annotations – tell agent which values are parameterized
        for dv in dest_vars:
            lines.append(
                f"  ⚡ PARAM  : {dv['token_value'][:60]} "
                f"→ ${{{dv['variable_name']}}}  [in: {dv['field']}]"
            )

        lines.append("")

    return "\n".join(lines)


# ── Mode 2: Chunking ──────────────────────────────────────────────────────────

def build_chunks(data: dict, chunk_size: int = 10) -> list[dict]:
    """
    Split http_samplers into pages of chunk_size.
    Each chunk is a valid dict with the same top-level keys.
    """
    samplers = data.get("http_samplers", [])
    total    = len(samplers)
    pages    = math.ceil(total / chunk_size) if total else 1
    chunks   = []

    base = {k: v for k, v in data.items() if k != "http_samplers"}

    for page in range(pages):
        start = page * chunk_size
        end   = min(start + chunk_size, total)
        chunk = dict(base)
        chunk["chunk_info"] = {
            "page":        page + 1,
            "total_pages": pages,
            "sampler_range": f"{start + 1}–{end} of {total}",
        }
        chunk["http_samplers"] = samplers[start:end]
        chunks.append(chunk)

    return chunks


# ── Mode 3: Index ─────────────────────────────────────────────────────────────

def build_index(data: dict) -> str:
    """
    Ultra-compact single-line-per-sampler index.
    Useful for asking the agent: 'which request authenticates the user?'
    """
    lines = [
        f"INDEX  {data.get('scenario','?')} / {data.get('group','?')}  "
        f"({len(data.get('http_samplers', []))} samplers)",
        "",
        f"  {'#':<4}  {'Method':<7}  {'Server':<45}  {'Path':<50}  {'Assert':>6}",
        f"  {'-'*4}  {'-'*7}  {'-'*45}  {'-'*50}  {'-'*6}",
    ]
    for i, s in enumerate(data.get("http_samplers", []), 1):
        hs = s.get("http_sampler", {})
        a  = s.get("assertion", {})
        server = f"{hs.get('server_name','?')}"
        path   = hs.get("path", "?")
        if len(path) > 50:
            path = path[:47] + "…"
        lines.append(
            f"  {i:<4}  {hs.get('method','?'):<7}  {server:<45}  "
            f"{path:<50}  {a.get('response_code','?'):>6}"
        )
    return "\n".join(lines)


# ── Output writers ────────────────────────────────────────────────────────────

def write_summary(data: dict, out_dir: Path, stem: str,
                  body_limit: int = 300,
                  correlations: dict = None,
                  skip_static: bool = True) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_summary.txt"
    out_path.write_text(
        build_summary(data, body_limit, correlations=correlations,
                      skip_static=skip_static),
        encoding="utf-8",
    )
    sz = out_path.stat().st_size
    print(f"  ✓ Summary  → {out_path}  ({sz:,} bytes)")
    return out_path


def write_chunks(data: dict, out_dir: Path, stem: str,
                 chunk_size: int = 10) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks(data, chunk_size)
    paths  = []
    for chunk in chunks:
        page     = chunk["chunk_info"]["page"]
        total_p  = chunk["chunk_info"]["total_pages"]
        fname    = f"{stem}_chunk_{page:03d}_of_{total_p:03d}.json"
        out_path = out_dir / fname
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunk, f, indent=2, ensure_ascii=False)
        sz = out_path.stat().st_size
        print(f"  ✓ Chunk {page}/{total_p} → {out_path}  ({sz:,} bytes)")
        paths.append(out_path)
    return paths


def write_index(data: dict, out_dir: Path, stem: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_index.txt"
    out_path.write_text(build_index(data), encoding="utf-8")
    sz = out_path.stat().st_size
    print(f"  ✓ Index    → {out_path}  ({sz:,} bytes)")
    return out_path


# ── Batch: process all JMeter files in a folder tree ─────────────────────────

def _load_flow_order(scenario_dir: Path) -> list:
    """
    Read the _flow_order.json written by extract_jmeter_data.py.
    Returns a list of group names in UI-flow order, or [] if not found.
    """
    order_file = scenario_dir / "_flow_order.json"
    if order_file.exists():
        try:
            with open(order_file, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return []


def _ordered_json_files(scenario_dir: Path) -> list:
    """
    Return JSON files for one scenario directory in UI-flow order.
    Falls back to alphabetical if no _flow_order.json is present.
    Skips files whose names start with '_' (helper/summary files).
    """
    flow_order = _load_flow_order(scenario_dir)
    candidates = {
        jf.stem: jf
        for jf in scenario_dir.glob("*.json")
        if not jf.name.startswith("_")
    }

    if flow_order:
        ordered = [candidates[grp] for grp in flow_order if grp in candidates]
        # Append any file not mentioned in the order list (safety net)
        in_order = {jf.stem for jf in ordered}
        ordered += [jf for stem, jf in sorted(candidates.items())
                    if stem not in in_order]
        return ordered

    # No flow order available – fall back to alphabetical
    return sorted(candidates.values())


def process_folder(input_dir: Path, output_dir: Path,
                   mode: str, chunk_size: int, body_limit: int,
                   correlations_dir: Path = None,
                   skip_static: bool = True) -> None:
    """
    Process all JMeter JSON files under *input_dir*, preserving UI-flow order.

    When *correlations_dir* is supplied, each group's summary is enriched
    with correlation data (extractors + ${variable} substitutions).
    """
    scenario_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir())

    if not scenario_dirs:
        flat_files = sorted(
            jf for jf in input_dir.rglob("*.json")
            if not jf.name.startswith("_")
        )
        if not flat_files:
            print(f"  ✗ No JSON files found in {input_dir}")
            return
        print(f"\n  Found {len(flat_files)} file(s) to process\n")
        for idx, jf in enumerate(flat_files, 1):
            prefixed = f"{idx:03d}_{jf.stem}"
            out_sub  = output_dir / prefixed
            corr     = _load_correlations_for_group(correlations_dir, jf.stem)
            print(f"  Processing: {jf.name}  →  {out_sub}")
            _run_modes(load_jmeter_file(jf), out_sub, prefixed,
                       mode, chunk_size, body_limit, correlations=corr,
                       skip_static=skip_static)
            print()
        return

    total_files = sum(len(_ordered_json_files(sd)) for sd in scenario_dirs)
    print(f"\n  Found {total_files} file(s) to process\n")

    for sd in scenario_dirs:
        scenario_name  = sd.name
        ordered_files  = _ordered_json_files(sd)

        for idx, jf in enumerate(ordered_files, 1):
            prefixed = f"{idx:03d}_{jf.stem}"
            out_sub  = output_dir / scenario_name / prefixed
            # Load correlations for this group (strip prefix for matching)
            corr = _load_correlations_for_group(correlations_dir, jf.stem)
            print(f"  Processing: {scenario_name}/{jf.name}  →  {out_sub}"
                  + (f"  [+{len(corr.get('dynamic_variables',[]))} corr vars]"
                     if corr else ""))
            _run_modes(load_jmeter_file(jf), out_sub, prefixed,
                       mode, chunk_size, body_limit, correlations=corr,
                       skip_static=skip_static)
            print()


def _run_modes(data, out_dir, stem, mode, chunk_size, body_limit,
               correlations=None, skip_static=True):
    if mode in ("summary", "all"):
        write_summary(data, out_dir, stem, body_limit,
                      correlations=correlations, skip_static=skip_static)
        write_reference(data, out_dir, stem, correlations=correlations)
    if mode in ("chunks", "all"):
        write_chunks(data, out_dir, stem, chunk_size)
    if mode in ("index", "all"):
        write_index(data, out_dir, stem)


DEFAULT_GROUPED_ROOT = Path("json(.saz)_Files/grouped")
DEFAULT_OUTPUT_ROOT  = Path("json(.saz)_Files/JMeter Specific Data")

# ── Filtering rules ───────────────────────────────────────────────────────────

# These HTTP methods are always skipped (Fiddler internals / noise)
SKIP_METHODS = {"CONNECT"}

# Headers that JMeter manages automatically – do NOT add to Header Manager
JMETER_AUTO_HEADERS = {
    "host",
    # NOTE: "user-agent" is intentionally NOT filtered here.
    # JMeter's default User-Agent (Apache-HttpClient) is commonly blocked by
    # WAFs and servers, causing 403 Forbidden.  The browser UA captured in the
    # SAZ file must be preserved in the Header Manager so JMeter replays it.
    # NOTE: "accept" is intentionally NOT filtered here.
    # Application-specific Accept headers (e.g. Accept: application/json)
    # are meaningful and must be replicated in JMeter, otherwise some servers
    # return HTML instead of JSON.  Generic browser Accept headers (text/html,
    # */*) will naturally be absent from API requests captured via Fiddler.
    "accept-encoding",
    "accept-language",
    "content-length",
    "connection",
    "cache-control",
    "pragma",
    "upgrade-insecure-requests",
    "sec-fetch-dest",
    "sec-fetch-mode",
    "sec-fetch-site",
    "sec-fetch-user",
    "priority",
    # Conditional cache headers – usually handled by JMeter Cache Manager
    "if-modified-since",
    "if-none-match",
    "if-range",
}

# ── Utility helpers ───────────────────────────────────────────────────────────

def is_binary(text: str) -> bool:
    """Return True when text contains > 5 % replacement characters (gzip etc.)."""
    if not text:
        return False
    return text.count("\ufffd") > len(text) * 0.05


def _derive_response_hint(body: str, content_type: str) -> str:
    """
    Extract a short, assertion-worthy keyword/phrase from the response body.

    Strategy by content type:
      JSON  → first top-level key name  e.g. 'JSON key: "session"'
      HTML  → <title> text or first <h1>  e.g. 'HTML title: "Buzz Login"'
      XML   → root element name  e.g. 'XML root: <settings>'
      Other → empty string (no hint)

    The returned string is stored in assertion.response_hint so the agent
    can add a Response Body Contains assertion in JMeter.
    """
    if not body or is_binary(body):
        return ""

    ct = content_type.lower()
    b  = body.strip()

    # ── JSON ──────────────────────────────────────────────────────────────────
    if "json" in ct or b.startswith(("{", "[")):
        try:
            obj = json.loads(b)
            if isinstance(obj, dict):
                first_key = next(iter(obj.keys()), None)
                if first_key:
                    return f'JSON key: "{first_key}"'
            elif isinstance(obj, list) and obj:
                first = obj[0]
                if isinstance(first, dict):
                    first_key = next(iter(first.keys()), None)
                    if first_key:
                        return f'JSON key (array[0]): "{first_key}"'
        except Exception:
            pass

    # ── HTML ──────────────────────────────────────────────────────────────────
    if "html" in ct or b.lower().startswith("<!doctype") or "<html" in b.lower():
        m = re.search(r'<title[^>]*>([^<]+)</title>', b, re.IGNORECASE)
        if m:
            return f'HTML title: "{m.group(1).strip()[:70]}"'
        m = re.search(r'<h1[^>]*>([^<]+)</h1>', b, re.IGNORECASE)
        if m:
            return f'HTML h1: "{m.group(1).strip()[:70]}"'
        return 'HTML response'

    # ── XML ───────────────────────────────────────────────────────────────────
    if "xml" in ct or b.startswith("<"):
        m = re.match(r'<([a-zA-Z][a-zA-Z0-9:_\-]*)', b)
        if m:
            return f'XML root: <{m.group(1)}>'

    return ""


def detect_body_format(body: str, content_type: str) -> str:
    """Identify body encoding: json / xml / form / text / binary / none."""
    if not body:
        return "none"
    if is_binary(body):
        return "binary"
    ct = content_type.lower()
    b  = body.strip()
    if "json" in ct or b.startswith(("{", "[")):
        return "json"
    if "xml" in ct or "text/plain" in ct and b.startswith("<"):
        return "xml"
    if b.startswith("<") and b.endswith(">"):
        return "xml"
    if "form" in ct or (re.search(r'^\w+=', b) and "&" in b):
        return "form"
    if "text" in ct:
        return "text"
    return "raw"


def parse_url(url: str) -> dict:
    """Split a full URL into JMeter HTTP Request fields."""
    if not url.startswith("http"):
        return {}
    try:
        p = urlsplit(url)
        port = p.port
        if port is None:
            port = 443 if p.scheme == "https" else 80

        query_params: dict = {}
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            if k in query_params:
                existing = query_params[k]
                if isinstance(existing, list):
                    existing.append(v)
                else:
                    query_params[k] = [existing, v]
            else:
                query_params[k] = v

        return {
            "protocol":    p.scheme,
            "server_name": p.hostname or "",
            "port":        str(port),
            "path":        p.path or "/",
            "query_params": query_params,
        }
    except Exception:
        return {}


def parse_cookies(cookie_header: str) -> dict:
    """Parse a Cookie: request header into {name: value} pairs."""
    _COOKIE_ATTRS = {"path", "domain", "expires", "max-age", "samesite",
                     "secure", "httponly", "version"}
    cookies: dict = {}
    for part in cookie_header.split(";"):
        part = part.strip()
        if "=" in part:
            name, _, val = part.partition("=")
            name = name.strip()
            val  = val.strip()
            if name.lower() not in _COOKIE_ATTRS:
                cookies[name] = val
    return cookies


def filter_headers(headers: dict, skip_cookies: bool = False) -> dict:
    """Return only the headers JMeter needs in its Header Manager."""
    result: dict = {}
    for k, v in (headers or {}).items():
        kl = k.lower()
        if kl in JMETER_AUTO_HEADERS:
            continue
        if skip_cookies and kl == "cookie":
            continue
        result[k] = v
    return result


def make_label(session: dict, parsed: dict) -> str:
    """Build a descriptive JMeter sampler label."""
    req     = session.get("request", {})
    method  = req.get("method", "")
    path    = parsed.get("path", "")
    comment = session.get("comment", "")
    sid     = session.get("session_id", "")
    if len(path) > 55:
        path = path[:55] + "…"
    return f"{comment} | {sid} | {method} {path}"


def parse_fiddler_time(time_str: str):
    """Parse Fiddler StartTime 'HH:MM:SS.mmm' → datetime (today's date)."""
    try:
        t = datetime.strptime(time_str.strip(), "%H:%M:%S.%f")
        return t
    except Exception:
        return None


# ── Per-session extraction ────────────────────────────────────────────────────

def extract_session(session: dict, skip_options: bool) -> dict | None:
    """
    Extract all JMeter-relevant data from one session.
    Returns None if the session should be skipped.
    """
    req    = session.get("request", {})
    res    = session.get("response", {})
    method = req.get("method", "").upper()
    url    = req.get("url", "")

    # ── Skip rules ────────────────────────────────────────────────────────────
    if method in SKIP_METHODS:
        return None                              # TLS CONNECT tunnel

    if skip_options and method == "OPTIONS":
        return None                              # CORS preflight

    if not url.startswith("http"):
        return None                              # non-HTTP (relative, CONNECT)

    parsed = parse_url(url)
    if not parsed:
        return None

    # ── Request body ─────────────────────────────────────────────────────────
    body         = req.get("body", "") or ""
    content_type = req.get("headers", {}).get("Content-Type", "")
    body_format  = detect_body_format(body, content_type)

    if body_format == "binary":
        body = None          # discard binary blobs (gzip HTML etc.)

    # ── Headers for JMeter Header Manager ────────────────────────────────────
    raw_headers     = req.get("headers", {}) or {}
    cookie_header   = raw_headers.get("Cookie", "")
    cookies         = parse_cookies(cookie_header) if cookie_header else {}
    # Exclude Cookie from Header Manager – it is tracked separately in the
    # 'cookies' dict so the agent uses JMeter Cookie Manager (not Header Manager)
    # for cookie handling, avoiding duplication.
    jmeter_headers  = filter_headers(raw_headers, skip_cookies=True)

    # ── Authorization ─────────────────────────────────────────────────────────
    auth_header = raw_headers.get("Authorization", "")
    auth_type   = ""
    auth_token  = ""
    if auth_header:
        parts     = auth_header.split(" ", 1)
        auth_type = parts[0]
        auth_token = parts[1] if len(parts) > 1 else ""

    # ── Fiddler start-time (for think time estimation) ────────────────────────
    # StartTime appears in Fiddler-injected CONNECT response headers
    fiddler_start = session.get("response", {}).get("headers", {}).get("StartTime", "")

    return {
        "label":          make_label(session, parsed),
        "session_id":     session.get("session_id"),
        "comment":        session.get("comment", ""),

        # ── HTTP Request Sampler fields ───────────────────────────────────────
        "http_sampler": {
            "method":        method,
            "protocol":      parsed["protocol"],
            "server_name":   parsed["server_name"],
            "port":          parsed["port"],
            "path":          parsed["path"],
            "query_params":  parsed["query_params"],
            "body_format":   body_format,
            "body":          body,
            "content_type":  content_type,
            "follow_redirects": True,
            "keep_alive":       True,
            "use_multipart":    body_format == "form",
        },

        # ── HTTP Header Manager ───────────────────────────────────────────────
        "header_manager": jmeter_headers,

        # ── Cookie breakdown (helps set up JMeter Cookie Manager) ─────────────
        "cookies": cookies,

        # ── Authorization info ────────────────────────────────────────────────
        "authorization": {
            "type":     auth_type,
            "token":    auth_token[:80] + ("…" if len(auth_token) > 80 else ""),
        } if auth_header else None,

        # ── Response Assertion ────────────────────────────────────────────────
        "assertion": {
            "response_code":    res.get("status_code", ""),
            "response_message": res.get("status_text", ""),
            "response_hint":    _derive_response_hint(
                res.get("body", "") or "",
                res.get("headers", {}).get("Content-Type", ""),
            ),
        },

        # ── Think time hint (from Fiddler, if present) ────────────────────────
        "_fiddler_start_time": fiddler_start or None,
    }


# ── Think time calculator ─────────────────────────────────────────────────────

def attach_think_times(samplers: list) -> None:
    """
    Mutate each sampler in-place, adding think_time_ms where computable.
    Think time = gap between consecutive Fiddler start-times (ms).
    """
    times = []
    for s in samplers:
        t = parse_fiddler_time(s.get("_fiddler_start_time") or "")
        times.append(t)

    for i, sampler in enumerate(samplers):
        if i > 0 and times[i] and times[i - 1]:
            delta_ms = (times[i] - times[i - 1]).total_seconds() * 1000
            # Ignore negative or unrealistically large gaps (> 5 min)
            if 0 <= delta_ms <= 300_000:
                sampler["think_time_ms"] = round(delta_ms)
            else:
                sampler["think_time_ms"] = None
        else:
            sampler["think_time_ms"] = None

        # Remove internal field
        sampler.pop("_fiddler_start_time", None)


# ── JSON loader with auto-repair ─────────────────────────────────────────────

def _repair_json(text: str) -> str:
    """
    Best-effort repair of common JSON corruption found in SAZ exports.
    Currently handles: bare (unquoted) identifiers that appear between
    a JSON string value and the next comma, e.g.
        "x-processinfo": "firefox:26136",admin_test1,
    is repaired to:
        "x-processinfo": "firefox:26136",
    """
    # Remove bare-word identifiers sitting after a string-value comma pair
    # Pattern: "string_value", bare_word,   →  "string_value",
    repaired = re.sub(
        r'("(?:[^"\\]|\\.)*")\s*,\s*([A-Za-z_]\w*)\s*,',
        r'\1,',
        text,
    )
    return repaired


def load_json_file(path: Path) -> tuple:
    """
    Load a JSON file.  On failure, attempt auto-repair and retry.
    Returns (data_dict, warnings_list).
    """
    warnings: list = []
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    try:
        return json.loads(raw), warnings
    except json.JSONDecodeError as first_err:
        warnings.append(f"JSON parse error on first attempt: {first_err}")
        repaired = _repair_json(raw)
        try:
            data = json.loads(repaired)
            warnings.append("Auto-repair succeeded (stripped stray bare-word tokens).")
            return data, warnings
        except json.JSONDecodeError as second_err:
            warnings.append(f"Auto-repair failed: {second_err}")
            return {}, warnings


# ── Group-file processor ──────────────────────────────────────────────────────

def process_group_file(json_path: Path, scenario_name: str,
                       output_dir: Path, skip_options: bool) -> dict:
    """
    Process one group JSON file (e.g. Login.json) and write output.
    Returns a summary dict for the scenario-level summary.
    """
    data, load_warnings = load_json_file(json_path)
    if load_warnings:
        for w in load_warnings:
            print(f"    ⚠  {json_path.name}: {w}")

    sessions   = data.get("sessions", [])
    # Strip numeric flow-order prefix (e.g. "001_Login" → "Login")
    group_name = re.sub(r'^\d+_', '', json_path.stem)

    # Counters
    n_connect = n_options = n_no_url = 0
    samplers: list = []

    for session in sessions:
        method = session.get("request", {}).get("method", "").upper()
        url    = session.get("request", {}).get("url", "")

        if method == "CONNECT":
            n_connect += 1
            continue
        if skip_options and method == "OPTIONS":
            n_options += 1
            continue
        if not url.startswith("http"):
            n_no_url += 1
            continue

        result = extract_session(session, skip_options)
        if result:
            samplers.append(result)

    attach_think_times(samplers)

    # Collect unique servers used in this group
    unique_servers = sorted({
        f"{s['http_sampler']['protocol']}://"
        f"{s['http_sampler']['server_name']}:"
        f"{s['http_sampler']['port']}"
        for s in samplers
    })

    # Collect all header keys used
    header_keys = sorted({
        k for s in samplers for k in s["header_manager"].keys()
    })

    # Collect all methods used
    methods_used = sorted({s["http_sampler"]["method"] for s in samplers})

    output = {
        "scenario":   scenario_name,
        "group":      group_name,
        "source_file": str(json_path),

        # ── Statistics ────────────────────────────────────────────────────────
        "statistics": {
            "total_sessions":      len(sessions),
            "actionable_requests": len(samplers),
            "skipped_connect":     n_connect,
            "skipped_options":     n_options,
            "skipped_no_url":      n_no_url,
        },

        # ── JMeter Test Plan hints ─────────────────────────────────────────────
        "jmeter_hints": {
            "unique_servers":       unique_servers,
            "http_request_defaults": _suggest_defaults(unique_servers),
            "header_keys_used":     header_keys,
            "methods_used":         methods_used,
            "has_auth_headers":     any(s["authorization"] for s in samplers),
            "has_cookies":          any(s["cookies"] for s in samplers),
            "has_post_body":        any(
                s["http_sampler"]["body"] for s in samplers
            ),
        },

        # ── HTTP Samplers ─────────────────────────────────────────────────────
        "http_samplers": samplers,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{group_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"    ✓ {group_name}.json  "
          f"({len(samplers)} samplers  |  "
          f"{n_connect} CONNECT skipped  |  "
          f"{n_options} OPTIONS skipped)")
    return output


def _suggest_defaults(servers: list) -> dict:
    """
    Suggest HTTP Request Defaults values if all requests share one server.
    """
    if len(servers) == 1:
        try:
            p = urlsplit(servers[0])
            return {
                "protocol":    p.scheme,
                "server_name": p.hostname or "",
                "port":        str(p.port or ""),
                "note": "Single server – safe to set as HTTP Request Defaults",
            }
        except Exception:
            pass
    return {
        "note": f"{len(servers)} different servers – set per-sampler or use variables"
    }


# ── Scenario-level summary ────────────────────────────────────────────────────

def write_scenario_summary(scenario_name: str, group_outputs: list,
                            output_dir: Path) -> None:
    """Write _summary.json for the whole scenario."""
    all_servers: set = set()
    all_header_keys: set = set()
    total_sessions = 0
    total_samplers = 0
    groups_summary = []

    for g in group_outputs:
        stats = g["statistics"]
        hints = g["jmeter_hints"]
        total_sessions += stats["total_sessions"]
        total_samplers += stats["actionable_requests"]
        all_servers.update(hints["unique_servers"])
        all_header_keys.update(hints["header_keys_used"])
        groups_summary.append({
            "group":               g["group"],
            "total_sessions":      stats["total_sessions"],
            "actionable_requests": stats["actionable_requests"],
            "skipped_connect":     stats["skipped_connect"],
            "skipped_options":     stats["skipped_options"],
            "has_auth":            hints["has_auth_headers"],
            "has_cookies":         hints["has_cookies"],
            "has_post_body":       hints["has_post_body"],
        })

    summary = {
        "scenario": scenario_name,

        "totals": {
            "total_sessions":      total_sessions,
            "actionable_requests": total_samplers,
        },

        "groups": groups_summary,

        "jmeter_test_plan_guide": {
            "thread_group_name":    scenario_name,
            "all_servers":          sorted(all_servers),
            "http_request_defaults": _suggest_defaults(sorted(all_servers)),
            "all_header_keys_seen": sorted(all_header_keys),
            "recommended_elements": _recommend_elements(group_outputs),
        },
    }

    out_path = output_dir / "_summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✓ _summary.json  (totals: {total_sessions} sessions, "
          f"{total_samplers} actionable requests)")


def _recommend_elements(group_outputs: list) -> list:
    """Return a list of JMeter element recommendations based on the data."""
    recs: list = []
    has_auth    = any(g["jmeter_hints"]["has_auth_headers"] for g in group_outputs)
    has_cookies = any(g["jmeter_hints"]["has_cookies"]      for g in group_outputs)
    has_post    = any(g["jmeter_hints"]["has_post_body"]     for g in group_outputs)
    servers     = {
        s for g in group_outputs for s in g["jmeter_hints"]["unique_servers"]
    }

    recs.append("HTTP Request Defaults – set base URL / server")

    if len(servers) == 1:
        recs.append("HTTP Request Defaults – single server detected, "
                    "configure protocol/host/port globally")

    if has_cookies:
        recs.append("HTTP Cookie Manager – cookies detected in requests")

    if has_auth:
        recs.append("HTTP Header Manager (Thread Group level) – "
                    "Authorization header detected; use correlation variable")

    if has_post:
        recs.append("HTTP Request Body – POST/PUT bodies detected; "
                    "review body_format field for JSON/XML/form handling")

    recs.append("Response Assertion – use assertion.response_code per sampler")
    recs.append("Gaussian Random Timer – add think time between samplers")
    recs.append("CSV Data Set Config – parameterize credentials/user-data fields")
    recs.append("Regular Expression / JSON Extractor – handle correlations "
                "(see report.json from analyze_dynamic.py)")

    return recs


# ── Main ──────────────────────────────────────────────────────────────────────

def main(grouped_root: Path = DEFAULT_GROUPED_ROOT,
         output_root:  Path = DEFAULT_OUTPUT_ROOT,
         skip_options: bool = False) -> None:

    grouped_root = Path(grouped_root)
    output_root  = Path(output_root)

    if not grouped_root.exists():
        print(f"\n  ✗ Grouped folder not found: {grouped_root}")
        return

    scenario_dirs = sorted(d for d in grouped_root.iterdir() if d.is_dir())
    if not scenario_dirs:
        print(f"\n  ✗ No scenario sub-folders found in: {grouped_root}")
        return

    SEP  = "=" * 68
    THIN = "-" * 68
    print(f"\n{SEP}")
    print("  JMETER DATA EXTRACTION")
    print(SEP)
    print(f"  Source  : {grouped_root}")
    print(f"  Output  : {output_root}")
    print(f"  Scenarios found : {len(scenario_dirs)}")
    print(f"  Skip OPTIONS    : {skip_options}")
    print(THIN)

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        print(f"\n  Scenario : {scenario_name}")
        print(f"  {THIN[:50]}")

        # ── Discover group files ──────────────────────────────────────────
        # Sort by filename so numeric-prefixed files (001_Login.json, …)
        # are processed in UI-flow order.  Skip _flow_order.json and other
        # underscore-prefixed helper files.
        json_files = sorted(
            f for f in scenario_dir.glob("*.json")
            if not f.name.startswith("_")
        )
        if not json_files:
            print("    ⚠  No JSON files found – skipping")
            continue

        output_scenario_dir = output_root / scenario_name
        group_outputs: list = []
        flow_order:    list = []   # group names in UI-flow order

        for json_file in json_files:
            result = process_group_file(
                json_file, scenario_name, output_scenario_dir, skip_options
            )
            group_outputs.append(result)
            flow_order.append(result.get("group", json_file.stem))

        # Write flow-order index so downstream stages (agent_input_prep) can
        # process files in the same UI-flow order without relying on sort.
        order_path = output_scenario_dir / "_flow_order.json"
        output_scenario_dir.mkdir(parents=True, exist_ok=True)
        with open(order_path, "w", encoding="utf-8") as _f:
            json.dump(flow_order, _f, indent=2, ensure_ascii=False)

        write_scenario_summary(scenario_name, group_outputs, output_scenario_dir)

    print(f"\n{SEP}")
    print(f"  ✓ All done!  Output folder → {output_root}")
    print(SEP + "\n")



# ─── Config dataclass ─────────────────────────────────────────────────────────

@dataclass
class Config:
    """
    All tunable parameters in one place.
    Defaults are generic / standard – no site-specific values.
    Override via config file (--config), CLI flags, or directly in Python.
    """

    # Token length thresholds
    min_token_len: int = 8      # shorter values are too generic to track
    max_token_len: int = 1500   # longer values are likely binary noise

    # Fiddler-internal session flags – always run-specific, never correlatable
    # (these come from the SAZ format itself, not from any particular website)
    skip_fiddler_flags: set = field(default_factory=lambda: {
        "x-clientport", "x-egressport", "x-hostip", "x-clientip",
        "x-processinfo", "starttime", "endtime",
        "https-client-sessionid", "x-responsebodytransferlength",
        "x-serversocket",
    })

    # Query params that change every run but carry NO correlation value.
    # Only truly standard/protocol-level params here – NO site-specific ones.
    volatile_query_keys: set = field(default_factory=lambda: {
        # OAuth 2 PKCE – always different per run, not worth tracking
        "code_challenge", "code_challenge_method",
        # Common anti-cache / anti-replay params used across many frameworks
        "nonce", "ts", "_", "rand", "t", "nocache", "v",
    })

    # Request headers that are noise for dynamic-value comparison
    skip_request_headers: set = field(default_factory=lambda: {
        "content-length", "user-agent", "accept",
        "accept-encoding", "accept-language",
        "cache-control", "pragma",
    })

    # Response headers that are noise for dynamic-value comparison
    # (CDN / infrastructure headers that change per request but have no
    #  correlation value – add vendor-specific ones via config file)
    skip_response_headers: set = field(default_factory=lambda: {
        "date", "content-length", "etag", "last-modified",
        # AWS CloudFront / S3 – common CDN headers
        "x-amz-version-id", "x-amz-request-id",
        "x-amz-cf-id", "x-cache", "via",
        # Generic infrastructure
        "x-request-id", "x-correlation-id",
        "traceparent", "tracestate",
    })


# ─── Config file & CLI builder ────────────────────────────────────────────────

def load_config_file(path) -> dict:
    """Load an optional JSON config file and return it as a dict."""
    if path and Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_config(config_file_data=None, args=None) -> Config:
    """
    Build a Config by merging: defaults → config file → CLI args.

    Config file keys (all optional):
        min_token_length        int
        max_token_length        int
        volatile_query_keys     list[str]
        skip_request_headers    list[str]
        skip_response_headers   list[str]
        skip_fiddler_flags      list[str]
    """
    cfg = Config()
    data = config_file_data or {}

    if data.get("min_token_length"):
        cfg.min_token_len = int(data["min_token_length"])
    if data.get("max_token_length"):
        cfg.max_token_len = int(data["max_token_length"])
    for k in data.get("volatile_query_keys", []):
        cfg.volatile_query_keys.add(k.lower())
    for h in data.get("skip_request_headers", []):
        cfg.skip_request_headers.add(h.lower())
    for h in data.get("skip_response_headers", []):
        cfg.skip_response_headers.add(h.lower())
    for f in data.get("skip_fiddler_flags", []):
        cfg.skip_fiddler_flags.add(f.lower())

    if args:
        if getattr(args, "min_token", None) is not None:
            cfg.min_token_len = args.min_token
        if getattr(args, "max_token", None) is not None:
            cfg.max_token_len = args.max_token
        for p in getattr(args, "volatile_param", []):
            cfg.volatile_query_keys.add(p.lower())
        for h in getattr(args, "skip_header", []):
            cfg.skip_response_headers.add(h.lower())

    return cfg


# ─── Default folder & auto-discovery ─────────────────────────────────────────

DEFAULT_JSON_FOLDER = Path("json(.saz)_Files")


def discover_json_files(folder: Path = DEFAULT_JSON_FOLDER) -> tuple:
    """
    Scan *folder* for JSON files and return the first two found
    (sorted by name for deterministic ordering).

    Raises FileNotFoundError  if the folder does not exist.
    Raises ValueError         if fewer than 2 JSON files are found.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"JSON folder not found: {folder}")

    json_files = sorted(folder.glob("*.json"))

    if len(json_files) < 2:
        raise ValueError(
            f"Expected at least 2 JSON files in '{folder}', "
            f"found {len(json_files)}: {[f.name for f in json_files]}"
        )

    return json_files[0], json_files[1]


# ─── Body / text utilities ────────────────────────────────────────────────────

def strip_chunked(text: str) -> str:
    """Remove Transfer-Encoding: chunked size lines before JSON parsing."""
    chunk_line = re.compile(r'^[0-9a-fA-F]+\r?\n')
    return "\n".join(
        line for line in text.split("\n") if not chunk_line.match(line)
    ).strip()


def try_parse_json(text: str):
    """Return parsed JSON object/array or None on failure."""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        try:
            return json.loads(strip_chunked(text))
        except Exception:
            return None


def flatten_json(obj, prefix="") -> dict:
    """Recursively flatten a JSON object to {dotted.path: str_value} pairs."""
    result = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            result.update(flatten_json(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            result.update(flatten_json(v, f"{prefix}[{i}]"))
    elif isinstance(obj, (str, int, float)) and obj not in (True, False, None):
        result[prefix] = str(obj)
    return result


def try_parse_xml(text: str) -> dict:
    """
    Parse XML body (SOAP / XML REST) and flatten to {tag_path: text_value}.
    Returns empty dict on failure.
    """
    try:
        root = ET.fromstring(text.strip())
        return _flatten_xml(root)
    except Exception:
        return {}


def _flatten_xml(el, prefix="") -> dict:
    """Recursively flatten an XML element tree to {path: value}."""
    # Strip XML namespace if present: {http://…}tag → tag
    tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
    path = f"{prefix}.{tag}" if prefix else tag
    result = {}
    if el.text and el.text.strip():
        result[path] = el.text.strip()
    for attr, val in el.attrib.items():
        result[f"{path}[@{attr}]"] = val
    for child in el:
        result.update(_flatten_xml(child, path))
    return result


def parse_cookie_header(cookie_str: str) -> dict:
    """
    Parse Cookie / Set-Cookie header into {name: value} pairs.
    Skips cookie attributes (path, domain, expires, …).
    """
    _ATTR = {"path", "domain", "expires", "max-age", "samesite",
             "secure", "httponly", "version"}
    values = {}
    for part in cookie_str.split(";"):
        part = part.strip()
        if "=" in part:
            name, _, val = part.partition("=")
            name, val = name.strip(), val.strip()
            if name.lower() not in _ATTR and val:
                values[name] = val
    return values


def parse_form_body(body: str) -> dict:
    """URL-form-encoded body → {key: value}."""
    try:
        return dict(parse_qsl(body, keep_blank_values=True))
    except Exception:
        return {}


def is_binary(text: str) -> bool:
    """Heuristic: body is binary/gzip if > 5% replacement characters."""
    return text.count("\ufffd") > len(text) * 0.05


# ─── Session loading & matching ───────────────────────────────────────────────

def load_sessions(json_file_path) -> list:
    with open(json_file_path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sessions", [])


def _norm_url(url: str, cfg: Config) -> str:
    """Normalise URL for session matching – strip cfg.volatile_query_keys."""
    if not url.startswith("http"):
        return url.lower()
    p = urlsplit(url)
    q = sorted(
        (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
        if k.lower() not in cfg.volatile_query_keys
    )
    return urlunsplit((p.scheme.lower(), p.netloc.lower(), p.path, urlencode(q), ""))


def _session_key(session: dict, cfg: Config) -> tuple:
    req = session.get("request", {})
    return (req.get("method", "").upper(), _norm_url(req.get("url", ""), cfg))


def match_sessions(sessions1: list, sessions2: list, cfg: Config) -> list:
    """
    Pair sessions between run 1 and run 2 by (method, normalised URL).
    Returns [(s1, s2), …] preserving order of run 1.
    """
    map2 = defaultdict(list)
    for s in sessions2:
        map2[_session_key(s, cfg)].append(s)

    used = set()
    pairs = []
    for s1 in sessions1:
        for c in map2.get(_session_key(s1, cfg), []):
            if id(c) not in used:
                used.add(id(c))
                pairs.append((s1, c))
                break
    return pairs


# ─── Dynamic value detection ──────────────────────────────────────────────────

def _make_dynamic(location, field_name, s1, value_run1, value_run2):
    req = s1.get("request", {})
    return {
        "location":        location,
        "field":           field_name,
        "comment":         s1.get("comment", ""),
        "session_id_run1": s1.get("session_id"),
        "method":          req.get("method", ""),
        "url":             req.get("url", ""),
        "value_run1":      str(value_run1)[:300],
        "value_run2":      str(value_run2)[:300],
    }


def _compare_dicts(dict1, dict2, location, s1, cfg: Config, skip_keys=None):
    """Yield dynamic records for every key whose value differs."""
    skip = {k.lower() for k in (skip_keys or set())}
    for key in set(dict1) | set(dict2):
        if key.lower() in skip:
            continue
        v1 = str(dict1.get(key, ""))
        v2 = str(dict2.get(key, ""))
        if v1 != v2 and len(v1) >= cfg.min_token_len:
            yield _make_dynamic(location, key, s1, v1, v2)


def find_dynamic_values(pairs: list, cfg: Config) -> list:
    """
    Compare matched (run1, run2) session pairs field by field.
    Supports: request headers, URL query params, request body (JSON / form / XML),
              response headers, response body (JSON / XML).
    Returns a deduplicated list of dynamic field records.
    """
    seen    = set()
    results = []

    def _add(record):
        key = (record["location"], record["field"],
               record["method"], record["url"])
        if key not in seen:
            seen.add(key)
            results.append(record)

    for s1, s2 in pairs:
        req1 = s1.get("request", {})
        req2 = s2.get("request", {})
        res1 = s1.get("response", {})
        res2 = s2.get("response", {})

        # ── Request headers ──────────────────────────────────────────────────
        for rec in _compare_dicts(
            req1.get("headers", {}), req2.get("headers", {}),
            "request.headers", s1, cfg,
            skip_keys=cfg.skip_request_headers
        ):
            _add(rec)

        # ── Request URL query params ─────────────────────────────────────────
        u1, u2 = req1.get("url", ""), req2.get("url", "")
        if u1 != u2 and u1.startswith("http"):
            q1 = dict(parse_qsl(urlsplit(u1).query, keep_blank_values=True))
            q2 = dict(parse_qsl(urlsplit(u2).query, keep_blank_values=True))
            for rec in _compare_dicts(q1, q2, "request.url.query", s1, cfg,
                                      skip_keys=cfg.volatile_query_keys):
                _add(rec)

        # ── Request body (JSON / XML / form) ─────────────────────────────────
        b1 = req1.get("body", "") or ""
        b2 = req2.get("body", "") or ""
        ct = req1.get("headers", {}).get("Content-Type", "").lower()
        if b1 and b2 and b1 != b2 and not is_binary(b1):
            if "json" in ct or b1.strip().startswith(("{", "[")):
                j1, j2 = try_parse_json(b1), try_parse_json(b2)
                if j1 and j2:
                    for rec in _compare_dicts(
                        flatten_json(j1), flatten_json(j2),
                        "request.body(json)", s1, cfg
                    ):
                        _add(rec)
            elif "xml" in ct or b1.strip().startswith("<"):
                x1, x2 = try_parse_xml(b1), try_parse_xml(b2)
                if x1 and x2:
                    for rec in _compare_dicts(
                        x1, x2, "request.body(xml)", s1, cfg
                    ):
                        _add(rec)
            elif "form" in ct or ("=" in b1 and "&" in b1):
                for rec in _compare_dicts(
                    parse_form_body(b1), parse_form_body(b2),
                    "request.body(form)", s1, cfg
                ):
                    _add(rec)

        # ── Response headers ─────────────────────────────────────────────────
        for rec in _compare_dicts(
            res1.get("headers", {}), res2.get("headers", {}),
            "response.headers", s1, cfg,
            skip_keys=cfg.skip_fiddler_flags | cfg.skip_response_headers
        ):
            _add(rec)

        # ── Response body (JSON / XML; skip binary/gzip) ─────────────────────
        rb1 = res1.get("body", "") or ""
        rb2 = res2.get("body", "") or ""
        rct = res1.get("headers", {}).get("Content-Type", "").lower()
        if rb1 and rb2 and rb1 != rb2 and not is_binary(rb1):
            if "json" in rct:
                rj1, rj2 = try_parse_json(rb1), try_parse_json(rb2)
                if rj1 and rj2:
                    for rec in _compare_dicts(
                        flatten_json(rj1), flatten_json(rj2),
                        "response.body(json)", s1, cfg
                    ):
                        _add(rec)
            elif "xml" in rct or rb1.strip().startswith("<"):
                rx1, rx2 = try_parse_xml(rb1), try_parse_xml(rb2)
                if rx1 and rx2:
                    for rec in _compare_dicts(
                        rx1, rx2, "response.body(xml)", s1, cfg
                    ):
                        _add(rec)

    return results


# ─── Token extraction (for correlation search) ────────────────────────────────

def extract_tokens(df: dict, cfg: Config) -> list:
    """
    Extract the concrete token string(s) from a dynamic field record.
    Returns [(token_string, extraction_note), …].
    """
    val   = df.get("value_run1", "")
    loc   = df.get("location", "")
    fname = df.get("field", "").lower()
    tokens = []

    if not val or len(val) < cfg.min_token_len or len(val) > cfg.max_token_len:
        return tokens

    # Cookie / Set-Cookie – extract individual name→value pairs
    if "cookie" in fname:
        for name, cval in parse_cookie_header(val).items():
            if len(cval) >= cfg.min_token_len:
                tokens.append((cval, f"cookie[{name}]"))
        return tokens

    # Authorization Bearer token
    if fname == "authorization" and val.lower().startswith("bearer "):
        bearer = val.split(" ", 1)[1].strip()
        if len(bearer) >= cfg.min_token_len:
            tokens.append((bearer, "bearer_token"))
        return tokens

    # Location redirect – extract each meaningful query param value
    if fname == "location":
        try:
            for k, v in parse_qsl(urlsplit(val).query, keep_blank_values=True):
                if len(v) >= cfg.min_token_len:
                    tokens.append((v, f"redirect_param[{k}]"))
        except Exception:
            pass
        if not tokens:
            tokens.append((val, "location_url"))
        return tokens

    # URL query param
    if "url.query" in loc:
        tokens.append((val, f"query[{df['field']}]"))
        return tokens

    # Body field (JSON / XML / form)
    if "body" in loc:
        tokens.append((val, f"body_field[{df['field']}]"))
        return tokens

    # Default – whole value
    tokens.append((val, f"{loc}.{fname}"))
    return tokens


# ─── Text blobs for quick search ─────────────────────────────────────────────

def _request_blob(session: dict) -> str:
    req  = session.get("request", {})
    body = req.get("body", "") or ""
    if req.get("method") == "CONNECT":
        body = ""   # suppress TLS handshake noise
    parts = [req.get("url", "")]
    parts.extend(str(v) for v in (req.get("headers", {}) or {}).values())
    parts.append(body)
    return "\n".join(parts)


def _response_blob(session: dict) -> str:
    res  = session.get("response", {})
    body = res.get("body", "") or ""
    if is_binary(body):
        body = ""
    parts = list(str(v) for v in (res.get("headers", {}) or {}).values())
    parts.append(body)
    return "\n".join(parts)


def _find_field_in(obj, token: str, prefix=""):
    """Return the first dotted JSON path in obj whose string value contains token."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else k
            hit = _find_field_in(v, token, p)
            if hit:
                return hit
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hit = _find_field_in(v, token, f"{prefix}[{i}]")
            if hit:
                return hit
    elif isinstance(obj, str) and token in obj:
        return prefix
    return None


def _find_in_url(url: str, token: str) -> str | None:
    """
    Decompose a URL and return a precise field path for where *token* appears.

    Priority (most specific first):
        request.url.query[<param>]  – token found in a specific query-param value
        request.url.path            – token found in the URL path
        request.url.host            – token found in the hostname / netloc
        request.url                 – token somewhere in the URL (fallback)

    Returns None if token is not in url at all.
    """
    if not url or token not in url:
        return None
    try:
        p = urlsplit(url)
        # Query params – most actionable for JMeter parameterization
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            if token in v:
                return f"request.url.query[{k}]"
        # Path segment
        if token in p.path:
            return "request.url.path"
        # Hostname (less common to parameterize, but still precise)
        if token in p.netloc:
            return "request.url.host"
    except Exception:
        pass
    return "request.url"


def _find_in_request(req: dict, token: str) -> str | None:
    """
    URL-aware, precise field finder for a request dict.

    Checks in priority order:
      1. URL components  → request.url.query[param] / .path / .host
      2. Individual request headers → request.headers.<HeaderName>
      3. Body (JSON-flattened if possible) → request.body.<dotted.path>

    Returns None if token is not found anywhere.
    """
    # 1. URL – decompose properly
    url = req.get("url", "")
    if token in url:
        return _find_in_url(url, token)

    # 2. Headers – each header is its own field
    for hname, hval in (req.get("headers", {}) or {}).items():
        if token in str(hval):
            return f"request.headers.{hname}"

    # 3. Body – try to flatten JSON for a dotted path
    body = req.get("body", "") or ""
    if token in body:
        j = try_parse_json(body)
        if j:
            fp = _find_field_in(j, token, "request.body")
            if fp:
                return fp
        return "request.body"

    return None


def _find_in_response(res: dict, token: str) -> str | None:
    """
    Precise field finder for a response dict.

    Checks in priority order:
      1. Individual response headers → response.headers.<HeaderName>
      2. Body (JSON-flattened if possible) → response.body.<dotted.path>

    Returns None if token is not found anywhere.
    """
    # 1. Headers
    for hname, hval in (res.get("headers", {}) or {}).items():
        if token in str(hval):
            return f"response.headers.{hname}"

    # 2. Body – try to flatten JSON for a dotted path
    body = res.get("body", "") or ""
    if token in body:
        j = try_parse_json(body)
        if j:
            fp = _find_field_in(j, token, "response.body")
            if fp:
                return fp
        return "response.body"

    return None


# ─── Correlation engine ───────────────────────────────────────────────────────

# Methods skipped by extract_jmeter_data.py – correlation sources/destinations
# must only reference sessions that will actually appear in the JMeter plan.
_JMETER_SKIP_METHODS = {"OPTIONS", "CONNECT"}


def _is_jmeter_actionable(session: dict) -> bool:
    """
    Return True only for sessions that extract_jmeter_data.py keeps.
    Mirrors the skip rules in extract_jmeter_data.extract_session():
      - skip CONNECT  (TLS tunnel)
      - skip OPTIONS  (CORS preflight)
      - skip sessions whose URL does not start with 'http'
    """
    req    = session.get("request", {})
    method = req.get("method", "").upper()
    url    = req.get("url", "")
    if method in _JMETER_SKIP_METHODS:
        return False
    if not url.startswith("http"):
        return False
    return True


def find_correlations(sessions: list, dynamic_fields: list, cfg: Config) -> list:
    """
    For every token extracted from dynamic fields:
      SOURCE  – earliest ACTIONABLE response in the run that contains the token.
                (OPTIONS / CONNECT sessions are excluded because they are skipped
                 by extract_jmeter_data.py and never appear in agent_input files.)
      USED IN – every subsequent ACTIONABLE request that contains the token.
    Returns a deduplicated list of correlation records.
    """
    seen_tokens  = set()
    correlations = []

    # Pre-filter: only keep sessions that will exist in the JMeter plan
    actionable = [s for s in sessions if _is_jmeter_actionable(s)]

    for df in dynamic_fields:
        for token, extraction_note in extract_tokens(df, cfg):
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if token_hash in seen_tokens:
                continue

            # ── Find SOURCE – earliest actionable response containing the token ──
            source = None
            for session in actionable:
                if token in _response_blob(session):
                    fp = _find_in_response(session.get("response", {}), token)
                    source = {
                        "session_id": session.get("session_id"),
                        "comment":    session.get("comment", ""),
                        "method":     session.get("request", {}).get("method", ""),
                        "url":        session.get("request", {}).get("url", "")[:100],
                        "field":      fp or "response(unknown)",
                        "value":      token,   # actual value generated at this source
                    }
                    break

            if not source:
                continue

            seen_tokens.add(token_hash)
            source_sid = source["session_id"]

            # ── Find DESTINATIONS – later actionable requests containing the token ──
            destinations = []
            for session in actionable:
                sid = session.get("session_id")
                if sid <= source_sid:
                    continue
                if token in _request_blob(session):
                    fp = _find_in_request(session.get("request", {}), token)
                    destinations.append({
                        "session_id": sid,
                        "comment":    session.get("comment", ""),
                        "method":     session.get("request", {}).get("method", ""),
                        "url":        session.get("request", {}).get("url", "")[:100],
                        "field":      fp or "request(unknown)",
                    })

            correlations.append({
                "dynamic_field_location": df["location"],
                "dynamic_field_name":     df["field"],
                "extraction_note":        extraction_note,
                "token_value":            token,
                "token_preview":          token[:100] + ("..." if len(token) > 100 else ""),
                "token_length":           len(token),
                "source":                 source,
                "destinations":           destinations,
                "correlated":             bool(destinations),
            })

    return correlations


# ─── Structural correlation (bearer / scoped tokens) ─────────────────────────

def find_structural_correlations(sessions: list, existing_correlations: list,
                                  cfg: Config) -> list:
    """
    Structural detection of bearer/auth tokens that may NOT be caught by
    cross-run comparison (e.g. if both SAZ recordings used the same token value).

    Scans all actionable run-1 sessions for:
      - Authorization: Bearer <token>  in request headers
      - Traces each unique token back to the earliest response that produced it
      - Collects every subsequent request that sends the same token

    Returns only NEW correlation records not already tracked in
    *existing_correlations* (deduplication by SHA-256 of the token value).
    """
    existing_hashes = {
        hashlib.sha256(c["token_value"].encode()).hexdigest()
        for c in existing_correlations
    }

    actionable   = [s for s in sessions if _is_jmeter_actionable(s)]
    new_corrs    = []
    seen_hashes  = set(existing_hashes)

    for session in actionable:
        req  = session.get("request", {})
        auth = (req.get("headers") or {}).get("Authorization", "") or ""
        if not auth.lower().startswith("bearer "):
            continue

        # Strip any suffix appended after the token (e.g. ~bzcbwebMailMiniClient)
        # so we can trace the base token independently.
        raw_token = auth.split(" ", 1)[1].strip()

        # Also collect "base" token if the raw token has a tilde suffix (scoped tokens)
        candidates = [raw_token]
        if "~" in raw_token:
            base = raw_token.split("~")[0]
            if len(base) >= cfg.min_token_len:
                candidates.insert(0, base)   # try base first

        for token in candidates:
            if len(token) < cfg.min_token_len or len(token) > cfg.max_token_len:
                continue

            token_hash = hashlib.sha256(token.encode()).hexdigest()
            if token_hash in seen_hashes:
                continue
            seen_hashes.add(token_hash)

            first_use_sid = session.get("session_id")

            # ── Find SOURCE: earliest response that contains this token ──────
            source = None
            for s in actionable:
                if s.get("session_id") >= first_use_sid:
                    break
                if token in _response_blob(s):
                    fp = _find_in_response(s.get("response", {}), token)
                    source = {
                        "session_id": s.get("session_id"),
                        "comment":    s.get("comment", ""),
                        "method":     s.get("request", {}).get("method", ""),
                        "url":        s.get("request", {}).get("url", "")[:100],
                        "field":      fp or "response(unknown)",
                        "value":      token,
                    }
                    break

            if not source:
                continue

            source_sid = source["session_id"]

            # ── Find DESTINATIONS: later requests that carry this token ──────
            destinations   = []
            seen_dest_sids: set = set()
            for s in actionable:
                dsid = s.get("session_id")
                if dsid <= source_sid or dsid in seen_dest_sids:
                    continue
                if token in _request_blob(s):
                    fp = _find_in_request(s.get("request", {}), token)
                    if fp:
                        seen_dest_sids.add(dsid)
                        destinations.append({
                            "session_id": dsid,
                            "comment":    s.get("comment", ""),
                            "method":     s.get("request", {}).get("method", ""),
                            "url":        s.get("request", {}).get("url", "")[:100],
                            "field":      fp,
                        })

            new_corrs.append({
                "dynamic_field_location": "request.headers",
                "dynamic_field_name":     "Authorization",
                "extraction_note":        "bearer_token",
                "token_value":            token,
                "token_preview":          token[:100] + ("..." if len(token) > 100 else ""),
                "token_length":           len(token),
                "source":                 source,
                "destinations":           destinations,
                "correlated":             bool(destinations),
                "detected_by":            "structural",
            })

    return new_corrs


# ─── Grouping & verification ─────────────────────────────────────────────────

def group_sessions_by_comment(sessions: list) -> dict:
    """
    Split a session list into groups keyed by the 'comment' field.
    Sessions with an empty/missing comment go under the key '(no comment)'.
    Returns an OrderedDict preserving insertion order.
    """
    groups = defaultdict(list)
    for s in sessions:
        key = s.get("comment", "").strip() or "(no comment)"
        groups[key].append(s)
    return dict(groups)


def save_groups(sessions: list, source_file: Path,
                grouped_root: Path = None) -> Path:
    """
    Write one JSON file per comment-group under:
        <grouped_root>/<source_file_stem>/<NNN_Comment>.json

    Files are prefixed with a zero-padded index (001_, 002_, …) so that
    file-system ordering matches the user's UI flow captured in Run 1.

    A companion  _flow_order.json  file is written listing group names
    in flow order – used by downstream stages to preserve that order.

    Returns the folder created for this file.
    """
    if grouped_root is None:
        grouped_root = source_file.parent / "grouped"

    folder = grouped_root / source_file.stem
    folder.mkdir(parents=True, exist_ok=True)

    groups = group_sessions_by_comment(sessions)
    group_order = list(groups.keys())   # insertion order = UI flow order

    # Write flow-order index
    order_path = folder / "_flow_order.json"
    with open(order_path, "w", encoding="utf-8") as f:
        json.dump(group_order, f, indent=2, ensure_ascii=False)

    for idx, (comment, grp_sessions) in enumerate(groups.items(), 1):
        safe_name = re.sub(r'[^\w\-. ]', '_', comment).strip()
        out_path  = folder / f"{idx:03d}_{safe_name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"sessions": grp_sessions}, f, indent=2, ensure_ascii=False)

    return folder


def verify_group_counts(sessions1: list, sessions2: list,
                        file_a: Path, file_b: Path) -> bool:
    """
    Compare total and per-group (comment) session counts between two runs.
    Prints a human-readable table and returns True if all counts match.
    """
    THIN = "-" * 72
    g1 = group_sessions_by_comment(sessions1)
    g2 = group_sessions_by_comment(sessions2)

    # Preserve Run-1 flow order, then append any Run-2-only groups
    seen: set = set()
    all_groups: list = []
    for grp in list(g1.keys()) + list(g2.keys()):
        if grp not in seen:
            seen.add(grp)
            all_groups.append(grp)

    print(f"\n{THIN}")
    print("  REQUEST COUNT VERIFICATION")
    print(THIN)
    print(f"  {'Group':<25} {'Run 1':>8} {'Run 2':>8}  {'Match?':>8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}  {'-'*8}")

    all_match = True
    for grp in all_groups:
        c1 = len(g1.get(grp, []))
        c2 = len(g2.get(grp, []))
        match = "✓" if c1 == c2 else "✗ DIFF"
        if c1 != c2:
            all_match = False
        print(f"  {grp:<25} {c1:>8} {c2:>8}  {match:>8}")

    total1, total2 = len(sessions1), len(sessions2)
    t_match = "✓" if total1 == total2 else "✗ DIFF"
    if total1 != total2:
        all_match = False
    print(f"  {'TOTAL':<25} {total1:>8} {total2:>8}  {t_match:>8}")
    print(THIN)

    if all_match:
        print("  ✓ All group counts match between the two runs.\n")
    else:
        print("  ⚠  Count mismatch detected – analysis proceeds per group.\n")

    return all_match


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(dynamic_fields: list, correlations: list, file_a, file_b,
                 group_name: str = None):
    SEP  = "=" * 72
    THIN = "-" * 72

    group_label = f"  Group            : {group_name}\n" if group_name else ""
    print(f"\n{SEP}")
    print("  DYNAMIC VALUE & CORRELATION ANALYSIS")
    print(SEP)
    print(f"  Run 1 (baseline) : {file_a}")
    print(f"  Run 2 (compare)  : {file_b}")
    if group_label:
        print(group_label, end="")
    print(THIN)
    print(f"  Dynamic fields detected  : {len(dynamic_fields)}")
    corr_with_dest = sum(1 for c in correlations if c["correlated"])
    print(f"  Correlations found       : {len(correlations)}")
    print(f"  Correlated (source->dest): {corr_with_dest}")
    print(SEP)

    if not dynamic_fields:
        print("\n  No dynamic values detected.\n")
    else:
        print(f"\n{THIN}")
        print("  SECTION 1 - DYNAMIC VALUES")
        print("  (fields whose values differ between Run 1 and Run 2)")
        print(THIN)
        for i, df in enumerate(dynamic_fields, 1):
            print(f"\n  [{i:02d}] {df['location']}.{df['field']}")
            print(f"       Session : {df['session_id_run1']} | "
                  f"{df['comment']} | {df['method']} {df['url'][:65]}")
            print(f"       Run 1   : {df['value_run1'][:110]}")
            print(f"       Run 2   : {df['value_run2'][:110]}")

    if not correlations:
        print("\n  No correlations found.\n")
    else:
        print(f"\n{THIN}")
        print("  SECTION 2 - CORRELATIONS")
        print("  (how each dynamic value flows from a response into later requests)")
        print(THIN)
        for i, c in enumerate(correlations, 1):
            src  = c["source"]
            icon = "[+]" if c["correlated"] else "[ ]"
            print(f"\n  [{i:02d}] {icon} {c['dynamic_field_location']}.{c['dynamic_field_name']}"
                  f"  [{c['extraction_note']}]")
            print(f"       Token ({c['token_length']} chars) : {c['token_preview']}")
            print(f"       SOURCE  --> Session {src['session_id']} | "
                  f"{src['comment']} | {src['method']} {src['url'][:60]}")
            print(f"                   field: {src['field']}")
            if c["destinations"]:
                for dest in c["destinations"]:
                    print(f"       USED IN --> Session {dest['session_id']} | "
                          f"{dest['comment']} | {dest['method']} {dest['url'][:60]}")
                    print(f"                   field: {dest['field']}")
            else:
                print("       USED IN --> (not found in any subsequent request)")

    print(f"\n{SEP}\n")


def _build_dynamic_variables(all_real_correlations: list) -> list:
    """
    Build a flat, deduplicated list of dynamic variables from all correlated entries.

    Variable names are derived from the extraction_note (most specific) rather
    than the raw field name, so:
        cookie[ASP.NET_SessionId]   → ASP_NET_SessionId   (not Set_Cookie)
        redirect_param[token]       → token               (not Location)
        bearer_token                → bearer_token
        body_field[session.userid]  → userid

    HTML body attributes (html.*, [@href], etc.) are skipped – they are
    not API tokens and produce noisy false correlations for JMeter.
    """
    seen   = set()
    result = []
    for c in all_real_correlations:
        token           = c.get("token_value", "")
        extraction_note = c.get("extraction_note", "")
        raw_name        = c.get("dynamic_field_name", "var")

        # ── Skip HTML body attribute correlations (noise for JMeter) ──────────
        # e.g. body_field[html.body.h2.a[@href]] — these are page links, not tokens
        if "html." in extraction_note or "[@" in extraction_note:
            continue
        # Skip if the token value looks like a URL path (full-URL false positive)
        if token.startswith("/") and ("?" in token or "&" in token):
            continue

        if token in seen:
            continue
        seen.add(token)

        # ── Derive variable name from extraction_note (most precise) ──────────
        m = re.search(r'cookie\[([^\]]+)\]', extraction_note)
        if m:
            # cookie[ASP.NET_SessionId] → ASP_NET_SessionId
            var_name = re.sub(r'\W+', '_', m.group(1)).strip("_")
        else:
            m = re.search(r'redirect_param\[([^\]]+)\]', extraction_note)
            if m:
                # redirect_param[token] → token
                var_name = re.sub(r'\W+', '_', m.group(1)).strip("_")
            else:
                m = re.search(r'query\[([^\]]+)\]', extraction_note)
                if m:
                    # query[domainid] → domainid
                    var_name = re.sub(r'\W+', '_', m.group(1)).strip("_")
                elif extraction_note == "bearer_token":
                    var_name = "bearer_token"
                elif extraction_note.startswith("body_field["):
                    # body_field[response.session.userid] → userid
                    inner = re.sub(r'^body_field\[', '', extraction_note).rstrip("]")
                    var_name = re.sub(r'\W+', '_', inner.split(".")[-1]).strip("_")
                else:
                    # Fallback: last segment of the raw field name
                    var_name = re.sub(r'\W+', '_',
                                      raw_name.split(".")[-1]).strip("_")

        if not var_name:
            var_name = "dynamic_var"

        src = c.get("source", {})
        result.append({
            "variable_name":   var_name,
            "field_name":      raw_name,
            "extraction_note": extraction_note,
            "token_value":     token,
            "source_session":  src.get("session_id"),
            "source_method":   src.get("method", ""),
            "source_url":      src.get("url", "")[:120],
            "source_field":    src.get("field", ""),
        })
    return result


def save_group_correlations(group_results: list, file_a, file_b,
                            correlations_dir: Path) -> None:
    """
    Save one JSON file per group under *correlations_dir*/<GroupName>.json.

    Each file contains:
        group             – group name
        run1 / run2       – source file paths
        summary           – count of correlations for this group
        dynamic_variables – deduplicated variable list for this group only
        correlations      – full correlated entries for this group
    """
    correlations_dir = Path(correlations_dir)
    correlations_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for idx, (grp_name, _dyn, corrs) in enumerate(group_results, 1):
        real = [c for c in corrs if c["correlated"]]
        if not real:
            continue

        # Numeric prefix keeps files in UI-flow order on the file system
        safe_name = re.sub(r'[^\w\-. ]', '_', grp_name).strip()
        out_path  = correlations_dir / f"{idx:03d}_{safe_name}.json"

        group_doc = {
            "group":             grp_name,
            "run1":              str(file_a),
            "run2":              str(file_b),
            "summary": {
                "correlations_found": len(real),
            },
            "dynamic_variables": _build_dynamic_variables(real),
            "correlations":      real,
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(group_doc, f, indent=2, ensure_ascii=False)
        print(f"  ✓ [{grp_name}] correlations → {out_path}  ({len(real)} entries)")
        saved += 1

    if saved == 0:
        print(f"  –  No correlated groups to save in {correlations_dir}")


def save_report(dynamic_fields, correlations, file_a, file_b, output_path,
                group_results: list = None,
                correlations_dir: Path = None):
    """
    Save report.json  AND  per-group correlation files.

    If group_results is provided (per-group analysis):
      • report.json          – full combined report with correlations_by_group
      • correlations/<Group>.json – one file per group (new)

    correlations_dir defaults to  <output_path.parent>/correlations/

    Always includes a top-level 'dynamic_variables' list for quick searching.
    """
    output_path = Path(output_path)

    if group_results is not None:
        # Build a per-group section with only real correlations
        groups_out     = {}
        all_real_corrs = []
        total_real     = 0
        for grp_name, _dyn, corrs in group_results:
            real = [c for c in corrs if c["correlated"]]
            total_real += len(real)
            all_real_corrs.extend(real)
            if real:
                groups_out[grp_name] = real

        report = {
            "run1": str(file_a),
            "run2": str(file_b),
            "summary": {
                "groups_analysed":    len(group_results),
                "correlations_found": total_real,
            },
            "dynamic_variables":     _build_dynamic_variables(all_real_corrs),
            "correlations_by_group": groups_out,
        }

        # ── Per-group files ───────────────────────────────────────────────
        corr_dir = Path(correlations_dir) if correlations_dir \
                   else output_path.parent / "correlations"
        print(f"\n  Saving per-group correlation files → {corr_dir}/")
        save_group_correlations(group_results, file_a, file_b, corr_dir)

    else:
        real_correlations = [c for c in correlations if c["correlated"]]
        report = {
            "run1": str(file_a),
            "run2": str(file_b),
            "summary": {
                "correlations_found": len(real_correlations),
            },
            "dynamic_variables": _build_dynamic_variables(real_correlations),
            "correlations":      real_correlations,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Correlations report saved → {output_path}")


# ─── Main API ─────────────────────────────────────────────────────────────────

def group_only(file_a, file_b, grouped_root=None, verbose=True) -> tuple:
    """
    Stage 2 (Grouping) – Load two run JSON files, split sessions by comment,
    write one JSON file per group under <grouped_root>/<run_stem>/<Group>.json.
    Prints a count-verification table.

    Returns (sessions1, sessions2, grouped_root_path).
    Does NOT run any correlation or dynamic-value analysis.
    """
    file_a = Path(file_a)
    file_b = Path(file_b)

    def _log(msg):
        if verbose:
            print(f"  {msg}")

    _log(f"Loading run 1 : {file_a}")
    sessions1 = load_sessions(file_a)
    _log(f"Loading run 2 : {file_b}")
    sessions2 = load_sessions(file_b)

    _log("Grouping sessions by comment …")

    root = Path(grouped_root) if grouped_root else file_a.parent / "grouped"
    folder1 = save_groups(sessions1, file_a, root)
    folder2 = save_groups(sessions2, file_b, root)
    _log(f"Groups saved  → {folder1}")
    _log(f"Groups saved  → {folder2}")

    verify_group_counts(sessions1, sessions2, file_a, file_b)

    return sessions1, sessions2, root


def correlate_only(file_a, file_b, grouped_root=None,
                   output_report=None, verbose=True, cfg: Config = None,
                   correlations_dir: Path = None) -> list:
    """
    Stage 4 (Correlation) – Reads already-grouped sessions from <grouped_root>,
    compares run 1 vs run 2 per group, finds dynamic values and their correlations,
    prints a detailed report, and optionally saves report.json.

    Assumes group_only() (Stage 2) has already been called so the
    grouped JSON files exist on disk.

    Returns group_results: list of (group_name, dynamic_fields, correlations).
    """
    if cfg is None:
        cfg = Config()

    file_a = Path(file_a)
    file_b = Path(file_b)
    root   = Path(grouped_root) if grouped_root else file_a.parent / "grouped"

    def _log(msg):
        if verbose:
            print(f"  {msg}")

    _log(f"Run 1 (baseline) : {file_a}")
    _log(f"Run 2 (compare)  : {file_b}")
    _log(f"Grouped root     : {root}")

    # Load sessions from the original JSON files (same as group_only loaded)
    sessions1 = load_sessions(file_a)
    sessions2 = load_sessions(file_b)

    groups1 = group_sessions_by_comment(sessions1)
    groups2 = group_sessions_by_comment(sessions2)

    # Preserve Run-1 UI-flow order; append any Run-2-only groups at the end
    seen: set = set()
    all_groups: list = []
    for grp in list(groups1.keys()) + list(groups2.keys()):
        if grp not in seen:
            seen.add(grp)
            all_groups.append(grp)
    group_results = []

    for grp in all_groups:
        g1 = groups1.get(grp, [])
        g2 = groups2.get(grp, [])

        if not g1 or not g2:
            _log(f"  Skipping group '{grp}' – "
                 f"missing in {'run1' if not g1 else 'run2'}")
            continue

        _log(f"  ── Group '{grp}' : {len(g1)} vs {len(g2)} sessions")

        pairs          = match_sessions(g1, g2, cfg)
        dynamic_fields = find_dynamic_values(pairs, cfg)
        correlations   = find_correlations(g1, dynamic_fields, cfg)
        structural     = find_structural_correlations(g1, correlations, cfg)
        correlations   = correlations + structural
        real           = sum(1 for c in correlations if c["correlated"])

        _log(f"     Matched pairs  : {len(pairs)}")
        _log(f"     Dynamic fields : {len(dynamic_fields)}")
        _log(f"     Correlations   : {len(correlations)}  (real: {real}, structural: {len(structural)})")

        print_report(dynamic_fields, correlations, file_a, file_b, group_name=grp)
        group_results.append((grp, dynamic_fields, correlations))

    # Per-group summary table
    SEP  = "=" * 72
    THIN = "-" * 72
    print(f"\n{SEP}")
    print("  PER-GROUP CORRELATION SUMMARY")
    print(THIN)
    print(f"  {'Group':<25} {'Dynamic':>8} {'Corr':>6} {'Real':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*6}")
    for grp, dyn, corrs in group_results:
        real = sum(1 for c in corrs if c["correlated"])
        print(f"  {grp:<25} {len(dyn):>8} {len(corrs):>6} {real:>6}")
    print(SEP)

    if output_report:
        save_report(None, None, file_a, file_b, output_report,
                    group_results=group_results,
                    correlations_dir=correlations_dir)

    return group_results


def analyze(file_a=None, file_b=None,
            output_report=None, verbose=True, cfg: Config = None):
    if cfg is None:
        cfg = Config()

    # Auto-discover files from the default folder when not explicitly provided
    if file_a is None or file_b is None:
        discovered_a, discovered_b = discover_json_files()
        file_a = file_a or discovered_a
        file_b = file_b or discovered_b

    file_a = Path(file_a)
    file_b = Path(file_b)

    def _log(msg):
        if verbose:
            print(f"  {msg}")

    # ── Load sessions ────────────────────────────────────────────────────────
    _log(f"Loading run 1 : {file_a}")
    sessions1 = load_sessions(file_a)
    _log(f"Loading run 2 : {file_b}")
    sessions2 = load_sessions(file_b)

    # ── Group by comment & save group files ──────────────────────────────────
    _log("Grouping sessions by comment …")
    groups1 = group_sessions_by_comment(sessions1)
    groups2 = group_sessions_by_comment(sessions2)

    grouped_root = file_a.parent / "grouped"
    folder1 = save_groups(sessions1, file_a, grouped_root)
    folder2 = save_groups(sessions2, file_b, grouped_root)
    _log(f"Groups saved  → {folder1}")
    _log(f"Groups saved  → {folder2}")

    # ── Verify counts ────────────────────────────────────────────────────────
    verify_group_counts(sessions1, sessions2, file_a, file_b)

    # ── Per-group analysis ───────────────────────────────────────────────────
    # Preserve Run-1 UI-flow order; append any Run-2-only groups at the end
    seen_grp: set = set()
    all_groups: list = []
    for grp in list(groups1.keys()) + list(groups2.keys()):
        if grp not in seen_grp:
            seen_grp.add(grp)
            all_groups.append(grp)
    group_results = []   # list of (group_name, dynamic_fields, correlations)

    for grp in all_groups:
        g1 = groups1.get(grp, [])
        g2 = groups2.get(grp, [])

        if not g1 or not g2:
            _log(f"  Skipping group '{grp}' – "
                 f"missing in {'run1' if not g1 else 'run2'}")
            continue

        _log(f"  ── Group '{grp}' : {len(g1)} vs {len(g2)} sessions")

        pairs = match_sessions(g1, g2, cfg)
        _log(f"     Matched pairs : {len(pairs)}")

        dynamic_fields = find_dynamic_values(pairs, cfg)
        _log(f"     Dynamic fields : {len(dynamic_fields)}")

        correlations = find_correlations(g1, dynamic_fields, cfg)
        structural   = find_structural_correlations(g1, correlations, cfg)
        correlations = correlations + structural
        real = sum(1 for c in correlations if c["correlated"])
        _log(f"     Correlations   : {len(correlations)}  (real: {real}, structural: {len(structural)})")

        print_report(dynamic_fields, correlations, file_a, file_b,
                     group_name=grp)
        group_results.append((grp, dynamic_fields, correlations))

    # ── Summary across all groups ────────────────────────────────────────────
    SEP  = "=" * 72
    THIN = "-" * 72
    print(f"\n{SEP}")
    print("  PER-GROUP SUMMARY")
    print(THIN)
    print(f"  {'Group':<25} {'Dynamic':>8} {'Corr':>6} {'Real':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*6} {'-'*6}")
    for grp, dyn, corrs in group_results:
        real = sum(1 for c in corrs if c["correlated"])
        print(f"  {grp:<25} {len(dyn):>8} {len(corrs):>6} {real:>6}")
    print(SEP)

    if output_report:
        save_report(None, None, file_a, file_b, output_report,
                    group_results=group_results)

    return group_results

def normalize_url_filters(url_filter):
    if not url_filter:
        return []

    if isinstance(url_filter, str):
        filter_values = [value.strip() for value in url_filter.split(",")]
    else:
        filter_values = []
        for value in url_filter:
            if value is None:
                continue
            filter_values.extend(part.strip() for part in str(value).split(","))

    return [value.lower() for value in filter_values if value]


def parse_raw_request(raw_bytes):
    if not raw_bytes:
        return {}
    try:
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return {"raw": str(raw_bytes)}

    lines = text.split("\r\n") if "\r\n" in text else text.split("\n")
    if not lines:
        return {}

    request_line = lines[0]
    parts = request_line.split(" ", 2)
    method = parts[0] if len(parts) > 0 else ""
    url = parts[1] if len(parts) > 1 else ""
    http_version = parts[2] if len(parts) > 2 else ""
    headers = {}
    body_start = 1
    for i, line in enumerate(lines[1:], 1):
        if line == "":
            body_start = i + 1
            break
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip()] = value.strip()

    body = "\r\n".join(lines[body_start:]) if body_start < len(lines) else ""

    return {
        "method": method,
        "url": url,
        "http_version": http_version,
        "headers": headers,
        "body": body,
    }


def parse_raw_response(raw_bytes):
    if not raw_bytes:
        return {}

    # ── Split headers and body at the BINARY level ────────────────────────────
    # This prevents corrupting compressed (gzip/deflate/br) response bodies
    # when they are decoded as text.
    sep = b"\r\n\r\n"
    sep_idx = raw_bytes.find(sep)
    sep_len = 4
    if sep_idx == -1:
        sep = b"\n\n"
        sep_idx = raw_bytes.find(sep)
        sep_len = 2

    if sep_idx == -1:
        # No header/body separator – treat everything as a text blob
        try:
            return {"body": raw_bytes.decode("utf-8", errors="replace")}
        except Exception:
            return {"raw": repr(raw_bytes)}

    header_bytes = raw_bytes[:sep_idx]
    body_bytes   = raw_bytes[sep_idx + sep_len:]

    # ── Parse status line + headers (always ASCII/UTF-8 safe) ────────────────
    try:
        header_text = header_bytes.decode("utf-8", errors="replace")
    except Exception:
        return {"raw": repr(raw_bytes)}

    lines = header_text.split("\r\n") if "\r\n" in header_text else header_text.split("\n")

    status_line = lines[0] if lines else ""
    parts = status_line.split(" ", 2)
    http_version = parts[0] if len(parts) > 0 else ""
    status_code  = parts[1] if len(parts) > 1 else ""
    status_text  = parts[2] if len(parts) > 2 else ""

    headers: dict = {}
    for line in lines[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            headers[key.strip()] = value.strip()

    # ── Decompress body if Content-Encoding says so ───────────────────────────
    content_encoding = headers.get("Content-Encoding", "").lower()
    content_type     = headers.get("Content-Type", "").lower()

    if body_bytes:
        try:
            if "gzip" in content_encoding:
                body_bytes = gzip.decompress(body_bytes)
            elif "deflate" in content_encoding:
                # RFC 7230: "deflate" is actually zlib-wrapped deflate
                try:
                    body_bytes = zlib.decompress(body_bytes)
                except zlib.error:
                    body_bytes = zlib.decompress(body_bytes, -zlib.MAX_WBITS)
            elif "br" in content_encoding:
                try:
                    import brotli  # pip install brotli
                    body_bytes = brotli.decompress(body_bytes)
                except ImportError:
                    pass  # brotli not installed – leave as-is
        except Exception:
            pass  # decompression failed – keep raw bytes

    # ── Decode body to string ─────────────────────────────────────────────────
    # Prefer UTF-8; fall back to latin-1 for binary-ish payloads
    body = ""
    if body_bytes:
        is_text = (
            "text"       in content_type
            or "json"    in content_type
            or "xml"     in content_type
            or "script"  in content_type
            or "html"    in content_type
        )
        try:
            body = body_bytes.decode("utf-8")
        except UnicodeDecodeError:
            if is_text:
                body = body_bytes.decode("utf-8", errors="replace")
            else:
                # Binary content (images, fonts, etc.) – store as base64 marker
                import base64
                body = "[base64:" + base64.b64encode(body_bytes).decode("ascii") + "]"

    return {
        "http_version": http_version,
        "status_code":  status_code,
        "status_text":  status_text,
        "headers":      headers,
        "body":         body,
    }


def parse_metadata_xml(xml_bytes):
    metadata = {
        "comment": "",
        "flags": {}
    }

    if not xml_bytes:
        return metadata

    try:
        text = xml_bytes.decode("utf-8", errors="replace")
        root = ET.fromstring(text)

        # Extract SessionComment — this is where Fiddler stores the comment
        comment_el = root.find("SessionComment")
        if comment_el is not None and comment_el.text:
            metadata["comment"] = comment_el.text.strip()

        # Extract all flags (key-value pairs Fiddler stores per session)
        flags_el = root.find("SessionFlags")
        if flags_el is not None:
            for flag in flags_el.findall("SessionFlag"):
                name = flag.get("N", "")
                value = flag.get("V", "")
                if name:
                    metadata["flags"][name] = value

            # Also check for comment inside flags (some Fiddler versions store it here)
            comment_flag = metadata["flags"].get("ui-comments", "")
            if comment_flag and not metadata["comment"]:
                metadata["comment"] = comment_flag

    except ET.ParseError as e:
        metadata["parse_error"] = str(e)

    return metadata


def parse_saz_to_json(saz_file_path, output_json_path=None, verbose=True, url_filter=None):

    normalized_filters = normalize_url_filters(url_filter)

    if output_json_path is None:
        output_json_path = Path(saz_file_path).stem + ".json"

    if not os.path.exists(saz_file_path):
        print(f"[ERROR] File not found: {saz_file_path}")
        raise FileNotFoundError(f"File not found: {saz_file_path}")

    sessions = []

    with zipfile.ZipFile(saz_file_path, "r") as saz:
        all_files = saz.namelist()

        # Find all session IDs by looking at request files (####_c.txt pattern)
        session_ids = set()
        for f in all_files:
            match = re.match(r"raw/(\d+)_c\.txt", f)
            if match:
                session_ids.add(match.group(1))

        if verbose:
            print(f"  Found {len(session_ids)} sessions in {Path(saz_file_path).name}")

        for sid in sorted(session_ids, key=lambda x: int(x)):
            request_file = f"raw/{sid}_c.txt"  # client request
            response_file = f"raw/{sid}_s.txt"  # server response
            metadata_file = f"raw/{sid}_m.xml"  # metadata with comment

            # Read request
            request_data = {}
            if request_file in all_files:
                raw_req = saz.read(request_file)
                request_data = parse_raw_request(raw_req)

            # Read response
            response_data = {}
            if response_file in all_files:
                raw_res = saz.read(response_file)
                response_data = parse_raw_response(raw_res)

            # Read metadata / comment
            metadata = {}
            if metadata_file in all_files:
                raw_meta = saz.read(metadata_file)
                metadata = parse_metadata_xml(raw_meta)
            else:
                metadata = {"comment": "", "flags": {}}

            # Apply URL filter if specified
            session_url = request_data.get("url", "")
            if normalized_filters and not any(filter_value in session_url.lower() for filter_value in normalized_filters):
                if verbose:
                    print(f"    Session {sid}: skipped (URL does not contain any of {normalized_filters})")
                continue

            session = {
                "session_id": int(sid),
                "comment": metadata.get("comment", ""),
                "flags": metadata.get("flags", {}),
                "request": request_data,
                "response": response_data,
            }

            sessions.append(session)

            if verbose:
                comment_display = metadata.get("comment", "") or "(no comment)"
                method = request_data.get('method', '?')
                url = request_data.get('url', '?')[:60]
                print(f"    Session {sid}: {method} {url}  | comment: {comment_display}")

    output = {
        "source_file": os.path.basename(saz_file_path),
        "total_sessions": len(sessions),
        "sessions": sessions
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    if verbose:
        print(f"  ✓ Converted {len(sessions)} sessions → {output_json_path}")

    return output


def batch_convert_saz_files(input_folder='.saz_Files', output_folder='json(.saz)_Files', verbose=True, url_filter=None):

    # Create input folder if it doesn't exist
    input_path = Path(input_folder)
    if not input_path.exists():
        if verbose:
            print(f"Creating input folder: {input_folder}")
        input_path.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"Please place your .saz files in the '{input_folder}' folder and run again.")
        return {
            'total': 0,
            'success': 0,
            'failed': 0,
            'files': []
        }

    # Create output folder if it doesn't exist
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all .saz files in input folder
    saz_files = list(input_path.glob('*.saz'))

    if not saz_files:
        if verbose:
            print(f"No .saz files found in '{input_folder}' folder.")
        return {
            'total': 0,
            'success': 0,
            'failed': 0,
            'files': []
        }

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Found {len(saz_files)} .saz file(s) to convert")
        print(f"Input folder:  {input_folder}")
        print(f"Output folder: {output_folder}")
        print(f"{'=' * 60}\n")

    stats = {
        'total': len(saz_files),
        'success': 0,
        'failed': 0,
        'files': []
    }

    # Process each .saz file
    for idx, saz_file in enumerate(saz_files, 1):
        if verbose:
            print(f"[{idx}/{len(saz_files)}] Processing: {saz_file.name}")

        try:
            # Generate output filename
            output_filename = saz_file.stem + '.json'
            output_filepath = output_path / output_filename

            # Convert the file
            parse_saz_to_json(str(saz_file), str(output_filepath), verbose=verbose, url_filter=url_filter)

            stats['success'] += 1
            stats['files'].append({
                'input': str(saz_file),
                'output': str(output_filepath),
                'status': 'success'
            })

        except Exception as e:
            if verbose:
                print(f"  ✗ Failed to convert {saz_file.name}: {str(e)}")
            stats['failed'] += 1
            stats['files'].append({
                'input': str(saz_file),
                'output': None,
                'status': 'failed',
                'error': str(e)
            })

    # Print summary
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"CONVERSION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total files:      {stats['total']}")
        print(f"✓ Successful:     {stats['success']}")
        print(f"✗ Failed:         {stats['failed']}")
        print(f"{'=' * 60}\n")

        if stats['success'] > 0:
            print(f"JSON files saved to: {output_folder}/")

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE ENTRY POINT  –  python -X utf8 Tool.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import io as _io

    # Fix Windows terminal UTF-8 encoding
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  SAZ SUMMARY TOOL")
    print("=" * 60)

    _url1   = input("\nEnter GitHub URL of the FIRST  .saz file       : ").strip()
    _url2   = input("Enter GitHub URL of the SECOND .saz file       : ").strip()
    _token  = input("Enter GitHub Personal Access Token             : ").strip()
    _folder = input("Enter output folder name (blank = auto-generate): ").strip()

    if not _url1 or not _url2 or not _token:
        print("\n[ERROR] URL 1, URL 2, and GitHub Token are all required.")
        sys.exit(1)

    print("\n" + "-" * 60)
    print("Running pipeline … this may take a few minutes.")
    print("-" * 60 + "\n")

    _tool   = SazSummaryTool()
    _result = _tool._run(
        saz_url_1=_url1,
        saz_url_2=_url2,
        github_token=_token,
        output_folder_name=_folder,
    )

    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(_result)
