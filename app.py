"""After the Diploma - proof of concept v0.1.

Connecting education, skills, and regional opportunity.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from modules import (
    ai_client,
    careers,
    compare,
    curriculum,
    earnings,
    employers,
    intel,
    profile_ui,
    recommendations,
    report,
    scp,
    skills,
    theme as themes,
)
from modules.data_sources import (
    DataStatus,
    REFERENCE_POINT,
    get_provider,
)

st.set_page_config(
    page_title="After the Diploma",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- style
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 1200px;}
    h1 {font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0;}
    .subtitle {color:#4a5568; font-size:1.05rem; margin-top:0.1rem; margin-bottom:1rem;}
    .demo-banner {background:#fff8e1; border:1px solid #f0d98c; border-radius:8px;
                  padding:0.55rem 0.9rem; color:#6b5d1f; font-size:0.9rem; margin-bottom:1rem;}
    .live-banner {background:#e8f5e9; border:1px solid #a5d6a7; border-radius:8px;
                  padding:0.55rem 0.9rem; color:#2e5d33; font-size:0.9rem; margin-bottom:1rem;}
    .status-chip {display:inline-block; border-radius:10px; padding:1px 9px; font-size:0.72rem;
                  font-weight:600; margin-left:6px; vertical-align:middle;}
    .chip-doc {background:#e3f2fd; color:#1565c0;}
    .chip-inf {background:#fff3e0; color:#e65100;}
    .chip-rec {background:#f3e5f5; color:#6a1b9a;}
    .section-note {color:#5a6472; font-size:0.85rem;}
    .caveat {background:#f8f9fa; border-left:3px solid #94a3b8; padding:0.5rem 0.8rem;
             color:#475569; font-size:0.85rem; border-radius:0 6px 6px 0;}
    .footer {margin-top:3rem; padding-top:1rem; border-top:1px solid #e2e8f0;
             color:#64748b; font-size:0.78rem;}
    .brandbar {display:flex; align-items:center; gap:14px; padding-left:14px;}
    .emblem {width:52px; height:52px; border-radius:10px; display:flex;
             align-items:center; justify-content:center; flex-shrink:0;}
    .emblem-letters {color:#fff; font-weight:700; font-size:1.05rem; letter-spacing:.03em;}
    .platform {font-size:1.5rem; font-weight:800; letter-spacing:.02em; color:#0f172a;}
    .platform-sub {font-size:.85rem; color:#475569; letter-spacing:.04em;}
    .env {font-size:.8rem; font-weight:600; margin-top:2px;}
    .demo-strip {border:1px solid; border-radius:6px; padding:.45rem .8rem;
                 font-size:.78rem; font-weight:700; letter-spacing:.02em;
                 margin:.6rem 0 .8rem 0; background:#fff;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data
provider = get_provider()


@st.cache_data(show_spinner=False)
def load_reference_data():
    return {
        "taxonomy": provider.skills_taxonomy(),
        "occupations": provider.occupations(),
        "wages": provider.wages(),
        "employers": provider.employers(),
        "job_listings": provider.job_listings(),
        "sources": provider.sources(),
    }


REF = load_reference_data()

DEFAULT_GEO_REF = {
    "name": REFERENCE_POINT["name"],
    "latitude": REFERENCE_POINT["latitude"],
    "longitude": REFERENCE_POINT["longitude"],
}


def get_geo_ref() -> dict:
    """Geographic reference for the current analysis (configurable)."""
    return st.session_state.get("geo_ref", DEFAULT_GEO_REF)


def ref_employers() -> pd.DataFrame:
    g = get_geo_ref()
    return employers.employers_with_distance(
        REF["employers"], g["latitude"], g["longitude"]
    )


def curriculum_employers() -> pd.DataFrame:
    """Employers relevant to the ACTIVE curriculum's eligible careers and
    present skills - the curriculum-scoped employer set for all modules."""
    a = st.session_state.get("analysis") or {}
    ca, sk = a.get("careers"), a.get("skills")
    df = ref_employers()
    if ca is None or sk is None:
        return df
    present = set(sk.loc[sk.status.isin(["documented", "inferred"]), "skill_id"])
    return employers.relevant_employers(df, set(ca["title"]), present)


def _render_lifestyle(occ_row, careers_df, emp_geo) -> None:
    """Career Lifestyle Profile v1.0 - evidence-scored where data exists."""
    from modules import lifestyle
    pw = earnings.pathway_wages(
        careers_df[careers_df.soc_code == occ_row["soc_code"]], REF["wages"])
    wage = pw.iloc[0].to_dict() if len(pw) else {}
    prof = lifestyle.build_profile(
        occ_row, careers_df, wage, emp_geo,
        get_geo_ref().get("state", "Vermont"))
    st.markdown("#### Career Lifestyle Profile")
    qdims = lifestyle.quantitative_dims(prof)
    ddims = lifestyle.descriptive_dims(prof)
    if len(qdims) >= 3:
        fig = go.Figure(go.Scatterpolar(
            r=[d["score"] for d in qdims] + [qdims[0]["score"]],
            theta=[f'{d["label"]} ({d["band"] or ""})' for d in qdims]
                  + [qdims[0]["label"]],
            fill="toself", line_color="#2563eb", name=occ_row["title"]))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])),
                          height=300, margin=dict(t=20, b=10), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption(
            "Fewer than three lifestyle dimensions have sufficient evidence "
            "for a comparison chart - shown as individual indicators instead.")
    for d in prof["dimensions"]:
        chip = STATUS_CHIP.get(d["evidence_status"], "")
        score_txt = (f" **{d['score']:.0f}/100** — *{d['band']}*"
                     if d["score"] is not None else " *not assessed*")
        with st.expander(f"{d['label']}{score_txt}{chip}", expanded=False):
            st.write(d["summary"])
            if d["limitations"]:
                st.caption("Limitations: " + "; ".join(d["limitations"]))
    st.caption(
        f"Missing evidence — "
        + ", ".join(d["label"] for d in prof["dimensions"]
                    if d["evidence_status"] == "insufficient_evidence")
        + " (shown honestly rather than estimated).")
    for q in prof["student_reflection"]:
        st.markdown(f"• *{q}*")
    st.info(lifestyle.CAVEAT)


def nav_options(has_analysis: bool) -> list[str]:
    """Sidebar navigation; intelligence modules unlock after analysis."""
    options = ["Home", "Select Curriculum", "Curriculum Library"]
    if has_analysis:
        options += [
            "Executive Dashboard",
            "1 · Curriculum Intelligence",
            "2 · Career Pathways",
            "3 · Regional Employment",
            "4 · Earnings Intelligence",
            "5 · Curriculum Advisor",
            "AI Comparison",
            "Regional Intelligence",
            "Reports",
        ]
    options.append("Data & Methodology")
    return options


# ---------------------------------------------------------------- state
def activate_profile(profile_id: str):
    """Make an SCP the active curriculum input for analysis."""
    p = st.session_state.profiles[profile_id]
    st.session_state.active_profile_id = profile_id
    st.session_state.curriculum_text = scp.corpus_text(p)
    st.session_state.curriculum_label = f"SCP {profile_id} · {p['program'].get('name','')}"
    st.session_state.curriculum_version += 1


def init_state():
    """Cold start: land on Home with NO curriculum selected."""
    if "initialized" in st.session_state:
        return
    st.session_state.profiles = {p["profile_id"]: p for p in scp.list_profiles()}
    st.session_state.draft_profile = None
    st.session_state.review_profile = None
    st.session_state.select_mode = "analyze"
    st.session_state.analysis = None
    st.session_state.active_profile_id = None
    st.session_state.curriculum_text = ""
    st.session_state.curriculum_label = ""
    st.session_state.curriculum_version = 0
    st.session_state.use_ai = False
    st.session_state.geo_ref = dict(DEFAULT_GEO_REF)
    st.session_state.theme_key = themes.DEFAULT_THEME
    st.session_state.initialized = True


init_state()


def analysis_fingerprint() -> str:
    """Invalidation key: curriculum, profile, model, prompt + schema versions,
    and whether AI enhancement is enabled. Any change => fresh analysis."""
    parts = [
        st.session_state.curriculum_text,
        st.session_state.get("active_profile_id") or "",
        ai_client.configured_model() if ai_client.openai_available() else "no-key",
        ai_client.PROMPT_VERSION,
        scp.SCP_VERSION,
        skills.ENGINE_VERSION,
        "ai" if st.session_state.get("use_ai") else "rules",
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def run_analysis(force: bool = False) -> None:
    """Single orchestration point for the full pipeline.

    Always runs the deterministic rules engine (baseline). When AI
    enhancement is enabled and a key is present, also runs the OpenAI
    extraction and merges it as an enhancement layer. Both result sets are
    stored in one analysis object consumed by every module - Streamlit
    reruns and navigation never trigger a second API call because the
    fingerprint guard in ensure_analysis skips fresh results.
    """
    text = st.session_state.curriculum_text
    if not text:
        return
    if (
        not force
        and st.session_state.analysis is not None
        and st.session_state.analysis.get("fingerprint") == analysis_fingerprint()
    ):
        return  # results are current - dedup guard

    use_ai = bool(st.session_state.get("use_ai"))
    key_ok = ai_client.openai_available()
    rules_df = skills.extract_skills_rules(text, REF["taxonomy"])

    skills_ai = careers_ai = None
    ai_error = None
    if use_ai and key_ok:
        try:
            ai_skills = ai_client.analyze_curriculum(text)
            skills_ai = skills.merge_ai_results(rules_df, ai_skills, REF["taxonomy"])
            ai_status = "ok"
        except ai_client.AIClientError as exc:
            ai_status = "error"
            ai_error = str(exc)
    elif use_ai and not key_ok:
        ai_status = "unavailable"
    else:
        ai_status = "disabled"

    engine_df = skills_ai if skills_ai is not None else rules_df
    method = (
        "Deterministic rules engine + AI enhancement"
        if ai_status == "ok"
        else "Deterministic rules engine"
        + (" (AI attempted - fell back)" if ai_status == "error" else "")
    )

    # Educator-reviewed SCP statuses layer onto the final output only;
    # engine-level sets stay pure so comparisons compare engines.
    final_df = engine_df
    pid = st.session_state.get("active_profile_id")
    if pid and pid in st.session_state.get("profiles", {}):
        final_df = scp.apply_profile_overrides(
            st.session_state.profiles[pid], engine_df, REF["taxonomy"]
        )
        method += " + reviewed profile statuses"

    careers_final = careers.match_occupations(final_df, REF["occupations"])
    careers_rules = careers.match_occupations(rules_df, REF["occupations"])
    if skills_ai is not None:
        careers_ai = careers.match_occupations(skills_ai, REF["occupations"])

    st.session_state.analysis = {
        "skills": final_df,
        "careers": careers_final,
        "careers_ids": recommendations.careers_with_skill_ids(
            careers_final, REF["occupations"]
        ),
        "skills_rules": rules_df,
        "careers_rules": careers_rules,
        "skills_ai": skills_ai,
        "careers_ai": careers_ai,
        "method": method,
        "used_ai": ai_status == "ok",
        "ai_status": ai_status,  # ok | error | unavailable | disabled
        "ai_error": ai_error,
        "fingerprint": analysis_fingerprint(),
        "meta": {
            "model": ai_client.configured_model() if key_ok else None,
            "prompt_version": ai_client.PROMPT_VERSION,
            "scp_version": scp.SCP_VERSION,
            "ran_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def ensure_analysis() -> dict | None:
    """Run the pipeline only when a curriculum has been explicitly selected
    and the stored result's fingerprint no longer matches current inputs."""
    if not st.session_state.curriculum_text:
        return None
    a = st.session_state.analysis
    if a is None or a.get("fingerprint") != analysis_fingerprint():
        run_analysis()
    return st.session_state.analysis


def require_analysis() -> bool:
    if st.session_state.analysis is None:
        st.info(
            "No analysis has run yet. Choose a curriculum source from the "
            "Home page or the Select Curriculum section to begin."
        )
        return False
    return True


A = ensure_analysis()


# ---------------------------------------------------------------- chrome
THEME = themes.get_theme(st.session_state.get("theme_key"))

st.markdown(
    f"""
    <div class="brandbar" style="border-left:6px solid {THEME.primary}">
      <div class="emblem" style="background:{THEME.primary}">
        <span class="emblem-letters">{THEME.emblem_initials}</span>
      </div>
      <div>
        <div class="platform">AFTER THE DIPLOMA</div>
        <div class="platform-sub">Curriculum &amp; Workforce Intelligence</div>
        {"" if not THEME.institution_name else
         f'<div class="env" style="color:{THEME.primary}">{THEME.institution_name} — {THEME.environment}</div>'}
      </div>
    </div>
    <div class="subtitle">Connecting education, skills, and regional opportunity.</div>
    """,
    unsafe_allow_html=True,
)
if THEME.demo_label:
    st.markdown(
        f'<div class="demo-strip" style="border-color:{THEME.primary};color:{THEME.primary}">'
        f"{THEME.demo_label}</div>",
        unsafe_allow_html=True,
    )
if THEME.emblem_url is None and THEME.institution_name:
    st.caption(THEME.emblem_status)

live = ai_client.openai_available()
if live:
    st.markdown(
        f'<div class="live-banner"><b>Live Analysis Mode available</b> - OpenAI key configured '
        f"(model: {ai_client.configured_model()}). AI extraction can be enabled in Module 1 and "
        "compared side-by-side under AI Comparison. Employment and wage data remain "
        "reference datasets regardless of AI mode.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        '<div class="demo-banner"><b>Demonstration Mode</b> - no OpenAI API key configured. '
        "Skills are extracted by a deterministic rules engine. Wage data are verified BLS OEWS "
        "figures; employer and job-listing data are labeled demonstration datasets.</div>",
        unsafe_allow_html=True,
    )

with st.sidebar:
    analyzed = st.session_state.analysis is not None
    options = nav_options(analyzed)
    if st.session_state.get("nav") not in options:
        st.session_state.nav = "Home"
    st.caption("Intelligence modules" if analyzed else "Get started")
    page = st.radio(
        "Section",
        options,
        key="nav",
        label_visibility="collapsed",
    )
    if analyzed:
        st.sidebar.button(
            "Change curriculum / new analysis",
            on_click=lambda: setattr(st.session_state, "nav", "Select Curriculum"),
        )
    st.divider()
    g = get_geo_ref()
    st.markdown(
        f"**Geographic reference**  \n{g['name']}  \n"
        f"{g['latitude']:.4f}, {g['longitude']:.4f}",
        help="Configurable per analysis — set it during profile review or in Module 3.",
    )
    if analyzed:
        st.markdown(
            f"**Analysis engine**  \n{A['method']}",
            help="Shown on every page so results are never misread as AI-generated when they are not.",
        )
        st.caption("Curriculum: " + st.session_state.curriculum_label)
    ai_note = (
        f"AI analysis: available ({ai_client.configured_model()})"
        if ai_client.openai_available()
        else "AI analysis: not configured"
    )
    st.caption(ai_note)
    st.selectbox(
        "Theme", list(themes.THEMES), key="theme_key",
        format_func=lambda k: "Lyndon Institute (demo)" if k == "lyndon" else "After the Diploma (generic)",
    )


STATUS_CHIP = {
    "documented": '<span class="status-chip chip-doc">documented</span>',
    "inferred": '<span class="status-chip chip-inf">inferred</span>',
    "recommended_addition": '<span class="status-chip chip-rec">suggested</span>',
}


def _fmt_money(v):
    return f"${int(v):,}" if pd.notna(v) else "—"


# ---------------------------------------------------------------- pages
def page_home():
    st.markdown(
        "A curriculum and workforce intelligence platform for career and "
        "technical education. Turn any program of study into evidence of what "
        "students can do, where those skills lead, and how the curriculum "
        "could go further."
    )
    st.caption(
        "Journey: Home → Curriculum Selection → Profile Review → Intelligence "
        "Analysis → Executive Dashboard → Detailed Modules"
    )
    st.divider()

    def go(mode, review=None):
        st.session_state.select_mode = mode
        st.session_state.review_profile = review
        st.session_state.nav = "Select Curriculum"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### Analyze a Curriculum")
        st.write(
            "Upload a PDF, DOCX, or TXT curriculum — or paste program text — and "
            "build a Standard Curriculum Profile for review."
        )
        st.button("Analyze a curriculum", key="home-analyze", type="primary",
                  on_click=go, args=("analyze",))
    with c2:
        st.markdown("### Explore a Program")
        st.write(
            "Select an existing Standard Curriculum Profile from the library, "
            "inspect its evidence, then run the intelligence analysis."
        )
        st.button("Explore a program", key="home-explore",
                  on_click=go, args=("explore",))
    with c3:
        st.markdown("### Try a Demonstration")
        st.write(
            "Open the representative Lyndon Institute Welding example built from "
            "the school's public program description."
        )
        li = st.session_state.profiles.get("LI-WELD-001")
        st.button("Try the Welding demo", key="home-demo",
                  on_click=go, args=("demo", li), disabled=li is None)

    if analyzed:
        st.divider()
        st.info(
            f"An analysis is already active for **{st.session_state.curriculum_label}**."
        )
        st.button("Continue to Executive Dashboard →",
                  on_click=lambda: setattr(st.session_state, "nav", "Executive Dashboard"))

    st.divider()
    st.markdown(
        '<div class="caveat">Everything downstream carries source-transparency labels — '
        "documented vs. inferred skills, verified vs. demonstration data. A public "
        "program description is never presented as a complete curriculum.</div>",
        unsafe_allow_html=True,
    )


def _geo_editor(key_prefix: str):
    """Editable geographic reference for the current analysis."""
    g = dict(get_geo_ref())
    g["name"] = st.text_input("Reference name", g["name"], key=f"{key_prefix}-goname")
    c1, c2 = st.columns(2)
    g["latitude"] = c1.number_input(
        "Latitude", value=float(g["latitude"]), format="%.5f", key=f"{key_prefix}-lat"
    )
    g["longitude"] = c2.number_input(
        "Longitude", value=float(g["longitude"]), format="%.5f", key=f"{key_prefix}-lon"
    )
    st.session_state.geo_ref = g
    if st.button("Reset to Lyndon Institute", key=f"{key_prefix}-goreset"):
        st.session_state.geo_ref = dict(DEFAULT_GEO_REF)
        st.rerun()


def _review_summary(p: dict):
    """Pre-analysis profile review: identity, source type, skill availability."""
    inst = p.get("institution", {})
    prog = p.get("program", {})
    src = p.get("source", {})
    sks = p.get("skills", [])
    counts = pd.Series([s.get("status", "unknown") for s in sks]).value_counts()

    c1, c2 = st.columns(2)
    c1.markdown(f"**Institution:** {inst.get('name') or '—'}")
    c1.markdown(f"**Location:** {inst.get('location') or '—'}")
    c2.markdown(f"**Program:** {prog.get('name') or '—'}")
    c2.markdown(
        f"**Source type:** `{scp.SOURCE_TYPES.get(src.get('source_type'), 'unknown')}`"
    )
    hrs = prog.get("instructional_hours")
    st.markdown(
        f"**Duration:** {prog.get('duration') or '—'} · "
        f"**Instructional hours:** {hrs if hrs else 'Unknown — not documented'}"
    )
    st.markdown(
        "**Skills in profile:** "
        + " · ".join(
            f"{counts.get(s, 0)} {s}"
            for s in ["documented", "inferred", "unknown", "recommended"]
        )
        if sks
        else "**Skills in profile:** none recorded yet — extraction will run on the source text."
    )
    if src.get("url"):
        st.caption(f"Source: {src.get('title', '')} — {src['url']}")
    if src.get("notes"):
        st.caption(src["notes"])


def page_select():
    st.subheader("Select Curriculum")
    mode = st.session_state.select_mode

    if mode == "demo":
        li = st.session_state.profiles.get("LI-WELD-001")
        if li and st.session_state.review_profile is None:
            st.session_state.review_profile = li
        st.caption(
            "Demonstration: Lyndon Institute Welding profile, built from the school's "
            "public program description."
        )
    elif mode == "explore":
        profiles = st.session_state.profiles
        if not profiles:
            st.info("The library is empty — analyze a curriculum to create a profile.")
        else:
            ids = list(profiles.keys())
            sel = st.selectbox(
                "Standard Curriculum Profile", ids,
                format_func=lambda k: f"{k} · {profiles[k]['program'].get('name', '')}",
                key="sel-explore",
            )
            if st.button("Review this profile", key="sel-explore-btn"):
                st.session_state.review_profile = profiles[sel]
    else:  # analyze
        tab_sample, tab_upload, tab_text = st.tabs(
            ["Sample curriculum", "Upload file", "Paste text"]
        )
        with tab_sample:
            st.markdown(
                "Bundled **representative welding curriculum** — clearly labeled "
                "fictional sample, **not** an official Lyndon Institute document."
            )
            if st.button("Create profile from sample", key="sel-sample"):
                draft = scp.draft_from_text(
                    curriculum.load_sample_curriculum(),
                    "manual_entry",
                    title="Representative sample welding curriculum",
                    profile_id="SAMPLE-WELD-001",
                )
                draft["institution"]["name"] = "Demonstration institution"
                draft["source"]["notes"] = (
                    "Fictional representative sample bundled with the application."
                )
                st.session_state.review_profile = draft
        with tab_upload:
            up = st.file_uploader("Curriculum file", type=["pdf", "docx", "txt"])
            stype = st.selectbox(
                "What kind of document is this?",
                list(scp.SOURCE_TYPES.keys()),
                index=list(scp.SOURCE_TYPES.keys()).index("uploaded_document"),
                format_func=lambda s: scp.SOURCE_TYPES[s], key="sel-stype",
            )
            if up and st.button("Create profile from file", key="sel-upload"):
                try:
                    text = curriculum.extract_text(up.getvalue(), up.name)
                    st.session_state.review_profile = scp.draft_from_text(
                        text, stype, title=up.name
                    )
                except ValueError as exc:
                    st.error(str(exc))
        with tab_text:
            pasted = st.text_area("Curriculum text", height=160,
                                  placeholder="Paste curriculum text here...")
            if st.button("Create profile from text", key="sel-text") and pasted.strip():
                st.session_state.review_profile = scp.draft_from_text(
                    curriculum.normalize_text(pasted), "manual_entry",
                    title="Pasted curriculum",
                )

    # ------------------------------------------------ profile review step
    review = st.session_state.review_profile
    if review is None:
        return

    st.divider()
    st.markdown(f"### Review profile `{review['profile_id']}`")
    _review_summary(review)

    with st.expander("Geographic reference for employer matching"):
        _geo_editor("sel")

    with st.expander("Edit profile before analysis"):
        edited = profile_ui.profile_editor(review, key_prefix="sel-review")
        if st.button("Apply edits", key="sel-apply"):
            st.session_state.review_profile = edited
            st.success("Edits applied — review the updated summary above.")

    def _confirm_profile():
        p = st.session_state.review_profile
        errors = scp.validate_profile(p)
        if errors:
            st.session_state.review_errors = errors
            return
        st.session_state.review_errors = []
        pid = p["profile_id"]
        st.session_state.profiles[pid] = p
        try:
            scp.save_profile(p)
        except OSError:
            pass  # library folder not writable (e.g. read-only deploy)
        st.session_state.review_profile = None
        activate_profile(pid)
        st.session_state.nav = "Executive Dashboard"

    st.button(
        "Confirm profile & run analysis", type="primary",
        key="sel-confirm", on_click=_confirm_profile,
    )
    for e in st.session_state.get("review_errors", []):
        st.caption(f"· {e}")


def _pathway_tooltips(careers_df: pd.DataFrame) -> list[str]:
    """One multiline tooltip per pathway category: every career title with
    its alignment score, built dynamically from the active analysis."""
    out = []
    for p in ("primary", "adjacent", "extended"):
        sub = careers_df[careers_df.pathway == p].sort_values(
            "compatibility_score", ascending=False
        )
        lines = "<br>".join(
            f"{r['title']} — {r['compatibility_score']}/100"
            for _, r in sub.iterrows()
        )
        out.append(lines or "No pathways")
    return out


def page_dashboard():
    if not require_analysis():
        return
    sk, ca = A["skills"], A["careers"]
    emp_geo = curriculum_employers()
    doc_core = sk[(sk.status == "documented") & (sk.category == "core")]
    adjacent = sk[
        (sk.category == "adjacent") & (sk.status.isin(["documented", "inferred"]))
    ]
    emp100 = emp_geo[emp_geo.distance_miles <= 100]
    pw = earnings.pathway_wages(ca, REF["wages"])
    recs = recommendations.generate_recommendations(
        sk, A["careers_ids"], emp_geo
    )
    lo, hi = earnings.wage_range_summary(pw)

    st.subheader("Executive overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documented core skills", len(doc_core))
    c2.metric("Adjacent skills identified", len(adjacent))
    c3.metric("Potential career pathways", len(ca))
    c4.metric("Regional employers (≤100 mi)", len(emp100))
    c5, c6, c7 = st.columns(3)
    c5.metric("Verified current openings", 0, help="No live job-posting feed in v0.1.")
    c6.metric(
        "Wage range (p10–p90, all pathways)",
        f"{_fmt_money(lo)} – {_fmt_money(hi)}" if lo else "n/a",
        help="Spread of published BLS OEWS percentiles across all matched pathways.",
    )
    c7.metric("Improvement recommendations", len(recs))

    st.divider()
    prim = ca[ca.pathway == "primary"].sort_values("compatibility_score", ascending=False)
    top = prim.iloc[0] if len(prim) else None
    n_doc = int((sk.status == "documented").sum())
    st.markdown(
        f"**What opportunities does this curriculum create?** The program documents "
        f"**{n_doc} skills** across four categories, supporting "
        f"**{len(ca)} candidate career pathways** — "
        f"{len(ca[ca.pathway=='primary'])} primary, {len(ca[ca.pathway=='adjacent'])} adjacent, "
        f"and {len(ca[ca.pathway=='extended'])} extended. "
        + (
            f"Strongest direct alignment: **{top['title']}** ({top['compatibility_score']}/100). "
            if top is not None
            else ""
        )
        + f"Within 100 straight-line miles of Lyndon Institute, **{len(emp100)} employers** "
        f"in the reference dataset operate in relevant sectors. "
        f"**{len(recs)} prioritized curriculum recommendations** identify where small additions "
        "could expand student options — all advisory for educator review."
    )

    st.button(
        "Generate Curriculum Opportunity Report (PDF)",
        type="primary", key="dash-report",
        on_click=lambda: setattr(st.session_state, "nav", "Reports"),
    )
    st.markdown("##### Pathways by category")
    counts = ca.groupby("pathway").size().reindex(["primary", "adjacent", "extended"]).fillna(0)
    tooltips = _pathway_tooltips(ca)
    # One trace per category so each bar's customdata maps 1:1 to its full
    # pathway list (per-category color traces break shared customdata).
    fig = go.Figure(
        data=[
            go.Bar(
                x=[careers.PATHWAY_LABELS[p]],
                y=[int(counts[p])],
                marker_color={
                    "primary": "#2563eb",
                    "adjacent": "#0891b2",
                    "extended": "#7c3aed",
                }[p],
                customdata=[tooltips[i]],
                hovertemplate=(
                    "<b>%{x}</b><br>Pathways: %{y}<br><br>%{customdata}"
                    "<extra></extra>"
                ),
            )
            for i, p in enumerate(counts.index)
        ]
    )
    fig.update_layout(
        yaxis_title="Pathways", showlegend=False, height=260,
        margin=dict(t=10, b=10), barmode="group",
    )
    st.plotly_chart(fig, width="stretch")
    st.selectbox(
        "Jump to detailed pathway view",
        ["", *counts.index],
        key="dash-jump",
        format_func=lambda v: "Select a category…" if not v else careers.PATHWAY_LABELS[v],
        on_change=lambda: st.session_state.update({"nav": "2 · Career Pathways"}),
    )

    st.markdown(
        '<div class="caveat">All compatibility scores are indicative, not validated employment '
        "probabilities. Employer entries are real regional employers compiled for demonstration "
        "and are <b>not</b> verified as currently hiring.</div>",
        unsafe_allow_html=True,
    )


def page_library():
    st.subheader("Curriculum Library")
    st.caption(
        "Standard Curriculum Profiles (SCP v1.0) — every curriculum source is converted "
        "into a consistent, validated, editable profile before analysis."
    )
    profiles = st.session_state.profiles

    with st.expander("Import a profile (JSON)"):
        imp = st.file_uploader("SCP JSON file", type=["json"], key="lib-import")
        if imp and st.button("Import profile", key="lib-imp-btn"):
            try:
                p = scp.import_profile(imp.getvalue().decode("utf-8"))
                profiles[p["profile_id"]] = p
                st.success(
                    f"Imported {p['profile_id']} — validated against SCP v1.0."
                )
            except (ValueError, UnicodeDecodeError) as exc:
                st.error(str(exc))

    if not profiles:
        st.info("No profiles in the library yet. Create one in Module 1.")
        return

    ids = list(profiles.keys())
    pid = st.selectbox(
        "Select profile", ids,
        index=ids.index(st.session_state.active_profile_id)
        if st.session_state.active_profile_id in ids else 0,
        format_func=lambda k: f"{k} · {profiles[k]['program'].get('name', '')}",
    )
    p = profiles[pid]

    def _activate_and_go():
        activate_profile(pid)
        st.session_state.nav = "Executive Dashboard"

    st.button(
        "Run analysis with this profile → Executive Dashboard",
        type="primary", on_click=_activate_and_go,
    )

    t_ov, t_units, t_skills, t_cred, t_src, t_edit, t_exp = st.tabs(
        ["Overview", "Instructional units", "Skills", "Credentials & assessments",
         "Source & evidence", "Edit profile", "Export"]
    )
    with t_ov:
        st.markdown(f"**Institution:** {p['institution'].get('name', '—')}")
        st.markdown(f"**Location:** {p['institution'].get('location', '—')}")
        st.markdown(f"**Institution type:** {p['institution'].get('institution_type', '—')}")
        st.markdown(f"**Program:** {p['program'].get('name', '—')}")
        st.markdown(f"**Level:** {p['program'].get('level', '—')}")
        st.markdown(f"**Duration:** {p['program'].get('duration', '—')}")
        hrs = p["program"].get("instructional_hours")
        st.markdown(
            f"**Instructional hours:** {hrs if hrs else 'Unknown — not documented'}"
        )
        st.markdown(f"**Description:** {p['program'].get('description', '—')}")
        st.markdown(
            f"**Review status:** `{p['metadata'].get('review_status', 'draft')}` · "
            f"**Source type:** `{scp.SOURCE_TYPES.get(p['source']['source_type'], '?')}`"
        )
        st.caption(p["metadata"].get("data_status", ""))
    with t_units:
        if p.get("instructional_units"):
            st.dataframe(pd.DataFrame(p["instructional_units"]),
                         width="stretch", hide_index=True)
        else:
            st.info("No instructional units documented.")
    with t_skills:
        if p.get("skills"):
            st.dataframe(pd.DataFrame(p["skills"]),
                         width="stretch", hide_index=True)
            st.caption(
                "Statuses: documented = explicit in source · inferred = implied, not "
                "confirmed · unknown = cannot be determined · recommended = suggested addition."
            )
        else:
            st.info("No skills recorded yet — add them in the Edit tab.")
    with t_cred:
        st.markdown("**Credentials**")
        st.dataframe(pd.DataFrame(p.get("credentials", [])),
                     width="stretch", hide_index=True)
        st.markdown("**Assessments**")
        st.dataframe(pd.DataFrame(p.get("assessments", [])),
                     width="stretch", hide_index=True)
    with t_src:
        st.markdown(f"**Source type:** {scp.SOURCE_TYPES.get(p['source']['source_type'], '?')}")
        st.markdown(f"**Title:** {p['source'].get('title', '—')}")
        st.markdown(f"**URL:** {p['source'].get('url', '—')}")
        st.markdown(f"**Retrieved:** {p['source'].get('retrieved_date', '—')}")
        st.markdown(f"**Notes:** {p['source'].get('notes', '—')}")
        if p["source"].get("evidence"):
            st.markdown("**Evidence register**")
            st.dataframe(pd.DataFrame(p["source"]["evidence"]),
                         width="stretch", hide_index=True)
        with st.expander("Verbatim source text (preserved, read-only)"):
            st.text(p["source"].get("original_text", "") or "—")
    with t_edit:
        st.caption(
            "Edit the profile, then save. Fields the source does not document should be "
            "left blank or 'unknown' — they are never invented."
        )
        edited = profile_ui.profile_editor(p, key_prefix=f"lib-{pid}")
        if st.button("Validate & save profile", key=f"lib-save-{pid}"):
            edited["profile_id"] = pid
            errors = scp.validate_profile(edited)
            if errors:
                st.error("Profile failed SCP v1.0 validation:")
                for e in errors[:15]:
                    st.caption(f"· {e}")
            else:
                st.session_state.profiles[pid] = edited
                try:
                    scp.save_profile(edited)
                    st.success(f"Saved {pid} (validated against SCP v1.0).")
                except OSError:
                    st.warning("Saved for this session only — library folder is not writable.")
                if st.session_state.active_profile_id == pid:
                    activate_profile(pid)
    with t_exp:
        st.download_button(
            "Download SCP JSON",
            scp.export_profile(p),
            file_name=f"{pid}.json",
            mime="application/json",
        )
        st.caption("The exported file validates against SCP v1.0 and can be re-imported.")


def page_curriculum():
    if not require_analysis():
        return
    st.subheader("Module 1 · Curriculum Intelligence")
    st.caption(
        "Every curriculum source is converted into a Standard Curriculum Profile (SCP) "
        "that can be reviewed, edited, and exported before analysis."
    )

    tab_lib, tab_sample, tab_upload, tab_text = st.tabs(
        ["Library profile", "Sample curriculum", "Upload file", "Paste text"]
    )
    with tab_lib:
        profiles = st.session_state.profiles
        if profiles:
            ids = list(profiles.keys())
            sel = st.selectbox(
                "Profile", ids,
                index=ids.index(st.session_state.active_profile_id)
                if st.session_state.active_profile_id in ids else 0,
                format_func=lambda k: f"{k} · {profiles[k]['program'].get('name', '')}",
                key="m1-libsel",
            )
            st.button(
                "Use this profile for analysis", key="m1-uselib",
                on_click=lambda: activate_profile(sel),
            )
            st.caption("Manage and edit profiles in the Curriculum Library.")
        else:
            st.info("The library is empty — create a profile from any source below.")
    with tab_sample:
        st.markdown(
            "Bundled **representative welding curriculum** — clearly labeled fictional "
            "sample, **not** an official Lyndon Institute document."
        )
        if st.button("Create profile from sample", key="m1-sample"):
            draft = scp.draft_from_text(
                curriculum.load_sample_curriculum(),
                "manual_entry",
                title="Representative sample welding curriculum",
                profile_id="SAMPLE-WELD-001",
            )
            draft["institution"]["name"] = "Demonstration institution"
            draft["source"]["notes"] = (
                "Fictional representative sample bundled with the application; "
                "not an official Lyndon Institute document."
            )
            st.session_state.draft_profile = draft
    with tab_upload:
        up = st.file_uploader(
            "Curriculum file", type=["pdf", "docx", "txt"],
            help="PDF, DOCX, or TXT. Scanned image-only PDFs will not extract.",
        )
        stype = st.selectbox(
            "What kind of document is this?",
            list(scp.SOURCE_TYPES.keys()),
            index=list(scp.SOURCE_TYPES.keys()).index("uploaded_document"),
            format_func=lambda s: scp.SOURCE_TYPES[s], key="m1-stype",
        )
        if up and st.button("Create profile from file", key="m1-upload"):
            try:
                text = curriculum.extract_text(up.getvalue(), up.name)
                st.session_state.draft_profile = scp.draft_from_text(
                    text, stype, title=up.name
                )
            except ValueError as exc:
                st.error(str(exc))
    with tab_text:
        pasted = st.text_area("Curriculum text", height=180,
                              placeholder="Paste curriculum text here...")
        if st.button("Create profile from text", key="m1-text") and pasted.strip():
            st.session_state.draft_profile = scp.draft_from_text(
                curriculum.normalize_text(pasted), "manual_entry", title="Pasted curriculum"
            )

    if st.session_state.draft_profile is not None:
        st.divider()
        draft = st.session_state.draft_profile
        st.markdown(f"### Review draft profile `{draft['profile_id']}`")
        st.caption(
            "Convert the source into a validated Standard Curriculum Profile. "
            "Leave undocumented fields blank or 'unknown'."
        )
        edited = profile_ui.profile_editor(draft, key_prefix="draft")
        c1, c2 = st.columns(2)
        if c1.button("Validate profile", key="draft-validate"):
            errors = scp.validate_profile(edited)
            if errors:
                for e in errors[:15]:
                    st.caption(f"· {e}")
            else:
                st.success("Valid against SCP v1.0.")
        if c2.button("Save profile & use for analysis", type="primary", key="draft-save"):
            errors = scp.validate_profile(edited)
            if errors:
                st.error("Profile failed SCP v1.0 validation:")
                for e in errors[:15]:
                    st.caption(f"· {e}")
            else:
                pid = edited["profile_id"]
                if pid in st.session_state.profiles and pid != draft["profile_id"]:
                    pid = scp.new_profile_id("DRAFT")
                    edited["profile_id"] = pid
                st.session_state.profiles[pid] = edited
                try:
                    scp.save_profile(edited)
                except OSError:
                    st.warning("Saved for this session only — library folder is not writable.")
                st.session_state.draft_profile = None
                activate_profile(pid)
                st.rerun()

    st.divider()
    st.toggle(
        "Use AI extraction (OpenAI)",
        key="use_ai",
        disabled=not live,
        help="Requires OPENAI_API_KEY. Responses are JSON-schema validated; "
        "failures fall back to the deterministic engine. Changing this "
        "re-runs the analysis automatically.",
    )
    if live:
        if st.button("Test OpenAI connection", key="m1-aitest"):
            with st.spinner("Sending a small authenticated request..."):
                result = ai_client.test_connection()
            (st.success if result["ok"] else st.error)(result["message"])
        st.caption(
            "Results feed both this module and AI Comparison - one analysis, "
            "no duplicate API calls."
        )
    if st.button("Re-run analysis", type="primary"):
        run_analysis(force=True)
        st.session_state.curriculum_version += 1
        st.rerun()
    if A and A.get("ai_error"):
        st.warning(f"AI enhancement failed; deterministic results shown: {A['ai_error']}")

    sk = A["skills"]
    st.caption(f"Active curriculum: **{st.session_state.curriculum_label}** · Engine: {A['method']}")

    n_doc = int((sk.status == "documented").sum())
    n_inf = int((sk.status == "inferred").sum())
    n_rec = int((sk.status == "recommended_addition").sum())
    n_und = int((sk.status == "not_documented").sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documented", n_doc)
    c2.metric("Inferred", n_inf)
    c3.metric("Suggested additions", n_rec)
    c4.metric("Total skills tracked", len(sk))
    st.caption(
        f"Total = {n_doc} documented + {n_inf} inferred + {n_rec} suggested "
        f"additions + {n_und} not established by available evidence."
    )

    # Chart intentionally plots ONLY documented+inferred skills.
    counts = sk[sk.status.isin(["documented", "inferred"])].groupby("category").size()
    counts = counts.reindex(["core", "supporting", "transferable", "adjacent"]).fillna(0)
    fig = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Skill category", "y": "Skills"},
        color_discrete_sequence=["#2563eb"],
        title=f"Documented + inferred skills by category "
        f"({int(counts.sum())} of {len(sk)} tracked)",
    )
    fig.update_layout(height=220, margin=dict(t=30, b=10), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Chart scope: documented + inferred skills only. Suggested additions "
        "and not-established skills are excluded by design - see the expanders "
        "below for the full categorized list."
    )

    for cat, label in [
        ("core", "Core technical skills"),
        ("supporting", "Supporting technical skills"),
        ("transferable", "Transferable skills"),
        ("adjacent", "Adjacent skills"),
    ]:
        # Expanders list curriculum-scoped skills only - absent catalog
        # rows are not displayed, so counts match what users see.
        sub = sk[(sk.category == cat) &
                 sk.status.isin(["documented", "inferred", "recommended_addition"])]
        with st.expander(f"{label} ({len(sub)})", expanded=False):
            for _, r in sub.iterrows():
                chip = STATUS_CHIP.get(r["status"], "")
                st.markdown(f"**{r['skill_name']}**{chip}", unsafe_allow_html=True)
                if r["evidence"]:
                    st.caption(f'Evidence: "{r["evidence"]}"')
                if r["rationale"]:
                    st.caption(f"Note: {r['rationale']}")
                if r["confidence"]:
                    st.caption(f"Confidence: {r['confidence']}")


def page_careers():
    if not require_analysis():
        return
    st.subheader("Module 2 · Career Pathway Intelligence")
    ca = A["careers"]
    st.caption(
        "Scores are explainable and indicative — not validated employment probabilities. "
        "Documented skills count 1.0, inferred skills 0.5 toward each occupation's requirements."
    )

    for path, label in careers.PATHWAY_LABELS.items():
        sub = ca[ca.pathway == path]
        st.markdown(f"### {label}s")
        view = sub[
            ["title", "soc_code", "compatibility_score", "n_documented", "n_inferred",
             "missing_skills"]
        ].rename(
            columns={
                "title": "Occupation",
                "soc_code": "SOC",
                "compatibility_score": "Score /100",
                "n_documented": "Doc. skills",
                "n_inferred": "Inferred",
                "missing_skills": "Missing / additional skills",
            }
        )
        st.dataframe(view, width="stretch", hide_index=True)

    st.divider()
    st.markdown("### Compare pathway alignment")
    fig = px.bar(
        ca.sort_values("compatibility_score"),
        x="compatibility_score", y="title", orientation="h", color="pathway",
        color_discrete_map={"primary": "#2563eb", "adjacent": "#0891b2", "extended": "#7c3aed"},
        labels={"compatibility_score": "Compatibility score", "title": "", "pathway": "Pathway"},
    )
    fig.update_layout(height=max(320, 30 * len(ca)), margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Inspect an occupation")
    choice = st.selectbox("Occupation", ca["title"].tolist())
    row = ca[ca.title == choice].iloc[0]
    st.markdown(f"**{row['title']}** ({row['soc_code']}) — {careers.PATHWAY_LABELS[row['pathway']]}")
    _render_lifestyle(row, ca, curriculum_employers())
    st.markdown(f"**Score:** {row['compatibility_score']}/100")
    st.info(careers.explain_score(row))
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Documented in curriculum**")
    c1.write(row["matched_documented"] or "—")
    c2.markdown("**Inferred (not confirmed)**")
    c2.write(row["matched_inferred"] or "—")
    c3.markdown("**Missing / to develop**")
    c3.write(row["missing_skills"] or "—")
    st.markdown("**Training & certification considerations**")
    st.write(row["training_notes"])
    st.markdown("**Why this occupation is relevant**")
    st.write(row["why_relevant"])


def page_employers():
    if not require_analysis():
        return
    st.subheader("Module 3 · Regional Employment Intelligence")
    g = get_geo_ref()
    st.caption(
        f"Reference point: {g['name']} ({g['latitude']:.4f}, {g['longitude']:.4f}). "
        "Distances are straight-line miles — not driving distances."
    )
    with st.expander("Change geographic reference"):
        _geo_editor("m3")

    emp_geo = curriculum_employers()

    radius = st.radio("Search radius (straight-line)", [25, 50, 100],
                      format_func=lambda m: f"{m} miles", horizontal=True)
    within = emp_geo[emp_geo.distance_miles <= radius]
    st.markdown(
        f"**{len(within)}** of {len(emp_geo)} dataset employers within {radius} mi."
    )

    industries = sorted(within["industry"].unique().tolist())
    sel_ind = st.multiselect("Industry", industries)
    ca_titles = ["(any)"] + A["careers"]["title"].tolist()
    sel_occ = st.selectbox("Career pathway", ca_titles)
    filtered = employers.filter_employers(
        within, industries=sel_ind or None,
        occupation=None if sel_occ == "(any)" else sel_occ,
    )

    st.dataframe(
        filtered[
            ["name", "city", "state", "industry", "distance_miles",
             "relevant_occupations", "information_source", "data_status"]
        ].rename(
            columns={
                "name": "Employer", "city": "City", "state": "State",
                "industry": "Industry", "distance_miles": "Mi (straight-line)",
                "relevant_occupations": "Relevant pathways",
                "information_source": "Source", "data_status": "Data status",
            }
        ),
        width="stretch", hide_index=True,
    )

    st.map(filtered.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])

    st.divider()
    st.markdown("### Job opportunities")
    st.warning(
        "No live job-posting feed is connected in v0.1. The entries below are **illustrative "
        "examples only** — they do NOT indicate that any employer is currently hiring.",
        icon="⚠️",
    )
    jl = employers.relevant_job_listings(
        REF["job_listings"], set(A["careers"]["title"])
    )
    if jl.empty:
        st.info(
            "No illustrative listings are relevant to this curriculum's "
            "pathways - irrelevant demonstration entries are filtered out."
        )
    else:
        st.dataframe(jl, width="stretch", hide_index=True)
    st.caption(
        "A replaceable data-provider interface (`modules.data_sources.DataProvider`) is in place "
        "for a future live job-posting feed."
    )


def page_earnings():
    if not require_analysis():
        return
    st.subheader("Module 4 · Earnings Intelligence")
    pw = earnings.pathway_wages(A["careers"], REF["wages"])
    st.caption(
        "Source: U.S. Bureau of Labor Statistics OEWS via O*NET OnLine. "
        "Entry proxy = 25th percentile; experienced proxy = 75th percentile. "
        "A percentile is a position in the wage distribution, not a guaranteed wage."
    )
    complete = pw.dropna(subset=["pct10", "entry_proxy", "median_annual",
                               "experienced_proxy", "pct90"])
    st.info(
        f"Earnings coverage: **{len(complete)} of {len(pw)}** occupations have "
        "all five published percentiles (P10–P90). The remainder show the "
        "percentiles actually published by the retrieved source — blanks are "
        "labeled gaps, not hidden estimates."
    )

    view = pw[
        ["title", "soc_code", "pathway", "wage_geographic_scope", "pct10",
         "entry_proxy", "median_annual", "experienced_proxy", "pct90",
         "reporting_period", "data_status"]
    ].copy()
    for col in ["pct10", "entry_proxy", "median_annual", "experienced_proxy", "pct90"]:
        view[col] = view[col].map(_fmt_money)
    st.dataframe(
        view.rename(
            columns={
                "title": "Occupation", "soc_code": "SOC", "pathway": "Pathway",
                "wage_geographic_scope": "Geography", "pct10": "P10",
                "entry_proxy": "P25", "median_annual": "Median",
                "experienced_proxy": "P75", "pct90": "P90",
                "reporting_period": "Reporting period", "data_status": "Status",
            }
        ),
        width="stretch", hide_index=True,
    )
    missing = pw[pw.data_status.str.contains("not published", na=False)]
    if len(missing):
        st.caption(
            "Coverage note: the retrieved source did not publish p25/median/p75 "
            f"for {len(missing)} occupation(s) — blank cells show only what the "
            "source published; the source mean is recorded in the Status column."
        )

    # Range chart: bar = published p10-p90 range; median marked when published.
    chart = pw.dropna(subset=["pct10", "pct90"]).sort_values("pct10")
    if len(chart):
        fig = go.Figure()
        fig.add_bar(
            x=chart["pct90"] - chart["pct10"], y=chart["title"],
            base=chart["pct10"], orientation="h", name="P10–P90 range",
            marker_color="#93c5fd",
            customdata=chart[["soc_code", "wage_geographic_scope",
                              "pct10", "pct90", "median_annual",
                              "reporting_period"]],
            hovertemplate=(
                "%{y}<br>SOC %{customdata[0]} · %{customdata[1]}<br>"
                "P10: $%{customdata[2]:,.0f} · P90: $%{customdata[3]:,.0f}<br>"
                "Median: $%{customdata[4]:,.0f}<br>%{customdata[5]}<extra></extra>"),
        )
        med = chart.dropna(subset=["median_annual"])
        if len(med):
            fig.add_scatter(
                x=med["median_annual"], y=med["title"], mode="markers",
                marker=dict(color="#1e40af", size=9, symbol="diamond"),
                name="Published median",
            )
        fig.update_layout(
            height=max(340, 30 * len(chart)), margin=dict(t=10, b=10),
            xaxis_title="Annual wage — published P10–P90 range (diamond = median)",
            yaxis_title="", showlegend=True,
        )
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "Extended pathways typically require additional education, certification, or "
        "experience — see Module 2 training notes."
    )

    st.markdown("### Vermont vs. national medians")
    cmp_df = earnings.compare_scopes(REF["wages"], pw["soc_code"].tolist())
    if "Vermont" in cmp_df.columns:
        cmp_df = cmp_df.dropna(subset=["Vermont"])
        fig2 = px.bar(
            cmp_df.melt(id_vars=["soc_code", "occupation_title"],
                        value_vars=[c for c in ["Vermont", "United States"] if c in cmp_df],
                        var_name="Scope", value_name="Median annual wage"),
            x="occupation_title", y="Median annual wage", color="Scope", barmode="group",
            color_discrete_map={"Vermont": "#059669", "United States": "#64748b"},
        )
        fig2.update_layout(height=360, xaxis_tickangle=-35, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, width="stretch")
    else:
        st.info("No Vermont-level wage rows available in the dataset.")


def page_recommendations():
    if not require_analysis():
        return
    st.subheader("Module 5 · Curriculum Improvement Advisor")
    recs = recommendations.generate_recommendations(
        A["skills"], A["careers_ids"], curriculum_employers()
    )
    st.caption(
        "Advisory recommendations for educator review — not curriculum requirements. "
        "Prioritized by implementation effort and number of pathways strengthened."
    )
    if recs.empty:
        st.info("No gaps detected against the recommendation catalog for this curriculum.")
        return

    for group in ["Quick wins", "Moderate enhancements", "Strategic opportunities"]:
        sub = recs[recs.group == group]
        if sub.empty:
            continue
        st.markdown(f"### {group}")
        for _, r in sub.iterrows():
            with st.expander(
                f"{r['recommendation']} — strengthens {r['n_pathways'] or 'multiple'} pathway(s)",
                expanded=(group == "Quick wins"),
            ):
                st.markdown(f"**Skill gap addressed:** {r['gap_addressed']}")
                st.markdown(f"**Pathways strengthened:** {r['pathways_strengthened']}")
                st.markdown(f"**Potential student benefit:** {r['student_benefit']}")
                st.markdown(f"**Rationale:** {r['rationale']}")
                st.markdown(f"**Demand evidence:** {r['employer_demand_evidence']}")
                st.markdown(f"**Implementation effort:** {r['effort']}")
                st.markdown(f"**Confidence & limitations:** {r['confidence_limitations']}")
                st.caption(f"Status: {r['data_status']}")

    st.markdown("### Opportunity matrix")
    fig = px.scatter(
        recs,
        x="effort", y="n_pathways",
        size="n_pathways", hover_name="recommendation", color="group",
        color_discrete_map={
            "Quick wins": "#059669",
            "Moderate enhancements": "#d97706",
            "Strategic opportunities": "#7c3aed",
        },
        labels={"effort": "Relative implementation effort",
                "n_pathways": "Career pathways strengthened", "group": ""},
        category_orders={"effort": ["Low", "Moderate", "High"]},
    )
    fig.update_layout(height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, width="stretch")


def page_compare():
    if not require_analysis():
        return
    st.subheader("AI vs Deterministic — side-by-side")

    # Deliberate re-run first (button state is evaluated this run).
    if st.button("Re-run analysis", key="cmp-rerun"):
        run_analysis(force=True)
        st.rerun()

    a = st.session_state.analysis
    meta = a.get("meta", {})
    st.caption(
        f"Engine: {a['method']}"
        + (f" · model {meta.get('model')}" if meta.get("model") else "")
        + (f" · ran {meta.get('ran_at')}" if meta.get("ran_at") else "")
        + " — one analysis feeds every module; navigating here makes no "
        "additional API call."
    )
    if ai_client.openai_available():
        if st.button("Test OpenAI connection", key="cmp-test"):
            with st.spinner("Sending a small authenticated request..."):
                result = ai_client.test_connection()
            (st.success if result["ok"] else st.error)(result["message"])

    rules_df = a["skills_rules"]
    present = rules_df[rules_df.status.isin(["documented", "inferred", "recommended_addition"])]

    if a["ai_status"] != "ok":
        msgs = {
            "disabled": "AI enhancement is **off** — enable 'Use AI extraction' in "
            "Module 1 (or below) and the comparison appears automatically.",
            "unavailable": "No OPENAI_API_KEY is configured — AI enhancement cannot run. "
            "Deterministic results are shown below.",
            "error": f"AI enhancement **failed** ({a.get('ai_error')}). Deterministic "
            "results are shown below.",
        }
        (st.warning if a["ai_status"] == "error" else st.info)(
            msgs.get(a["ai_status"], "")
        )
        if ai_client.openai_available() and a["ai_status"] != "ok":
            def _enable_ai():
                st.session_state.use_ai = True  # fingerprint change -> auto re-run

            st.button(
                "Enable AI & re-run analysis", type="primary",
                key="cmp-enable-ai", on_click=_enable_ai,
            )
        st.markdown("### Deterministic engine results")
        st.dataframe(
            present[["skill_name", "category", "status", "evidence"]].rename(
                columns={"skill_name": "Skill", "category": "Category",
                         "status": "Status", "evidence": "Evidence"}
            ),
            width="stretch", hide_index=True,
        )
        return

    # ai_status == "ok" -> render diffs from the SAME stored analysis object.
    sd = compare.diff_skills(rules_df, a["skills_ai"])
    cd = compare.diff_careers(a["careers_rules"], a["careers_ai"])
    emp_geo = curriculum_employers()
    rules_ids = recommendations.careers_with_skill_ids(
        a["careers_rules"], REF["occupations"]
    )
    ai_ids = recommendations.careers_with_skill_ids(
        a["careers_ai"], REF["occupations"]
    )
    rd = compare.diff_recommendations(
        recommendations.generate_recommendations(rules_df, rules_ids, emp_geo),
        recommendations.generate_recommendations(a["skills_ai"], ai_ids, emp_geo),
    )

    changed = sd[sd.change != "agreement"]
    st.markdown("### Skills")
    m1, m2, m3 = st.columns(3)
    m1.metric("Skills where engines agree", int((sd.change == "agreement").sum()))
    m2.metric("AI-upgraded findings", int((sd.change == "ai_upgraded").sum()))
    m3.metric(
        "AI-only / downgraded",
        int((sd.change.isin(["ai_only", "ai_downgraded"])).sum()),
    )
    show_all = st.checkbox("Show all skills (not just differences)", key="cmp-all")
    view = sd if show_all else changed
    st.dataframe(
        view[["skill", "category", "rules_status", "ai_status", "change",
              "ai_evidence", "ai_rationale"]],
        width="stretch", hide_index=True,
    )
    st.caption(
        "ai_evidence / ai_rationale are the model's own support for its finding — "
        "inferred skills are never confirmed competencies."
    )

    st.markdown("### Career pathway score differences")
    deltas = cd[cd.score_delta != 0]
    if deltas.empty:
        st.info("No score differences between engines.")
    else:
        st.dataframe(
            deltas[["title", "pathway", "rules_score", "ai_score", "score_delta",
                    "ai_missing"]],
            width="stretch", hide_index=True,
        )
    st.markdown("### Curriculum recommendations")
    st.dataframe(rd, width="stretch", hide_index=True)


def page_intel():
    if not require_analysis():
        return
    st.subheader("Regional Intelligence")
    g = get_geo_ref()
    a = st.session_state.analysis
    scope = intel.research_scope(a, a["skills"], a["careers"])
    st.caption(
        f"{st.session_state.curriculum_label} · center: {g['name']} "
        f"({g['latitude']:.4f}, {g['longitude']:.4f}) · scope derived from "
        f"{len(scope['skills'])} skills and {len(scope['primary_occupations'])} "
        "primary pathways."
    )

    res = st.session_state.get("intel_result")
    cur_fp = intel.fingerprint(scope, g, intel.DEFAULT_CONFIG)
    if res and res["fingerprint"] != cur_fp:
        st.info("Curriculum or geography changed — intelligence may be stale.")
        res = None
    if not res:
        st.info(
            "No regional intelligence collected yet. One deliberate research "
            "operation gathers all three layers; results are then reused by every "
            "module and the PDF report."
        )
        if st.button("Run regional research", type="primary", key="intel-run"):
            st.session_state.intel_result = intel.run_research(
                scope, g, intel.DEFAULT_CONFIG
            )
            st.session_state.intel_runs = st.session_state.get("intel_runs", 0) + 1
            st.rerun()
        st.caption(
            "Provider: bundled demonstration dataset (labeled). A live retrieval "
            "adapter can be configured via the provider interface — paid sources "
            "require approval and credentials."
        )
        return

    # Curriculum-scoped relevance: display only observations tied to this
    # curriculum's eligible occupations or present skills (stored evidence
    # is preserved; presentation is filtered).
    pres_ids = set(
        a["skills"].loc[a["skills"].status.isin(["documented", "inferred"]),
                        "skill_id"]
    )
    titles = set(a["careers"]["title"])
    obs_all = intel.relevant_observations(res["observations"], titles, pres_ids)
    by_layer = {
        l["key"]: [o for o in obs_all if o.layer == l["key"]] for l in intel.LAYERS
    }
    unlocated = [o for o in obs_all if o.layer is None]
    st.caption(
        f"Collected {res['collected_at']} · provider: {res['provider']} · "
        f"{len(obs_all)} of {len(res['observations'])} observations relevant to "
        f"this curriculum · research runs: {st.session_state.get('intel_runs', 1)}"
    )
    if st.button("Re-run research", key="intel-rerun"):
        st.session_state.intel_result = intel.run_research(scope, g)
        st.session_state.intel_runs += 1
        st.rerun()

    for layer in intel.LAYERS:
        obs = by_layer[layer["key"]]
        st.markdown(f"### {layer['label']} — {layer['band']}")
        st.caption(layer["question"])
        st.markdown(intel.summarize_layer(layer["key"], obs, scope))
        for o in obs[:3]:
            st.markdown(
                f"**{o.organization}** ({o.city}, {o.state} — {o.distance_miles} mi) "
                f"· `{o.category}` · {o.observation} "
                f"*Implication:* {o.curriculum_implication}"
            )
        if obs:
            with st.expander(f"View supporting intelligence ({len(obs)})"):
                st.dataframe(
                    pd.DataFrame([{
                        "Organization": o.organization,
                        "Development": o.observation,
                        "Skills": "; ".join(o.skills),
                        "Source": o.source_url or "—",
                        "Published": o.source_published or "—",
                        "Evidence": o.evidence_class,
                        "Confidence": o.confidence,
                    } for o in obs]),
                    width="stretch", hide_index=True,
                )
    if unlocated:
        with st.expander(f"Unlocated evidence ({len(unlocated)})"):
            for o in unlocated:
                st.caption(f"{o.organization}: {o.observation} — {o.evidence_class}")


def page_reports():
    if not require_analysis():
        return
    st.subheader("Curriculum Opportunity Report")
    st.caption(
        "A professional PDF assessment built from the completed analysis — "
        "no additional API calls are made. Demonstration and unverified data "
        "are labeled inside the report."
    )
    a = st.session_state.analysis
    profile = st.session_state.profiles.get(st.session_state.active_profile_id)
    st.markdown(
        f"**Subject:** {st.session_state.curriculum_label} · "
        f"**Engine:** {a['method']} · "
        f"**Analysis ran:** {a.get('meta', {}).get('ran_at', 'unknown')}"
    )

    if st.button("Generate PDF report", type="primary", key="rep-gen"):
        with st.spinner("Building report..."):
            pw = earnings.pathway_wages(a["careers"], REF["wages"])
            emp_geo = curriculum_employers()
            recs = recommendations.generate_recommendations(
                a["skills"], a["careers_ids"], emp_geo
            )
            st.session_state.report_pdf = report.build_report(
                a,
                profile,
                {
                    "wages": pw,
                    "employers": emp_geo,
                    "recommendations": recs,
                    "sources": REF["sources"],
                    "job_listings": REF["job_listings"],
                    "intel": st.session_state.get("intel_result"),
                },
                get_geo_ref(),
                theme=THEME,
            )
            st.session_state.report_fingerprint = a["fingerprint"]

    pdf = st.session_state.get("report_pdf")
    if pdf:
        if st.session_state.report_fingerprint != a["fingerprint"]:
            st.warning("Analysis changed since this PDF was generated — regenerate.")
        pid = (profile or {}).get("profile_id", "curriculum")
        st.download_button(
            "Download Curriculum Opportunity Report",
            data=pdf,
            file_name=f"ATD-opportunity-report-{pid}.pdf",
            mime="application/pdf",
            type="primary",
        )
        st.caption(
            "Sections: executive summary · skills intelligence · career pathways · "
            "earnings · regional employment · enhancement opportunities · "
            "institutional actions · methodology & sources."
        )


def page_sources():
    st.subheader("Data & Methodology")
    st.markdown(
        "Every important claim in this application carries a source status. "
        "Statuses used: **Verified source data**, **Derived analysis**, "
        "**AI-generated recommendation**, **Demonstration example**, "
        "**Unavailable or unverified**."
    )
    for s in REF["sources"]:
        st.markdown(
            f"**{s.name}**  \n"
            f"Scope: {s.geographic_scope} · Period: {s.publication_period or 'n/a'} · "
            f"Retrieved: {s.retrieved_date or 'n/a'} · Status: `{s.status}`"
            + (f"  \n{s.url}" if s.url else "")
        )
    st.divider()
    st.markdown("#### Method notes")
    st.markdown(
        "- **Skill extraction:** deterministic keyword matching against a skills taxonomy; "
        "inference rules mark implied skills so they are never confused with documented ones.\n"
        "- **Compatibility score:** weighted share of an occupation's required skills present "
        "in the curriculum (documented = 1.0, inferred = 0.5).\n"
        "- **Employer distance:** haversine (straight-line) miles from Lyndon Institute; "
        "employer coordinates are municipality centroids (approximate).\n"
        "- **Wages:** published BLS OEWS percentile figures; Vermont preferred, U.S. fallback, "
        "scope always labeled.\n"
        "- **AI mode:** optional OpenAI extraction, JSON-schema validated, graceful fallback. "
        "AI analysis of a document is separate from live labor-market verification."
    )


PAGES = {
    "Home": page_home,
    "Select Curriculum": page_select,
    "Executive Dashboard": page_dashboard,
    "Curriculum Library": page_library,
    "1 · Curriculum Intelligence": page_curriculum,
    "2 · Career Pathways": page_careers,
    "3 · Regional Employment": page_employers,
    "4 · Earnings Intelligence": page_earnings,
    "5 · Curriculum Advisor": page_recommendations,
    "AI Comparison": page_compare,
    "Regional Intelligence": page_intel,
    "Reports": page_reports,
    "Data & Methodology": page_sources,
}

PAGES[page]()

st.markdown(
    '<div class="footer">After the Diploma — proof of concept v0.1 · '
    "Analysis is decision-support only; verify all employer, wage, and pathway claims "
    "against primary sources before use.</div>",
    unsafe_allow_html=True,
)
