"""After the Diploma - proof of concept v0.1.

Connecting education, skills, and regional opportunity.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules import (
    ai_client,
    careers,
    curriculum,
    earnings,
    employers,
    profile_ui,
    recommendations,
    scp,
    skills,
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
REF_EMPLOYERS = employers.employers_with_distance(REF["employers"])


# ---------------------------------------------------------------- state
def activate_profile(profile_id: str):
    """Make an SCP the active curriculum input for analysis."""
    p = st.session_state.profiles[profile_id]
    st.session_state.active_profile_id = profile_id
    st.session_state.curriculum_text = scp.corpus_text(p)
    st.session_state.curriculum_label = f"SCP {profile_id} · {p['program'].get('name','')}"
    st.session_state.curriculum_version += 1


def init_state():
    if "initialized" in st.session_state:
        return
    st.session_state.profiles = {p["profile_id"]: p for p in scp.list_profiles()}
    st.session_state.draft_profile = None
    st.session_state.analysis = None
    st.session_state.curriculum_version = 0
    st.session_state.last_run_version = -1
    st.session_state.use_ai = False
    if "LI-WELD-001" in st.session_state.profiles:
        activate_profile("LI-WELD-001")
    else:
        st.session_state.active_profile_id = None
        st.session_state.curriculum_text = curriculum.load_sample_curriculum()
        st.session_state.curriculum_label = "Representative sample welding curriculum (demo)"
    st.session_state.initialized = True


init_state()


def run_analysis(use_ai: bool) -> str | None:
    """Run the skill + career pipeline; returns a warning string or None."""
    text = st.session_state.curriculum_text
    warning = None
    rules_df = skills.extract_skills_rules(text, REF["taxonomy"])

    if use_ai and ai_client.openai_available():
        try:
            ai_skills = ai_client.analyze_curriculum(text)
            skills_df = skills.merge_ai_results(rules_df, ai_skills, REF["taxonomy"])
            method = "Rules engine + AI analysis (OpenAI)"
        except ai_client.AIClientError as exc:
            skills_df = rules_df
            method = "Rules engine (AI failed - fallback)"
            warning = f"AI analysis failed, using deterministic extraction instead: {exc}"
    else:
        skills_df = rules_df
        method = "Deterministic rules engine" if not use_ai else "Rules engine (no API key)"

    # Educator-reviewed SCP skill statuses override engine output.
    pid = st.session_state.get("active_profile_id")
    if pid and pid in st.session_state.get("profiles", {}):
        skills_df = scp.apply_profile_overrides(
            st.session_state.profiles[pid], skills_df, REF["taxonomy"]
        )
        method += " + reviewed profile statuses"

    careers_df = careers.match_occupations(skills_df, REF["occupations"])
    careers_ids = recommendations.careers_with_skill_ids(careers_df, REF["occupations"])

    st.session_state.analysis = {
        "skills": skills_df,
        "careers": careers_df,
        "careers_ids": careers_ids,
        "method": method,
        "used_ai": use_ai and ai_client.openai_available() and warning is None,
    }
    st.session_state.last_run_version = st.session_state.curriculum_version
    return warning


def ensure_analysis() -> dict:
    if (
        st.session_state.analysis is None
        or st.session_state.last_run_version != st.session_state.curriculum_version
    ):
        run_analysis(st.session_state.get("use_ai", False))
    return st.session_state.analysis


A = ensure_analysis()


# ---------------------------------------------------------------- chrome
st.markdown("# After the Diploma")
st.markdown(
    '<div class="subtitle">Connecting education, skills, and regional opportunity.</div>',
    unsafe_allow_html=True,
)

live = ai_client.openai_available()
if live:
    st.markdown(
        '<div class="live-banner"><b>Live Analysis Mode available</b> - an OpenAI API key is '
        "configured. AI extraction can be enabled in Module 1. Employment and wage data remain "
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
    st.header("Navigate")
    page = st.radio(
        "Section",
        key="nav",
        [
            "Executive Dashboard",
            "Curriculum Library",
            "1 · Curriculum Intelligence",
            "2 · Career Pathways",
            "3 · Regional Employment",
            "4 · Earnings Intelligence",
            "5 · Curriculum Advisor",
            "Data & Methodology",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown(
        f"**Geographic reference**  \n{REFERENCE_POINT['name']}  \n"
        f"{REFERENCE_POINT['latitude']:.4f}, {REFERENCE_POINT['longitude']:.4f}"
    )
    st.markdown(
        f"**Analysis engine**  \n{A['method']}",
        help="Shown on every page so results are never misread as AI-generated when they are not.",
    )
    st.caption("Curriculum: " + st.session_state.curriculum_label)


STATUS_CHIP = {
    "documented": '<span class="status-chip chip-doc">documented</span>',
    "inferred": '<span class="status-chip chip-inf">inferred</span>',
    "recommended_addition": '<span class="status-chip chip-rec">suggested</span>',
}


def _fmt_money(v):
    return f"${int(v):,}" if pd.notna(v) else "—"


# ---------------------------------------------------------------- pages
def page_dashboard():
    sk, ca = A["skills"], A["careers"]
    doc_core = sk[(sk.status == "documented") & (sk.category == "core")]
    adjacent = sk[
        (sk.category == "adjacent") & (sk.status.isin(["documented", "inferred"]))
    ]
    emp100 = REF_EMPLOYERS[REF_EMPLOYERS.distance_miles <= 100]
    pw = earnings.pathway_wages(ca, REF["wages"])
    recs = recommendations.generate_recommendations(
        sk, A["careers_ids"], REF_EMPLOYERS
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

    st.markdown("##### Pathways by category")
    counts = ca.groupby("pathway").size().reindex(["primary", "adjacent", "extended"]).fillna(0)
    fig = px.bar(
        x=[careers.PATHWAY_LABELS[p] for p in counts.index],
        y=counts.values,
        labels={"x": "", "y": "Pathways"},
        color=counts.index,
        color_discrete_map={
            "primary": "#2563eb",
            "adjacent": "#0891b2",
            "extended": "#7c3aed",
        },
    )
    fig.update_layout(showlegend=False, height=260, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

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
                         use_container_width=True, hide_index=True)
        else:
            st.info("No instructional units documented.")
    with t_skills:
        if p.get("skills"):
            st.dataframe(pd.DataFrame(p["skills"]),
                         use_container_width=True, hide_index=True)
            st.caption(
                "Statuses: documented = explicit in source · inferred = implied, not "
                "confirmed · unknown = cannot be determined · recommended = suggested addition."
            )
        else:
            st.info("No skills recorded yet — add them in the Edit tab.")
    with t_cred:
        st.markdown("**Credentials**")
        st.dataframe(pd.DataFrame(p.get("credentials", [])),
                     use_container_width=True, hide_index=True)
        st.markdown("**Assessments**")
        st.dataframe(pd.DataFrame(p.get("assessments", [])),
                     use_container_width=True, hide_index=True)
    with t_src:
        st.markdown(f"**Source type:** {scp.SOURCE_TYPES.get(p['source']['source_type'], '?')}")
        st.markdown(f"**Title:** {p['source'].get('title', '—')}")
        st.markdown(f"**URL:** {p['source'].get('url', '—')}")
        st.markdown(f"**Retrieved:** {p['source'].get('retrieved_date', '—')}")
        st.markdown(f"**Notes:** {p['source'].get('notes', '—')}")
        if p["source"].get("evidence"):
            st.markdown("**Evidence register**")
            st.dataframe(pd.DataFrame(p["source"]["evidence"]),
                         use_container_width=True, hide_index=True)
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
    use_ai = st.toggle(
        "Use AI extraction (OpenAI)",
        value=False,
        disabled=not live,
        help="Requires OPENAI_API_KEY. Responses are JSON-schema validated; "
        "failures fall back to the deterministic engine.",
    )
    st.session_state.use_ai = use_ai
    if st.button("Re-run analysis", type="primary"):
        warning = run_analysis(use_ai)
        if warning:
            st.warning(warning)
        else:
            st.success("Analysis updated.")

    sk = A["skills"]
    st.caption(f"Active curriculum: **{st.session_state.curriculum_label}** · Engine: {A['method']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Documented", int((sk.status == "documented").sum()))
    c2.metric("Inferred", int((sk.status == "inferred").sum()))
    c3.metric("Suggested additions", int((sk.status == "recommended_addition").sum()))
    c4.metric("Total skills tracked", len(sk))

    counts = sk[sk.status.isin(["documented", "inferred"])].groupby("category").size()
    counts = counts.reindex(["core", "supporting", "transferable", "adjacent"]).fillna(0)
    fig = px.bar(
        x=counts.index, y=counts.values,
        labels={"x": "Skill category", "y": "Skills"},
        color_discrete_sequence=["#2563eb"],
    )
    fig.update_layout(height=220, margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    for cat, label in [
        ("core", "Core technical skills"),
        ("supporting", "Supporting technical skills"),
        ("transferable", "Transferable skills"),
        ("adjacent", "Adjacent skills"),
    ]:
        sub = sk[sk.category == cat]
        with st.expander(f"{label} ({len(sub)})", expanded=(cat == "core")):
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
        st.dataframe(view, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Compare pathway alignment")
    fig = px.bar(
        ca.sort_values("compatibility_score"),
        x="compatibility_score", y="title", orientation="h", color="pathway",
        color_discrete_map={"primary": "#2563eb", "adjacent": "#0891b2", "extended": "#7c3aed"},
        labels={"compatibility_score": "Compatibility score", "title": "", "pathway": "Pathway"},
    )
    fig.update_layout(height=max(320, 30 * len(ca)), margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Inspect an occupation")
    choice = st.selectbox("Occupation", ca["title"].tolist())
    row = ca[ca.title == choice].iloc[0]
    st.markdown(f"**{row['title']}** ({row['soc_code']}) — {careers.PATHWAY_LABELS[row['pathway']]}")
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
    st.subheader("Module 3 · Regional Employment Intelligence")
    st.caption(
        f"Reference point: {REFERENCE_POINT['name']}. "
        "Distances are straight-line miles — not driving distances."
    )

    radius = st.radio("Search radius (straight-line)", [25, 50, 100],
                      format_func=lambda m: f"{m} miles", horizontal=True)
    within = REF_EMPLOYERS[REF_EMPLOYERS.distance_miles <= radius]
    st.markdown(
        f"**{len(within)}** of {len(REF_EMPLOYERS)} dataset employers within {radius} mi."
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
        use_container_width=True, hide_index=True,
    )

    st.map(filtered.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])

    st.divider()
    st.markdown("### Job opportunities")
    st.warning(
        "No live job-posting feed is connected in v0.1. The entries below are **illustrative "
        "examples only** — they do NOT indicate that any employer is currently hiring.",
        icon="⚠️",
    )
    jl = REF["job_listings"]
    st.dataframe(jl, use_container_width=True, hide_index=True)
    st.caption(
        "A replaceable data-provider interface (`modules.data_sources.DataProvider`) is in place "
        "for a future live job-posting feed."
    )


def page_earnings():
    st.subheader("Module 4 · Earnings Intelligence")
    pw = earnings.pathway_wages(A["careers"], REF["wages"])
    st.caption(
        "Source: U.S. Bureau of Labor Statistics OEWS via O*NET OnLine. "
        "Entry proxy = 25th percentile; experienced proxy = 75th percentile. "
        "A percentile is a position in the wage distribution, not a guaranteed wage."
    )

    view = pw[
        ["title", "soc_code", "pathway", "wage_geographic_scope", "entry_proxy",
         "median_annual", "experienced_proxy", "reporting_period", "data_status"]
    ].copy()
    for col in ["entry_proxy", "median_annual", "experienced_proxy"]:
        view[col] = view[col].map(_fmt_money)
    st.dataframe(
        view.rename(
            columns={
                "title": "Occupation", "soc_code": "SOC", "pathway": "Pathway",
                "wage_geographic_scope": "Wage scope", "entry_proxy": "Entry proxy (p25)",
                "median_annual": "Median", "experienced_proxy": "Experienced proxy (p75)",
                "reporting_period": "Reporting period", "data_status": "Status",
            }
        ),
        use_container_width=True, hide_index=True,
    )

    chart = pw.dropna(subset=["median_annual"]).sort_values("median_annual")
    fig = px.bar(
        chart, x="median_annual", y="title", orientation="h", color="pathway",
        error_x_minus=chart["median_annual"] - chart["pct10"],
        error_x=chart["pct90"] - chart["median_annual"],
        color_discrete_map={"primary": "#2563eb", "adjacent": "#0891b2", "extended": "#7c3aed"},
        labels={"median_annual": "Median annual wage (bar) with p10–p90 range",
                "title": "", "pathway": "Pathway"},
    )
    fig.update_layout(height=max(340, 30 * len(chart)), margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No Vermont-level wage rows available in the dataset.")


def page_recommendations():
    st.subheader("Module 5 · Curriculum Improvement Advisor")
    recs = recommendations.generate_recommendations(
        A["skills"], A["careers_ids"], REF_EMPLOYERS
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
    st.plotly_chart(fig, use_container_width=True)


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
    "Executive Dashboard": page_dashboard,
    "Curriculum Library": page_library,
    "1 · Curriculum Intelligence": page_curriculum,
    "2 · Career Pathways": page_careers,
    "3 · Regional Employment": page_employers,
    "4 · Earnings Intelligence": page_earnings,
    "5 · Curriculum Advisor": page_recommendations,
    "Data & Methodology": page_sources,
}

PAGES[page]()

st.markdown(
    '<div class="footer">After the Diploma — proof of concept v0.1 · '
    "Analysis is decision-support only; verify all employer, wage, and pathway claims "
    "against primary sources before use.</div>",
    unsafe_allow_html=True,
)
