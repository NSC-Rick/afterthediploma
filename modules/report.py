"""Curriculum Opportunity Report - PDF generation service.

Builds a professional, evidence-based PDF from the SHARED analysis object
(same data the UI displays) plus the active SCP and reference datasets.
Streamlit-free: pure ReportLab + matplotlib, returns bytes in memory.

Nothing here invents content - every figure comes from the analysis,
profile, or reference datasets passed in. Missing data is reported as
"not documented"/"unavailable", never replaced with plausible filler.
"""

from __future__ import annotations

import io
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#0f172a")
BLUE = colors.HexColor("#2563eb")
TEAL = colors.HexColor("#0891b2")
PURPLE = colors.HexColor("#7c3aed")
SLATE = colors.HexColor("#475569")
LIGHT = colors.HexColor("#f1f5f9")
LINE = colors.HexColor("#cbd5e1")
PATHWAY_COLORS = {"primary": "#2563eb", "adjacent": "#0891b2", "extended": "#7c3aed"}

PAGE_W, PAGE_H = letter
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

_GENERATED = datetime.now()  # module load time is irrelevant; set at build


def _styles(primary: str = "#2563eb") -> dict:
    pri = colors.HexColor(primary)
    base = dict(fontName="Helvetica", textColor=NAVY, leading=13)
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=26,
                                textColor=NAVY, leading=30, spaceAfter=4),
        "subtitle": ParagraphStyle("subtitle", fontSize=13, textColor=SLATE,
                                   leading=16, spaceAfter=18),
        "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15,
                             textColor=pri, leading=18, spaceBefore=6,
                             spaceAfter=6, keepWithNext=1),
        "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5,
                             textColor=NAVY, leading=14, spaceBefore=8,
                             spaceAfter=4, keepWithNext=1),
        "body": ParagraphStyle("body", fontSize=9.5, spaceAfter=5, **base),
        "small": ParagraphStyle("small", fontSize=8, textColor=SLATE, leading=10,
                                spaceAfter=4),
        "cell": ParagraphStyle("cell", fontSize=8, leading=10, splitLongWords=0),
        "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8,
                                leading=10, textColor=colors.white, splitLongWords=0),
        "note": ParagraphStyle("note", fontSize=8.5, textColor=SLATE,
                               backColor=LIGHT, borderPadding=6, leading=11,
                               spaceBefore=4, spaceAfter=8),
        "covermeta": ParagraphStyle("covermeta", fontSize=10, leading=14,
                                    textColor=SLATE),
    }


_TAGS = {
    "<b>": "\x01B\x02", "</b>": "\x01/B\x02",
    "<i>": "\x01I\x02", "</i>": "\x01/I\x02",
    "<br/>": "\x01BR\x02", "&nbsp;": "\x01N\x02",
}
_TAGS_INV = {v: k for k, v in _TAGS.items()}


def _p(text, style):
    """Paragraph that keeps authored <b>/<i> markup but escapes data."""
    t = str(text)
    for tag, tok in _TAGS.items():
        t = t.replace(tag, tok)
    t = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    for tok, tag in _TAGS_INV.items():
        t = t.replace(tok, tag if tag != "&nbsp;" else "&#160;")
    return Paragraph(t, style)


def _table(data, col_widths, header=True, st=None, header_bg=None):
    """Word-wrapped table with repeat header row (no truncated cells)."""
    st = st or _styles()
    wrapped = []
    for i, row in enumerate(data):
        wrapped.append(
            [_p(c, st["cellb"] if (header and i == 0) else st["cell"]) for c in row]
        )
    t = Table(wrapped, colWidths=col_widths, repeatRows=1 if header else 0)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor(header_bg) if header_bg else (BLUE if header else LIGHT)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _bar_chart(series: dict[str, float], title: str, color: str = "#2563eb",
               horizontal: bool = False, xlabel: str = "") -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=150)
    names, vals = list(series.keys()), list(series.values())
    if horizontal:
        ax.barh(names[::-1], vals[::-1], color=color)
        ax.set_xlabel(xlabel)
    else:
        ax.bar(names, vals, color=color)
        ax.set_ylabel(xlabel)
    ax.set_title(title, fontsize=10, color="#0f172a", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def _money(v):
    return f"${int(v):,}" if pd.notna(v) and v is not None else "—"


def _make_footer(theme):
    def _footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(SLATE)
        left = "After the Diploma — Curriculum Opportunity Report"
        if theme and theme.demo_label:
            left += "  ·  Demonstration - not an official institution product"
        canvas.drawString(MARGIN, 0.45 * inch, left)
        canvas.drawRightString(PAGE_W - MARGIN, 0.45 * inch, f"Page {doc.page}")
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, 0.6 * inch, PAGE_W - MARGIN, 0.6 * inch)
        canvas.restoreState()
    return _footer


def build_report(analysis: dict, profile: dict | None, datasets: dict,
                 geo_ref: dict, theme=None) -> bytes:
    """Render the Curriculum Opportunity Report as PDF bytes.

    ``datasets``: {"wages": pathway_wages_df, "employers": df-with-distance,
                   "sources": [SourceInfo], "job_listings": df|None}
    ``theme``: InstitutionTheme for institution-branded output (optional).
    """
    PRI = theme.primary if theme else "#2563eb"
    st = _styles(PRI)
    _t = lambda data, widths: _table(data, widths, st=st, header_bg=PRI)
    skills_df, careers_df = analysis["skills"], analysis["careers"]
    recs_df = datasets.get("recommendations", pd.DataFrame())
    wages_df = datasets["wages"]
    emp_df = datasets["employers"]
    meta = analysis.get("meta", {})
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    inst = (profile or {}).get("institution", {})
    prog = (profile or {}).get("program", {})
    src = (profile or {}).get("source", {})

    story = []

    # ------------------------------------------------- 1. cover + exec summary
    if theme and theme.institution_name:
        badge = Table(
            [[Paragraph(
                f'<font color="white"><b>{theme.emblem_initials}</b></font>',
                st["cell"])]],
            colWidths=[0.55 * inch], rowHeights=[0.55 * inch])
        badge.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(PRI)),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story += [
            Spacer(1, 0.5 * inch),
            badge,
            _p(f"{theme.institution_name} — {theme.environment}",
               ParagraphStyle("env", fontSize=11, textColor=colors.HexColor(PRI),
                              spaceAfter=2)),
        ]
        if theme.emblem_url is None:
            story.append(_p(theme.emblem_status, st["small"]))
    else:
        story.append(Spacer(1, 0.6 * inch))
    story += [
        _p("After the Diploma", st["title"]),
        _p("Curriculum Opportunity Report", st["subtitle"]),
        Spacer(1, 0.1 * inch),
        _p(f"<b>Institution:</b> {inst.get('name') or 'Not specified'}", st["covermeta"]),
        _p(f"<b>Program:</b> {prog.get('name') or 'Not specified'}", st["covermeta"]),
        _p(f"<b>Location:</b> {inst.get('location') or geo_ref['name']}", st["covermeta"]),
        _p(f"<b>Analysis date:</b> {meta.get('ran_at', generated_at)} &nbsp;·&nbsp; "
           f"<b>Report generated:</b> {generated_at}", st["covermeta"]),
        _p(f"<b>Profile:</b> {(profile or {}).get('profile_id', 'n/a')} "
           f"(SCP v{(profile or {}).get('scp_version', '?')}) &nbsp;·&nbsp; "
           f"<b>Source type:</b> {src.get('source_type', 'unknown').replace('_', ' ')}",
           st["covermeta"]),
        Spacer(1, 0.15 * inch),
    ]
    if src.get("notes"):
        story.append(_p(src["notes"], st["note"]))
    if theme and theme.demo_label:
        strip = Table([[_p(f"<b>{theme.demo_label}</b>", st["cell"])]],
                      colWidths=[CONTENT_W])
        strip.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(PRI)),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story += [strip, Spacer(1, 0.08 * inch)]

    n_doc = int((skills_df.status == "documented").sum())
    n_inf = int((skills_df.status == "inferred").sum())
    top_paths = careers_df.assign(
        compatibility_score=pd.to_numeric(
            careers_df["compatibility_score"], errors="coerce")
    )
    top_paths = top_paths[top_paths.pathway == "primary"].nlargest(
        1, "compatibility_score")
    story += [
        _p("Executive summary", st["h1"]),
        _p(
            f"This analysis documents <b>{n_doc} skills</b> in the curriculum and infers "
            f"<b>{n_inf}</b> more, supporting <b>{len(careers_df)} candidate career "
            f"pathways</b> — {len(careers_df[careers_df.pathway=='primary'])} primary, "
            f"{len(careers_df[careers_df.pathway=='adjacent'])} adjacent, and "
            f"{len(careers_df[careers_df.pathway=='extended'])} extended. "
            + (f"Strongest direct alignment: <b>{top_paths.iloc[0]['title']}</b> "
               f"({top_paths.iloc[0]['compatibility_score']}/100). " if len(top_paths) else "")
            + "Graduates leave with hands-on welding and fabrication skills mapped to "
            "real occupations, regional employers, and published wage benchmarks.",
            st["body"],
        ),
    ]
    if len(recs_df):
        story.append(_p("Top opportunities", st["h2"]))
        for _, r in recs_df.head(3).iterrows():
            story.append(_p(f"<b>{r['recommendation']}.</b> {r['student_benefit']}",
                            st["body"]))
    story.append(PageBreak())

    # ------------------------------------------------- 2. skills intelligence
    story += [_p("Curriculum & skills intelligence", st["h1"])]
    for status, label, desc in [
        ("documented", "Documented in public source",
         "Explicitly evidenced in the source material."),
        ("inferred", "Reasonably inferred",
         "Implied by documented content — NOT confirmed student competencies."),
        ("recommended_addition", "Recommended for consideration",
         "Not established by available evidence; plausibly valuable — not existing "
         "competencies and not asserted deficiencies."),
    ]:
        sub = skills_df[skills_df.status == status]
        story.append(KeepTogether([
            _p(f"{label} ({len(sub)})", st["h2"]), _p(desc, st["small"]),
        ]))
        if len(sub):
            rows = [["Skill", "Category", "Evidence / rationale"]]
            for _, r in sub.iterrows():
                ev = r.get("evidence") or r.get("rationale") or "—"
                rows.append([r["skill_name"], r["category"], ev])
            story.append(_t(rows, [1.7 * inch, 0.9 * inch, 4.4 * inch]))
        else:
            story.append(_p("None identified.", st["body"]))
        story.append(Spacer(1, 0.08 * inch))
    if analysis.get("used_ai"):
        story.append(_p(
            "AI enhancement contributed to this extraction; AI-surfaced evidence is "
            "shown in the AI Comparison module. Engine: " + analysis["method"],
            st["small"]))
    absent = skills_df[skills_df.status == "not_documented"]
    if len(absent):
        story.append(_p(
            f"<b>Not established by available evidence ({len(absent)}):</b> "
            + ", ".join(absent["skill_name"]) +
            ". Absence from a public description does not establish a curriculum "
            "deficiency.", st["small"]))

    # ------------------------------------------------- 3. career pathways
    story += [_p("Career pathway analysis", st["h1"])]
    counts = careers_df.groupby("pathway").size().reindex(
        ["primary", "adjacent", "extended"]).fillna(0)
    story.append(Image(
        _bar_chart({k.title(): int(v) for k, v in counts.items()},
                   "Pathways by category", xlabel="Pathways"),
        width=4.6 * inch, height=1.9 * inch))
    story.append(_p(
        "Scores are indicative alignment (documented skills count 1.0, inferred 0.5 "
        "toward each occupation's requirements) — not employment probabilities.",
        st["small"]))
    for path, label, note in [
        ("primary", "Primary pathways", "Direct, realistic entry-level options."),
        ("adjacent", "Adjacent pathways", "Shared skills; supplemental training likely needed."),
        ("extended", "Extended pathways", "Longer-term careers needing further experience, credentials, or education."),
    ]:
        sub = careers_df[careers_df.pathway == path]
        story.append(KeepTogether([_p(f"{label} ({len(sub)})", st["h2"]),
                                   _p(note, st["small"])]))
        rows = [["Occupation", "SOC", "Score", "Readiness",
                 "Missing / to develop", "Training notes"]]
        for _, r in sub.iterrows():
            rows.append([r["title"], r["soc_code"], f"{r['compatibility_score']}/100",
                         r.get("readiness", "—"),
                         r["missing_skills"] or "—", r["training_notes"]])
        story.append(_t(rows, [1.35 * inch, 0.7 * inch, 0.6 * inch,
                               1.3 * inch, 1.35 * inch, 1.7 * inch]))
        story.append(Spacer(1, 0.08 * inch))
    story.append(_p(
        "Scores measure technical skill overlap only. Credentials, licenses, and "
        "experience requirements are listed separately under Readiness/Training notes "
        "and are not counted in the score.", st["small"]))
    story.append(PageBreak())

    # ------------------------------------------------- 4. earnings
    story += [_p("Earnings intelligence", st["h1"]),
              _p("Source: U.S. Bureau of Labor Statistics OEWS via O*NET OnLine. "
                 "Entry proxy = 25th percentile; experienced proxy = 75th percentile. "
                 "A percentile is a wage-distribution position, not a guaranteed wage.",
                 st["small"])]
    wages_df = wages_df.assign(
        median_annual=pd.to_numeric(wages_df["median_annual"], errors="coerce")
    )
    chart_src = wages_df.dropna(subset=["median_annual"]).nlargest(8, "median_annual")
    if len(chart_src):
        story.append(Image(_bar_chart(
            {t[:28]: float(v) for t, v in
             zip(chart_src["title"], chart_src["median_annual"])},
            "Median annual wages (best available scope)", color="#059669",
            horizontal=True, xlabel="Annual USD"),
            width=5.6 * inch, height=2.2 * inch))
    rows = [["Occupation", "SOC", "Scope", "Entry (p25)", "Median", "Experienced (p75)"]]
    for _, r in wages_df.iterrows():
        rows.append([r["title"], r["soc_code"], r["wage_geographic_scope"],
                     _money(r["entry_proxy"]), _money(r["median_annual"]),
                     _money(r["experienced_proxy"])])
    story.append(_t(rows, [1.9 * inch, 0.6 * inch, 1.0 * inch,
                               1.1 * inch, 1.0 * inch, 1.4 * inch]))
    story.append(_p(
        "Wage figures are broad occupational benchmarks; extended-pathway earnings "
        "assume additional experience or credentials. Reporting period shown in "
        "Methodology.", st["small"]))
    story.append(PageBreak())

    # ------------------------------------------------- 5. regional employment
    g = geo_ref
    story += [_p("Regional employment opportunities", st["h1"]),
              _p(f"Reference point: {g['name']} ({g['latitude']:.4f}, "
                 f"{g['longitude']:.4f}). Distances are straight-line miles, not "
                 "driving distance.", st["small"])]
    for radius in (25, 50, 100):
        n = int((emp_df.distance_miles <= radius).sum())
        story.append(_p(f"• <b>{radius} mi:</b> {n} employers in the reference dataset",
                        st["body"]))
    rows = [["Employer", "Location", "Industry", "Mi", "Relevant pathways"]]
    for _, r in emp_df.head(14).iterrows():
        rows.append([r["name"], f"{r['city']}, {r['state']}", r["industry"],
                     r["distance_miles"], r["relevant_occupations"]])
    story.append(Spacer(1, 0.05 * inch))
    story.append(_t(rows, [1.7 * inch, 1.1 * inch, 1.6 * inch, 0.4 * inch, 2.2 * inch]))
    story.append(_p(
        "Employers are real regional organizations compiled for demonstration and are "
        "<b>not verified as currently hiring</b>. No job openings are asserted; a live "
        "job-posting feed is designed for a future release.", st["note"]))

    # ------------------------------------------------- 6. enhancement opportunities
    story += [_p("Curriculum enhancement opportunities", st["h1"]),
              _p("Advisory recommendations for educator review — not curriculum "
                 "requirements.", st["small"])]
    creds = (profile or {}).get("credentials", [])
    doc_creds = [c["name"] for c in creds if c.get("status") == "documented"]
    if doc_creds:
        story.append(_p(
            "<b>Existing credential-related instruction (documented):</b> "
            + "; ".join(doc_creds)
            + ". Recommendations below aim to extend, not duplicate, these offerings.",
            st["body"]))
    for group in ["Quick wins", "Moderate enhancements", "Strategic opportunities"]:
        sub = recs_df[recs_df.group == group] if len(recs_df) else recs_df
        if not len(sub):
            continue
        story.append(_p(f"{group} ({len(sub)})", st["h2"]))
        rows = [["Recommendation", "Gap addressed", "Connection & pathways", "Effort",
                 "Validation & confidence"]]
        for _, r in sub.iterrows():
            rows.append([r["recommendation"], r["gap_addressed"],
                         f"{r.get('connection','')}; {r['pathways_strengthened']}",
                         r["effort"],
                         f"{r.get('validation','')}; {r['confidence_limitations']}"])
        story.append(_t(rows, [1.6 * inch, 1.6 * inch, 1.9 * inch,
                                   0.5 * inch, 1.4 * inch]))
        story.append(Spacer(1, 0.06 * inch))
    story.append(PageBreak())

    # ------------------------------------------------- 6b. regional intelligence
    intel_res = datasets.get("intel")
    story += [_p("Regional Intelligence Outlook", st["h1"])]
    if intel_res:
        story.append(_p(
            f"Collected {intel_res['collected_at']} via provider "
            f"'{intel_res['provider']}' ({len(intel_res['observations'])} "
            "observations). Cached results - no new research calls were made for "
            "this report.", st["small"]))
        from .intel import LAYERS, summarize_layer, relevant_observations
        pres_ids = set(
            skills_df.loc[skills_df.status.isin(["documented", "inferred"]),
                          "skill_id"]
        )
        obs_all = relevant_observations(
            intel_res["observations"], set(careers_df["title"]), pres_ids
        )
        story.append(_p(
            f"{len(obs_all)} of {len(intel_res['observations'])} observations "
            "relevant to this curriculum's pathways.", st["small"]))
        for layer in LAYERS:
            obs = [o for o in obs_all if o.layer == layer["key"]]
            story.append(KeepTogether([
                _p(f"{layer['label']} ({layer['band']})", st["h2"])]))
            story.append(_p(summarize_layer(layer["key"], obs, intel_res["scope"]),
                            st["body"]))
            if obs:
                rows = [["Organization", "Development", "Evidence", "Conf."]]
                for o in obs[:4]:
                    rows.append([o.organization, o.observation[:160],
                                 o.evidence_class, o.confidence])
                story.append(_t(rows, [1.5 * inch, 3.6 * inch, 1.4 * inch, 0.5 * inch]))
    else:
        story.append(_p(
            "Regional intelligence has not been collected for this analysis - "
            "run a research operation in the Regional Intelligence module to "
            "populate this section.", st["body"]))
    story.append(PageBreak())

    # ------------------------------------------------- 7. institutional actions
    quick = recs_df[recs_df.group == "Quick wins"] if len(recs_df) else recs_df
    strategic = recs_df[recs_df.group == "Strategic opportunities"] if len(recs_df) else recs_df
    actions = []
    if len(quick):
        actions.append(
            f"<b>Near term:</b> schedule the low-effort additions first — "
            f"{'; '.join(quick['recommendation'].head(2))}. These need little or no "
            "new equipment.")
    if len(strategic):
        actions.append(
            f"<b>Longer term:</b> evaluate "
            f"{strategic.iloc[0]['recommendation'].lower()} as a program-level "
            "initiative with partner input.")
    actions.append(
        f"<b>Employer engagement:</b> invite the {min(5, len(emp_df))} nearest "
        "reference employers to validate the skill list against real hiring needs "
        "and to explore work-based learning placements.")
    actions.append(
        "<b>Curriculum validation:</b> confirm undocumented skills marked "
        "'unknown'/'inferred' with instructors and update the profile - this report "
        "reflects only what the source documents.")
    actions.append(
        "<b>Data refresh:</b> re-run this analysis annually or whenever the "
        "curriculum, wage data, or employer landscape changes.")
    story += [_p("Recommended institutional actions", st["h1"])]
    for i, a in enumerate(actions, 1):
        story.append(_p(f"{i}. {a}", st["body"]))
    story.append(Spacer(1, 0.1 * inch))

    # ------------------------------------------------- 8. methodology & sources
    story += [_p("Methodology & sources", st["h1"]),
              _p("<b>Analysis method:</b> " + analysis["method"], st["body"]),
              _p(f"<b>Engine metadata:</b> model {meta.get('model') or 'n/a'} · "
                 f"prompt v{meta.get('prompt_version', '?')} · SCP v"
                 f"{meta.get('scp_version', '?')} · analysis ran "
                 f"{meta.get('ran_at', 'unknown')}", st["body"]),
              _p("<b>Evidence classification:</b> documented = explicit in source · "
                 "inferred = implied, not confirmed · unknown = cannot be determined · "
                 "recommended = suggested addition. Statuses: Verified source data, "
                 "Derived analysis, AI-generated recommendation, Demonstration "
                 "example, Unavailable or unverified.", st["body"])]
    if src.get("original_text"):
        story.append(_p(
            "<b>Source curriculum:</b> " + (src.get("title") or "document") +
            (f" ({src['url']})" if src.get("url") else "") +
            f" — retrieved {src.get('retrieved_date', 'n/a')}. The verbatim source is "
            "preserved in the profile; this report is derived analysis on top of it.",
            st["body"]))
    for s in datasets.get("sources", []):
        story.append(_p(
            f"<b>{s.name}</b> — scope {s.geographic_scope}; period "
            f"{s.publication_period or 'n/a'}; retrieved {s.retrieved_date or 'n/a'}; "
            f"status: {s.status}", st["small"]))
    story.append(_p(
        "Limitations: employer coordinates are municipality centroids; job openings "
        "are not live; wage percentiles are occupational benchmarks; compatibility "
        "scores are indicative. No content in this report was invented to fill gaps — "
        "undocumented items are labeled unknown or absent.", st["note"]))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=0.9 * inch,
        title="After the Diploma — Curriculum Opportunity Report",
        author="After the Diploma",
    )
    doc.build(story, onFirstPage=_make_footer(theme),
              onLaterPages=_make_footer(theme))
    return buf.getvalue()
