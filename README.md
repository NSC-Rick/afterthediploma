# After the Diploma — Proof of Concept v0.1

**Connecting education, skills, and regional opportunity.**

An interactive workforce-intelligence application that demonstrates how a Career and
Technical Education (CTE) institution can transform an existing curriculum into
actionable career and regional-opportunity insight. The architecture accepts any
program without code changes; a **representative Lyndon Institute Welding profile**
(`LI-WELD-001`) ships as the bundled demonstration.

### User journey

**Home → Curriculum Selection → Profile Review → Intelligence Analysis →
Executive Dashboard → Detailed Modules**

The app opens on a general-purpose landing page with three entry points:
*Analyze a Curriculum* (upload/paste), *Explore a Program* (SCP library), and
*Try a Demonstration* (the Welding profile). No curriculum is pre-selected; the
Executive Dashboard and the five intelligence modules unlock only after an
explicit profile confirmation. The geographic reference for employer matching is
configurable per analysis.

## What it answers

1. What skills is the curriculum developing? (documented / inferred / suggested)
2. What careers do those skills prepare students for? (primary / adjacent / extended)
3. Which regional employers align with those skills? (25 / 50 / 100-mile radii)
4. What are the potential earnings across pathways? (BLS OEWS percentiles)
5. How could the curriculum be enhanced? (prioritized, advisory recommendations)

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app runs end-to-end in **Demonstration Mode** with no credentials. Set
`OPENAI_API_KEY` (see `.env.example`) to enable optional AI-assisted extraction in
Module 1. The deterministic rules engine remains the fallback and default.

## Architecture

```
app.py                     Streamlit UI shell: dashboard + curriculum library + 5 modules + sources
modules/
  curriculum.py            PDF / DOCX / TXT text extraction
  scp.py                   Standard Curriculum Profile v1.0: schema validation,
                           drafting, import/export, profile->corpus conversion
  profile_ui.py            shared editable profile review screen
  skills.py                taxonomy-driven extraction (documented / inferred / suggested)
  careers.py               occupation matching + explainable compatibility score
  employers.py             haversine distance math, radius & attribute filters
  earnings.py              BLS OEWS wage join, percentile labeling, scope fallback
  recommendations.py       rule-catalog gap analysis -> prioritized advisor output
  ai_client.py             optional OpenAI extraction, JSON-schema validated
  data_sources.py          provider interface (swap CSVs for live feeds later)
data/
  scp_schema.json          versioned JSON Schema for SCP v1.0
  profiles/LI-WELD-001.json  Lyndon Institute Welding profile (public description)
  sample_welding_curriculum.txt, skills_taxonomy.csv, occupations.csv,
  wages.csv, employers.csv, job_listings.csv
tests/                     pytest suite for every analysis module
```

### Standard Curriculum Profile (SCP) v1.0

Every curriculum source — PDF, DOCX, TXT, pasted text, or a public program
description — is converted into a validated, editable **SCP** before analysis:

- Versioned JSON Schema (`data/scp_schema.json`); all profiles validate against it.
- Source-type classification: complete curriculum / course catalog /
  public program description / manual entry / uploaded document.
- Skill statuses: `documented`, `inferred`, `unknown`, `recommended`.
- Original source text preserved verbatim under `source.original_text`.
- Undocumented hours, credentials, proficiency levels and assessments stay
  null/`unknown` — never invented.
- Curriculum Library UI: inspect units/skills, review evidence, edit, save,
  import/export JSON, and run the five-module analysis on any profile.
- Bundled prototype `LI-WELD-001` built from Lyndon Institute's public Welding
  program page (summary description only — not the complete curriculum).

### Replaceable data providers

`modules.data_sources.DataProvider` defines the contract (`occupations()`, `wages()`,
`employers()`, `job_listings()`, `skills_taxonomy()`, `sources()`). v0.1 ships
`BundledDataProvider` backed by CSVs; a live provider can be dropped in behind
`get_provider()` without touching analysis or UI code.

## Data provenance

| Dataset | Source | Status |
|---|---|---|
| Wages | U.S. BLS OEWS via O*NET OnLine (CC-BY 4.0), 2025 wage data, retrieved 2026-10-31 | Verified source data |
| Employer list | Real regional employers from public listings/websites; **not verified for hiring** | Demonstration example |
| Job listings | Illustrative entries only — no live feed in v0.1 | Demonstration example |
| Sample curriculum | Fictional representative document, not an official Lyndon Institute document | Demonstration example |
| Scores & recommendations | In-app analysis | Derived analysis |

Distances are straight-line (haversine) miles from Lyndon Institute
(44.5387, -72.0163), never driving distances. Compatibility scores are indicative,
not validated employment probabilities.

## Deployment (Render)

`render.yaml` is included. Exact service configuration:

- **Type:** Web Service, Python
- **Python:** 3.11 (`PYTHON_VERSION=3.11.9`)
- **Build:** `pip install -r requirements.txt`
- **Start:** `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT --server.headless true`
- **Health check path:** `/_stcore/health` (Streamlit's built-in endpoint, returns `ok`)
- **Secrets:** `OPENAI_API_KEY` only if Live Analysis Mode is desired — never commit it.

If a Render service already exists for this repo, connect it to the same GitHub
repository and branch (`main`); do not create a second service.

## Tests

```bash
pytest -q
```

## Known limitations (v0.1)

- No live job-posting or employer-verification feed (provider interface ready).
- Employer coordinates are municipality centroids (approximate).
- Skill extraction keywords cover the welding domain; other programs need taxonomy additions.
- Vermont wage estimates are suppressed by BLS for some occupations; the app falls back
  to national figures and labels the scope.
