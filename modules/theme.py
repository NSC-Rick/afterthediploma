"""Institution theming for After the Diploma.

An InstitutionTheme carries everything the UI and PDF report need to
present the platform as configured for a specific institution - colors,
emblem, and the required demonstration disclaimer. Themes are pure
configuration; no module hardcodes a school's branding.

Lyndon Institute colors were verified against public brand assets:
the school seal (LI-Seal-LI-Maroon.png) uses maroon #842b37 with dark
accents ~#43151b. The official emblem is NOT bundled - usage rights are
unconfirmed, so a flagged placeholder is shown instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InstitutionTheme:
    key: str
    institution_name: str          # "" = generic After the Diploma theme
    environment: str               # e.g. "Demonstration Environment"
    primary: str                   # headings, header band
    secondary: str                 # darker support color
    accent: str                    # badges, chips, chart highlight
    website: str = ""
    emblem_url: str | None = None  # approved asset only - never assumed
    emblem_status: str = "placeholder - pending approval"
    emblem_initials: str = "ATD"
    demo_label: str = ""           # empty -> no institution disclaimer needed


GENERIC = InstitutionTheme(
    key="afterthediploma",
    institution_name="",
    environment="",
    primary="#2563eb",
    secondary="#0f172a",
    accent="#0891b2",
    emblem_initials="ATD",
)

LYNDON = InstitutionTheme(
    key="lyndon",
    institution_name="Lyndon Institute",
    environment="Demonstration Environment",
    primary="#842b37",      # verified LI maroon (Seal-LI-Maroon.png)
    secondary="#43151b",    # darker seal accent
    accent="#a97c2f",       # design accent (muted gold); not an official LI color
    website="https://www.lyndoninstitute.org",
    emblem_url=None,  # official seal: usage rights unconfirmed -> placeholder
    emblem_status="Official emblem not bundled - usage approval pending",
    emblem_initials="LI",
    demo_label="DEMONSTRATION ONLY — NOT AN OFFICIAL LYNDON INSTITUTE PRODUCT",
)

THEMES: dict[str, InstitutionTheme] = {t.key: t for t in (GENERIC, LYNDON)}
DEFAULT_THEME = "lyndon"


def get_theme(key: str | None) -> InstitutionTheme:
    return THEMES.get(key) or THEMES[DEFAULT_THEME]
