"""Tests for institution theming: registry, app branding, PDF branding."""

from pathlib import Path

import pytest
import pymupdf
from streamlit.testing.v1 import AppTest

from modules import theme as themes
from modules.report import build_report
from tests.test_report import _payload, GEO

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_theme_registry():
    assert "lyndon" in themes.THEMES and "afterthediploma" in themes.THEMES
    li = themes.get_theme("lyndon")
    assert li.primary == "#842b37"  # verified LI maroon
    assert "DEMONSTRATION ONLY" in li.demo_label
    assert li.emblem_url is None  # no unapproved emblem bundled


def test_generic_theme_has_no_demo_label():
    assert themes.get_theme("afterthediploma").demo_label == ""


def test_get_theme_fallback():
    assert themes.get_theme("nope").key == themes.DEFAULT_THEME
    assert themes.get_theme(None).key == themes.DEFAULT_THEME


def test_app_shows_demo_disclaimer():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    text = " ".join(m.value for m in at.markdown)
    assert "DEMONSTRATION ONLY" in text
    assert "NOT AN OFFICIAL LYNDON INSTITUTE PRODUCT" in text


def test_app_generic_theme_hides_disclaimer():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.session_state["theme_key"] = "afterthediploma"
    at.run()
    text = " ".join(m.value for m in at.markdown)
    assert "DEMONSTRATION ONLY" not in text


def test_pdf_uses_institution_branding():
    li = themes.get_theme("lyndon")
    analysis, profile, ds = _payload()
    pdf = build_report(analysis, profile, ds, GEO, theme=li)
    text = " ".join(
        p.get_text() for p in pymupdf.open(stream=pdf, filetype="pdf")
    )
    flat = " ".join(text.split())
    assert "NOT AN OFFICIAL LYNDON INSTITUTE PRODUCT" in flat
    assert "Lyndon Institute" in flat


def test_pdf_generic_theme_still_works():
    analysis, profile, ds = _payload()
    pdf = build_report(analysis, profile, ds, GEO, theme=themes.GENERIC)
    assert pdf[:5] == b"%PDF-"
    text = " ".join(
        p.get_text() for p in pymupdf.open(stream=pdf, filetype="pdf")
    )
    assert "NOT AN OFFICIAL" not in text


def test_pdf_no_theme_backwards_compatible():
    analysis, profile, ds = _payload()
    pdf = build_report(analysis, profile, ds, GEO)
    assert pdf[:5] == b"%PDF-"
