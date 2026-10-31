"""Tests for geographic employer matching."""

import pytest

from modules import employers
from modules.data_sources import REFERENCE_POINT, get_provider


@pytest.fixture(scope="module")
def dataset():
    return employers.employers_with_distance(get_provider().employers())


def test_haversine_known_distance():
    # Lyndonville to St. Johnsbury is roughly 8 miles straight-line.
    d = employers.haversine_miles(44.5337, -72.0031, 44.4195, -72.0157)
    assert 6 < d < 11


def test_haversine_zero_for_same_point():
    assert employers.haversine_miles(44.5, -72.0, 44.5, -72.0) == pytest.approx(0)


def test_reference_point_is_lyndon(dataset):
    assert "Lyndon" in REFERENCE_POINT["name"]
    assert dataset["distance_miles"].min() < 10  # something local exists


def test_radius_filters_monotonic(dataset):
    r25 = dataset[dataset.distance_miles <= 25]
    r50 = dataset[dataset.distance_miles <= 50]
    r100 = dataset[dataset.distance_miles <= 100]
    assert len(r25) < len(r50) <= len(r100) <= len(dataset)
    assert r100["distance_miles"].max() <= 100


def test_distances_labeled_straight_line(dataset):
    assert "distance_miles" in dataset.columns  # UI relabels as straight-line


def test_no_employer_claims_verified_hiring(dataset):
    assert dataset["data_status"].str.lower().str.contains(
        "not verified for current hiring"
    ).all()


def test_occupation_filter(dataset):
    filtered = employers.filter_employers(dataset, occupation="Welder / Welding Fabricator")
    assert len(filtered) > 0
    assert all("Welder / Welding Fabricator" in o for o in filtered["relevant_occupations"])
