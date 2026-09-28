"""
tests/test_license_text.py — spec B7.1, family "license-text".

`scripts/core/license.py::_cc_type_from_text` must match whole canonical
phrases with word boundaries, let the most restrictive match win, map
"all rights reserved" to PROPRIETARY, treat bare "attribution" as CC_BY only
in a Creative Commons context, and return no permitted type for anything else.

The gate tests drive the real ingest path: ccMixter hands classify_license a
free text `license_name` as `license_str` (scripts/download.py:136), so every
label below is fed through classify_license("ccmixter", license_str=...) and
judged by is_publishable, exactly as scripts/download.py:328 does.
"""

from __future__ import annotations

import pytest

from scripts.core.license import (
    PERMITTED_LICENSES,
    LicenseType,
    _cc_type_from_text,
    classify_license,
    is_publishable,
)

# --- The six verified false positives (audit ledger, ingest and licence) ---------
# Each one was classified publishable (cc_by or cc0) by the substring matcher.
FALSE_POSITIVES = [
    "all rights reserved by the artist",
    "Attribution Share-Alike",
    "Attribution No Derivatives",
    "Attribution Non-Commercial",
    "Ground Zero Records",
    "Sampling Plus",
]

# --- Real ccMixter / Internet Archive allow labels -------------------------------
ALLOW_LABELS = [
    "Attribution (3.0)",
    "Creative Commons Attribution",
    "CC BY 4.0",
    "Public Domain Dedication (CC0)",
]


@pytest.mark.parametrize("label", FALSE_POSITIVES)
def test_false_positive_labels_are_not_publishable(label: str) -> None:
    info = classify_license("ccmixter", license_str=label)
    assert not is_publishable(info), (
        f"{label!r} classified {info.license_type.value} and passed the publish gate"
    )
    assert info.license_type not in PERMITTED_LICENSES


@pytest.mark.parametrize("label", ALLOW_LABELS)
def test_real_allow_labels_are_publishable(label: str) -> None:
    info = classify_license("ccmixter", license_str=label)
    assert is_publishable(info), (
        f"{label!r} classified {info.license_type.value} and was refused"
    )


@pytest.mark.parametrize("label", FALSE_POSITIVES)
def test_false_positives_not_publishable_via_archive_source(label: str) -> None:
    # The Internet Archive deny cell is text only (no licenseurl), same fallback.
    info = classify_license("archive", license_str=label)
    assert not is_publishable(info), label


# --- "all rights reserved" is PROPRIETARY, and it wins over any CC word ------------

@pytest.mark.parametrize(
    "label",
    [
        "all rights reserved",
        "All Rights Reserved",
        "all rights reserved by the artist",
        "(c) 2024 Some Label. All rights reserved.",
        # most restrictive match wins: a CC word next to a reservation is still a reservation
        "Creative Commons Attribution, all rights reserved",
    ],
)
def test_all_rights_reserved_is_proprietary(label: str) -> None:
    assert _cc_type_from_text(label) is LicenseType.PROPRIETARY, label
    info = classify_license("ccmixter", license_str=label)
    assert info.license_type is LicenseType.PROPRIETARY
    assert not is_publishable(info)


# --- Most restrictive match wins on real ccMixter spellings -------------------------

@pytest.mark.parametrize(
    "label,expected",
    [
        ("Attribution Share-Alike", LicenseType.CC_BY_SA),
        ("Attribution No Derivatives", LicenseType.CC_BY_ND),
        ("Attribution Non-Commercial", LicenseType.CC_BY_NC),
        ("Attribution NonCommercial", LicenseType.CC_BY_NC),
        ("Attribution-NonCommercial-ShareAlike", LicenseType.CC_BY_NC_SA),
        ("Attribution Noncommercial No Derivatives", LicenseType.CC_BY_NC_ND),
        ("CC BY-NC 3.0", LicenseType.CC_BY_NC),
        ("CC BY-SA 4.0", LicenseType.CC_BY_SA),
        ("CC BY-ND 4.0", LicenseType.CC_BY_ND),
        # CC0 and PUBLIC_DOMAIN are equally permissive; both permitted
        ("CC0 1.0", LicenseType.CC0),
        ("Public Domain", LicenseType.PUBLIC_DOMAIN),
    ],
)
def test_most_restrictive_match_wins(label: str, expected: LicenseType) -> None:
    assert _cc_type_from_text(label) is expected, label


def test_public_domain_dedication_cc0_is_a_permitted_type() -> None:
    lt = _cc_type_from_text("Public Domain Dedication (CC0)")
    assert lt in (LicenseType.CC0, LicenseType.PUBLIC_DOMAIN), lt


# --- Bare "attribution" is CC_BY only in a Creative Commons context -----------------

@pytest.mark.parametrize(
    "label",
    [
        "Creative Commons Attribution",
        "Creative Commons Attribution 3.0",
        "CC Attribution",
        "CC BY 4.0",
        "cc by",
    ],
)
def test_attribution_after_cc_context_is_cc_by(label: str) -> None:
    assert _cc_type_from_text(label) is LicenseType.CC_BY, label


# --- Word boundaries: slugs and CC words never match inside other words -------------

@pytest.mark.parametrize(
    "label",
    [
        "Ground Zero Records",   # "zero" inside a record label name, not the CC0 slug
        "Sampling Plus",         # retired ccMixter licence, no CC grant
        "Bygone Era",            # "by" inside "bygone"
        "Standby",               # "by" as a suffix
        "Rugby Anthems",         # "by" inside "rugby"
        "Sandstone",             # "sa" and "nd" inside a word
        "Zero Gravity Records",  # bare "zero" outside a CC context
        "Byzantine Chant Society",  # "by" prefix and "sa" inside
        "",                      # empty label: no grant
    ],
)
def test_unrelated_text_is_never_a_permitted_type(label: str) -> None:
    lt = _cc_type_from_text(label)
    assert lt in (None, LicenseType.UNKNOWN), f"{label!r} -> {lt}"
    info = classify_license("ccmixter", license_str=label)
    assert info.license_type is LicenseType.UNKNOWN, label
    assert not is_publishable(info)


def test_none_and_whitespace_labels_are_unknown() -> None:
    for label in (None, "", "   "):
        info = classify_license("ccmixter", license_str=label)
        assert info.license_type is LicenseType.UNKNOWN, repr(label)
        assert not is_publishable(info)
