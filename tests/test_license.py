"""
tests/test_license.py — the hosted-catalogue publish gate (deployment plan P1).

Covers what `test_core_modules.py::TestLicense` does not: the CC-community source
branches (ccMixter / FMA / Internet Archive / Musopen), the PUBLIC_DOMAIN type,
`is_publishable`, and the `REMIXMATE_LICENSE_ENFORCE` flag.
"""

from __future__ import annotations

import pytest

from scripts.core.license import (
    LicenseInfo,
    LicenseType,
    _cc_type_from_text,
    classify_license,
    is_publishable,
    license_enforce_enabled,
    parse_cc_url,
)

# --- The 8-case source matrix (deployment plan P1 "classify_license matrix") ---------
#
# One CC-BY (allow) and one CC-BY-NC (deny) per source, each fed the licence in
# the *shape that source actually hands to* classify_license (see the fetchers in
# scripts/download.py):
#   • ccmixter  — license_url (CC deed) + license_str ("license_name": "Attribution")
#   • archive   — license_url ("licenseurl" from item metadata); the deny cell is
#                 text-only, which also guards the "archive" source alias in
#                 classify_license (without it the branch falls through to the
#                 generic else, which has no _cc_type_from_text fallback)
#   • fma       — license_url (embedded CC deed), stubbed fetcher
#   • musopen   — usually no field -> PUBLIC_DOMAIN; here we pass a contradicting
#                 CC signal to prove the PD default does not mask it
#
# Asserted per cell, not once globally: the resolved LicenseType AND the publish
# gate (is_publishable). "deny + allow per source" so a gate mutated to accept or
# reject everything fails this test.
_MATRIX = [
    # id,                     source,      kwargs,                                                        expected_type,           publishable
    ("ccmixter-by",           "ccmixter",  dict(license_url="http://creativecommons.org/licenses/by/3.0/",
                                                license_str="Attribution"),                               LicenseType.CC_BY,       True),
    ("ccmixter-by-nc",        "ccmixter",  dict(license_str="Attribution-NonCommercial"),                 LicenseType.CC_BY_NC,    False),
    ("archive-by",            "archive",   dict(license_url="https://creativecommons.org/licenses/by/4.0/"), LicenseType.CC_BY,     True),
    ("archive-by-nc",         "archive",   dict(license_str="Attribution-NonCommercial"),                 LicenseType.CC_BY_NC,    False),
    ("fma-by",                "fma",       dict(license_url="https://creativecommons.org/licenses/by/4.0/"), LicenseType.CC_BY,     True),
    ("fma-by-nc",             "fma",       dict(license_url="https://creativecommons.org/licenses/by-nc/4.0/"), LicenseType.CC_BY_NC, False),
    ("musopen-by",            "musopen",   dict(license_url="https://creativecommons.org/licenses/by/4.0/"), LicenseType.CC_BY,     True),
    ("musopen-by-nc",         "musopen",   dict(license_str="Attribution-NonCommercial"),                 LicenseType.CC_BY_NC,    False),
]


@pytest.mark.parametrize(
    "source,kwargs,expected_type,publishable",
    [(m[1], m[2], m[3], m[4]) for m in _MATRIX],
    ids=[m[0] for m in _MATRIX],
)
def test_source_licence_matrix(source, kwargs, expected_type, publishable) -> None:
    info = classify_license(source, **kwargs)
    assert info.license_type is expected_type, (
        f"{source} {kwargs} -> {info.license_type}, expected {expected_type}"
    )
    assert is_publishable(info) is publishable, (
        f"{source} {kwargs}: gate says {is_publishable(info)}, expected {publishable}"
    )


def test_matrix_covers_allow_and_deny_for_every_source() -> None:
    """Guards the guard: 4 sources x {allow, deny} = 8 cells, no source all-one-way."""
    seen: dict[str, set[bool]] = {}
    for _id, source, _kwargs, _lt, publishable in _MATRIX:
        seen.setdefault(source, set()).add(publishable)
    assert len(_MATRIX) == 8
    assert set(seen) == {"ccmixter", "archive", "fma", "musopen"}
    assert all(outcomes == {True, False} for outcomes in seen.values()), seen


def test_musopen_without_any_licence_field_is_public_domain() -> None:
    lic = classify_license("musopen")
    assert lic.license_type is LicenseType.PUBLIC_DOMAIN
    assert lic.commercial_ok is True
    assert lic.attribution_required is False
    assert is_publishable(lic)


def test_unresolved_cc_community_licence_stays_unknown_and_is_rejected() -> None:
    lic = classify_license("ccmixter", license_str="some bespoke label")
    assert lic.license_type is LicenseType.UNKNOWN
    assert not is_publishable(lic)


def test_parse_cc_url_distinguishes_cc0_from_public_domain_mark() -> None:
    assert parse_cc_url("https://creativecommons.org/publicdomain/zero/1.0/") is LicenseType.CC0
    assert (
        parse_cc_url("https://creativecommons.org/publicdomain/mark/1.0/")
        is LicenseType.PUBLIC_DOMAIN
    )


@pytest.mark.parametrize(
    "text,expected",
    [
        ("CC BY 3.0", LicenseType.CC_BY),
        ("Attribution-ShareAlike", LicenseType.CC_BY_SA),
        ("Attribution-NonCommercial-NoDerivatives", LicenseType.CC_BY_NC_ND),
        ("Public Domain", LicenseType.PUBLIC_DOMAIN),
        ("CC0 1.0", LicenseType.CC0),
        # B7.1 (2026-09-27): the text fallback denies "all rights reserved" as PROPRIETARY.
        ("all rights reserved", LicenseType.PROPRIETARY),
    ],
)
def test_cc_type_from_text(text: str, expected) -> None:
    assert _cc_type_from_text(text) is expected


def test_is_publishable_allow_deny() -> None:
    def _info(lt: LicenseType) -> LicenseInfo:
        return LicenseInfo(license_type=lt, source="test")

    for lt in (LicenseType.CC0, LicenseType.PUBLIC_DOMAIN, LicenseType.CC_BY):
        assert is_publishable(_info(lt)), lt
    for lt in (
        LicenseType.CC_BY_SA,
        LicenseType.CC_BY_NC,
        LicenseType.CC_BY_ND,
        LicenseType.CC_BY_NC_SA,
        LicenseType.CC_BY_NC_ND,
        LicenseType.YOUTUBE_TOS,
        LicenseType.PROPRIETARY,
        LicenseType.UNKNOWN,
    ):
        assert not is_publishable(_info(lt)), lt


def test_license_enforce_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REMIXMATE_LICENSE_ENFORCE", raising=False)
    assert license_enforce_enabled() is False
    for truthy in ("1", "true", "TRUE", "yes"):
        monkeypatch.setenv("REMIXMATE_LICENSE_ENFORCE", truthy)
        assert license_enforce_enabled() is True
    for falsy in ("0", "false", "", "no"):
        monkeypatch.setenv("REMIXMATE_LICENSE_ENFORCE", falsy)
        assert license_enforce_enabled() is False
