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

_SOURCES = ["ccmixter", "fma", "internetarchive", "musopen"]
_BY_URL = "https://creativecommons.org/licenses/by/4.0/"
_BY_NC_URL = "https://creativecommons.org/licenses/by-nc/4.0/"


@pytest.mark.parametrize("source", _SOURCES)
def test_classify_matrix_by_url(source: str) -> None:
    # allow: CC-BY resolves and is publishable
    allow = classify_license(source, license_url=_BY_URL)
    assert allow.license_type is LicenseType.CC_BY
    assert is_publishable(allow)
    # deny: CC-BY-NC resolves and is NOT publishable
    deny = classify_license(source, license_url=_BY_NC_URL)
    assert deny.license_type is LicenseType.CC_BY_NC
    assert not is_publishable(deny)


@pytest.mark.parametrize("source", _SOURCES)
def test_classify_matrix_by_text(source: str) -> None:
    allow = classify_license(source, license_str="Attribution 3.0")
    assert allow.license_type is LicenseType.CC_BY
    deny = classify_license(source, license_str="Attribution-NonCommercial")
    assert deny.license_type is LicenseType.CC_BY_NC


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
        ("all rights reserved", None),
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
