"""
tests/test_local_jwks.py — spec B6.1: local JWKS and token minter.

`scripts/system/local_jwks.py` is the tool hosted-mode simulation (B6.2)
uses in place of a Supabase project:

  keygen --out DIR                       -> DIR/private.pem, DIR/jwks.json (RS256, kid)
  mint --key DIR/private.pem --sub UUID --email E [--verified] [--minutes 120]
                                         -> prints a token (aud/role authenticated, exp)

The tool is exercised through its CLI (``python -m scripts.system.local_jwks``)
so the tests fail on an assertion when the tool is missing, not on an import.
Verification goes through the real ``auth._verify_token`` with
``auth._get_jwks_client`` patched to a ``PyJWKClient`` whose fetch reads
``jwks.json`` from disk; the client's own ``kid`` lookup stays in play.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from pathlib import Path

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from scripts.api import auth as auth_mod

_ROOT = Path(__file__).resolve().parents[1]
_MODULE = "scripts.system.local_jwks"
_AUD = "authenticated"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", _MODULE, *args],
        cwd=str(_ROOT), capture_output=True, text=True, timeout=60,
    )


def _keygen(out: Path) -> Path:
    proc = _run("keygen", "--out", str(out))
    assert proc.returncode == 0, (
        f"keygen exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    return out


def _mint(key: Path, *, sub: str, email: str, verified: bool = False,
          minutes: int | None = None) -> str:
    args = ["mint", "--key", str(key), "--sub", sub, "--email", email]
    if verified:
        args.append("--verified")
    if minutes is not None:
        args.append(f"--minutes={minutes}")   # '=' form so a negative value is unambiguous
    proc = _run(*args)
    assert proc.returncode == 0, (
        f"mint exited {proc.returncode}\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
    token = proc.stdout.strip()
    assert token.count(".") == 2, f"mint did not print a compact JWS token: {token!r}"
    return token


class _FileJWKSClient(jwt.PyJWKClient):
    """The real PyJWKClient (kid matching, key parsing) with its network fetch
    swapped for a read of jwks.json on disk: what a file:// served copy gives."""

    def __init__(self, path: Path):
        # PyJWT accepts only http(s) URIs; this one is never fetched because
        # fetch_data below reads the file instead (no network, loopback host).
        super().__init__("http://127.0.0.1:8999/jwks.json", cache_keys=False, cache_jwk_set=False)
        self._path = path

    def fetch_data(self):  # type: ignore[override]
        return json.loads(self._path.read_text())


@pytest.fixture(autouse=True)
def _reset_jwks_client():
    auth_mod._jwks_client = None
    auth_mod._jwks_client_url = None
    yield
    auth_mod._jwks_client = None
    auth_mod._jwks_client_url = None


@pytest.fixture
def jwks_dir(tmp_path: Path) -> Path:
    return _keygen(tmp_path / "jwks")


@pytest.fixture
def auth_reads_jwks_file(jwks_dir: Path, monkeypatch):
    monkeypatch.setenv("REMIXMATE_AUTH_ENABLED", "1")
    monkeypatch.setenv("SUPABASE_JWKS_URL", f"file://{jwks_dir / 'jwks.json'}")
    monkeypatch.delenv("SUPABASE_JWT_AUD", raising=False)
    monkeypatch.delenv("SUPABASE_JWT_ISS", raising=False)
    client = _FileJWKSClient(jwks_dir / "jwks.json")
    monkeypatch.setattr(auth_mod, "_get_jwks_client", lambda: client)
    return jwks_dir


# ---------------------------------------------------------------------------
# keygen
# ---------------------------------------------------------------------------

def test_keygen_writes_private_pem_and_jwks_with_a_kid(tmp_path: Path):
    out = _keygen(tmp_path / "jwks")
    pem = out / "private.pem"
    jwks_path = out / "jwks.json"
    assert pem.is_file(), "keygen did not write private.pem"
    assert jwks_path.is_file(), "keygen did not write jwks.json"

    priv = serialization.load_pem_private_key(pem.read_bytes(), password=None)
    assert isinstance(priv, rsa.RSAPrivateKey), "private.pem is not an RSA private key"
    assert priv.key_size >= 2048

    jwks = json.loads(jwks_path.read_text())
    keys = jwks.get("keys")
    assert isinstance(keys, list) and len(keys) == 1, f"jwks.json needs one key: {jwks!r}"
    jwk = keys[0]
    assert jwk.get("kty") == "RSA"
    assert jwk.get("alg") == "RS256"
    assert jwk.get("use") == "sig"
    assert isinstance(jwk.get("kid"), str) and jwk["kid"], "jwk has no kid"
    assert jwk.get("n") and jwk.get("e"), "jwk lacks the public numbers n / e"

    # The published public key is the private key's public half.
    published = jwt.PyJWK.from_dict(jwk).key
    assert published.public_numbers() == priv.public_key().public_numbers()


def test_keygen_creates_the_out_dir_and_refuses_to_leak_private_key_into_jwks(tmp_path: Path):
    out = _keygen(tmp_path / "nested" / "does" / "not" / "exist")
    jwks_text = (out / "jwks.json").read_text()
    for private_member in ("\"d\"", "\"p\"", "\"q\"", "\"dp\"", "\"dq\"", "\"qi\""):
        assert private_member not in jwks_text, f"jwks.json carries private member {private_member}"


# ---------------------------------------------------------------------------
# mint -> auth._verify_token
# ---------------------------------------------------------------------------

def test_minted_token_verifies_through_auth_verify_token(auth_reads_jwks_file: Path):
    sub = str(uuid.uuid4())
    before = int(time.time())
    token = _mint(auth_reads_jwks_file / "private.pem", sub=sub,
                  email="a@example.test", minutes=120)

    header = jwt.get_unverified_header(token)
    kid = json.loads((auth_reads_jwks_file / "jwks.json").read_text())["keys"][0]["kid"]
    assert header.get("alg") == "RS256"
    assert header.get("kid") == kid, "token kid must match the published jwk kid"

    assert auth_mod._verify_token(token) == sub

    claims = auth_mod._verify_token_claims(token)
    assert claims["aud"] == _AUD
    assert claims["role"] == "authenticated"
    assert claims["email"] == "a@example.test"
    # exp = now + 120 minutes (a few seconds of slack for the subprocess)
    assert before + 120 * 60 - 5 <= claims["exp"] <= before + 120 * 60 + 60, claims["exp"]


def test_mint_default_lifetime_is_120_minutes(auth_reads_jwks_file: Path):
    before = int(time.time())
    token = _mint(auth_reads_jwks_file / "private.pem", sub=str(uuid.uuid4()),
                  email="d@example.test")
    claims = auth_mod._verify_token_claims(token)
    assert before + 120 * 60 - 5 <= claims["exp"] <= before + 120 * 60 + 60, claims["exp"]


def test_expired_token_is_rejected(auth_reads_jwks_file: Path):
    token = _mint(auth_reads_jwks_file / "private.pem", sub=str(uuid.uuid4()),
                  email="x@example.test", minutes=-5)
    # Prove the token is otherwise well formed: exp is in the past, signature is ours.
    unverified = jwt.decode(token, options={"verify_signature": False, "verify_exp": False},
                            audience=_AUD)
    assert unverified["exp"] < int(time.time())

    with pytest.raises(auth_mod.HTTPException) as excinfo:
        auth_mod._verify_token(token)
    assert excinfo.value.status_code == 401
    assert "expired" in str(excinfo.value.detail).lower()


def test_token_signed_by_another_key_is_rejected(auth_reads_jwks_file: Path, tmp_path: Path):
    # A second keygen: a different private key and a different kid. auth must
    # not accept it against the first jwks.json.
    other = _keygen(tmp_path / "other")
    token = _mint(other / "private.pem", sub=str(uuid.uuid4()), email="o@example.test")
    with pytest.raises(auth_mod.HTTPException) as excinfo:
        auth_mod._verify_token(token)
    assert excinfo.value.status_code == 401


# ---------------------------------------------------------------------------
# --verified
# ---------------------------------------------------------------------------

def test_verified_flag_sets_email_verified(auth_reads_jwks_file: Path):
    key = auth_reads_jwks_file / "private.pem"
    verified = auth_mod._verify_token_claims(
        _mint(key, sub=str(uuid.uuid4()), email="v@example.test", verified=True))
    assert verified.get("email_verified") is True
    assert auth_mod._email_verified(verified) is True

    plain = auth_mod._verify_token_claims(
        _mint(key, sub=str(uuid.uuid4()), email="p@example.test"))
    assert plain.get("email_verified") is not True, "email_verified set without --verified"
    assert auth_mod._email_verified(plain) is False


def test_mint_requires_sub_and_email(auth_reads_jwks_file: Path):
    key = auth_reads_jwks_file / "private.pem"
    no_sub = _run("mint", "--key", str(key), "--email", "e@example.test")
    assert no_sub.returncode != 0
    assert "--sub" in no_sub.stderr, f"expected a missing --sub error, got: {no_sub.stderr!r}"
    no_email = _run("mint", "--key", str(key), "--sub", str(uuid.uuid4()))
    assert no_email.returncode != 0
    assert "--email" in no_email.stderr, f"expected a missing --email error, got: {no_email.stderr!r}"
