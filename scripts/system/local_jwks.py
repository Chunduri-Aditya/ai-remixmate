"""
scripts.system.local_jwks — Local JWKS and access token minter (spec B6.1).

Stands in for a Supabase project when simulating hosted mode:

  python -m scripts.system.local_jwks keygen --out DIR
      writes DIR/private.pem (RSA 2048, mode 0600) and DIR/jwks.json
      (one RS256 signing key, kid = RFC 7638 thumbprint)

  python -m scripts.system.local_jwks mint --key DIR/private.pem \\
      --sub UUID --email E [--verified] [--minutes 120]
      prints one RS256 token with aud and role "authenticated" and exp

The private key never leaves DIR/private.pem; jwks.json carries only n and e.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

_AUDIENCE = "authenticated"
_ROLE = "authenticated"
_DEFAULT_MINUTES = 120


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8 or 1, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _public_jwk(public_key: rsa.RSAPublicKey) -> dict:
    """Public JWK for ``public_key`` with an RFC 7638 thumbprint as its kid."""
    numbers = public_key.public_numbers()
    n = _b64url_uint(numbers.n)
    e = _b64url_uint(numbers.e)
    canonical = json.dumps({"e": e, "kty": "RSA", "n": n}, separators=(",", ":"), sort_keys=True)
    kid = base64.urlsafe_b64encode(
        hashlib.sha256(canonical.encode("ascii")).digest()
    ).rstrip(b"=").decode("ascii")
    return {"kty": "RSA", "alg": "RS256", "use": "sig", "kid": kid, "n": n, "e": e}


def keygen(out: Path) -> None:
    """Write ``out/private.pem`` and ``out/jwks.json``; create ``out`` if absent."""
    out.mkdir(parents=True, exist_ok=True)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pem_path = out / "private.pem"
    fd = os.open(pem_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(pem)
    os.chmod(pem_path, 0o600)

    jwks = {"keys": [_public_jwk(private_key.public_key())]}
    (out / "jwks.json").write_text(json.dumps(jwks, indent=2) + "\n")


def mint(key: Path, *, sub: str, email: str, verified: bool, minutes: int) -> str:
    """Return an RS256 token signed by ``key`` whose kid matches keygen's jwks.json."""
    private_key = serialization.load_pem_private_key(key.read_bytes(), password=None)
    if not isinstance(private_key, rsa.RSAPrivateKey):
        raise SystemExit(f"{key}: not an RSA private key")
    kid = _public_jwk(private_key.public_key())["kid"]
    now = int(time.time())
    claims: dict = {
        "sub": sub,
        "email": email,
        "aud": _AUDIENCE,
        "role": _ROLE,
        "iat": now,
        "exp": now + minutes * 60,
    }
    if verified:
        claims["email_verified"] = True
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": kid})


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``keygen`` or ``mint``."""
    parser = argparse.ArgumentParser(prog="python -m scripts.system.local_jwks",
                                     description=__doc__.strip().splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_keygen = sub.add_parser("keygen", help="write private.pem and jwks.json")
    p_keygen.add_argument("--out", required=True, type=Path, help="output directory")

    p_mint = sub.add_parser("mint", help="print a signed access token")
    p_mint.add_argument("--key", required=True, type=Path, help="path to private.pem")
    p_mint.add_argument("--sub", required=True, help="user id (UUID)")
    p_mint.add_argument("--email", required=True, help="email claim")
    p_mint.add_argument("--verified", action="store_true", help="set email_verified: true")
    p_mint.add_argument("--minutes", type=int, default=_DEFAULT_MINUTES,
                        help=f"lifetime in minutes (default {_DEFAULT_MINUTES}; < 0 is expired)")

    args = parser.parse_args(argv)
    if args.command == "keygen":
        keygen(args.out)
        return 0
    token = mint(args.key, sub=args.sub, email=args.email,
                 verified=args.verified, minutes=args.minutes)
    sys.stdout.write(token + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
