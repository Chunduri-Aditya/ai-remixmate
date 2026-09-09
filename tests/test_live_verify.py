"""tests/test_live_verify.py — mocked HTTP checks for the live verify harness."""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from unittest import mock

from scripts.system import live_verify as lv

TOKEN_A = "token-user-a-aaaaaaaa"
TOKEN_B = "token-user-b-bbbbbbbb"


class LiveVerifyTests(unittest.TestCase):
    def test_health_ready_pass(self):
        with mock.patch.object(lv, "_make_request", return_value=(200, {"status": "ready"})):
            check = lv.check_health_ready("http://127.0.0.1:8000")
        self.assertTrue(check.passed)

    def test_cross_tenant_leak_fails(self):
        crate_id = "crate_A"

        def fake(url, method="GET", token=None, body=None):
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": crate_id, "name": "t"}
            if method == "GET" and url.endswith("/crates"):
                if token == TOKEN_A:
                    return 200, {"crates": [{"crate_id": crate_id}]}
                return 200, {"crates": [{"crate_id": crate_id}]}  # leak
            if method == "DELETE":
                return 200, {"deleted": True}
            return 404, {}

        with mock.patch.object(lv, "_make_request", side_effect=fake):
            check = lv.check_cross_tenant_isolation(
                "http://127.0.0.1:8000", TOKEN_A, TOKEN_B
            )
        self.assertFalse(check.passed)
        self.assertIn("LEAK", check.detail)

    def test_cross_tenant_isolated_pass(self):
        crate_id = "crate_A"

        def fake(url, method="GET", token=None, body=None):
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": crate_id, "name": "t"}
            if method == "GET" and url.endswith("/crates"):
                if token == TOKEN_A:
                    return 200, {"crates": [{"crate_id": crate_id}]}
                return 200, {"crates": []}
            if "/songs" in url:
                return 200, {"songs": []}
            if method == "DELETE":
                return 200, {"deleted": True}
            return 404, {}

        with mock.patch.object(lv, "_make_request", side_effect=fake):
            check = lv.check_cross_tenant_isolation(
                "http://127.0.0.1:8000", TOKEN_A, TOKEN_B
            )
        self.assertTrue(check.passed)

    def test_vacuous_shape_does_not_pass(self):
        """B response missing crates key must fail (not pass vacuously)."""
        crate_id = "crate_A"

        def fake(url, method="GET", token=None, body=None):
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": crate_id}
            if method == "GET" and url.endswith("/crates"):
                if token == TOKEN_A:
                    return 200, {"crates": [{"crate_id": crate_id}]}
                return 200, {"items": [{"crate_id": crate_id}]}  # wrong shape + leak
            if method == "DELETE":
                return 200, {}
            return 404, {}

        with mock.patch.object(lv, "_make_request", side_effect=fake):
            check = lv.check_cross_tenant_isolation(
                "http://127.0.0.1:8000", TOKEN_A, TOKEN_B
            )
        self.assertFalse(check.passed)
        self.assertIn("shape", check.detail.lower())

    def test_positive_control_required(self):
        crate_id = "crate_A"

        def fake(url, method="GET", token=None, body=None):
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": crate_id}
            if method == "GET" and url.endswith("/crates"):
                return 200, {"crates": []}  # A can't see own crate
            if method == "DELETE":
                return 200, {}
            return 404, {}

        with mock.patch.object(lv, "_make_request", side_effect=fake):
            check = lv.check_cross_tenant_isolation(
                "http://127.0.0.1:8000", TOKEN_A, TOKEN_B
            )
        self.assertFalse(check.passed)
        self.assertIn("Positive control", check.detail)

    def test_validate_base_url_requires_https_remote(self):
        self.assertIsNotNone(lv._validate_base_url("http://api.example.com"))
        self.assertIsNone(lv._validate_base_url("https://api.example.com"))
        self.assertIsNone(lv._validate_base_url("http://127.0.0.1:8000"))

    def test_no_raw_jwt_in_stdout(self):
        jwtish = "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"

        def fake(url, method="GET", token=None, body=None):
            if "health" in url:
                return 200, {"status": "ready"}
            if "jobs" in url:
                return 200, []
            if method == "POST" and url.endswith("/crates"):
                return 201, {"crate_id": "c1"}
            if method == "GET" and url.endswith("/crates"):
                if token == jwtish:
                    return 200, {"crates": [{"crate_id": "c1"}]}
                return 200, {"crates": []}
            if "/songs" in url:
                return 200, {"songs": []}
            if method == "DELETE":
                return 200, {}
            return 404, {}

        buf = io.StringIO()
        with mock.patch.object(lv, "_make_request", side_effect=fake):
            with mock.patch(
                "sys.argv",
                [
                    "live_verify",
                    "--base-url",
                    "http://127.0.0.1:8000",
                    "--token-a",
                    jwtish,
                    "--token-b",
                    jwtish + "b",
                    "--skip-render",
                ],
            ):
                with redirect_stdout(buf):
                    code = lv.main()
        out = buf.getvalue()
        self.assertEqual(code, 0)
        self.assertNotIn(jwtish, out)
        self.assertNotIn(jwtish + "b", out)

    def test_json_mode(self):
        def fake(url, method="GET", token=None, body=None):
            if "health" in url:
                return 200, {"status": "ready"}
            if "jobs" in url:
                return 200, []
            if method == "POST":
                return 201, {"crate_id": "c1"}
            if method == "GET" and url.endswith("/crates"):
                if token == TOKEN_A:
                    return 200, {"crates": [{"crate_id": "c1"}]}
                return 200, {"crates": []}
            if "/songs" in url:
                return 200, {"songs": []}
            if method == "DELETE":
                return 200, {}
            return 404, {}

        buf = io.StringIO()
        with mock.patch.object(lv, "_make_request", side_effect=fake):
            with mock.patch(
                "sys.argv",
                [
                    "live_verify",
                    "--base-url",
                    "http://127.0.0.1:8000",
                    "--token-a",
                    TOKEN_A,
                    "--token-b",
                    TOKEN_B,
                    "--skip-render",
                    "--json",
                ],
            ):
                with redirect_stdout(buf):
                    code = lv.main()
        data = json.loads(buf.getvalue())
        self.assertEqual(code, 0)
        self.assertTrue(data["all_passed"])


if __name__ == "__main__":
    unittest.main()
