#!/usr/bin/env python3
"""Unit tests for the attest-helper launch-token gate."""
from __future__ import annotations

import hashlib
import unittest

from server import EMPTY_TOKEN_HASH, check_auth

TOKEN = "6f1a" * 16
TOKEN_HASH = hashlib.sha256(TOKEN.encode()).hexdigest()


class CheckAuthTest(unittest.TestCase):
    def test_correct_token_accepted(self):
        self.assertIsNone(check_auth(TOKEN_HASH, f"Bearer {TOKEN}"))

    def test_wrong_token_rejected(self):
        denied = check_auth(TOKEN_HASH, "Bearer not-the-token")
        self.assertIsNotNone(denied)
        self.assertEqual(denied[0], 401)

    def test_missing_header_rejected(self):
        self.assertEqual(check_auth(TOKEN_HASH, None)[0], 401)
        self.assertEqual(check_auth(TOKEN_HASH, TOKEN)[0], 401)

    def test_empty_token_hash_refuses_to_serve(self):
        for header in (None, "Bearer ", f"Bearer {TOKEN}"):
            denied = check_auth(EMPTY_TOKEN_HASH, header)
            self.assertIsNotNone(denied)
            self.assertEqual(denied[0], 503)

    def test_malformed_configured_hash_refuses_to_serve(self):
        for configured in (None, "", "  ", "zz" * 32, TOKEN_HASH.upper(), TOKEN_HASH[:63]):
            denied = check_auth(configured, f"Bearer {TOKEN}")
            self.assertIsNotNone(denied)
            self.assertEqual(denied[0], 503)

    def test_denied_body_never_echoes_the_token(self):
        for configured in (TOKEN_HASH, EMPTY_TOKEN_HASH):
            denied = check_auth(configured, f"Bearer {TOKEN}") or (200, "")
            self.assertNotIn(TOKEN, denied[1])


if __name__ == "__main__":
    unittest.main()
