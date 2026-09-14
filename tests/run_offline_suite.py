"""Run every test with real HTTP, DNS, and socket connections prohibited."""

import json
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    attempts = []
    def blocked(*args, **kwargs):
        # Do not retain arguments: HTTP calls could contain credentials.
        attempts.append(True)
        raise AssertionError("Real network/provider API calls forbidden during tests")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex",
                       "socket.create_connection", "socket.getaddrinfo", "urllib.request.urlopen"):
            stack.enter_context(patch(target, side_effect=blocked))
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = {"tests_run": result.testsRun, "failures": len(result.failures),
               "errors": len(result.errors), "skipped": len(result.skipped),
               "network_access_blocked": True, "unexpected_network_attempts": len(attempts),
               "passed": result.wasSuccessful() and not attempts,
               "command": "python3 tests/run_offline_suite.py"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
