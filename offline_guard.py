"""Fail before network access in reproducible offline workflows."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch


@contextmanager
def offline_only():
    audit = {"provider_calls": 0, "blocked_network_attempts": 0}

    def blocked(*args, **kwargs):
        audit["blocked_network_attempts"] += 1
        raise RuntimeError("Network/provider access is forbidden in this offline workflow")

    with ExitStack() as stack:
        for target in (
            "socket.socket.connect",
            "socket.socket.connect_ex",
            "socket.create_connection",
            "socket.getaddrinfo",
            "urllib.request.urlopen",
            "providers.call_model",
            "providers.call_model_with_retries",
            "generate_dataset.call_model_with_retries",
        ):
            stack.enter_context(patch(target, side_effect=blocked))
        yield audit
