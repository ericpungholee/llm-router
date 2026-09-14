"""Ensure the privileged diagnostic is validation-selected and has an OOD fallback."""

import unittest

import numpy as np

from experiments.router_domain_diagnostic import select_domain_models
from routing_ml.training import MODEL_IDS
from test_router_v1 import synthetic


class DomainDiagnosticTests(unittest.TestCase):
    def test_domain_winners_use_validation_and_global_fallback(self):
        part = synthetic("validation")
        part.y[:] = 0
        part.y["gpt-5"] = 1
        part.y.loc[part.prompts.dataset == "math", MODEL_IDS[0]] = 1
        choices, fallback = select_domain_models(part, np.arange(1, 9))
        self.assertEqual(choices["math"], MODEL_IDS[0])
        self.assertEqual(choices["knowledge"], "gpt-5")
        self.assertEqual(choices.get("unseen_code", fallback), "gpt-5")
        with self.assertRaisesRegex(ValueError, "validation only"):
            select_domain_models(synthetic("test"), np.arange(1, 9))


if __name__ == "__main__":
    unittest.main()
