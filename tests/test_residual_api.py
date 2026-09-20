import io
import json
import unittest
import urllib.error
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from databricks_integration.scripts.bronze_to_silver_ubuntu import (
    build_residual_labels_from_counts,
)
from pipeline.audit import append_run_audit
from pipeline.residual_stage import (
    DEFAULT_RESIDUAL_CLASSIFIER_MODEL,
    ResidualClassifierError,
    _classify_residual_words_api,
    apply_api_labels,
    classify_residuals,
    residual_api_provenance,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self._payload


def _response(classifications, *, response_id="resp_fixture"):
    return {
        "id": response_id,
        "status": "completed",
        "output": [{
            "type": "message",
            "content": [{
                "type": "output_text",
                "text": json.dumps({"classifications": classifications}),
            }],
        }],
    }


class ResidualResponsesAPITests(unittest.TestCase):
    def test_request_uses_responses_strict_schema_and_no_storage(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(_response([
                {"word": "aptfoo", "label": "JARGON"},
                {"word": "basura", "label": "SPANISH"},
            ]))

        provenance = {}
        with patch("pipeline.residual_stage.urllib.request.urlopen", fake_urlopen):
            labels = _classify_residual_words_api(
                ["aptfoo", "basura"],
                Counter({"aptfoo": 3, "basura": 1}),
                api_key="sk-test-secret",
                provenance=provenance,
            )

        request = captured["request"]
        body = json.loads(request.data.decode("utf-8"))
        self.assertTrue(request.full_url.endswith("/responses"))
        self.assertEqual(body["model"], DEFAULT_RESIDUAL_CLASSIFIER_MODEL)
        self.assertFalse(body["store"])
        self.assertEqual(body["text"]["format"]["type"], "json_schema")
        self.assertTrue(body["text"]["format"]["strict"])
        self.assertFalse(
            body["text"]["format"]["schema"]["additionalProperties"]
        )
        self.assertNotIn("messages", body)
        self.assertNotIn("seed", body)
        self.assertEqual(
            json.loads(body["input"]),
            [{"word": "aptfoo", "count": 3}, {"word": "basura", "count": 1}],
        )
        self.assertEqual(labels, {"aptfoo": "JARGON", "basura": "SPANISH"})
        self.assertEqual(provenance["api"], "responses")
        self.assertEqual(provenance["response_ids"], ["resp_fixture"])
        self.assertNotIn("sk-test-secret", json.dumps(provenance))

    def test_transient_failure_retries_then_succeeds(self):
        effects = [
            urllib.error.URLError("temporary"),
            _FakeResponse(_response([{"word": "aptfoo", "label": "JARGON"}])),
        ]
        with patch(
            "pipeline.residual_stage.urllib.request.urlopen",
            side_effect=effects,
        ) as urlopen, patch("pipeline.residual_stage.time.sleep") as sleep:
            labels = _classify_residual_words_api(
                ["aptfoo"],
                Counter({"aptfoo": 1}),
                api_key="sk-test-secret",
                retry_backoff=0.01,
            )

        self.assertEqual(labels, {"aptfoo": "JARGON"})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_permanent_http_failure_is_not_retried_or_leaked(self):
        error = urllib.error.HTTPError(
            "https://api.openai.com/v1/responses",
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":"invalid key"}'),
        )
        with patch(
            "pipeline.residual_stage.urllib.request.urlopen",
            side_effect=error,
        ) as urlopen:
            with self.assertRaises(ResidualClassifierError) as raised:
                _classify_residual_words_api(
                    ["aptfoo"],
                    Counter({"aptfoo": 1}),
                    api_key="sk-test-secret",
                )

        self.assertEqual(urlopen.call_count, 1)
        self.assertIn("HTTP 401", str(raised.exception))
        self.assertNotIn("sk-test-secret", str(raised.exception))

    def test_partial_or_duplicate_output_fails_closed(self):
        incomplete = _FakeResponse(_response([
            {"word": "aptfoo", "label": "JARGON"},
        ]))
        with patch(
            "pipeline.residual_stage.urllib.request.urlopen",
            return_value=incomplete,
        ):
            with self.assertRaisesRegex(ResidualClassifierError, "omitted"):
                _classify_residual_words_api(
                    ["aptfoo", "basura"],
                    Counter({"aptfoo": 1, "basura": 1}),
                    api_key="sk-test-secret",
                )

    def test_api_labels_are_marked_as_review_candidates(self):
        frame = pd.DataFrame({"text_cleaned": ["aptfoo basura"]})
        response = _FakeResponse(_response([
            {"word": "aptfoo", "label": "JARGON"},
            {"word": "basura", "label": "SPANISH"},
        ]))
        with patch(
            "pipeline.residual_stage.urllib.request.urlopen",
            return_value=response,
        ):
            result, labels, sources = classify_residuals(
                frame,
                Counter({"aptfoo": 1, "basura": 1}),
                policy="api",
                api_key="sk-test-secret",
            )

        self.assertEqual(labels["aptfoo"], "JARGON")
        self.assertEqual(set(sources.values()), {"api_candidate"})
        self.assertEqual(result.loc[0, "text_cleaned"], "aptfoo SPANISH")
        self.assertEqual(
            result.attrs["residual_api_provenance"]["classification_status"],
            "candidate_for_human_review",
        )

    def test_digit_only_tokens_are_never_sent_or_replaced(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(_response([
                {"word": "aptfoo", "label": "JARGON"},
            ]))

        frame = pd.DataFrame({"text_cleaned": ["version 64 aptfoo"]})
        with patch("pipeline.residual_stage.urllib.request.urlopen", fake_urlopen):
            result, labels, sources = classify_residuals(
                frame,
                Counter({"64": 1, "aptfoo": 1}),
                policy="api",
                api_key="sk-test-secret",
            )

        self.assertEqual(
            json.loads(captured["body"]["input"]),
            [{"word": "aptfoo", "count": 1}],
        )
        self.assertEqual(labels["64"], "UNCERTAIN")
        self.assertEqual(sources["64"], "deterministic_numeric")
        self.assertEqual(result.loc[0, "text_cleaned"], "version 64 aptfoo")
        provenance = result.attrs["residual_api_provenance"]
        self.assertEqual(provenance["vocabulary_size"], 2)
        self.assertEqual(provenance["submitted_vocabulary_size"], 1)
        self.assertEqual(provenance["deterministic_numeric_passthroughs"], 1)

    def test_apply_api_labels_defensively_preserves_digits(self):
        frame = pd.DataFrame({"text_cleaned": ["2 or 3 sentences"]})

        result = apply_api_labels(frame, {"2": "NONWORD", "3": "SPANISH"})

        self.assertEqual(result.loc[0, "text_cleaned"], "2 or 3 sentences")

    def test_all_numeric_vocabulary_makes_no_api_request(self):
        provenance = {}
        with patch("pipeline.residual_stage.urllib.request.urlopen") as urlopen:
            labels = _classify_residual_words_api(
                ["2", "3", "64"],
                Counter({"2": 1, "3": 1, "64": 1}),
                api_key="sk-test-secret",
                provenance=provenance,
            )

        urlopen.assert_not_called()
        self.assertEqual(labels, {
            "2": "UNCERTAIN",
            "3": "UNCERTAIN",
            "64": "UNCERTAIN",
        })
        self.assertEqual(provenance["submitted_vocabulary_size"], 0)
        self.assertEqual(provenance["deterministic_numeric_passthroughs"], 3)

    def test_databricks_driver_path_uses_same_candidate_contract(self):
        response = _FakeResponse(_response([
            {"word": "aptfoo", "label": "JARGON"},
        ]))
        provenance = {}
        with patch(
            "pipeline.residual_stage.urllib.request.urlopen",
            return_value=response,
        ):
            labels, sources = build_residual_labels_from_counts(
                {"aptfoo": 2},
                policy="api",
                api_key="sk-test-secret",
                api_provenance=provenance,
            )

        self.assertEqual(labels, {"aptfoo": "JARGON"})
        self.assertEqual(sources, {"aptfoo": "api_candidate"})
        self.assertEqual(provenance["api"], "responses")
        self.assertNotIn("sk-test-secret", json.dumps(provenance))

    def test_local_audit_retains_provenance_without_key(self):
        frame = pd.DataFrame({"text_cleaned": ["aptfoo"]})
        frame.attrs["residual_api_provenance"] = {
            "model": DEFAULT_RESIDUAL_CLASSIFIER_MODEL,
            "request_sha256": ["abc123"],
        }
        with TemporaryDirectory() as directory:
            output = Path(directory) / "silver.csv"
            output.write_text("text_cleaned\naptfoo\n", encoding="utf-8")
            record = append_run_audit(
                Path(directory) / "audit.jsonl",
                df=frame,
                stage="silver_message_features",
                output_path=output,
                validation={"passed": True},
                config={"residual_policy": "api"},
            )

        self.assertEqual(
            record["residual_api_provenance"]["model"],
            DEFAULT_RESIDUAL_CLASSIFIER_MODEL,
        )

    def test_provenance_is_allowlisted_and_contains_no_base_url(self):
        provenance = residual_api_provenance(
            base_url="https://example.invalid/tenant-secret",
            api_key="sk-test-secret",
        )

        serialized = json.dumps(provenance)
        self.assertNotIn("tenant-secret", serialized)
        self.assertNotIn("sk-test-secret", serialized)
        self.assertEqual(provenance["model"], DEFAULT_RESIDUAL_CLASSIFIER_MODEL)


if __name__ == "__main__":
    unittest.main()
