# SPDX-License-Identifier: Apache-2.0


"""Coverage guard for the TFLite frontend.

Compares every TFLite builtin operator (tf2onnx's vendored ``BuiltinOperator``
enum) against the handlers registered with ``@tfl_op``. It does NOT convert or
run any model - it only introspects tf2onnx's own source, so it is
version-independent and never flaky. It fails only when a change alters the set
of handled TFLite ops: deleting a handler, adding one without updating the list
below, or a schema bump that introduces a new builtin. Such failures always
point at the change that caused them, never at unrelated PRs.

Runs only in its dedicated (non-blocking) CI job via ``TF2ONNX_TFLITE_COVERAGE``;
skipped in the normal test matrix so it can never block unrelated PRs.
"""

import os
import unittest

# TFLite builtins that currently have NO ``@tfl_op`` handler. Emitting one of
# these yields an invalid ``TFL_<NAME>`` passthrough node that no ONNX runtime
# can load (e.g. TFL_GELU). Keep this list honest:
#   * implement a handler for one of these -> REMOVE it here
#   * a TF/tflite upgrade adds a new builtin -> ADD it here (or add a handler)
KNOWN_UNSUPPORTED = {
    "ASSIGN_VARIABLE", "BIDIRECTIONAL_SEQUENCE_LSTM", "BIDIRECTIONAL_SEQUENCE_RNN", "BITCAST",
    "BROADCAST_ARGS", "BUCKETIZE", "CALL", "CALL_ONCE", "CONCAT_EMBEDDINGS", "DENSIFY",
    "DYNAMIC_UPDATE_SLICE", "EMBEDDING_LOOKUP", "EMBEDDING_LOOKUP_SPARSE", "FAKE_QUANT", "GELU",
    "HASHTABLE", "HASHTABLE_FIND", "HASHTABLE_IMPORT", "HASHTABLE_LOOKUP", "HASHTABLE_SIZE",
    "IMAG", "L2_POOL_2D", "LSH_PROJECTION", "LSTM", "READ_VARIABLE", "REAL", "RELU_0_TO_1",
    "RELU_N1_TO_1", "RNN", "SKIP_GRAM", "STABLEHLO_ADD", "STABLEHLO_DIVIDE",
    "STABLEHLO_LOGISTIC", "STABLEHLO_MAXIMUM", "STABLEHLO_MULTIPLY", "SVDF",
    "UNIDIRECTIONAL_SEQUENCE_LSTM", "UNIDIRECTIONAL_SEQUENCE_RNN", "VAR_HANDLE",
}

# Enum entries that are not real operators and never need a handler.
_SENTINELS = {"CUSTOM", "DELEGATE", "PLACEHOLDER_FOR_GREATER_OP_CODES"}


def _builtins_and_handled():
    """Return (all builtin op names, names that have a TFL_ handler)."""
    import tf2onnx.convert  # noqa: F401  pylint: disable=unused-import  (registers handlers)
    from tf2onnx.handler import tf_op
    from tf2onnx.tflite import BuiltinOperator as bo

    builtins = {
        k for k, v in vars(bo.BuiltinOperator).items()
        if not k.startswith("_") and isinstance(v, int) and k not in _SENTINELS
    }
    registered = set()
    for opset_list in tf_op.get_opsets().values():
        for ops in opset_list:
            registered.update(ops)
    handled = {n[len("TFL_"):] for n in registered if n.startswith("TFL_")}
    return builtins, handled


# unittest.skipUnless (not pytest.mark.skipif) so the guard is honored under both
# runners -- `pytest` and a direct `python -m unittest` / `unittest.main()` via the
# __main__ block below.
@unittest.skipUnless(
    os.environ.get("TF2ONNX_TFLITE_COVERAGE"),
    "TFLite coverage guard runs only in its dedicated non-blocking CI job",
)
class TFLiteOpCoverageTests(unittest.TestCase):

    def test_no_new_unsupported_builtins(self):
        # A builtin with no handler that we did not already know about: either a
        # regressed handler or a newly added builtin.
        builtins, handled = _builtins_and_handled()
        new_gaps = (builtins - handled) - KNOWN_UNSUPPORTED
        self.assertEqual(
            set(), new_gaps,
            "Unhandled TFLite builtin(s) not in KNOWN_UNSUPPORTED: %s. "
            "Implement an @tfl_op handler, or add them to KNOWN_UNSUPPORTED if "
            "leaving them unsupported is intentional." % sorted(new_gaps))

    def test_allowlist_still_accurate(self):
        builtins, handled = _builtins_and_handled()
        # Someone added a handler but forgot to shrink the list.
        now_supported = KNOWN_UNSUPPORTED & handled
        self.assertEqual(
            set(), now_supported,
            "TFLite builtin(s) now have a handler; remove them from "
            "KNOWN_UNSUPPORTED: %s" % sorted(now_supported))
        # Guard against typos / renamed enum entries in the list.
        stale = KNOWN_UNSUPPORTED - builtins
        self.assertEqual(
            set(), stale,
            "KNOWN_UNSUPPORTED has name(s) not in BuiltinOperator "
            "(typo or renamed op): %s" % sorted(stale))


if __name__ == "__main__":
    unittest.main()
