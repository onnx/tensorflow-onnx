# SPDX-License-Identifier: Apache-2.0

"""Tests for protobuf runtime_version compatibility shim."""

import sys
import types

import pytest


def test_runtime_version_shim_provides_validator():
    """When google.protobuf.runtime_version is missing, the shim injects one."""
    # Simulate an older protobuf runtime by removing runtime_version from sys.modules.
    real_module = sys.modules.pop("google.protobuf.runtime_version", None)
    protobuf_pkg = sys.modules.get("google.protobuf")
    original_has_runtime_version = hasattr(protobuf_pkg, "runtime_version") if protobuf_pkg else False
    if protobuf_pkg and original_has_runtime_version:
        delattr(protobuf_pkg, "runtime_version")

    try:
        # Clear any cached tf2onnx.protobuf_compat module so the shim path runs again.
        sys.modules.pop("tf2onnx.protobuf_compat", None)
        import tf2onnx.protobuf_compat  # noqa: F401  # side-effect under test

        shim = sys.modules["google.protobuf.runtime_version"]
        assert isinstance(shim, types.ModuleType)
        assert hasattr(shim, "ValidateProtobufRuntimeVersion")
        # The stubbed validator must be callable and not raise.
        shim.ValidateProtobufRuntimeVersion(1, 2, 3, "", "dummy")
    finally:
        # Restore original state.
        if real_module is not None:
            sys.modules["google.protobuf.runtime_version"] = real_module
        elif "google.protobuf.runtime_version" in sys.modules:
            del sys.modules["google.protobuf.runtime_version"]
        if protobuf_pkg and original_has_runtime_version:
            protobuf_pkg.runtime_version = real_module


def test_runtime_version_passthrough_when_present():
    """If google.protobuf.runtime_version already exists, the shim leaves it alone."""
    try:
        from google.protobuf import runtime_version as real_runtime_version
    except ImportError:
        pytest.skip("protobuf runtime_version is not available in this environment")

    sys.modules.pop("tf2onnx.protobuf_compat", None)
    import tf2onnx.protobuf_compat  # noqa: F401  # side-effect under test

    assert sys.modules["google.protobuf.runtime_version"] is real_runtime_version
