# SPDX-License-Identifier: Apache-2.0

"""Compatibility shim for protobuf runtime_version.

ONNX 1.16+ ships protobuf-generated files that import
``google.protobuf.runtime_version`` and call ``ValidateProtobufRuntimeVersion``.
That module was introduced in protobuf 5.27, so importing ONNX on older
protobuf releases raises ``ImportError`` at load time.

This module injects a no-op stub when the real ``runtime_version`` is missing,
allowing tf2onnx to be imported with older protobuf runtimes.
"""

import sys
import types

try:
    from google.protobuf import runtime_version  # type: ignore
except ImportError:
    # Protobuf runtime is older than 5.27 and does not expose runtime_version.
    # Provide a minimal stub so ONNX's generated modules can still be imported.
    _stub = types.ModuleType("google.protobuf.runtime_version")
    _stub.MAJOR = 0
    _stub.MINOR = 0
    _stub.PATCH = 0
    _stub.SUFFIX = ""
    _stub.OSS_VERSION = 0

    def ValidateProtobufRuntimeVersion(*args, **kwargs):  # pylint: disable=invalid-name
        """No-op runtime version check for older protobuf installations."""

    _stub.ValidateProtobufRuntimeVersion = ValidateProtobufRuntimeVersion
    sys.modules["google.protobuf.runtime_version"] = _stub
