# SPDX-License-Identifier: Apache-2.0

"""Generate the checked-in TensorFlow operator handler inventory."""

from __future__ import annotations

import argparse
import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

HANDLER_ROOTS = (
    Path("tf2onnx/custom_opsets"),
    Path("tf2onnx/onnx_opset"),
    Path("tf2onnx/tflite_handlers"),
)

# The default-domain ceiling is the highest opset exercised by required CI in
# unit_test_ci.yml. Keep the two in sync when required opset coverage changes.
MAX_DOCUMENTED_OPSET = {
    "": 21,
    "com.microsoft": 1,
    "ai.onnx.contrib": 1,
}

DOMAIN_EXPRESSIONS = {
    "constants.ONNX_DOMAIN": "",
    "constants.MICROSOFT_DOMAIN": "com.microsoft",
    "constants.CONTRIB_OPS_DOMAIN": "ai.onnx.contrib",
}

DOMAIN_ORDER = ("", "com.microsoft", "ai.onnx.contrib")
VERSION_METHOD = re.compile(r"version_(\d+)$")


@dataclass(frozen=True)
class ClassInfo:
    """Static information needed to resolve handler class versions."""

    module: str
    node: ast.ClassDef
    aliases: dict[str, str]


def _module_name(repo_root: Path, path: Path) -> str:
    relative = path.relative_to(repo_root).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _import_aliases(tree: ast.Module, module: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    package = module.split(".")[:-1]

    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                local_name = imported.asname or imported.name.split(".")[0]
                aliases[local_name] = imported.name if imported.asname else local_name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                keep = len(package) - node.level + 1
                prefix_parts = package[:keep]
                if node.module:
                    prefix_parts.extend(node.module.split("."))
                prefix = ".".join(prefix_parts)
            else:
                prefix = node.module or ""

            for imported in node.names:
                if imported.name == "*":
                    continue
                local_name = imported.asname or imported.name
                aliases[local_name] = ".".join(part for part in (prefix, imported.name) if part)

    return aliases


def _load_classes(repo_root: Path) -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}
    for source_root in HANDLER_ROOTS:
        for path in sorted((repo_root / source_root).rglob("*.py")):
            module = _module_name(repo_root, path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            aliases = _import_aliases(tree, module)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    classes[f"{module}.{node.name}"] = ClassInfo(module, node, aliases)
    return classes


def _qualified_name(expression: ast.expr, info: ClassInfo) -> str | None:
    if isinstance(expression, ast.Name):
        return info.aliases.get(expression.id, f"{info.module}.{expression.id}")

    if isinstance(expression, ast.Attribute):
        parts = []
        current: ast.expr = expression
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        root = info.aliases.get(current.id, f"{info.module}.{current.id}")
        return ".".join([root, *reversed(parts)])

    return None


def _raises_not_implemented(function: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for node in ast.walk(function):
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        exception = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(exception, ast.Name) and exception.id == "NotImplementedError":
            return True
    return False


def _class_versions(
    class_name: str,
    classes: dict[str, ClassInfo],
    cache: dict[str, dict[int, bool]],
    resolving: set[str],
) -> dict[int, bool]:
    if class_name in cache:
        return cache[class_name]
    if class_name in resolving:
        raise ValueError(f"cyclic handler inheritance involving {class_name}")

    info = classes[class_name]
    resolving.add(class_name)
    versions: dict[int, bool] = {}

    for base in info.node.bases:
        base_name = _qualified_name(base, info)
        if base_name in classes:
            versions.update(_class_versions(base_name, classes, cache, resolving))

    for child in info.node.body:
        if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        match = VERSION_METHOD.fullmatch(child.name)
        if match:
            versions[int(match.group(1))] = _raises_not_implemented(child)

    resolving.remove(class_name)
    cache[class_name] = versions
    return versions


def _decorator_name(decorator: ast.expr) -> str | None:
    if not isinstance(decorator, ast.Call):
        return None
    if isinstance(decorator.func, ast.Name):
        return decorator.func.id
    if isinstance(decorator.func, ast.Attribute):
        return decorator.func.attr
    return None


def _handler_names(decorator: ast.Call) -> list[str]:
    if not decorator.args:
        raise ValueError("tf_op/tfl_op decorator is missing its operator name")
    value = ast.literal_eval(decorator.args[0])
    names = [value] if isinstance(value, str) else list(value)
    if not names or not all(isinstance(name, str) for name in names):
        raise ValueError("tf_op/tfl_op operator names must be string literals")
    return names


def _handler_domain(decorator: ast.Call) -> str:
    for keyword in decorator.keywords:
        if keyword.arg != "domain":
            continue
        if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
        expression = ast.unparse(keyword.value)
        if expression in DOMAIN_EXPRESSIONS:
            return DOMAIN_EXPRESSIONS[expression]
        raise ValueError(f"unknown handler domain expression: {expression}")
    return ""


def collect_handler_versions(repo_root: Path) -> dict[str, dict[str, dict[int, bool]]]:
    """Collect registered version methods without importing project dependencies."""

    classes = _load_classes(repo_root)
    version_cache: dict[str, dict[int, bool]] = {}
    handlers: dict[str, dict[str, dict[int, bool]]] = defaultdict(lambda: defaultdict(dict))

    for class_name, info in classes.items():
        versions = _class_versions(class_name, classes, version_cache, set())
        if not versions:
            # tfl_op classes that only define to_tf rewrite TFLite into TensorFlow
            # operations and are not direct ONNX conversion handlers.
            continue

        for decorator in info.node.decorator_list:
            if _decorator_name(decorator) not in {"tf_op", "tfl_op"}:
                continue
            assert isinstance(decorator, ast.Call)
            domain = _handler_domain(decorator)
            if domain not in MAX_DOCUMENTED_OPSET:
                raise ValueError(f"no documented opset ceiling for domain {domain!r}")
            for operator_name in _handler_names(decorator):
                handlers[domain][operator_name].update(versions)

    return handlers


def render_support_status(handlers: dict[str, dict[str, dict[int, bool]]]) -> str:
    """Render the operator handler inventory as Markdown."""

    lines = [
        "<!--- SPDX-License-Identifier: Apache-2.0 -->",
        "<!-- Generated by tools/gen_doc.py. Do not edit manually. -->",
        "",
        "# `tf2onnx` operator handler coverage",
        "",
        "This inventory is generated from the converter's `@tf_op` and `@tfl_op` registrations.",
        "A registered handler means the converter has an implementation for that operator and opset range;",
        "it does not guarantee that every model using the operator converts successfully.",
        "See the [README](README.md#supported-versions) for the versions exercised by CI.",
        "",
    ]

    domains = [*DOMAIN_ORDER, *sorted(set(handlers) - set(DOMAIN_ORDER))]
    for domain in domains:
        operators = handlers.get(domain)
        if not operators:
            continue
        label = f'"{domain}"' if domain else '"" (default domain)'
        lines.extend(
            [
                f"## Domain: {label}",
                "",
                "| TensorFlow operator | Registered ONNX opset range |",
                "| ------------------- | --------------------------- |",
            ]
        )

        maximum = MAX_DOCUMENTED_OPSET[domain]
        for operator_name in sorted(operators):
            versions = operators[operator_name]
            minimum = min(versions)
            if minimum > maximum:
                raise ValueError(
                    f"{operator_name} has a version {minimum} handler above the documented {domain!r} ceiling {maximum}"
                )
            version_text = str(minimum) if minimum == maximum else f"{minimum}–{maximum}"
            unsupported = sorted(version for version, blocked in versions.items() if blocked and version <= maximum)
            if unsupported:
                version_text += " (except {})".format(", ".join(str(version) for version in unsupported))
            lines.append(f"| {operator_name} | {version_text} |")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filename", type=Path, help="Path to the generated Markdown file")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    handlers = collect_handler_versions(repo_root)
    args.filename.write_text(render_support_status(handlers), encoding="utf-8")


if __name__ == "__main__":
    main()
