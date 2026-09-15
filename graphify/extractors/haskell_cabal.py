"""Small, conservative Cabal component model for static Haskell resolution.

This intentionally does not try to solve Cabal flags.  Values from conditional
branches are unioned and marked conditional so static resolution retains real
ambiguity instead of selecting a build configuration it cannot know.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from typing import Iterable


_CACHE: dict[str, tuple[int, int, dict]] = {}
_COMPONENT_KINDS = {"library", "executable", "test-suite", "benchmark"}
_LIST_FIELDS = {
    "hs-source-dirs", "exposed-modules", "other-modules", "build-depends", "import",
    "main-is", "visibility", "default-extensions",
}


def _words(value: str) -> list[str]:
    return [part for part in re.split(r"[\s,]+", value.strip()) if part]


def _dependency(value: str) -> str:
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9_.+-]*(?::[A-Za-z0-9_.+-]+)?", value)
    return match.group(0) if match else ""


def _parse(path: Path) -> dict:
    package = ""
    commons: dict[str, dict] = {}
    components: list[dict] = []
    current: dict | None = None
    current_field: str | None = None
    conditional_indents: list[int] = []

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.split("--", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        stripped = line.strip()
        header = re.match(r"^(common|library|executable|test-suite|benchmark)(?:\s+(.+))?$", stripped)
        if indent == 0 and header:
            kind, written_name = header.groups()
            name = (written_name or package).strip()
            current = {
                "name": name,
                "kind": kind,
                "fields": {},
                "conditional_fields": set(),
                "conditional_values": {},
            }
            if kind == "common":
                commons[name] = current
            else:
                components.append(current)
            current_field = None
            conditional_indents = []
            continue
        top_field = re.match(r"^([A-Za-z][A-Za-z0-9-]*):\s*(.*)$", stripped)
        if indent == 0 and current is None and top_field:
            if top_field.group(1).lower() == "name":
                package = top_field.group(2).strip()
            continue
        if current is None:
            continue
        if stripped.startswith(("if ", "else", "elif ")):
            while conditional_indents and conditional_indents[-1] >= indent:
                conditional_indents.pop()
            conditional_indents.append(indent)
            current_field = None
            continue
        while conditional_indents and indent <= conditional_indents[-1]:
            conditional_indents.pop()
        field = re.match(r"^([A-Za-z][A-Za-z0-9-]*):\s*(.*)$", stripped)
        if field:
            current_field = field.group(1).lower()
            if current_field in _LIST_FIELDS:
                values = _words(field.group(2))
                current["fields"].setdefault(current_field, []).extend(values)
                if conditional_indents:
                    current["conditional_fields"].add(current_field)
                    current["conditional_values"].setdefault(current_field, []).extend(values)
            continue
        if current_field in _LIST_FIELDS:
            values = _words(stripped)
            current["fields"].setdefault(current_field, []).extend(values)
            if conditional_indents:
                current["conditional_fields"].add(current_field)
                current["conditional_values"].setdefault(current_field, []).extend(values)

    def expand_common(name: str, seen: frozenset[str] = frozenset()) -> tuple[dict, set, dict]:
        if name in seen or name not in commons:
            return {}, set(), {}
        common = commons[name]
        merged: dict[str, list[str]] = {}
        conditional_fields = set(common["conditional_fields"])
        conditional_values = {
            key: list(values) for key, values in common["conditional_values"].items()
        }
        conditional_imports = Counter(common["conditional_values"].get("import", []))
        for parent in common["fields"].get("import", []):
            import_is_conditional = conditional_imports[parent] > 0
            if import_is_conditional:
                conditional_imports[parent] -= 1
            parent_fields, parent_conditional, parent_values = expand_common(
                parent, seen | {name}
            )
            for key, values in parent_fields.items():
                merged.setdefault(key, []).extend(values)
            conditional_fields.update(parent_conditional)
            for key, values in parent_values.items():
                conditional_values.setdefault(key, []).extend(values)
            if import_is_conditional:
                conditional_fields.update(parent_fields)
                for key, values in parent_fields.items():
                    conditional_values.setdefault(key, []).extend(values)
        for key, values in common["fields"].items():
            if key != "import":
                merged.setdefault(key, []).extend(values)
        return merged, conditional_fields, conditional_values

    # Package name may occur before the first stanza, which is normal.  Fill
    # unnamed main libraries after the whole file has been seen.
    for component in components:
        if component["kind"] == "library" and not component["name"]:
            component["name"] = package
        fields = component["fields"]
        conditional_imports = Counter(component["conditional_values"].get("import", []))
        for common_name in fields.get("import", []):
            import_is_conditional = conditional_imports[common_name] > 0
            if import_is_conditional:
                conditional_imports[common_name] -= 1
            common_fields, common_conditional, common_values = expand_common(common_name)
            for key, values in common_fields.items():
                fields.setdefault(key, []).extend(values)
            component["conditional_fields"].update(common_conditional)
            for key, values in common_values.items():
                component["conditional_values"].setdefault(key, []).extend(values)
            if import_is_conditional:
                component["conditional_fields"].update(common_fields)
                for key, values in common_fields.items():
                    component["conditional_values"].setdefault(key, []).extend(values)
        source_dirs = fields.get("hs-source-dirs", ["."])
        modules = fields.get("exposed-modules", []) + fields.get("other-modules", [])
        dependencies = [_dependency(item) for item in fields.get("build-depends", [])]
        conditional = component["conditional_values"]
        unconditional = {
            key: list((
                Counter(fields.get(key, [])) - Counter(conditional.get(key, []))
            ).elements())
            for key in _LIST_FIELDS
        }
        unconditional_dependencies = [
            _dependency(item) for item in unconditional["build-depends"]
        ]
        component.update({
            "source_dirs": list(dict.fromkeys(source_dirs)),
            "exposed_modules": list(dict.fromkeys(fields.get("exposed-modules", []))),
            "other_modules": list(dict.fromkeys(fields.get("other-modules", []))),
            "modules": list(dict.fromkeys(modules)),
            "main_is": list(dict.fromkeys(fields.get("main-is", []))),
            "build_depends": list(dict.fromkeys(item for item in dependencies if item)),
            "unconditional_source_dirs": list(dict.fromkeys(
                unconditional["hs-source-dirs"] or (
                    ["."] if "hs-source-dirs" not in component["conditional_fields"] else []
                )
            )),
            "unconditional_exposed_modules": list(dict.fromkeys(
                unconditional["exposed-modules"]
            )),
            "unconditional_other_modules": list(dict.fromkeys(
                unconditional["other-modules"]
            )),
            "unconditional_main_is": list(dict.fromkeys(unconditional["main-is"])),
            "unconditional_build_depends": list(dict.fromkeys(
                item for item in unconditional_dependencies if item
            )),
            "no_field_selectors": (
                "NoFieldSelectors" in unconditional["default-extensions"]
                and "FieldSelectors" not in fields.get("default-extensions", [])
            ),
            "visibility": (
                fields.get("visibility", [""])[-1]
                or ("public" if component["name"] == package else "private")
            ),
            "visibility_certain": "visibility" not in component["conditional_fields"],
            "conditional": bool(component["conditional_fields"]),
        })
        component.pop("fields", None)
        component.pop("conditional_values", None)
        component["conditional_fields"] = sorted(component["conditional_fields"])
    return {"cabal_file": str(path.resolve()), "package": package, "components": components}


def _package(path: Path) -> dict | None:
    path = path.resolve()
    for directory in (path.parent, *path.parents):
        cabal_files = sorted(directory.glob("*.cabal"))
        if len(cabal_files) > 1:
            return {"ambiguous_manifests": [path.name for path in cabal_files]}
        if cabal_files:
            cabal = cabal_files[0]
            stat = cabal.stat()
            key = str(cabal)
            cached = _CACHE.get(key)
            if cached is None or cached[:2] != (stat.st_mtime_ns, stat.st_size):
                _CACHE[key] = (stat.st_mtime_ns, stat.st_size, _parse(cabal))
            return _CACHE[key][2]
        # The nearest repository owns package discovery. Do not accidentally
        # adopt a manifest from an unrelated parent checkout.
        if (directory / ".git").exists():
            break
    return None


def _module_for(path: Path, cabal_dir: Path, source_dir: str) -> str | None:
    root = (cabal_dir / source_dir).resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError:
        return None
    if relative.suffix not in {".hs", ".lhs", ".hs-boot"}:
        return None
    text = str(relative)
    for suffix in (".hs-boot", ".lhs", ".hs"):
        if text.endswith(suffix):
            text = text[:-len(suffix)]
            break
    return text.replace("/", ".")


def _source_no_field_selectors(path: Path) -> bool | None:
    """Return a certain source LANGUAGE override from the pre-module header."""
    text = path.read_text(encoding="utf-8", errors="replace")
    extensions: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        if depth:
            if text.startswith("{-", index):
                depth += 1
                index += 2
            elif text.startswith("-}", index):
                depth -= 1
                index += 2
            else:
                index += 1
            continue
        if text.startswith("--", index):
            newline = text.find("\n", index + 2)
            index = len(text) if newline < 0 else newline + 1
            continue
        if text.startswith("{-", index) and not text.startswith("{-#", index):
            depth = 1
            index += 2
            continue
        if text.startswith("{-#", index):
            end = text.find("#-}", index + 3)
            if end < 0:
                break
            pragma = text[index + 3:end].strip()
            language = re.match(r"^LANGUAGE\s+(.+)$", pragma, re.DOTALL)
            options = re.match(r"^OPTIONS_GHC\s+(.+)$", pragma, re.DOTALL)
            if language:
                extensions.extend(_words(language.group(1)))
            elif options:
                extensions.extend(
                    option[2:] for option in _words(options.group(1))
                    if option.startswith("-X")
                )
            index = end + 3
            continue
        if text[index].isspace():
            index += 1
            continue
        # The pragma preamble is over at the first real source token. This also
        # treats CPP as unknown rather than searching generated/runtime text.
        break
    if "FieldSelectors" in extensions:
        return False
    if "NoFieldSelectors" in extensions:
        return True
    return None


def file_component_context(path: str | Path) -> dict:
    """Return package and all plausible Cabal component memberships for path."""
    source = Path(path).resolve()
    package = _package(source)
    if package is None:
        return {"manifest": "", "package": "", "components": []}
    if package.get("ambiguous_manifests"):
        return {
            "manifest": ",".join(package["ambiguous_manifests"]),
            "package": "",
            "components": [],
            "ambiguous": True,
        }
    cabal_dir = Path(package["cabal_file"]).parent
    source_override = _source_no_field_selectors(source)
    memberships: list[tuple[tuple[int, int], dict]] = []
    for component in package["components"]:
        matched_module = None
        membership_certain = False
        explicit_match = False
        matched_specificity = 0
        for source_dir in component["source_dirs"]:
            module = _module_for(source, cabal_dir, source_dir)
            if module is None:
                continue
            source_root = (cabal_dir / source_dir).resolve()
            if matched_module is not None and len(source_root.parts) <= matched_specificity:
                continue
            relative_source = str(source.relative_to(source_root)).replace("\\", "/")
            main_match = relative_source in {
                item.replace("\\", "/") for item in component["main_is"]
            }
            listed_match = module in component["modules"]
            # Main modules and other imported home modules may be omitted from
            # other-modules (Cabal warns, but still compiles them), so source
            # directory reachability remains a conservative membership.
            matched_module = module
            explicit_match = listed_match or main_match
            matched_specificity = len(source_root.parts)
            source_certain = source_dir in component["unconditional_source_dirs"]
            module_certain = (
                module in component["unconditional_exposed_modules"]
                or module in component["unconditional_other_modules"]
            )
            if not component["modules"]:
                module_certain = not component["main_is"]
            main_certain = relative_source in {
                item.replace("\\", "/") for item in component["unconditional_main_is"]
            }
            membership_certain = source_certain and (
                module_certain or main_certain or not explicit_match
            )
        if matched_module is not None:
            item = dict(component)
            if source_override is not None:
                item["no_field_selectors"] = source_override
            item["module"] = matched_module
            item["membership_certain"] = membership_certain
            item["inferred_from_source_dir"] = not explicit_match
            memberships.append(((1 if explicit_match else 0, matched_specificity), item))
    if memberships:
        best = max(rank for rank, _item in memberships)
        memberships = [(rank, item) for rank, item in memberships if rank == best]
    return {
        # Keep graph metadata portable. The absolute manifest path belongs
        # only to the private parse cache above.
        "manifest": Path(package["cabal_file"]).name,
        "package": package["package"],
        "components": [item for _rank, item in memberships],
    }


def _dependency_allows(caller: dict, candidate_package: str, candidate: dict,
                       *, same_package: bool) -> bool:
    dependencies = set(caller.get("unconditional_build_depends", []))
    library_name = candidate.get("name", "")
    if library_name == candidate_package:
        return candidate_package in dependencies
    if not library_name:
        return False
    qualified = f"{candidate_package}:{library_name}"
    if same_package:
        return library_name in dependencies or qualified in dependencies
    return (
        qualified in dependencies
        and candidate.get("visibility") == "public"
        and candidate.get("visibility_certain", False)
    )


def _membership_is_certain(component: dict) -> bool:
    return bool(component.get("membership_certain"))


def filter_visible_contexts(caller_context: dict,
                            candidate_contexts: Iterable[tuple[str, dict]]) -> set[str]:
    """Filter opaque candidate tokens using portable, attached Cabal metadata."""
    callers = caller_context.get("components", [])
    if not callers:
        if caller_context.get("manifest"):
            return set()
        return {token for token, _context in candidate_contexts}
    visible: set[str] = set()
    for token, candidate_context in candidate_contexts:
        candidates = candidate_context.get("components", [])
        if not candidates:
            # No manifest means unknown metadata. A known manifest with no
            # memberships means Cabal excludes this source from every component.
            if not candidate_context.get("manifest"):
                visible.add(token)
            continue
        for caller in callers:
            for candidate in candidates:
                same_package = bool(caller_context.get("package")) and (
                    caller_context.get("package") == candidate_context.get("package")
                )
                same_component = (
                    same_package and caller["kind"] == candidate["kind"]
                    and caller["name"] == candidate["name"]
                    and _membership_is_certain(caller)
                    and _membership_is_certain(candidate)
                )
                public_library = candidate["kind"] == "library" and (
                    candidate.get("module")
                    in candidate.get("unconditional_exposed_modules", [])
                ) and _membership_is_certain(candidate)
                if same_component or (
                    public_library
                    and _membership_is_certain(caller)
                    and _dependency_allows(
                        caller, str(candidate_context.get("package", "")), candidate,
                        same_package=same_package,
                    )
                ):
                    visible.add(token)
                    break
            if token in visible:
                break
    return visible


def filter_visible_candidate_paths(caller_path: str | Path,
                                   candidate_paths: Iterable[str | Path]) -> set[str]:
    """Path-discovering wrapper for callers that do not persist metadata."""
    original = {str(path) for path in candidate_paths}
    return filter_visible_contexts(
        file_component_context(caller_path),
        ((written, file_component_context(written)) for written in original),
    )
