"""Haskell extraction and import-aware cross-file resolution."""
from __future__ import annotations

from pathlib import Path

from graphify.extractors.base import _file_stem, _make_id, _read_text


def _symbol_key(name: str) -> str:
    """Keep operators and primed identifiers distinct after ID normalization."""
    if "'" in name or not all(char.isalnum() or char in "_. " for char in name):
        return "symbol_" + name.encode().hex()
    return name


def _symbol_id(owner: str, name: str) -> str:
    return _make_id(owner, _symbol_key(name))


def _first_parse_error(root) -> tuple[int, int, bool]:
    first: int | None = None
    count = 0
    multiline = False
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type == "ERROR" or node.is_missing:
            count += 1
            line = node.start_point[0] + 1
            first = line if first is None else min(first, line)
            multiline = multiline or node.end_point[0] > node.start_point[0]
        stack.extend(node.children)
    return first or 1, count, multiline


def extract_haskell(path: Path) -> dict:
    """Extract Haskell modules, declarations, imports and scoped call facts.

    Calls that need import knowledge are deliberately deferred to
    :func:`resolve_haskell_calls`. This preserves the written qualifier and
    import scope instead of asking the generic bare-name stub rewire to guess.
    """
    try:
        import tree_sitter_haskell as tshaskell
        from tree_sitter import Language, Parser
    except ImportError:
        return {"nodes": [], "edges": [], "error": "tree-sitter-haskell not installed"}
    try:
        source = path.read_bytes()
        root = Parser(Language(tshaskell.language())).parse(source).root_node
    except Exception as exc:
        return {"nodes": [], "edges": [], "error": str(exc)}

    nodes: list[dict] = []
    edges: list[dict] = []
    raw_calls: list[dict] = []
    raw_references: list[dict] = []
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str, str, int]] = set()
    seen_raw_calls: set[tuple[str, str, str, int, str]] = set()
    seen_raw_references: set[tuple[str, str, str, int, str, str]] = set()
    node_by_id: dict[str, dict] = {}
    stem, str_path = _file_stem(path), str(path)

    def text(node) -> str:
        return _read_text(node, source) if node is not None else ""

    def field(node, name):
        return node.child_by_field_name(name) if node is not None else None

    def add_node(nid: str, label: str, node=None, **metadata) -> str:
        if nid in seen_nodes:
            if (metadata.get("node_kind") == "value"
                    and node_by_id[nid].get("node_kind") == "function"):
                metadata.pop("node_kind")
            node_by_id[nid].update(metadata)
            if node is not None:
                node_by_id[nid]["end_line"] = max(
                    node_by_id[nid].get("end_line", 0), node.end_point[0] + 1
                )
            return nid
        seen_nodes.add(nid)
        item = {
            "id": nid,
            "label": label,
            "file_type": "code",
            "language": "haskell",
            "source_file": str_path if node is not None else "",
            "source_location": f"L{node.start_point[0] + 1}" if node is not None else "",
            **({"end_line": node.end_point[0] + 1} if node is not None else {}),
            **({"origin_file": str_path} if node is None else {}),
            **metadata,
        }
        nodes.append(item)
        node_by_id[nid] = item
        return nid

    def add_edge(src: str, dst: str, relation: str, node,
                 confidence: str = "EXTRACTED", context: str | None = None) -> None:
        line = node.start_point[0] + 1
        key = (src, dst, relation, line)
        if key in seen_edges:
            return
        seen_edges.add(key)
        item = {
            "source": src,
            "target": dst,
            "relation": relation,
            "confidence": confidence,
            "source_file": str_path,
            "source_location": f"L{line}",
            "weight": 1.0,
        }
        if context:
            item["context"] = context
        edges.append(item)

    def import_stub(name: str) -> str:
        return add_node(_make_id(name), name, _haskell_import_stub=True)

    def normalize_ref(name: str) -> str:
        name = name.strip()
        if len(name) >= 2 and name[0] == "(" and name[-1] == ")":
            name = name[1:-1]
        return name.strip("`")

    def variables(node) -> set[str]:
        if node is None:
            return set()
        if node.type == "variable":
            return {text(node)}
        found: set[str] = set()
        for child in node.named_children:
            found.update(variables(child))
        return found

    def callable_head(node):
        while node is not None:
            if node.type == "apply":
                node = field(node, "function")
                continue
            if node.type in {"parens", "prefix_id", "infix_id"} and node.named_children:
                inner = field(node, "expression")
                node = inner if inner is not None else node.named_children[0]
                continue
            return node
        return None

    def declaration_names(node) -> list[str]:
        group = field(node, "names")
        if group is not None:
            return [normalize_ref(text(child)) for child in group.named_children]
        name = field(node, "name")
        if name is not None:
            return [normalize_ref(text(name))]
        if node.type == "bind" and field(node, "pattern") is not None:
            return sorted(variables(field(node, "pattern")))
        if node.type == "pattern_synonym":
            equation = next((c for c in node.named_children if c.type == "equation"), None)
            head = callable_head(field(equation, "synonym"))
            return [normalize_ref(text(head))] if head is not None else []
        if node.type == "foreign_import":
            signature = next((c for c in node.named_children if c.type == "signature"), None)
            return declaration_names(signature) if signature is not None else []
        infix = next((c for c in node.named_children if c.type == "infix"), None)
        if infix is not None:
            return [normalize_ref(text(field(infix, "operator")))]
        return []

    def selection(node) -> dict:
        """Parse a Haskell import/export list into JSON-safe name filters."""
        result = {"names": [], "all_children": [], "children": {}}
        if node is None:
            return result
        for item in node.named_children:
            base_node = (
                field(item, "variable") or field(item, "type")
                or field(item, "operator")
            )
            if base_node is None:
                continue
            base = normalize_ref(text(base_node))
            if base:
                result["names"].append(base)
            children = field(item, "children")
            if children is None:
                continue
            if any(child.type == "all_names" for child in children.named_children):
                result["all_children"].append(base)
            else:
                result["children"][base] = [
                    normalize_ref(text(child)) for child in children.named_children
                ]
        return result

    header = next((child for child in root.named_children if child.type == "header"), None)
    written_module = text(field(header, "module")) if header is not None else ""
    module_name = written_module or "Main"
    export_node = field(header, "exports") if header is not None else None
    exports = selection(export_node)
    exports["explicit"] = export_node is not None
    exports["modules"] = []
    if export_node is not None:
        exports["modules"] = [
            text(field(child, "module"))
            for child in export_node.named_children
            if child.type == "module_export" and field(child, "module") is not None
        ]

    file_id = add_node(_make_id(str_path), path.name, root)
    node_by_id[file_id]["node_kind"] = "file"
    container = file_id
    if written_module:
        container = add_node(_make_id(stem, module_name), module_name, header)
        node_by_id[container]["node_kind"] = "module"
        add_edge(file_id, container, "defines", header)

    imports: list[dict] = []
    for child in root.named_children:
        if child.type != "imports":
            continue
        for item in child.named_children:
            imported_module = text(field(item, "module"))
            if not imported_module:
                continue
            names_node = field(item, "names")
            chosen = selection(names_node)
            spec = {
                "module": imported_module,
                "alias": text(field(item, "alias")) or None,
                "qualified": any(c.type == "qualified" for c in item.children),
                "hiding": any(c.type == "hiding" for c in item.children),
                "selection": chosen if names_node is not None else None,
            }
            imports.append(spec)
            add_edge(container, import_stub(imported_module), "imports_from", item, "INFERRED")

    add_node(
        container,
        module_name if written_module else path.name,
        header if written_module else root,
        _haskell_module=module_name,
        _haskell_imports=imports,
        _haskell_exports=exports,
        node_kind="module" if written_module else "file",
        exports=exports,
    )

    def is_exported(name: str, parent: str | None = None) -> bool:
        if not exports["explicit"]:
            return True
        if module_name in exports["modules"]:
            return True
        if name in exports["names"]:
            return True
        if parent is None:
            return False
        return (
            parent in exports["all_children"]
            or name in exports["children"].get(parent, [])
        )

    def define(owner: str, name: str, node, *, callable_: bool = False,
               module_symbol: bool = False, parent: str | None = None,
               identity_owner: str | None = None, node_kind: str = "value",
               written_signature: str | None = None,
               haddock_summary: str | None = None) -> str:
        metadata: dict = {"node_kind": node_kind}
        if written_signature:
            metadata["written_signature"] = written_signature
        if haddock_summary:
            metadata["haddock_summary"] = haddock_summary
        if callable_:
            metadata["_callable"] = True
        if module_symbol:
            metadata.update({
                "_haskell_owner_module": module_name,
                "_haskell_exported": is_exported(name, parent),
                "exported": is_exported(name, parent),
            })
            if parent:
                metadata["_haskell_parent"] = parent
                metadata["parent"] = parent
        symbol_owner = identity_owner or owner
        nid = _symbol_id(symbol_owner, name)
        # Haskell permits a record constructor and one of its fields to differ
        # only by case (FieldSchema / fieldSchema). Normalized IDs fold that
        # distinction, so salt only the colliding field rather than allowing it
        # to overwrite the callable constructor metadata.
        if (nid in node_by_id and node_kind in {"constructor", "field"}
                and node_by_id[nid].get("node_kind") in {"constructor", "field"}
                and node_by_id[nid].get("node_kind") != node_kind):
            nid = _symbol_id(_make_id(symbol_owner, node_kind), name)
        nid = add_node(nid, name, node, **metadata)
        add_edge(owner, nid, "defines" if owner == file_id else "contains", node)
        return nid

    declaration_types = {
        "function", "bind", "signature", "data_type", "newtype",
        "type_synomym", "type_family", "data_family", "class",
        "pattern_synonym", "foreign_import",
    }
    # A signature can be the only declaration of a class method. Treat it as
    # callable so imported methods resolve even when the class has no default
    # implementation in its defining module.
    callable_declarations = {
        "function", "bind", "signature", "pattern_synonym", "foreign_import"
    }
    module_env: dict[str, str | None] = {}

    kind_for_declaration = {
        "function": "function", "bind": "value", "signature": "function",
        "data_type": "type", "newtype": "type", "type_synomym": "type",
        "type_family": "type_family", "data_family": "type_family",
        "class": "class", "pattern_synonym": "constructor",
        "foreign_import": "function",
    }

    all_haddocks = []
    pending = [root]
    while pending:
        candidate = pending.pop()
        if candidate.type == "haddock":
            all_haddocks.append(candidate)
        pending.extend(candidate.named_children)

    def haddock_before(block, declaration) -> str | None:
        candidates = [doc for doc in all_haddocks if doc.end_byte <= declaration.start_byte]
        if not candidates:
            return None
        preceding = max(candidates, key=lambda item: item.end_byte)
        gap = source[preceding.end_byte:declaration.start_byte]
        if gap.strip() or gap.count(b"\n") > 2:
            return None
        doc = text(preceding).strip()
        for marker in ("-- |", "-- ^", "{-|", "{-^"):
            if doc.startswith(marker):
                doc = doc[len(marker):]
                break
        doc = doc.removesuffix("-}").strip()
        return next((line.strip() for line in doc.splitlines() if line.strip()), None)

    def register(block, owner: str, outer: dict[str, str | None], *, module_scope=False,
                 export_parent: str | None = None):
        scope = dict(outer)
        # Separate nested declaration groups can legally reuse a local name.
        # Include the group's location in local IDs while keeping module and
        # class declarations stable so signatures merge with implementations.
        identity_owner = (
            owner if module_scope
            else _make_id(owner, f"scope_l{block.start_point[0] + 1}")
        )
        for declaration in block.named_children:
            if declaration.type not in declaration_types:
                continue
            for name in declaration_names(declaration):
                signature = text(declaration) if declaration.type == "signature" else None
                scope[name] = define(
                    owner,
                    name,
                    declaration,
                    callable_=declaration.type in callable_declarations,
                    module_symbol=module_scope,
                    parent=export_parent,
                    identity_owner=identity_owner,
                    node_kind=(kind_for_declaration[declaration.type]
                               if module_scope or declaration.type not in {"function", "bind"}
                               else ("method" if node_by_id.get(owner, {}).get("node_kind") == "instance"
                                     else "local_binding")),
                    written_signature=signature,
                    haddock_summary=haddock_before(block, declaration),
                )

        def constructors(node, type_id: str, parent: str) -> None:
            if node.type in {
                "prefix", "record", "newtype_constructor", "gadt_constructor", "infix"
            }:
                name_node = field(node, "name") or field(node, "operator")
                if name_node is not None:
                    label = normalize_ref(text(name_node))
                    scope[label] = define(
                        type_id, label, node, callable_=True,
                        module_symbol=module_scope, parent=parent,
                        node_kind="constructor",
                    )
            if node.type == "field":
                for name_node in node.children_by_field_name("name"):
                    label = normalize_ref(text(name_node))
                    scope[label] = define(
                        type_id, label, name_node, callable_=True,
                        module_symbol=module_scope, parent=parent,
                        node_kind="field",
                    )
            for child in node.named_children:
                constructors(child, type_id, parent)

        for declaration in block.named_children:
            if declaration.type in {"data_type", "newtype"}:
                declared = declaration_names(declaration)
                if declared:
                    constructors(declaration, scope[declared[0]], declared[0])
        return scope

    def defer_call(caller: str, qualifier: str | None, name: str, node, context: str) -> None:
        line = node.start_point[0] + 1
        key = (caller, qualifier or "", name, line, context)
        if key in seen_raw_calls:
            return
        seen_raw_calls.add(key)
        raw_calls.append({
            "caller_nid": caller,
            "callee": name,
            "qualified_prefix": qualifier,
            "language": "haskell",
            "context": context,
            "source_file": str_path,
            "source_location": f"L{line}",
        })

    def defer_reference(caller: str, qualifier: str | None, name: str, node,
                        relation: str, namespace: str, context: str) -> None:
        line = node.start_point[0] + 1
        key = (caller, qualifier or "", name, line, relation, context)
        if not name or key in seen_raw_references:
            return
        seen_raw_references.add(key)
        raw_references.append({
            "caller_nid": caller, "callee": name,
            "qualified_prefix": qualifier, "language": "haskell",
            "source_file": str_path, "source_location": f"L{line}",
            "relation": relation, "namespace": namespace, "context": context,
        })

    def written_type_references(node):
        """Yield explicitly written type/class names, excluding type variables."""
        if node is None:
            return
        if node.type == "qualified":
            qualifier = text(field(node, "module")).rstrip(".")
            name = normalize_ref(text(field(node, "id")))
            if name:
                yield qualifier or None, name, node
            return
        if node.type in {"name", "constructor", "constructor_operator"}:
            name = normalize_ref(text(node))
            if name:
                yield None, name, node
            return
        for child in node.named_children:
            yield from written_type_references(child)

    def emit_type_references(caller: str, node, relation: str, context: str) -> None:
        for qualifier, name, ref_node in written_type_references(node):
            defer_reference(caller, qualifier, name, ref_node, relation, "type", context)

    def call(callee, caller: str, env: dict[str, str | None], node,
             context: str = "call") -> None:
        callee = callable_head(callee)
        if callee is not None and callee.type == "projection":
            receiver = text(field(callee, "expression"))
            member = normalize_ref(text(field(callee, "field")))
            if receiver and member:
                defer_call(caller, receiver, member, node, "record_field_call")
            return
        if callee is None or callee.type not in {
            "variable", "constructor", "operator", "constructor_operator", "qualified"
        }:
            return
        if callee.type == "qualified":
            qualifier = text(field(callee, "module")).rstrip(".")
            name = normalize_ref(text(field(callee, "id")))
            if qualifier == module_name and name in module_env:
                target = module_env[name]
                if target is not None:
                    add_edge(caller, target, "calls", node, context=context)
            else:
                defer_call(caller, qualifier, name, node, context)
            return
        name = normalize_ref(text(callee))
        if name in env:
            target = env[name]
            if target is not None:
                add_edge(caller, target, "calls", node, context=context)
        else:
            defer_call(caller, None, name, node, context)

    def expression(node, caller: str, env: dict[str, str | None]) -> None:
        if node.type in {"do", "qualifiers"}:
            sequential = dict(env)
            for child in node.named_children:
                expression(child, caller, sequential)
                sequential.update(dict.fromkeys(variables(field(child, "pattern"))))
                local = field(child, "binds")
                if local is not None:
                    sequential = register(local, caller, sequential)
            return
        if node.type == "list_comprehension":
            sequential = dict(env)
            qualifiers = field(node, "qualifiers")
            if qualifiers is not None:
                for child in qualifiers.named_children:
                    expression(child, caller, sequential)
                    sequential.update(dict.fromkeys(variables(field(child, "pattern"))))
            body = field(node, "expression")
            if body is not None:
                expression(body, caller, sequential)
            return
        if node.type == "alternative":
            env = {**env, **dict.fromkeys(variables(field(node, "pattern")))}
        binds = field(node, "binds")
        if binds is not None:
            env = register(binds, caller, env)
        patterns = field(node, "patterns")
        if patterns is not None:
            env = {**env, **dict.fromkeys(variables(patterns))}
        if node.type == "apply":
            call(field(node, "function"), caller, env, node)
        elif node.type == "record":
            # Record syntax covers both construction (`R { field = value }`)
            # and update (`value { field = value }`). Only constructors denote
            # a call; the variable form is an update of an existing value.
            record_head = field(node, "expression")
            if record_head is not None and record_head.type in {"constructor", "qualified"}:
                call(record_head, caller, env, node, "record_construction")
        elif node.type == "infix":
            operator_node = field(node, "operator")
            operator = normalize_ref(text(callable_head(operator_node)))
            call(operator_node, caller, env, node)
            if operator in {"|>", "&"}:
                call(field(node, "right_operand"), caller, env, node, "pipeline")
            elif operator == "$":
                call(field(node, "left_operand"), caller, env, node, "application_operator")
        for child in node.named_children:
            if child == binds:
                declarations(child, caller, env)
            elif child not in {patterns, field(node, "pattern")}:
                expression(child, caller, env)

    def declarations(block, owner: str, outer: dict[str, str | None], *, module_scope=False,
                     export_parent: str | None = None):
        nonlocal module_env
        env = register(
            block, owner, outer, module_scope=module_scope, export_parent=export_parent
        )
        if module_scope:
            module_env = env
        for declaration in block.named_children:
            declared = declaration_names(declaration)
            declaration_owners = [env.get(name) or owner for name in declared] or [owner]
            declaration_owner = declaration_owners[0]
            if declaration.type == "signature":
                signature_type = field(declaration, "type")
                constraint = field(signature_type, "context")
                signature_body = field(signature_type, "type") if constraint is not None else signature_type
                for signature_owner in declaration_owners:
                    emit_type_references(
                        signature_owner, constraint, "constrains", "signature_constraint"
                    )
                    emit_type_references(
                        signature_owner, signature_body, "references", "written_signature"
                    )
            elif declaration.type == "type_synomym":
                emit_type_references(
                    declaration_owner, field(declaration, "type"), "aliases", "type_alias_rhs"
                )
            elif declaration.type == "class":
                emit_type_references(
                    declaration_owner, field(declaration, "context"),
                    "constrains", "class_constraint"
                )
            elif declaration.type in {"data_type", "newtype"}:
                # Constructor names inhabit the value namespace. Traverse only
                # explicitly written field/GADT types here.
                pending_types = [declaration]
                while pending_types:
                    part = pending_types.pop()
                    if part.type in {"field", "gadt_constructor"}:
                        emit_type_references(
                            declaration_owner, field(part, "type"),
                            "references", "declaration_type"
                        )
                    elif part.type in {"prefix", "newtype_constructor", "infix"}:
                        constructor_name = field(part, "name") or field(part, "operator")
                        for argument in part.named_children:
                            if argument != constructor_name:
                                emit_type_references(
                                    declaration_owner, argument,
                                    "references", "declaration_type"
                                )
                    pending_types.extend(part.named_children)
            elif declaration.type == "type_family":
                for equation in (part for part in declaration.named_children
                                 if part.type == "equations"):
                    for item in equation.named_children:
                        emit_type_references(
                            declaration_owner, field(item, "patterns"),
                            "references", "type_family_lhs"
                        )
                        rhs = next((child for child in item.named_children
                                    if child not in {field(item, "name"), field(item, "patterns")}), None)
                        emit_type_references(
                            declaration_owner, rhs, "references", "type_family_rhs"
                        )
            if declaration.type in {"function", "bind"}:
                caller = env[declared[0]] if declared else owner
                if caller is None:
                    caller = owner
                local_env = dict(env)
                patterns = field(declaration, "patterns")
                local_env.update(dict.fromkeys(variables(patterns)))
                for child in declaration.named_children:
                    if child.type == "infix":
                        local_env.update(dict.fromkeys(variables(field(child, "left_operand"))))
                        local_env.update(dict.fromkeys(variables(field(child, "right_operand"))))
                binds = field(declaration, "binds")
                if binds is not None:
                    local_env = register(binds, caller, local_env)
                    declarations(binds, caller, local_env)
                for child in declaration.named_children:
                    if child.type != "match":
                        continue
                    body = field(child, "expression")
                    reference = body
                    while reference is not None and reference.type == "parens":
                        reference = field(reference, "expression")
                    if declaration.type == "bind" and reference is not None and reference.type in {
                        "variable", "constructor", "operator", "constructor_operator", "qualified"
                    }:
                        call(reference, caller, local_env, child, "point_free_alias")
                    expression(child, caller, local_env)
            elif declaration.type in {"class", "instance"}:
                body = field(declaration, "declarations")
                if declaration.type == "class":
                    class_owner = env[declaration_names(declaration)[0]]
                    child_module_scope = module_scope
                else:
                    heading = text(declaration).split("where", 1)[0]
                    label = "instance " + heading.removeprefix("instance").strip()
                    class_owner = define(
                        owner, label, declaration, node_kind="instance",
                        parent=export_parent,
                    )
                    class_name_node = field(declaration, "name")
                    if class_name_node is not None:
                        if class_name_node.type == "qualified":
                            class_qualifier = text(field(class_name_node, "module")).rstrip(".")
                            class_label = normalize_ref(text(field(class_name_node, "id")))
                        else:
                            class_qualifier = None
                            class_label = normalize_ref(text(class_name_node))
                        defer_reference(
                            class_owner, class_qualifier, class_label,
                            class_name_node, "instance_of", "type", "instance_class"
                        )
                    emit_type_references(
                        class_owner, field(declaration, "context"),
                        "constrains", "instance_constraint"
                    )
                    emit_type_references(
                        class_owner, field(declaration, "patterns"),
                        "references", "instance_head"
                    )
                    child_module_scope = False
                if body is not None and class_owner is not None:
                    class_name = (
                        declaration_names(declaration)[0]
                        if declaration.type == "class" else None
                    )
                    declarations(
                        body,
                        class_owner,
                        env,
                        module_scope=child_module_scope,
                        export_parent=class_name,
                    )
            elif declaration.type == "type_instance":
                family = text(field(declaration, "name"))
                patterns = text(field(declaration, "patterns"))
                label = "type instance " + " ".join(part for part in (family, patterns) if part)
                instance_id = define(owner, label, declaration, node_kind="type_instance")
                family_node = field(declaration, "name")
                if family_node is not None:
                    defer_reference(
                        instance_id, None, normalize_ref(text(family_node)), family_node,
                        "instance_of", "type", "type_family_instance"
                    )
                emit_type_references(
                    instance_id, field(declaration, "patterns"),
                    "references", "type_family_lhs"
                )
                rhs = next((child for child in declaration.named_children
                            if child not in {family_node, field(declaration, "patterns")}), None)
                emit_type_references(
                    instance_id, rhs, "references", "type_family_rhs"
                )
            elif declaration.type == "top_splice":
                expression(declaration, owner, env)

    for child in root.named_children:
        if child.type == "declarations":
            declarations(child, container, {}, module_scope=True)

    result = {
        "nodes": nodes, "edges": edges, "raw_calls": raw_calls,
        "raw_references": raw_references,
        "_haskell_schema": 3,
    }
    if root.has_error:
        first_line, count, multiline = _first_parse_error(root)
        result["parse_errors"] = {
            "first_error_line": first_line,
            "multiline_error": multiline,
            "error_count": count,
            "material_recovery": True,
        }
    return result


def resolve_haskell_calls(per_file: list[dict], all_nodes: list[dict],
                          all_edges: list[dict]) -> None:
    """Resolve deferred Haskell names through imports and re-exports."""
    node_by_id = {node.get("id"): node for node in all_nodes if node.get("id")}
    anchors = [node for node in all_nodes if node.get("_haskell_module")]
    anchors_by_module: dict[str, list[dict]] = {}
    anchors_by_source: dict[str, dict] = {}
    for anchor in anchors:
        anchors_by_module.setdefault(anchor["_haskell_module"], []).append(anchor)
        if anchor.get("source_file"):
            anchors_by_source[str(anchor["source_file"])] = anchor

    try:
        from graphify.extractors.haskell_cabal import filter_visible_contexts
    except ImportError:
        filter_visible_contexts = None

    def cabal_context(node: dict) -> dict:
        context = node.get("_haskell_cabal")
        if context:
            return context
        anchor = anchors_by_source.get(str(node.get("source_file", "")))
        return anchor.get("_haskell_cabal", {}) if anchor else {}

    def visible(ids: set[str], caller_context: dict | None) -> set[str]:
        ids = {nid for nid in ids if nid in node_by_id}
        if filter_visible_contexts is None or not caller_context:
            return ids
        return filter_visible_contexts(
            caller_context,
            ((nid, cabal_context(node_by_id[nid])) for nid in ids),
        )

    direct: dict[tuple[str, str], list[str]] = {}
    local_direct: dict[tuple[str, str], list[str]] = {}
    for node in all_nodes:
        module = node.get("_haskell_owner_module")
        if (module and node.get("source_file")
                and (node.get("_callable") or node.get("node_kind") in {
                    "type", "class", "type_family", "constructor", "field",
                })):
            key = (str(module), str(node.get("label", "")))
            local_direct.setdefault(key, []).append(node["id"])
            if node.get("_haskell_exported"):
                direct.setdefault(key, []).append(node["id"])

    path_scores: dict[tuple[str, str], int] = {}

    def path_score(left: str, right: str) -> int:
        cache_key = (left, right)
        if cache_key in path_scores:
            return path_scores[cache_key]
        a = Path(left).parts[:-1]
        b = Path(right).parts[:-1]
        best = 0
        for i in range(len(a)):
            for j in range(len(b)):
                size = 0
                while i + size < len(a) and j + size < len(b) and a[i + size] == b[j + size]:
                    size += 1
                best = max(best, size)
        path_scores[cache_key] = best
        return best

    def closest(ids: set[str], caller_file: str, *, require_same_module: bool = False) -> str | None:
        ids = {nid for nid in ids if nid in node_by_id}
        if len(ids) == 1:
            return next(iter(ids))
        if not ids:
            return None
        if require_same_module:
            # Path proximity may choose between duplicate copies of the same
            # module, but it must not turn two different exporting modules into
            # a resolved Haskell name.
            modules = {
                node_by_id[nid].get("_haskell_owner_module") for nid in ids
            }
            if len(modules) != 1:
                return None
        scored = sorted(
            ((path_score(caller_file, str(node_by_id[nid].get("source_file", ""))), nid)
             for nid in ids),
            reverse=True,
        )
        return scored[0][1] if len(scored) == 1 or scored[0][0] > scored[1][0] else None

    def allowed(spec: dict, symbol: str, candidate: dict | None = None) -> bool:
        selected = spec.get("selection")
        if selected is None:
            return True
        parent = candidate.get("_haskell_parent") if candidate else None
        raw_named = symbol in selected.get("names", [])
        # `import M (T)` imports the type constructor but not a data
        # constructor also named T. Conversely `hiding (T)` hides both.
        same_named_child = bool(parent) and parent == symbol
        named = raw_named and (spec.get("hiding") or not same_named_child)
        child = bool(parent) and (
            parent in selected.get("all_children", [])
            or symbol in selected.get("children", {}).get(parent, [])
        )
        included = named or child
        return not included if spec.get("hiding") else included

    def in_namespace(candidate: dict | None, namespace: str | None) -> bool:
        if candidate is None or namespace is None:
            return True
        kind = candidate.get("node_kind")
        if namespace == "type":
            return kind in {"type", "class", "type_family"}
        if namespace == "value":
            if kind == "field":
                components = cabal_context(candidate).get("components", [])
                if components and all(
                    component.get("no_field_selectors") for component in components
                ):
                    return False
            return bool(candidate.get("_callable")) or kind in {
                "constructor", "field", "function", "method", "value",
            }
        return True

    def explicitly_exported(exports: dict, symbol: str,
                            candidate: dict | None,
                            qualifier: str | None = None) -> bool:
        written_symbol = f"{qualifier}.{symbol}" if qualifier else symbol
        parent = candidate.get("_haskell_parent") if candidate else None
        if written_symbol in exports.get("names", []) and parent != symbol:
            return True
        written_parent = f"{qualifier}.{parent}" if qualifier and parent else parent
        return bool(parent) and (
            written_parent in exports.get("all_children", [])
            or symbol in exports.get("children", {}).get(written_parent, [])
        )

    def import_qualifier(spec: dict) -> str:
        # An `as` name replaces the module's original qualifier.
        return str(spec.get("alias") or spec.get("module", ""))

    def context_cache_key(context: dict | None) -> tuple:
        """Compact the caller fields that affect Cabal visibility decisions."""
        if not context:
            return ()
        components = []
        for component in context.get("components", []):
            components.append((
                component.get("kind"), component.get("name"),
                component.get("membership_certain"),
                tuple(component.get("unconditional_build_depends", [])),
            ))
        return (
            context.get("manifest"), context.get("package"),
            tuple(sorted(components, key=repr)),
        )

    # Identify modules whose complete import closure is acyclic. Repeated paths
    # through a diamond-shaped DAG can share recursive results safely; modules
    # in or leading to a cycle must continue to carry their path-specific
    # ``seen`` set. Unknown/external imports are leaves for this graph.
    known_modules = set(anchors_by_module)
    import_dependencies: dict[str, set[str]] = {
        module: {
            str(spec.get("module", ""))
            for anchor in module_anchors
            for spec in anchor.get("_haskell_imports", [])
            if str(spec.get("module", "")) in known_modules
        }
        for module, module_anchors in anchors_by_module.items()
    }
    import_parents: dict[str, set[str]] = {module: set() for module in known_modules}
    for module, dependencies in import_dependencies.items():
        for dependency in dependencies:
            import_parents[dependency].add(module)
    remaining_imports = {
        module: len(dependencies) for module, dependencies in import_dependencies.items()
    }
    pending_leaves = [module for module, count in remaining_imports.items() if count == 0]
    acyclic_import_modules: set[str] = set()
    while pending_leaves:
        leaf = pending_leaves.pop()
        if leaf in acyclic_import_modules:
            continue
        acyclic_import_modules.add(leaf)
        for parent in import_parents.get(leaf, ()):
            remaining_imports[parent] -= 1
            if remaining_imports[parent] == 0:
                pending_leaves.append(parent)

    candidate_cache: dict[tuple[str, str, str | None, tuple], frozenset[str]] = {}

    def module_candidates(module: str, symbol: str,
                          namespace: str | None = None,
                          caller_context: dict | None = None,
                          seen: frozenset[str] = frozenset()) -> set[str]:
        if module in seen:
            return set()
        context_key = context_cache_key(caller_context)
        cache_key = (module, symbol, namespace, context_key)
        cacheable = not seen or module in acyclic_import_modules
        if cacheable and cache_key in candidate_cache:
            return set(candidate_cache[cache_key])
        found = visible({
            nid for nid in direct.get((module, symbol), [])
            if in_namespace(node_by_id.get(nid), namespace)
        }, caller_context)
        visible_anchors = visible(
            {anchor["id"] for anchor in anchors_by_module.get(module, [])},
            caller_context,
        )
        for anchor in anchors_by_module.get(module, []):
            if anchor.get("id") not in visible_anchors:
                continue
            exports = anchor.get("_haskell_exports", {})
            imports = anchor.get("_haskell_imports", [])
            anchor_context = anchor.get("_haskell_cabal", {})
            imported_cache: dict[str, set[str]] = {}

            def imported_candidates(spec: dict) -> set[str]:
                # Several export rules can inspect the same import: explicit
                # names, unqualified scope, and one or more `module X` items.
                # The recursive candidate set is independent of that import's
                # list/hiding/qualification; those filters remain at each use.
                imported_module = str(spec["module"])
                if imported_module not in imported_cache:
                    imported_cache[imported_module] = module_candidates(
                        imported_module, symbol, namespace, anchor_context,
                        seen | {module},
                    )
                return imported_cache[imported_module]

            # An explicit export item may name an entity brought into scope by
            # an import. This is the common facade pattern (`Position(..)`),
            # and it composes transitively with other facades.
            export_names = exports.get("names", [])
            export_children = exports.get("children", {})
            may_export_named = (
                symbol in export_names
                or any(name.endswith(f".{symbol}") for name in export_names)
                or bool(exports.get("all_children"))
                or any(symbol in children for children in export_children.values())
            )
            if exports.get("explicit") and may_export_named:
                for spec in imports:
                    imported = imported_candidates(spec)
                    qualifier = import_qualifier(spec)
                    found.update(
                        nid for nid in imported
                        if allowed(spec, symbol, node_by_id.get(nid))
                        and (
                            explicitly_exported(
                                exports, symbol, node_by_id.get(nid), qualifier
                            )
                            or (
                                not spec.get("qualified")
                                and explicitly_exported(
                                    exports, symbol, node_by_id.get(nid)
                                )
                            )
                        )
                    )

            module_reexports = exports.get("modules", [])
            unqualified: set[str] = set()
            if module_reexports:
                # Haskell's `module X` form requires both X.name and the same
                # entity's unqualified name to be in scope. Build that
                # unqualified set across all imports before following every
                # import that supplies qualifier X. This also handles several
                # imports deliberately sharing an alias.
                for visible_spec in imports:
                    if visible_spec.get("qualified"):
                        continue
                    unqualified.update(
                        nid for nid in imported_candidates(visible_spec)
                        if allowed(visible_spec, symbol, node_by_id.get(nid))
                    )
            for reexport in module_reexports:
                for spec in imports:
                    if reexport != import_qualifier(spec):
                        continue
                    imported = imported_candidates(spec)
                    found.update(
                        nid for nid in imported
                        if nid in unqualified
                        and allowed(spec, symbol, node_by_id.get(nid))
                    )
        if cacheable:
            candidate_cache[cache_key] = frozenset(found)
        return found

    existing = {
        (edge.get("source"), edge.get("target"), edge.get("relation"), edge.get("source_location"))
        for edge in all_edges
    }

    def unresolved(label: str, namespace: str | None = None) -> str:
        # Encode the complete written name so `A.B`, `A_B`, operators, and
        # primed identifiers cannot collapse through generic ID normalization.
        encoded = f"{namespace or 'name'}:{label}".encode().hex()
        nid = _make_id("haskell_ref", encoded)
        if nid not in node_by_id:
            node = {
                "id": nid,
                "label": label,
                "file_type": "code",
                "source_file": "",
                "source_location": "",
                "_haskell_unresolved": True,
            }
            if namespace:
                node["_haskell_namespace"] = namespace
            all_nodes.append(node)
            node_by_id[nid] = node
        return nid

    def add_reference(raw: dict, target: str, confidence: str) -> None:
        relation = str(raw.get("relation") or "calls")
        key = (raw["caller_nid"], target, relation, raw["source_location"])
        if key in existing or raw["caller_nid"] == target:
            return
        existing.add(key)
        all_edges.append({
            "source": raw["caller_nid"],
            "target": target,
            "relation": relation,
            "context": raw.get("context", "call"),
            "confidence": confidence,
            "confidence_score": 1.0 if confidence == "EXTRACTED" else 0.85,
            "source_file": raw.get("source_file", ""),
            "source_location": raw.get("source_location"),
            "weight": 1.0,
        })

    import_edges: dict[tuple[str, str], list[dict]] = {}
    for edge in all_edges:
        if edge.get("relation") != "imports_from":
            continue
        current = node_by_id.get(edge.get("target"), {})
        import_edges.setdefault(
            (str(edge.get("source", "")), str(current.get("label", ""))), []
        ).append(edge)

    # Resolve module-level import edges to their exact internal module anchor.
    for anchor in anchors:
        caller_file = str(anchor.get("source_file", ""))
        for spec in anchor.get("_haskell_imports", []):
            choices = visible(
                {item["id"] for item in anchors_by_module.get(spec["module"], [])},
                anchor.get("_haskell_cabal", {}),
            )
            target = closest(choices, caller_file)
            if target is None:
                continue
            for edge in import_edges.get((str(anchor.get("id", "")), spec["module"]), []):
                edge["target"] = target

    raw_calls = [
        raw for result in per_file for raw in result.get("raw_calls", [])
        if raw.get("language") == "haskell"
    ]
    raw_references = [
        raw for result in per_file for raw in result.get("raw_references", [])
        if raw.get("language") == "haskell"
    ]
    for raw in [*raw_calls, *raw_references]:
        caller_file = str(raw.get("source_file", ""))
        anchor = anchors_by_source.get(caller_file)
        imports = anchor.get("_haskell_imports", []) if anchor else []
        symbol = str(raw.get("callee", ""))
        qualifier = raw.get("qualified_prefix")
        namespace = raw.get("namespace")
        if namespace is None and raw.get("relation", "calls") == "calls":
            namespace = "value"
        candidates: set[str] = set()
        referenced_modules: list[str] = []

        # Declarations in the caller's own module may be written either bare
        # or with the module's own qualifier (`Main.T`).
        own_module = str(anchor.get("_haskell_module", "")) if anchor else ""
        if anchor and (not qualifier or qualifier == own_module):
            candidates.update(
                nid for nid in local_direct.get((own_module, symbol), [])
                if in_namespace(node_by_id.get(nid), namespace)
                and str(node_by_id[nid].get("source_file", "")) == caller_file
            )

        if qualifier:
            for spec in imports:
                if qualifier != import_qualifier(spec):
                    continue
                referenced_modules.append(spec["module"])
                for nid in module_candidates(
                    spec["module"], symbol, namespace,
                    anchor.get("_haskell_cabal", {}) if anchor else {},
                ):
                    candidate = node_by_id.get(nid)
                    if allowed(spec, symbol, candidate):
                        candidates.add(nid)
        else:
            for spec in imports:
                if spec.get("qualified"):
                    continue
                module_ids = module_candidates(
                    spec["module"], symbol, namespace,
                    anchor.get("_haskell_cabal", {}) if anchor else {},
                )
                permitted = {
                    nid for nid in module_ids
                    if allowed(spec, symbol, node_by_id.get(nid))
                }
                if permitted:
                    referenced_modules.append(spec["module"])
                    candidates.update(permitted)

        target = closest(candidates, caller_file, require_same_module=True)
        if target is not None:
            add_reference(raw, target, "EXTRACTED")
            continue
        if referenced_modules:
            modules = sorted(set(referenced_modules))
            label = f"{modules[0]}.{symbol}" if len(modules) == 1 else symbol
        elif qualifier:
            label = f"{qualifier}.{symbol}"
        else:
            label = symbol
        add_reference(raw, unresolved(label, namespace), "INFERRED")

    # Import stubs that were repointed above have no remaining purpose.
    referenced = {
        endpoint for edge in all_edges
        for endpoint in (edge.get("source"), edge.get("target"))
    }
    all_nodes[:] = [
        node for node in all_nodes
        if not node.get("_haskell_import_stub") or node.get("id") in referenced
    ]
