from pathlib import Path

from graphify.extractors.haskell_cabal import (
    file_component_context,
    filter_visible_candidate_paths,
    filter_visible_contexts,
)


def _write(root: Path, relative: str, text: str = "") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def test_component_membership_and_duplicate_main_are_separate(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: src
  exposed-modules: App
executable one
  hs-source-dirs: app/one
  build-depends: demo
executable two
  hs-source-dirs: app/two
  build-depends: demo
""")
    app = _write(tmp_path, "src/App.hs")
    one = _write(tmp_path, "app/one/Main.hs")
    two = _write(tmp_path, "app/two/Main.hs")
    assert file_component_context(one)["components"][0]["name"] == "one"
    assert file_component_context(two)["components"][0]["name"] == "two"
    assert filter_visible_candidate_paths(one, [str(app), str(two)]) == {str(app)}


def test_common_stanza_and_internal_library_dependency(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
common shared
  build-depends: demo:util
library util
  hs-source-dirs: util
  exposed-modules: Util
test-suite spec
  import: shared
  hs-source-dirs: test
""")
    util = _write(tmp_path, "util/Util.hs")
    spec = _write(tmp_path, "test/Main.hs")
    context = file_component_context(spec)
    assert context["components"][0]["build_depends"] == ["demo:util"]
    assert filter_visible_candidate_paths(spec, [str(util)]) == {str(util)}


def test_unrelated_package_without_dependency_is_blocked(tmp_path):
    caller_root = tmp_path / "caller"
    other_root = tmp_path / "other"
    _write(caller_root, "caller.cabal", "name: caller\nexecutable app\n  hs-source-dirs: app\n")
    caller = _write(caller_root, "app/Main.hs")
    _write(other_root, "other.cabal", "name: other\nlibrary\n  hs-source-dirs: src\n  exposed-modules: App\n")
    other = _write(other_root, "src/App.hs")
    assert filter_visible_candidate_paths(caller, [str(other)]) == set()


def test_conditionals_union_unknown_branches(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
executable app
  hs-source-dirs: app
  if flag(new)
    build-depends: new-api
  else
    build-depends: old-api
""")
    main = _write(tmp_path, "app/Main.hs")
    component = file_component_context(main)["components"][0]
    assert set(component["build_depends"]) == {"new-api", "old-api"}
    assert component["conditional"] is True


def test_common_conditional_and_nonstandard_main_is(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
common choices
  if flag(new)
    build-depends: new-api
  else
    build-depends: old-api
executable app
  import: choices
  hs-source-dirs: app
  main-is: Start.hs
  other-modules: Helper
""")
    start = _write(tmp_path, "app/Start.hs")
    component = file_component_context(start)["components"][0]
    assert set(component["build_depends"]) == {"new-api", "old-api"}
    assert component["conditional"] is True


def test_cache_invalidates_when_cabal_stat_changes(tmp_path):
    cabal = _write(tmp_path, "demo.cabal", "name: demo\nexecutable old\n  hs-source-dirs: app\n")
    main = _write(tmp_path, "app/Main.hs")
    assert file_component_context(main)["components"][0]["name"] == "old"
    cabal.write_text("name: demo\nexecutable newer-name\n  hs-source-dirs: app\n")
    assert file_component_context(main)["components"][0]["name"] == "newer-name"


def test_context_is_portable_and_filters_without_source_paths(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: src
  exposed-modules: Api
executable app
  hs-source-dirs: app
  build-depends: demo
""")
    api_context = file_component_context(_write(tmp_path, "src/Api.hs"))
    app_context = file_component_context(_write(tmp_path, "app/Main.hs"))
    assert app_context["manifest"] == "demo.cabal"
    assert str(tmp_path) not in repr(app_context)
    assert filter_visible_contexts(app_context, [("candidate-id", api_context)]) == {
        "candidate-id"
    }


def test_conditional_dependency_does_not_certify_visibility(tmp_path):
    dependency = tmp_path / "dependency"
    caller = tmp_path / "caller"
    _write(dependency, "maybe.cabal", "name: maybe\nlibrary\n  hs-source-dirs: src\n  exposed-modules: Api\n")
    api_context = file_component_context(_write(dependency, "src/Api.hs"))
    _write(caller, "caller.cabal", """name: caller
executable app
  hs-source-dirs: app
  if flag(with-api)
    build-depends: maybe
""")
    app_context = file_component_context(_write(caller, "app/Main.hs"))
    assert filter_visible_contexts(app_context, [("maybe-api", api_context)]) == set()


def test_unconditional_dependency_survives_optional_dependency(tmp_path):
    dependency = tmp_path / "dependency"
    caller = tmp_path / "caller"
    _write(dependency, "stable.cabal", "name: stable\nlibrary\n  hs-source-dirs: src\n  exposed-modules: Api\n")
    api_context = file_component_context(_write(dependency, "src/Api.hs"))
    _write(caller, "caller.cabal", """name: caller
executable app
  hs-source-dirs: app
  build-depends: stable
  if flag(extra)
    build-depends: optional-api
""")
    app_context = file_component_context(_write(caller, "app/Main.hs"))
    assert filter_visible_contexts(app_context, [("stable-api", api_context)]) == {
        "stable-api"
    }


def test_nested_common_imports_expand_and_terminate_cycles(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
common base
  import: loop
  build-depends: demo
common loop
  import: base
  build-depends: text
common outer
  import: base
  build-depends: bytestring
executable app
  import: outer
  hs-source-dirs: app
""")
    component = file_component_context(_write(tmp_path, "app/Main.hs"))["components"][0]
    assert set(component["unconditional_build_depends"]) == {
        "demo", "text", "bytestring"
    }


def test_main_is_subpath_matches_relative_to_source_dir(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
executable app
  hs-source-dirs: app
  main-is: Driver/Start.hs
  other-modules: Helper
""")
    context = file_component_context(_write(tmp_path, "app/Driver/Start.hs"))
    assert context["components"][0]["membership_certain"] is True


def test_package_dependency_does_not_expose_named_sublibrary(tmp_path):
    dependency = tmp_path / "dependency"
    caller = tmp_path / "caller"
    _write(dependency, "demo.cabal", """name: demo
library util
  hs-source-dirs: util
  exposed-modules: Util
""")
    util_context = file_component_context(_write(dependency, "util/Util.hs"))
    _write(caller, "caller.cabal", """name: caller
executable app
  hs-source-dirs: app
  build-depends: demo
""")
    app_context = file_component_context(_write(caller, "app/Main.hs"))
    assert filter_visible_contexts(app_context, [("util", util_context)]) == set()


def test_external_sublibrary_requires_explicit_dependency_and_public_visibility(tmp_path):
    dependency = tmp_path / "dependency"
    caller = tmp_path / "caller"
    _write(dependency, "demo.cabal", """name: demo
library public-util
  visibility: public
  hs-source-dirs: public
  exposed-modules: PublicUtil
library private-util
  hs-source-dirs: private
  exposed-modules: PrivateUtil
""")
    public_context = file_component_context(_write(dependency, "public/PublicUtil.hs"))
    private_context = file_component_context(_write(dependency, "private/PrivateUtil.hs"))
    _write(caller, "caller.cabal", """name: caller
executable app
  hs-source-dirs: app
  build-depends: demo:public-util, demo:private-util
""")
    app_context = file_component_context(_write(caller, "app/Main.hs"))
    assert filter_visible_contexts(app_context, [
        ("public", public_context), ("private", private_context)
    ]) == {"public"}


def test_known_nonmember_fails_closed_but_missing_manifest_remains_unknown(tmp_path):
    package = tmp_path / "package"
    _write(package, "demo.cabal", """name: demo
executable app
  hs-source-dirs: app
""")
    caller = file_component_context(_write(package, "app/Main.hs"))
    known_nonmember = file_component_context(_write(package, "scratch/Unused.hs"))
    unknown = file_component_context(_write(tmp_path / "loose", "Loose.hs"))
    assert known_nonmember["manifest"] == "demo.cabal"
    assert known_nonmember["components"] == []
    assert filter_visible_contexts(caller, [
        ("known", known_nonmember), ("unknown", unknown)
    ]) == {"unknown"}
    assert filter_visible_contexts(known_nonmember, [("unknown", unknown)]) == set()


def test_conditional_common_import_does_not_create_unconditional_dependency(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
common extra
  build-depends: optional-api
common nested
  if flag(extra)
    import: extra
executable app
  hs-source-dirs: app
  if flag(nested)
    import: nested
""")
    component = file_component_context(_write(tmp_path, "app/Main.hs"))["components"][0]
    assert "optional-api" in component["build_depends"]
    assert "optional-api" not in component["unconditional_build_depends"]


def test_nested_condition_keeps_outer_fields_conditional(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
executable app
  hs-source-dirs: app
  if flag(outer)
    if flag(inner)
      build-depends: inner-api
    build-depends: still-outer-api
  build-depends: stable-api
""")
    component = file_component_context(_write(tmp_path, "app/Main.hs"))["components"][0]
    assert set(component["unconditional_build_depends"]) == {"stable-api"}


def test_multiple_nearest_manifests_are_explicitly_ambiguous(tmp_path):
    _write(tmp_path, "one.cabal", "name: one\nlibrary\n  hs-source-dirs: src\n")
    _write(tmp_path, "two.cabal", "name: two\nlibrary\n  hs-source-dirs: src\n")
    context = file_component_context(_write(tmp_path, "src/Thing.hs"))
    assert context == {
        "manifest": "one.cabal,two.cabal",
        "package": "",
        "components": [],
        "ambiguous": True,
    }
    assert filter_visible_contexts(context, [("candidate", {})]) == set()


def test_conditional_only_caller_membership_cannot_certify_dependency(tmp_path):
    dependency = tmp_path / "dependency"
    caller = tmp_path / "caller"
    _write(dependency, "api.cabal", "name: api\nlibrary\n  hs-source-dirs: src\n  exposed-modules: Api\n")
    api_context = file_component_context(_write(dependency, "src/Api.hs"))
    _write(caller, "caller.cabal", """name: caller
executable app
  if flag(build-app)
    hs-source-dirs: app
  build-depends: api
""")
    app_context = file_component_context(_write(caller, "app/Main.hs"))
    assert app_context["components"][0]["membership_certain"] is False
    assert filter_visible_contexts(app_context, [("api", api_context)]) == set()


def test_unlisted_home_module_uses_most_specific_source_component(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: .
  exposed-modules: Api
test-suite auth
  hs-source-dirs: test-auth, test
  other-modules: Auth.OAuth2.TypesSpec
  build-depends: demo
""")
    # The written module intentionally differs from the stale other-modules
    # entry, matching NeoHaskell's Auth.OAuth2TypesSpec regression.
    context = file_component_context(_write(tmp_path, "test-auth/Auth/OAuth2TypesSpec.hs"))
    assert [(item["kind"], item["name"]) for item in context["components"]] == [
        ("test-suite", "auth")
    ]
    assert context["components"][0]["membership_certain"] is True
    assert context["components"][0]["inferred_from_source_dir"] is True


def test_unconditional_no_field_selectors_propagates_from_common_stanza(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
common defaults
  default-extensions: NoFieldSelectors, OverloadedStrings
library
  import: defaults
  hs-source-dirs: src
  exposed-modules: Api
""")
    component = file_component_context(_write(tmp_path, "src/Api.hs"))["components"][0]
    assert component["no_field_selectors"] is True


def test_conditional_no_field_selectors_does_not_certify_extension(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: src
  exposed-modules: Api
  if flag(new-records)
    default-extensions: NoFieldSelectors
""")
    component = file_component_context(_write(tmp_path, "src/Api.hs"))["components"][0]
    assert component["no_field_selectors"] is False


def test_conditional_field_selectors_prevents_no_field_selector_certainty(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: src
  exposed-modules: Api
  default-extensions: NoFieldSelectors
  if flag(legacy-selectors)
    default-extensions: FieldSelectors
""")
    component = file_component_context(_write(tmp_path, "src/Api.hs"))["components"][0]
    assert component["no_field_selectors"] is False


def test_source_field_selectors_pragma_overrides_cabal_default(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: src
  exposed-modules: Api
  default-extensions: NoFieldSelectors
""")
    source = _write(
        tmp_path, "src/Api.hs",
        "{-# LANGUAGE FieldSelectors #-}\nmodule Api where\n",
    )
    assert file_component_context(source)["components"][0]["no_field_selectors"] is False


def test_source_no_field_selectors_multiline_pragma_certifies_override(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: src
  exposed-modules: Api
""")
    source = _write(
        tmp_path, "src/Api.hs",
        "{-# LANGUAGE\n  OverloadedStrings,\n  NoFieldSelectors\n#-}\nmodule Api where\n",
    )
    assert file_component_context(source)["components"][0]["no_field_selectors"] is True


def test_source_options_ghc_field_selector_overrides(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: src
  exposed-modules: Enabled Disabled
  default-extensions: NoFieldSelectors
""")
    enabled = _write(
        tmp_path, "src/Enabled.hs",
        "{-# OPTIONS_GHC -Wall -XFieldSelectors #-}\nmodule Enabled where\n",
    )
    disabled = _write(
        tmp_path, "src/Disabled.hs",
        "{-# OPTIONS_GHC -XNoFieldSelectors #-}\nmodule Disabled where\n",
    )
    assert file_component_context(enabled)["components"][0]["no_field_selectors"] is False
    assert file_component_context(disabled)["components"][0]["no_field_selectors"] is True


def test_commented_pragmas_do_not_certify_no_field_selectors(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
library
  hs-source-dirs: src
  exposed-modules: Api
""")
    source = _write(
        tmp_path, "src/Api.hs",
        "-- {-# LANGUAGE NoFieldSelectors #-}\n"
        "{- example: {-# OPTIONS_GHC -XNoFieldSelectors #-} -}\n"
        "module Api where\n",
    )
    assert file_component_context(source)["components"][0]["no_field_selectors"] is False


def test_implicit_main_runtime_string_is_not_read_as_language_pragma(tmp_path):
    _write(tmp_path, "demo.cabal", """name: demo
executable app
  hs-source-dirs: app
  main-is: Main.hs
""")
    source = _write(
        tmp_path, "app/Main.hs",
        'main = putStrLn "{-# LANGUAGE NoFieldSelectors #-}"\n',
    )
    assert file_component_context(source)["components"][0]["no_field_selectors"] is False
