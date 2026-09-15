"""Tests for studio.utils.severity — severity as a property of a finding.

Covers:
- DEFAULT_SEVERITY pinned as a whole-table golden, compared by value
- every error_codes constant has an entry, and no entry is orphaned
- default_severity: known code, unknown code, absent code
- both finding builders stamp severity from the table
- the required/optional code pairs carry different defaults
- the repo-wide invariant: the list a finding lands in equals its severity

The golden below is the point of this file. Asserting only that every code
*has* a default would let a rule ship as ``warning`` when it should be
``error``: every "fails on bad input" test would stay green, because those
tests assert exit codes and message text rather than the label itself. The
table is therefore compared by value, entry by entry.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pytest

from studio.utils import error_codes as EC
from studio.utils import severity as sev
from studio.utils.codebase import error as code_error
from studio.utils.constraints import error as constraints_error


# ---------------------------------------------------------------------------
# The golden table
# ---------------------------------------------------------------------------

EXPECTED_DEFAULT_SEVERITY: Dict[str, str] = {
    "LANG001": "error",
    "cdsl-code-syntax": "error",
    "cdsl-duplicate-inst-id": "error",
    "cdsl-incomplete-step-line": "warning",
    "cdsl-language-operator": "error",
    "cdsl-missing-checkbox": "warning",
    "cdsl-missing-inst-id": "warning",
    "cdsl-missing-phase-token": "warning",
    "cdsl-not-plain-english": "error",
    "cdsl-placeholder": "error",
    "cdsl-step-unchecked": "error",
    "cdsl-type-annotation": "error",
    "code-docs-only": "error",
    "code-inst-missing": "error",
    "code-inst-orphan": "error",
    "code-no-marker": "error",
    "code-orphan-ref": "error",
    "code-task-unchecked": "error",
    "codebase-entry-empty": "warning",
    "constraints-invalid": "error",
    "def-done-ref-not-done": "error",
    "def-missing-priority": "error",
    "def-missing-task": "error",
    "def-prohibited-priority": "error",
    "def-prohibited-task": "error",
    "def-wrong-headings": "error",
    "duplicate-definition": "error",
    "file-load-error": "error",
    "file-read-error": "error",
    "file-too-large": "error",
    "heading-missing": "error",
    "heading-number-not-consecutive": "error",
    "heading-numbering-mismatch": "error",
    "heading-prohibits-multiple": "error",
    "heading-requires-multiple": "error",
    "id-kind-not-allowed": "error",
    "id-not-referenced": "error",
    "id-not-referenced-no-scope": "warning",
    "id-system-unrecognized": "error",
    "kit-binding-error": "error",
    "kit-model-invalid": "error",
    "kit-path-not-accessible": "error",
    "kit-resource-path-not-found": "error",
    "kit-template-binding-missing": "warning",
    "marker-begin-no-end": "error",
    "marker-dup-begin": "error",
    "marker-dup-scope": "error",
    "marker-empty-block": "error",
    "marker-end-no-begin": "error",
    "missing-constraints": "error",
    "parent-checked-nested-unchecked": "error",
    "parent-unchecked-all-done": "error",
    "ref-done-def-not-done": "error",
    "ref-from-prohibited-kind": "error",
    "ref-missing-from-kind": "error",
    "ref-missing-priority": "error",
    "ref-missing-task": "error",
    "ref-missing-task-for-tracked": "error",
    "ref-no-definition": "error",
    "ref-prohibited-priority": "error",
    "ref-prohibited-task": "error",
    "ref-target-not-in-scope": "warning",
    "ref-task-def-no-task": "error",
    "ref-wrong-headings": "error",
    "registry-autodetect-failed": "error",
    "registry-autodetect-invalid": "error",
    "required-id-kind-missing": "error",
    "template-def-kind-not-in-constraints": "error",
    "template-def-placeholder-missing": "error",
    "template-def-placeholder-missing-optional": "warning",
    "template-def-placeholder-wrong-headings": "error",
    "template-id-kind-no-template": "error",
    "template-read-error": "error",
    "template-ref-kind-not-in-constraints": "error",
    "template-ref-placeholder-missing": "error",
    "template-ref-placeholder-missing-optional": "warning",
    "template-ref-placeholder-wrong-headings": "error",
    "toc-anchor-broken": "error",
    "toc-heading-depth-jump": "warning",
    "toc-heading-duplicate": "warning",
    "toc-heading-not-in-toc": "error",
    "toc-missing": "error",
    "toc-missing-description": "warning",
    "toc-section-too-long": "warning",
    "toc-stale": "warning",
}


def _all_code_constants() -> Dict[str, str]:
    return {
        name: value
        for name, value in vars(EC).items()
        if name.isupper() and isinstance(value, str)
    }


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------

def test_default_severity_matches_the_golden_table_by_value():
    """Every default is pinned. Changing one must fail here, by name."""
    assert sev.DEFAULT_SEVERITY == EXPECTED_DEFAULT_SEVERITY


def test_every_error_code_constant_has_a_default_severity():
    codes = set(_all_code_constants().values())
    assert codes - set(sev.DEFAULT_SEVERITY) == set()


def test_table_has_no_entry_for_a_code_that_does_not_exist():
    codes = set(_all_code_constants().values())
    assert set(sev.DEFAULT_SEVERITY) - codes == set()


def test_every_default_is_a_member_of_the_vocabulary():
    assert set(sev.DEFAULT_SEVERITY.values()) <= set(sev.VALIDATION_SEVERITIES)


def test_no_code_defaults_to_off_yet():
    """``off`` is declared vocabulary; suppression arrives with configuration.

    If this starts failing, a rule was made non-blocking by default — which is
    a behaviour change, not a refactor, and belongs in its own review.
    """
    assert "off" not in set(sev.DEFAULT_SEVERITY.values())


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("code,expected", [
    (EC.TOC_STALE, "warning"),
    (EC.CDSL_MISSING_INST_ID, "warning"),
    (EC.HEADING_MISSING, "error"),
    (EC.MARKER_DUP_BEGIN, "error"),
])
def test_default_severity_resolves_a_known_code(code: str, expected: str):
    assert sev.default_severity(code) == expected


@pytest.mark.parametrize("code", [None, "", "no-such-code-exists"])
def test_an_absent_or_unknown_code_resolves_to_error(code):
    """Fail closed: an unrecognised rule must not become silently non-blocking."""
    assert sev.default_severity(code) == "error"


# ---------------------------------------------------------------------------
# The two-code split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("required_code,optional_code", [
    (EC.TEMPLATE_DEF_PLACEHOLDER_MISSING, EC.TEMPLATE_DEF_PLACEHOLDER_MISSING_OPTIONAL),
    (EC.TEMPLATE_REF_PLACEHOLDER_MISSING, EC.TEMPLATE_REF_PLACEHOLDER_MISSING_OPTIONAL),
])
def test_required_and_optional_placeholders_are_two_codes_at_two_defaults(
    required_code: str, optional_code: str
):
    """One code must mean one default severity.

    These sites previously emitted a single codeless finding routed to either
    list by a ``required`` flag. A single code would make severity depend on
    call-site control flow again, which is exactly what this model removes.
    """
    assert required_code != optional_code
    assert sev.default_severity(required_code) == "error"
    assert sev.default_severity(optional_code) == "warning"


# ---------------------------------------------------------------------------
# Both builders stamp
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [constraints_error, code_error], ids=["constraints", "codebase"])
def test_builder_stamps_severity_from_the_table(builder):
    warn = builder("toc", "m", path=Path("a.md"), line=1, code=EC.TOC_STALE)
    err = builder("structure", "m", path=Path("a.md"), line=1, code=EC.HEADING_MISSING)
    assert warn["severity"] == "warning"
    assert err["severity"] == "error"


@pytest.mark.parametrize("builder", [constraints_error, code_error], ids=["constraints", "codebase"])
def test_builder_stamps_error_when_no_code_is_supplied(builder):
    finding = builder("structure", "m", path=Path("a.md"), line=1)
    assert finding["severity"] == "error"


def test_severity_is_stamped_not_defaulted_by_keyword():
    """A keyword default would silently agree with whatever the call site did.

    Stamping from the table is what makes the equivalence testable, so a
    warning-by-default code must come out as ``warning`` even though the
    builder was given no severity argument at all.
    """
    finding = constraints_error("toc", "m", path=Path("a.md"), code=EC.TOC_HEADING_DUPLICATE)
    assert finding["severity"] == "warning"


# ---------------------------------------------------------------------------
# The table is the only source of severity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("builder", [constraints_error, code_error], ids=["constraints", "codebase"])
@pytest.mark.parametrize("supplied", ["off", "warning", "error"])
def test_a_caller_cannot_override_the_stamped_severity(builder, supplied):
    """`**extra` is splatted over the finding after the stamp, so a call site
    passing `severity=` would silently beat the table. No call site does; this
    makes sure none can start to without being told."""
    with pytest.raises(TypeError, match="derived from the finding's code"):
        builder("toc", "m", path=Path("a.md"), code=EC.TOC_STALE, severity=supplied)


def test_other_extras_still_pass_through():
    """The guard must reject exactly one key, not harden the builder generally."""
    finding = constraints_error(
        "template", "m", path=Path("a.md"), code=EC.TEMPLATE_READ_ERROR,
        kit_id="sdlc", artifact_kind="PRD",
    )
    assert finding["kit_id"] == "sdlc"
    assert finding["artifact_kind"] == "PRD"
    assert finding["severity"] == "error"


# ---------------------------------------------------------------------------
# Each newly coded branch emits its code, at its declared severity
#
# Deterministic and self-contained: these call the emitting helpers directly
# with synthetic inputs, so they neither depend on this repository's own
# content nor shell out. Each fails if its call site loses the `code=` kwarg or
# names the wrong constant.
# ---------------------------------------------------------------------------

def test_missing_kit_resource_path_is_coded_and_an_error(tmp_path):
    from studio.commands.validate_kits import _missing_resource_binding_errors

    errors = _missing_resource_binding_errors("sdlc", {"prd-template": str(tmp_path / "nope.md")})
    assert [(e["code"], e["severity"]) for e in errors] == [
        (EC.KIT_RESOURCE_PATH_NOT_FOUND, "error")
    ]


def test_unbindable_artifact_kind_is_coded_and_advisory():
    """The one advisory member of the newly coded set.

    validate-kits reports this with status PASS and warning_count 1, so an
    `error` default here would turn a passing kit check into a failing one.
    """
    from studio.commands.validate_kits import _missing_bound_artifact_warnings

    results = _missing_bound_artifact_warnings(kit_id="sdlc", known_kinds={"PRD"}, artifacts={})
    warning = results[0]["warnings"][0]
    assert (warning["code"], warning["severity"]) == (EC.KIT_TEMPLATE_BINDING_MISSING, "warning")
    assert results[0]["status"] == "PASS"


def test_unloadable_constraints_is_coded_and_an_error(tmp_path):
    from studio.commands.self_check import _append_constraints_load_failure

    results: List[dict] = []
    _append_constraints_load_failure(
        results, kit_id="sdlc", kit_base=tmp_path,
        constraints_path=None, constraint_errors=["bad table"],
    )
    finding = results[0]["errors"][0]
    assert (finding["code"], finding["severity"]) == (EC.CONSTRAINTS_INVALID, "error")


def test_id_kind_without_a_template_is_coded_and_an_error(tmp_path):
    from studio.commands.self_check import _append_missing_template_id_issue

    issues: Dict[str, List[dict]] = {"errors": [], "warnings": []}
    _append_missing_template_id_issue(
        issues, template_path=tmp_path / "t.md", kit_id="sdlc", kind_u="PRD", id_kind="fr",
    )
    finding = issues["errors"][0]
    assert (finding["code"], finding["severity"]) == (EC.TEMPLATE_ID_KIND_NO_TEMPLATE, "error")


@pytest.mark.parametrize("required,expected_code,expected_severity,expected_list", [
    (True, EC.TEMPLATE_DEF_PLACEHOLDER_MISSING, "error", "errors"),
    (False, EC.TEMPLATE_DEF_PLACEHOLDER_MISSING_OPTIONAL, "warning", "warnings"),
])
def test_missing_definition_placeholder_splits_by_required(
    tmp_path, required, expected_code, expected_severity, expected_list
):
    """The two-code split, exercised at its real call site.

    This is the branch that previously emitted one codeless finding routed by
    `required`; the code must track the flag, or severity silently depends on
    call-site control flow again.
    """
    from studio.commands.self_check import _append_missing_definition_placeholder

    issues: Dict[str, List[dict]] = {"errors": [], "warnings": []}
    _append_missing_definition_placeholder(
        issues, required=required, template_path=tmp_path / "t.md",
        kit_id="sdlc", kind_u="PRD", id_kind="fr", template_id="cpt-{system}-fr-{slug}",
    )
    finding = issues[expected_list][0]
    assert (finding["code"], finding["severity"]) == (expected_code, expected_severity)


@pytest.mark.parametrize("required,expected_code,expected_severity", [
    (True, EC.TEMPLATE_REF_PLACEHOLDER_MISSING, "error"),
    (False, EC.TEMPLATE_REF_PLACEHOLDER_MISSING_OPTIONAL, "warning"),
])
def test_missing_reference_placeholder_splits_by_required(
    tmp_path, required, expected_code, expected_severity
):
    from studio.commands.self_check import _append_missing_reference_issue

    target = "errors" if required else "warnings"
    issues: Dict[str, List[dict]] = {"errors": [], "warnings": []}
    _append_missing_reference_issue(
        issues, target, required=required, template_path=tmp_path / "t.md",
        kit_id="sdlc", kind_u="PRD", id_kind="fr", template_id="cpt-{system}-fr-{slug}",
    )
    finding = issues[target][0]
    assert (finding["code"], finding["severity"]) == (expected_code, expected_severity)


# ---------------------------------------------------------------------------
# The stamped severity reaches the human report
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_error,code,expected", [
    (True, EC.HEADING_MISSING, "error"),
    (False, EC.TOC_STALE, "warning"),
])
def test_human_output_renders_the_stamped_severity(capsys, is_error, code, expected):
    """A stamped severity that never reaches the reader is not reported.

    ``severity`` was briefly listed in the human formatter's ``handled_keys``
    without anything rendering it, which suppressed it from validate output
    entirely — the worst of both, since the key looked handled. Pin it.
    """
    from studio.commands.validate import _format_issue
    from studio.utils.ui import set_json_mode

    set_json_mode(False)
    try:
        _format_issue(
            constraints_error("toc", "m", path=Path("a.md"), line=3, code=code),
            is_error=is_error,
        )
    finally:
        set_json_mode(True)

    assert f"severity: {expected}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The repo-wide invariant
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


#: Generous enough that only a genuine hang trips it — a full validate of this
#: repository runs in well under a second. Without it a deadlocked child would
#: block the run forever, and CI would report a timeout on the whole job rather
#: than on the test that caused it.
_VALIDATE_TIMEOUT_SECONDS = 300


@pytest.fixture(scope="module")
def repo_validate_report() -> Dict[str, object]:
    """Run this repository's own validate once and share the report.

    Uses ``sys.executable`` rather than a bare ``python3``: the interpreter
    first on PATH is commonly older than this project supports, and a run that
    silently measured the wrong thing is precisely the failure this module
    exists to make impossible.
    """
    import subprocess  # local: only this fixture shells out
    import sys

    try:
        proc = subprocess.run(
            [sys.executable, "skills/studio/scripts/studio.py", "validate", "--json", "--verbose"],
            cwd=_repo_root(), capture_output=True, text=True, check=False,
            timeout=_VALIDATE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"validate did not finish within {_VALIDATE_TIMEOUT_SECONDS}s — treat as a hang, "
            f"not a slow machine; it normally completes in under a second. "
            f"stderr tail: {(exc.stderr or b'')[-400:]!r}"
        ) from exc
    assert proc.stdout.strip(), f"validate produced no stdout (exit {proc.returncode}): {proc.stderr[:400]}"
    return json.loads(proc.stdout)


@pytest.mark.integration
def test_list_membership_equals_stamped_severity_over_this_repo(repo_validate_report):
    """The invariant the whole model rests on, measured on real output."""
    report = repo_validate_report

    mismatches: List[str] = []
    for list_name, expected in (("errors", "error"), ("warnings", "warning")):
        for finding in report.get(list_name) or []:
            if finding.get("severity") != expected:
                mismatches.append(
                    f"{finding.get('code')} in {list_name} stamped {finding.get('severity')}"
                )
    assert not mismatches, mismatches


@pytest.mark.integration
def test_every_finding_this_repo_emits_carries_a_code(repo_validate_report):
    report = repo_validate_report
    findings = (report.get("errors") or []) + (report.get("warnings") or [])
    assert findings, "validate produced no findings — the invariant would be vacuous"
    assert [f for f in findings if not f.get("code")] == []
