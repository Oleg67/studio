"""A traceability identifier written as a markdown link.

A reference may be spelled bare or as ``[`cpt-id`](target)``; both name the same
node, because the node is the id string. A *definition* may only be spelled bare,
and a link-form one is reported rather than quietly filed as a reference to itself.

Pins issue #177 AC1-AC3 and the narrowness of the accepted form.
"""
from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "studio" / "scripts"))

from studio.commands.where_used import cmd_where_used
from studio.utils import error_codes as EC
from studio.utils.constraints import parse_kit_constraints, validate_artifact_file
from studio.utils.document import LINK_FORM_DEFINITION, scan_cpt_id_lines
from studio.utils.ui import is_json_mode, set_json_mode

TARGET = "cpt-myapp-flow-login"


# --------------------------------------------------------------- the scan (AC2)
@pytest.mark.parametrize("bare, linked", [
    ("`{id}`", "[`{id}`](spec.md)"),
    ("[x] `p1` - `{id}`", "[x] `p1` - [`{id}`](spec.md#login)"),
    ("[ ] `p2` - `{id}`", "[ ] `p2` - [`{id}`](../other/spec.md)"),
    ("`p1` - `{id}`", "`p1` - [`{id}`](spec.md 'Login flow')"),
])
def test_a_link_form_reference_carries_what_the_bare_form_carries(bare, linked):
    """AC2: task marker, priority and checkbox survive the link.

    They were lost before, because a link-form reference only ever matched the
    inline backtick scan, which stamps an unconditional ``checked=False``.
    """
    bare_hits = scan_cpt_id_lines([bare.format(id=TARGET)])
    linked_hits = scan_cpt_id_lines([linked.format(id=TARGET)])
    assert linked_hits == bare_hits


def test_both_spellings_resolve_to_the_same_node():
    hits = scan_cpt_id_lines([f"`{TARGET}`", "", f"[`{TARGET}`](spec.md#login)"])
    assert [h["id"] for h in hits] == [TARGET, TARGET]
    assert [h["type"] for h in hits] == ["reference", "reference"]
    assert [h["line"] for h in hits] == [1, 3]


@pytest.mark.parametrize("line", [
    f"[the login flow](spec.md#{TARGET})",
    f"[the login flow]({TARGET}.md)",
])
def test_only_the_backticked_form_is_a_reference(line):
    """The accepted form is narrow on purpose: the id has to be marked up as an id.

    Any link whose *target* merely contains the string stays prose — inferring a
    reference from a URL would make every path that names an id a reference to it.
    """
    assert scan_cpt_id_lines([line]) == []


# ---------------------------------------------------- link-form definition (AC3)
def test_a_link_form_definition_is_neither_a_definition_nor_a_reference():
    hits = scan_cpt_id_lines([f"**ID**: [`{TARGET}`](spec.md)"])
    assert [h["type"] for h in hits] == [LINK_FORM_DEFINITION]
    assert hits[0]["id"] == TARGET


@pytest.mark.parametrize("line", [
    "**ID**: [`{id}`](spec.md)",
    "`p1` - **ID**: [`{id}`](spec.md)",
    "- [x] `p1` - **ID**: [`{id}`](spec.md#login)",
])
def test_every_definition_shape_is_recognised_in_link_form(line):
    hits = scan_cpt_id_lines([line.format(id=TARGET)])
    assert [h["type"] for h in hits] == [LINK_FORM_DEFINITION]


def _prd_constraints():
    kit, errors = parse_kit_constraints({"PRD": {"identifiers": {"flow": {}}}})
    assert errors == []
    return kit.by_kind["PRD"]


def test_validate_reports_a_link_form_definition(tmp_path: Path):
    """AC3 end to end: one finding naming the line, not silence."""
    artifact = tmp_path / "PRD.md"
    artifact.write_text(f"# PRD\n\n**ID**: [`{TARGET}`](spec.md)\n", encoding="utf-8")

    report = validate_artifact_file(
        artifact_path=artifact,
        artifact_kind="PRD",
        constraints=_prd_constraints(),
        registered_systems={"myapp"},
    )
    findings = [e for e in report["errors"] if e.get("code") == EC.DEF_LINK_FORM_NOT_ALLOWED]
    assert len(findings) == 1
    assert findings[0]["line"] == 3
    assert TARGET in str(findings[0]["message"])


def test_validate_leaves_a_bare_definition_alone(tmp_path: Path):
    artifact = tmp_path / "PRD.md"
    artifact.write_text(f"# PRD\n\n**ID**: `{TARGET}`\n", encoding="utf-8")

    report = validate_artifact_file(
        artifact_path=artifact,
        artifact_kind="PRD",
        constraints=_prd_constraints(),
        registered_systems={"myapp"},
    )
    assert [e for e in report["errors"] if e.get("code") == EC.DEF_LINK_FORM_NOT_ALLOWED] == []


def test_a_link_form_definition_does_not_define_the_id(tmp_path: Path):
    """The id really is undefined — that is why the line is worth reporting."""
    artifact = tmp_path / "PRD.md"
    artifact.write_text(f"# PRD\n\n**ID**: [`{TARGET}`](spec.md)\n", encoding="utf-8")

    hits = scan_cpt_id_lines(artifact.read_text(encoding="utf-8").splitlines())
    assert [h for h in hits if h["type"] == "definition"] == []


# ------------------------------------------------------------- where-used (AC1)
def _where_used(artifact: Path, argv: list[str]) -> dict:
    saved = is_json_mode()
    stdout = io.StringIO()
    try:
        set_json_mode(True)
        with patch(
            "studio.commands.where_used.resolve_target_and_artifacts",
            return_value=(TARGET, object(), [(artifact, "FEATURE")], {}, None),
        ), redirect_stdout(stdout):
            assert cmd_where_used(argv) == 0
        return json.loads(stdout.getvalue())
    finally:
        set_json_mode(saved)


def test_where_used_lists_the_bare_and_the_link_form_once_each(tmp_path: Path):
    """AC1: two spellings of one reference, two places to look, one record each."""
    artifact = tmp_path / "FEATURE.md"
    artifact.write_text(
        f"# Feature\n\n`{TARGET}`\n\n[`{TARGET}`](../prd/PRD.md#login)\n",
        encoding="utf-8",
    )
    data = _where_used(artifact, [TARGET])
    assert data["count"] == 2
    assert [r["line"] for r in data["references"]] == [3, 5]


def test_where_used_counts_one_line_once_however_often_it_names_the_id(tmp_path: Path):
    artifact = tmp_path / "FEATURE.md"
    artifact.write_text(
        f"# Feature\n\nThe `{TARGET}` flow supersedes `{TARGET}` in the old plan.\n",
        encoding="utf-8",
    )
    data = _where_used(artifact, [TARGET])
    assert data["count"] == 1
    assert data["references"][0]["line"] == 3


def test_where_used_does_not_count_a_link_form_definition_as_a_use(tmp_path: Path):
    artifact = tmp_path / "FEATURE.md"
    artifact.write_text(f"# Feature\n\n**ID**: [`{TARGET}`](spec.md)\n", encoding="utf-8")
    data = _where_used(artifact, [TARGET, "--include-definitions"])
    assert data["count"] == 0
