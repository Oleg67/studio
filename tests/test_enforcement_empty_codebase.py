"""An empty scan is not a pass: `validate` must reach its checks on 0 files.

Every check needed here already exists and is correct. What this corpus covers
is *reachability*: with a registered FULL codebase entry that resolves to no
files, cross-validation used to be skipped wholesale, so a FEATURE claiming
implemented code validated clean. Adding a single unmarked `.py` file to the
same repository produced the right errors, which is the evidence that the gap
was the 0-file bypass rather than missing validation.

Three questions are kept apart, matching the report semantics used by
`spec-coverage`:

* **"did anything fail?"** -- a repository that claims nothing owes nothing,
  even with no code at all. A spec-first project must stay green.
* **"was there anything to assess?"** -- a registered entry resolving to no
  files is a configuration mistake and is reported as one, whether or not any
  ID is claimed.
* **"was a guarantee demanded that cannot be given?"** -- a checked
  `to_code = true` ID with no marker anywhere is an unmet claim, and the scan
  size is irrelevant to that.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent.parent / "skills" / "studio" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

from _test_helpers import run_cli_in_project, write_constraints_toml

CLAIMED_ID = "cpt-test-flow-login"
# `algo` is not configured to_code, so this ID declares work without claiming code.
UNCLAIMED_ID = "cpt-test-algo-bucket"


def _project(
    root: Path,
    *,
    claim_id: bool,
    code: dict[str, str] | None = None,
    codebase_path: str | None = "src",
    declare_instructions: bool = False,
) -> None:
    """Build a project with a FULL artifact and a registered codebase entry.

    ``claim_id`` controls whether the artifact checks a ``to_code = true`` ID --
    that is, whether anything is being claimed. ``code`` maps relative paths to
    file contents; omitting it leaves the registered tree empty, which is the
    state under test. ``declare_instructions`` adds checked CDSL steps under an
    ID that is *not* ``to_code``, which is what an over-broad fix would turn
    into a wall of missing-instruction errors.
    """
    from studio.utils import toml_utils

    (root / "kits" / "sdlc").mkdir(parents=True)
    write_constraints_toml(
        root / "kits" / "sdlc",
        {
            "PRD": {
                "identifiers": {
                    # `flow` claims code; `algo` declares work without claiming it.
                    "flow": {"to_code": True},
                    "algo": {"to_code": False, "required": False},
                }
            }
        },
    )

    art_dir = root / "architecture"
    art_dir.mkdir(parents=True)
    checkbox = "x" if claim_id else " "
    content = f"- [{checkbox}] **ID**: `{CLAIMED_ID}`\n"
    if declare_instructions:
        content += (
            f"\n- [x] **ID**: `{UNCLAIMED_ID}`\n"
            "\n**Steps**:\n"
            "- [x] - `p1` - Put the thing in the bucket - `inst-bucket-put`\n"
            "- [x] - `p1` - Read the thing back out - `inst-bucket-get`\n"
        )
    (art_dir / "PRD.md").write_text(content, encoding="utf-8")

    for rel, content in (code or {}).items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    (root / ".git").mkdir(exist_ok=True)
    (root / "AGENTS.md").write_text(
        '<!-- @cf:root-agents -->\n```toml\ncf-studio-path = "adapter"\n```\n',
        encoding="utf-8",
    )
    config = root / "adapter" / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "AGENTS.md").write_text("# Test adapter\n", encoding="utf-8")
    system: dict = {
        "name": "Test",
        "slug": "test",
        "kit": "cypilot",
        "artifacts": [
            {"path": "architecture/PRD.md", "kind": "PRD", "traceability": "FULL"}
        ],
    }
    if codebase_path is not None:
        system["codebase"] = [{"path": codebase_path, "extensions": [".py"]}]
    toml_utils.dump(
        {
            "version": "1.0",
            "project_root": "..",
            "kits": {"cypilot": {"format": "CFS", "path": "kits/sdlc"}},
            "systems": [system],
        },
        config / "artifacts.toml",
    )


def _codes(report: dict, bucket: str) -> list[str]:
    """Issue codes from a report bucket. Requires ``--verbose`` to be populated."""
    assert bucket in report, f"report has no `{bucket}` array — pass --verbose"
    return [str(issue.get("code")) for issue in report[bucket]]


# ---------------------------------------------------------------------------
# Known-bad must go red. These are the states that used to validate clean.
# ---------------------------------------------------------------------------

class TestAClaimWithNoCodeIsNotClean:
    def test_a_checked_to_code_id_with_no_code_at_all_fails(self):
        """The maximal false-completion claim: everything marked done, nothing built.

        Before this change the identical repository returned exit 0 "All checks
        passed" while reporting ``to_code_ids_total: 1`` and ``code_ids_found: 0``
        in the same object.
        """
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=True)

            code, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        assert code == 2
        assert report["status"] == "FAIL"
        assert "code-no-marker" in _codes(report, "errors")

    def test_the_unmet_claim_is_named_not_just_counted(self):
        """A verdict has to say which claim is unbacked, or it cannot be acted on."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=True)

            _, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        unmet = [
            issue for issue in report["errors"]
            if issue.get("code") == "code-no-marker"
        ]
        assert len(unmet) == 1
        assert unmet[0].get("id") == CLAIMED_ID

    def test_the_scan_size_does_not_change_the_verdict(self):
        """0 files and 1 unmarked file are the same claim, so the same verdict.

        This is the invariant the whole change rests on: the checks were always
        right, they were just unreachable below one file.
        """
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=True)
            zero_code, zero_report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=True, code={"src/module.py": "def f():\n    return 1\n"})
            one_code, one_report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        assert zero_code == one_code == 2
        assert _codes(zero_report, "errors") == _codes(one_report, "errors")


class TestAnEmptyRegisteredTreeSaysSo:
    def test_a_registered_entry_that_matches_nothing_is_reported(self):
        """Nothing is claimed, so nothing fails -- but nothing was checked either."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=False)

            code, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        assert code == 0
        assert "codebase-entry-empty" in _codes(report, "warnings")

    def test_the_warning_names_the_entry_that_matched_nothing(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=False, codebase_path="does/not/exist")

            _, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        empty = [
            issue for issue in report["warnings"]
            if issue.get("code") == "codebase-entry-empty"
        ]
        assert len(empty) == 1
        assert "does/not/exist" in str(empty[0].get("message"))

    def test_a_populated_entry_produces_no_such_warning(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(
                root,
                claim_id=False,
                code={"src/module.py": "def f():\n    return 1\n"},
            )

            code, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        assert code == 0
        assert "codebase-entry-empty" not in _codes(report, "warnings")


# ---------------------------------------------------------------------------
# Known-good must stay green. A fix that breaks one of these is too broad.
# ---------------------------------------------------------------------------

class TestNothingClaimedIsNothingOwed:
    def test_a_spec_first_project_with_no_code_stays_green(self):
        """The leniency this change deliberately keeps.

        A repository with artifacts written and no implementation yet -- no
        checked ``to_code`` ID, no registered codebase entry -- must not be
        failed. This is the assertion most at risk from an over-broad fix, and
        the reason the guard keys on whether a claim exists rather than on
        whether files were scanned.
        """
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=False, codebase_path=None)

            code, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        assert code == 0
        assert report["status"] == "PASS"
        assert report["error_count"] == 0
        assert "codebase-entry-empty" not in _codes(report, "warnings")

    def test_declared_instructions_without_a_code_claim_are_not_errors(self):
        """The discriminating case for over-breadth.

        A spec-first repository can declare a full set of checked CDSL steps
        under IDs that are not ``to_code`` -- design work, written down, nothing
        built. Running cross-validation here anyway turns every declared
        instruction into ``code-inst-missing``, which is why the guard has to
        key on whether a *code claim* exists rather than on whether the scan
        was empty. Without this case a fix that simply deletes the file-count
        check looks correct.
        """
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=False, codebase_path=None, declare_instructions=True)

            code, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        assert code == 0
        assert report["status"] == "PASS"
        assert "code-inst-missing" not in _codes(report, "errors")
        assert report["error_count"] == 0

    def test_an_unchecked_claim_owes_nothing_even_with_an_empty_tree(self):
        """An ID present but *unchecked* is not yet a claim of completion."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(root, claim_id=False)

            code, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        assert code == 0
        assert "code-no-marker" not in _codes(report, "errors")

    def test_a_marked_implementation_is_clean(self):
        """Known-good: the claim is made and the marker backs it."""
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _project(
                root,
                claim_id=True,
                code={"src/module.py": f"# @cpt-flow:{CLAIMED_ID}:p1\ndef f():\n    return 1\n"},
            )

            code, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])

        assert code == 0
        assert report["status"] == "PASS"
        assert "code-no-marker" not in _codes(report, "errors")


class TestTheVerdictIsDeterministic:
    def test_repeated_runs_on_the_same_state_agree(self):
        verdicts = set()
        for _ in range(3):
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                _project(root, claim_id=True)
                code, report = run_cli_in_project(root, ["--json", "validate", "--verbose"])
                verdicts.add((code, report["status"], report["error_count"]))

        assert len(verdicts) == 1
