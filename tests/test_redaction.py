"""Tests for the one owner of `$HOME` collapsing.

Written against the shared helper rather than through either caller, because the defect
that produced it was two callers with the same code and only one of them repaired.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from studio.utils import decision_log
from studio.utils.redaction import home_collapsed

logger = logging.getLogger("test_redaction")


class TestAUsernameNeverReachesTheOutput:
    """Every branch, because the leak lived in the branches nothing exercised."""

    def test_a_path_under_home_is_collapsed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/someone")))

        assert home_collapsed("/home/someone/projects/kit", logger=logger,
                              subject="probe") == "~/projects/kit"

    def test_a_redactor_that_silently_did_nothing_is_not_trusted(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`capped_text` returning cleanly is not proof that it redacted anything.

        `decision_log._redact` catches its *own* `Path.home()` failure and returns the value
        unchanged without raising, so the success branch handed back a raw absolute path.
        Both copies of this helper trusted that clean return; it was found and fixed in one
        of them, and the other still leaked when this was written.
        """
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/someone")))
        # Exactly what `_redact` does when `Path.home()` raises inside it: hand the input
        # straight back. No exception, so no except branch runs.
        monkeypatch.setattr(decision_log, "capped_text", lambda value: value)

        rendered = home_collapsed("/home/someone/projects/kit", logger=logger, subject="probe")
        assert "someone" not in rendered, rendered
        assert rendered == "~/projects/kit", rendered

    @pytest.mark.parametrize("path", [
        "/home/someone/projects/kit",
        # The case the earlier repair missed: the path reported IS the home directory, so
        # its final component is the username. Returning "the final component" looked safe
        # and leaked here. Raised in review.
        "/home/someone",
        r"C:\Users\someone",
    ])
    def test_an_unreadable_home_reports_no_component_at_all(
            self, monkeypatch: pytest.MonkeyPatch, caplog, path: str) -> None:
        """No prefix to remove and no way to learn one, so nothing of the path is reported.

        Two repairs deep. Both original copies returned the path truncated, which puts the
        username straight in. The first fix returned only the final component — safe for a
        path *under* the home directory, and still the username for the home directory
        itself. Without `$HOME` the two cannot be told apart, so the only available answer
        is to report neither.
        """
        def _no_home(cls):
            raise RuntimeError("no home directory")

        monkeypatch.setattr(Path, "home", classmethod(_no_home))
        with caplog.at_level(logging.WARNING, logger=logger.name):
            rendered = home_collapsed(path, logger=logger, subject="probe")

        assert "someone" not in rendered, rendered
        assert rendered == "...", rendered
        assert any("could not be read" in r.getMessage() for r in caplog.records), \
            [r.getMessage() for r in caplog.records]

    def test_the_redactor_failing_is_reported_and_the_collapse_still_happens(
            self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        """A leak guard that stops running without saying so is worth a line."""
        def _boom(_value):
            raise RuntimeError("gone")

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/someone")))
        monkeypatch.setattr(decision_log, "capped_text", _boom)
        with caplog.at_level(logging.WARNING, logger=logger.name):
            rendered = home_collapsed("/home/someone/projects/kit", logger=logger,
                                      subject="probe")

        assert rendered == "~/projects/kit", rendered
        assert any("shared redactor is unavailable" in r.getMessage()
                   for r in caplog.records), [r.getMessage() for r in caplog.records]

    def test_it_never_raises_when_the_home_pattern_itself_fails(
            self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        """`home_collapsed`'s contract is 'never raise'; the direct `_home_pattern` call was
        the gap.

        It guarded the `capped_text` call but not the final `re.sub(_home_pattern(home), ...)`
        — the unguarded twin of `decision_log._redact`'s guarded call, which was fixed while
        this one was not (the A6 asymmetry review flagged). With `capped_text` passing the raw
        value through and `_home_pattern` made to raise, the final substitution is the only
        thing that can throw: the function must still fail safe — no raise, and never the raw
        path. Raised in review.
        """
        import studio.utils.redaction as rmod  # noqa: PLC0415

        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/someone")))
        # capped_text passes through, so the value reaches the final substitution still
        # carrying `$HOME`; only the `_home_pattern` call below can fail from here.
        monkeypatch.setattr(decision_log, "capped_text", lambda value: value)

        def _boom(_home):
            raise RuntimeError("pattern build failed")

        monkeypatch.setattr(rmod, "_home_pattern", _boom)
        with caplog.at_level(logging.WARNING, logger=logger.name):
            out = home_collapsed("/home/someone/project/kit", logger=logger, subject="probe")

        assert "someone" not in out, out       # never the raw, leaking value
        assert out == "...", out               # blanked safely
        assert any("home-redaction pattern is unavailable" in r.getMessage()
                   for r in caplog.records), [r.getMessage() for r in caplog.records]

    def test_the_fallback_collapses_home_wherever_it_appears(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The fallback substitutes the way the shared redactor does, not just at position 0.

        It used a whole-string prefix test, so `/home/x/p` collapsed and
        `git error: /home/x/p not found` did not — and this branch runs *exactly* when the
        shared redactor is down, which is the one case it exists for. The guard was leaking
        in its own failure mode. Raised in review.

        Asserted as **parity with `_redact`** rather than against hand-written expectations:
        the requirement is that the fallback behaves like the thing it substitutes for, and
        writing the answers out by hand would let the two drift while both stayed green.
        """
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path("/home/alice")))

        def _down(_value):
            raise RuntimeError("redactor down")

        monkeypatch.setattr(decision_log, "capped_text", _down)

        for value in ("/home/alice/project",
                      "git error: /home/alice/project not found",
                      "/home/alice",
                      # A sibling must not be mangled into `~sibling` -- the boundary
                      # lookahead is why, and `_redact` carries the same rule.
                      "/home/alicesibling/x",
                      "a /home/alice b /home/alice/c"):
            assert home_collapsed(value, logger=logger, subject="probe") == \
                decision_log._redact(value), value

    def test_the_subject_names_the_calling_guard(
            self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        """So a log line says which guard degraded, not merely that one did."""
        def _no_home(cls):
            raise OSError("no home directory")

        monkeypatch.setattr(Path, "home", classmethod(_no_home))
        with caplog.at_level(logging.WARNING, logger=logger.name):
            home_collapsed("/x/y", logger=logger, subject="armed reversal")

        assert any(r.getMessage().startswith("armed reversal:") for r in caplog.records), \
            [r.getMessage() for r in caplog.records]


class TestBothCallersGoThroughIt:
    """The duplication is the finding, so the absence of a second copy is the test."""

    #: The functions allowed to collapse `$HOME` themselves, **by name** -- not whole files.
    #: `_redact` is `decision_log`'s ledger redactor; `home_collapsed` (and `_home_pattern`) is
    #: `redaction`'s shared owner. Any *other* function that reads home and builds a `~` is a
    #: third copy -- **including one added inside those two files**, which a whole-file
    #: exemption used to hide: it skipped the entire owner module before the heuristic ran, so
    #: a second stray home-collapser there was invisible. Scoped to the sanctioned names now.
    #: Raised in review.
    SANCTIONED = {"_redact", "_home_pattern", "home_collapsed"}

    @staticmethod
    def _functions_that_collapse_home(tree: object) -> list:
        """Names of the functions in ``tree`` that both read `$HOME` and build a `~`.

        Scoped to one function body, not the whole file. Redacting reads the home directory
        and produces the `~` in the **same place**, so an unrelated home-read (a module
        locating a config directory) and an unrelated `~` literal (a CLI default, a glob) in
        *different* functions of one module are not a reimplementation -- the earlier
        whole-file co-occurrence test flagged exactly that pair. A module-scope redactor (both
        at top level, in no function) is out of scope, the same best-effort line the rest of
        this guard draws.
        """
        import ast  # noqa: PLC0415

        def reads_home(node: object) -> bool:
            return any(
                # `Path.home()`, `os.path.expanduser(...)`, or a bare `expanduser(...)`
                (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and n.func.attr in {"home", "expanduser"})
                or (isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "expanduser")
                for n in ast.walk(node))

        def makes_tilde(node: object) -> bool:
            return any(isinstance(n, ast.Constant) and isinstance(n.value, str)
                       and n.value in {"~", "~/"} for n in ast.walk(node))

        return [fn.name for fn in ast.walk(tree)
                if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
                and reads_home(fn) and makes_tilde(fn)]

    def test_no_third_module_collapses_home_for_itself(self) -> None:
        """Discovered across the tree, not asserted about two filenames.

        The first version named `armed_reversal.py` and `gate_surface.py` — the two copies
        that existed when it was written — so a third, anywhere else, would have been
        invisible. That is the defect this module exists to prevent, reproduced in its own
        guard. Raised in review.

        Detected as **reading the home directory and producing a `~`**, which is what
        redacting is. Reading it alone is ordinary — `commands/init.py`, `git_kit_source`
        and `mirrors` all locate a directory that way and are not redacting anything — so a
        rule banning `Path.home()` outright would be wrong and would be worked around.
        """
        import ast  # noqa: PLC0415

        # A best-effort tripwire, not a proof: it recognises the common ways to read the home
        # directory and produce a `~`, not every possible shape. Reading home through an
        # alias, or building `~` from `chr(126)`, would still evade it -- exhaustive AST
        # detection is impossible and would chase false positives. It is here to catch an
        # accidental re-implementation, which takes an ordinary shape; a determined evasion is
        # out of scope. Broadened past `Path.home()` to also see `expanduser`, the next most
        # common home-read, after review flagged the single-shape original as narrow. The
        # home-read and the `~` must land in the **same function** (`_a_function_reads_home_
        # and_makes_tilde`), not merely somewhere in the same file, so an unrelated pair does
        # not raise a false alarm -- review flagged the whole-file co-occurrence.
        repo = Path(__file__).resolve().parents[1]
        # Both shipped source trees, not only the studio kit. `src/studio_proxy` already reads
        # the home directory today -- telemetry, the update check, the cache and mirrors all
        # locate a directory under it -- so a redaction copy could grow there just as easily,
        # and scanning only the kit left that whole package unwatched. That is the same "half
        # the house" gap this test exists to close, one level out. Raised in review. The wider
        # net stays quiet because those reads locate a directory and never build a `~`, so they
        # are not redaction and `makes_tilde` is false for them.
        roots = [repo / "skills/studio/scripts/studio", repo / "src/studio_proxy"]
        offenders = []
        for root in roots:
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                # Skip anything that is not a regular file. A glob can match a FIFO or other
                # special file left in a local workspace, and `read_text` would then block on
                # it forever -- the same hang `gate_surface._read_regular` guards the
                # production scan against. Cannot happen in a clean checkout; cheap here.
                if not path.is_file():
                    continue
                tree = ast.parse(path.read_text(encoding="utf-8"))
                rel = path.relative_to(repo).as_posix()
                # Every file, owners included: the exemption is by sanctioned *function name*,
                # so a stray home-collapser added inside decision_log/redaction is still caught.
                offenders.extend(
                    f"{rel}::{name}" for name in self._functions_that_collapse_home(tree)
                    if name not in self.SANCTIONED)

        assert not offenders, (
            f"these functions collapse $HOME themselves instead of calling "
            f"`redaction.home_collapsed`: {offenders}. That is how the repair gets lost — "
            "one copy is fixed and the other keeps leaking."
        )

    def test_the_duplication_guard_is_scoped_to_one_function(self) -> None:
        """An unrelated home-read and `~` in *different* functions is not a reimplementation.

        The guard used two whole-file `ast.walk` scans, so a module that reads `$HOME` in one
        function (to locate a directory) and carried an unrelated `~` literal in another --
        a CLI default, a glob -- was flagged as a third redactor, forcing an OWNERS entry for
        a module that redacts nothing. As the tree grows that erodes the guard. Scoped to a
        single function now, and asserted on synthetic modules so the scoping itself is
        pinned: the split case is what the old whole-file test flagged and the new one must
        not, and the same-function case is a real reimplementation it must still catch.
        Raised in review.
        """
        import ast  # noqa: PLC0415

        split = ast.parse(
            "from pathlib import Path\n"
            "def find_config():\n"
            "    return Path.home() / '.config'\n"   # reads home, no tilde
            "def cli_default():\n"
            "    return '~'\n")                       # tilde, no home read -- unrelated
        same = ast.parse(
            "from pathlib import Path\n"
            "def collapse(p):\n"
            "    return str(p).replace(str(Path.home()), '~')\n")  # both, one function

        assert not self._functions_that_collapse_home(split), \
            "an unrelated home-read and `~` in different functions must not be flagged"
        assert self._functions_that_collapse_home(same) == ["collapse"], \
            "a real per-function reimplementation must still be caught, by name"

    def test_a_stray_collapser_inside_an_owner_file_is_still_caught(self) -> None:
        """The exemption is by function name, not by file — a whole-file skip hid this.

        The old guard skipped `decision_log.py`/`redaction.py` entirely via an OWNERS file
        list, so a *second*, ad-hoc home-collapsing function added inside one of those two
        files — the very files it exists to keep canonical — was invisible. Now only the
        sanctioned function names are exempt, so a sanctioned function coexists with a stray
        one and only the stray is reported. Raised in review.
        """
        import ast  # noqa: PLC0415

        owner_with_stray = ast.parse(
            "from pathlib import Path\n"
            "def _redact(value):\n"                                  # sanctioned -> exempt
            "    return str(value).replace(str(Path.home()), '~')\n"
            "def _sneaky(p):\n"                                      # stray -> must be caught
            "    return str(p).replace(str(Path.home()), '~')\n")
        offenders = [n for n in self._functions_that_collapse_home(owner_with_stray)
                     if n not in self.SANCTIONED]
        assert offenders == ["_sneaky"], offenders

    def test_both_callers_go_through_the_shared_helper(self) -> None:
        """And the two that were the copies now delegate, rather than merely not duplicating."""
        import ast  # noqa: PLC0415

        root = Path(__file__).resolve().parents[1] / "skills/studio/scripts/studio/utils"
        for name in ("armed_reversal.py", "gate_surface.py"):
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            uses = [n for n in ast.walk(tree)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == "home_collapsed"]
            assert uses, f"{name} no longer calls the shared helper"

    def test_the_cross_module_imports_stay_function_local(self) -> None:
        """Neither `redaction` nor `decision_log` may import the other at module scope.

        The two form a real cycle: `decision_log._redact` imports `redaction._home_pattern`
        and `redaction.home_collapsed` imports `decision_log.capped_text`. It works only
        because each import is deferred inside a function. Hoisting either to module scope --
        a normal-looking cleanup -- makes whichever module loads first import a
        half-initialised other and raise `ImportError` at load, breaking both the ledger and
        the shared redactor. Raised in review; asserted structurally so the placement can't
        drift back.
        """
        import ast  # noqa: PLC0415

        def module_scope_imports(node: object) -> list:
            """Every `import`/`from` that runs at module load: descends into module-level
            `if`/`try`/`with`/`for` (still load-time) but **not** into a def/class body, since
            a function-local import is the correct placement this test is protecting.
            """
            out = []
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    out.append(child)
                out.extend(module_scope_imports(child))
            return out

        def imports(imp: object, other: str) -> bool:
            # `import other` / `import pkg.other`, or `from pkg.other import x` / `from . import other`.
            if isinstance(imp, ast.ImportFrom):
                return (imp.module == other or (imp.module or "").endswith("." + other)
                        or any(a.name == other for a in imp.names))
            return any(a.name == other or a.name.endswith("." + other) for a in imp.names)

        root = Path(__file__).resolve().parents[1] / "skills/studio/scripts/studio/utils"
        for module, other in (("decision_log.py", "redaction"), ("redaction.py", "decision_log")):
            tree = ast.parse((root / module).read_text(encoding="utf-8"))
            # Catches an `ast.Import` and an import nested in a module-level `if`/`try` too, not
            # only a top-level `ast.ImportFrom` -- both still execute at load and reopen the
            # cycle. Function-local imports (under a def) are skipped, being the safe form.
            offenders = [n.lineno for n in module_scope_imports(tree) if imports(n, other)]
            assert not offenders, (
                f"{module} imports {other} at module scope (line {offenders}); it must stay "
                "function-local to avoid the decision_log<->redaction load-time cycle")


class TestItHoldsOnEitherSeparator:
    """The hole review found: the guard only understood the separator it was running on."""

    @pytest.mark.parametrize("home, text", [
        # A Windows home, and text written with the other separator. Git and plenty of
        # other tools emit forward slashes on Windows, so this is the ordinary case there.
        (r"C:\Users\someone", "C:/Users/someone/projects/kit"),
        (r"C:\Users\someone", r"C:\Users\someone\projects\kit"),
        (r"C:\Users\someone", "error: C:/Users/someone/p not found"),
        # And the POSIX case keeps working, including mid-sentence.
        ("/home/someone", "/home/someone/projects/kit"),
        ("/home/someone", "git error: /home/someone/p not found"),
    ])
    def test_the_username_is_collapsed_whichever_separator_is_used(
            self, monkeypatch: pytest.MonkeyPatch, home: str, text: str) -> None:
        """Every one of these left the username in the output before the fix.

        The old pattern matched the home prefix literally and accepted only the running
        platform's `os.sep` as a boundary. On Windows that meant a forward-slash path
        matched nothing at all, and the value went into a shared file whole. Marked Major
        in review, correctly: this module exists for exactly this.
        """
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(home)))
        out = home_collapsed(text, logger=logging.getLogger("t"), subject="t")
        assert "someone" not in out, out
        assert "~" in out, out

    @pytest.mark.parametrize("home, text", [
        # Windows filesystems are case-insensitive, and tools emit the path in any case, so a
        # differently-cased prefix must still redact. Left the username whole before `(?i)`.
        (r"C:\Users\Someone", r"c:\users\someone\projects\kit"),
        (r"C:\Users\Someone", "error: C:/USERS/SOMEONE/p not found"),
    ])
    def test_the_username_is_collapsed_regardless_of_case(
            self, monkeypatch: pytest.MonkeyPatch, home: str, text: str) -> None:
        """A differently-cased Windows path used to escape redaction whole; `(?i)` closes it."""
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(home)))
        out = home_collapsed(text, logger=logging.getLogger("t"), subject="t")
        assert "someone" not in out.lower(), out
        assert "~" in out, out

    @pytest.mark.parametrize("home, text", [
        ("/home/some", "/home/someone/projects"),
        (r"C:\Users\some", "C:/Users/someone/projects"),
    ])
    def test_a_longer_name_that_merely_starts_the_same_is_left_alone(
            self, monkeypatch: pytest.MonkeyPatch, home: str, text: str) -> None:
        """Widening the separator must not widen what counts as the end of the home path.

        `/home/some` is a prefix of `/home/someone` as plain text. Without a boundary the
        substitution turns another user's directory into `~one`, which is both wrong and a
        silent corruption of a path someone may act on. The boundary is the reason the
        pattern is not a plain prefix test, and it survives the fix.
        """
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: Path(home)))
        out = home_collapsed(text, logger=logging.getLogger("t"), subject="t")
        assert "someone" in out, out
        assert "~one" not in out, out
