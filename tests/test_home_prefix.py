"""The last-resort `$HOME` guard both redactors fall back to.

This replaced two inline copies of the same four lines. The copies were first defended as
deliberate — a last-resort guard must not depend on the thing whose absence triggers it — and
that reasoning was about the wrong code: it holds for `_home_pattern` and `capped_text`, not for
this, which touches only `home` and the string. Raised in review.

Because the module imports nothing it joins no cycle, so both callers take it at **module scope**
rather than inside the handler for a failed import. An import performed at load time cannot be
the thing that is missing later — if it ever failed, the module would not load and there would be
no fallback to reach. `test_the_module_imports_nothing` pins that property, because it is the one
thing that makes this design safe and nothing else would notice it going away.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

from studio.utils import decision_log
from studio.utils import redaction
from studio.utils.home_prefix import blank_if_could_carry_home
from studio.utils.redaction import home_collapsed

logger = logging.getLogger("test_home_prefix")

POSIX_HOME = "/home/alice"
WINDOWS_HOME = "C:\\Users\\alice"

#: `(home, value, blanked)`. **`blanked` is written out by hand, not computed**: deriving it with
#: `home.lower() in value.lower().replace(...)` would be the implementation's own expression, so a
#: wrong predicate would agree with itself and pass on both sides of the bug (§3b A5a).
#:
#: Both homes are exercised deliberately. With a POSIX home there is no backslash to fold, so the
#: separator-normalising half does no work at all — deleting it left every test green until a
#: Windows-style home was added. Found by mutation.
CASES = [
    pytest.param(POSIX_HOME, f"{POSIX_HOME}/project", True, id="posix-path-under-home"),
    pytest.param(POSIX_HOME, f"git error: {POSIX_HOME}/project not found", True,
                 id="posix-home-embedded-mid-string"),
    pytest.param(POSIX_HOME, POSIX_HOME, True, id="posix-the-home-directory-itself"),
    # Blunter than `_home_pattern`, which has a boundary lookahead: a substring test catches the
    # sibling too. Over-redaction, and the correct direction to err when the precise matcher is
    # already unavailable.
    pytest.param(POSIX_HOME, "/home/alicesibling/x", True, id="posix-sibling-over-redacted"),
    pytest.param(POSIX_HOME, "/var/log/syslog", False, id="posix-unrelated-path"),
    pytest.param(POSIX_HOME, "/HOME/ALICE/PROJECT", True, id="posix-upper-cased-home"),
    pytest.param(WINDOWS_HOME, "C:\\Users\\alice\\project", True, id="windows-same-separator"),
    # The case the separator-normalising exists for: a Windows home written forward-slashed,
    # which plenty of tools emit, git among them.
    pytest.param(WINDOWS_HOME, "C:/Users/alice/project", True, id="windows-forward-slashed-home"),
    pytest.param(WINDOWS_HOME, "c:\\users\\ALICE\\project", True, id="windows-mixed-case"),
    pytest.param(WINDOWS_HOME, "D:\\Other\\bob", False, id="windows-unrelated-drive"),
    pytest.param(POSIX_HOME, "nothing here at all", False, id="no-path-at-all"),
    pytest.param(POSIX_HOME, "", False, id="empty-string"),
    pytest.param("", f"{POSIX_HOME}/project", False, id="no-home-nothing-to-look-for"),
]


class TestTheGuardItself:

    @pytest.mark.parametrize("home,value,blanked", CASES)
    def test_it_blanks_exactly_what_could_carry_home(
            self, home: str, value: str, blanked: bool) -> None:
        out = blank_if_could_carry_home(value, home)
        if blanked:
            assert out == "...", f"{value!r} could carry {home!r} and came back {out!r}"
        else:
            assert out == value, f"{value!r} cannot carry {home!r} yet became {out!r}"

    def test_the_module_imports_nothing(self) -> None:
        """The property the whole design rests on, and the one nothing else would catch.

        Both callers import this at module scope *because* it imports nothing, so it cannot be
        half-initialised by a cycle and cannot itself be the missing dependency. An `import` added
        here later would reintroduce exactly the risk the inline copies were defending against,
        and every other test would still pass.
        """
        path = Path(redaction.__file__).with_name("home_prefix.py")
        try:
            source = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # "I could not check the invariant" is not "the invariant holds", and it is not the
            # same as "the invariant is broken" either. Left bare, a read failure surfaces as a
            # crash shaped like any other and the property's status goes unstated. Raised in
            # review as two findings on this line.
            pytest.fail(f"could not read {path} to check it imports nothing: "
                        f"{type(exc).__name__}: {exc}")
        found = [n for n in ast.walk(ast.parse(source))
                 if isinstance(n, (ast.Import, ast.ImportFrom))]
        assert found == [], [ast.dump(n) for n in found]


class TestBothFallbacksRouteThroughIt:
    """Each caller reached its own copy before. The point is that neither has one now."""

    @pytest.mark.parametrize("home,value,blanked", CASES)
    def test_decision_log_falls_back_to_the_guard(
            self, monkeypatch: pytest.MonkeyPatch, home: str, value: str, blanked: bool) -> None:
        """`_redact` with its lazy import of `_home_pattern` made to fail — the real trigger."""
        with monkeypatch.context() as m:
            m.setattr(Path, "home", classmethod(lambda cls, h=home: Path(h)))
            m.delattr(redaction, "_home_pattern")
            out = decision_log._redact(value)
        assert out == blank_if_could_carry_home(value, home), out

    @pytest.mark.parametrize("home,value,blanked", CASES)
    def test_home_collapsed_falls_back_to_the_guard(
            self, monkeypatch: pytest.MonkeyPatch, home: str, value: str, blanked: bool) -> None:
        """`home_collapsed` with `_home_pattern` raising — its own trigger, a different one."""
        with monkeypatch.context() as m:
            m.setattr(Path, "home", classmethod(lambda cls, h=home: Path(h)))
            m.setattr(decision_log, "capped_text", lambda v: v)

            def _boom(_home):
                raise RuntimeError("pattern build failed")

            m.setattr(redaction, "_home_pattern", _boom)
            out = home_collapsed(value, logger=logger, subject="probe")
        assert out == blank_if_could_carry_home(value, home), out

    def test_both_fallbacks_actually_ran(
            self, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
        """The guard against a green run that exercised neither.

        Both tests above reach their fallback through a deliberate failure. If a refactor moved
        either trigger they would still pass, by comparing a *normal* path against the helper —
        which is how the previous parity test came to cover the wrong pair. Each fallback warns
        before blanking, so the warnings are the evidence it ran.
        """
        with caplog.at_level(logging.WARNING):
            with monkeypatch.context() as m:
                m.setattr(Path, "home", classmethod(lambda cls: Path(POSIX_HOME)))
                m.delattr(redaction, "_home_pattern")
                decision_log._redact(f"{POSIX_HOME}/x")
            with monkeypatch.context() as m:
                m.setattr(Path, "home", classmethod(lambda cls: Path(POSIX_HOME)))
                m.setattr(decision_log, "capped_text", lambda v: v)

                def _boom(_home):
                    raise RuntimeError("pattern build failed")

                m.setattr(redaction, "_home_pattern", _boom)
                home_collapsed(f"{POSIX_HOME}/x", logger=logger, subject="probe")

        messages = [r.getMessage() for r in caplog.records]
        assert sum("unavailable" in m for m in messages) >= 2, messages
