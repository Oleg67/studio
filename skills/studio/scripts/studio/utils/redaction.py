"""One owner for collapsing `$HOME` out of a path before it is reported.

Two modules had grown their own copy of this — `armed_reversal._said` and
`gate_surface._said` — and the copies did not stay equal. Review found the leak in one,
it was fixed there, and the other kept it: at the time this module was written
`armed_reversal` still returned a raw absolute path under two separate conditions, with
a comment claiming the second of them had been fixed. That is the argument for one
owner, made by the thing it predicts. Raised in review.

What every caller wants is the same: a path with the home prefix replaced by `~`, and a
guarantee that a username never reaches the output. What they do afterwards differs —
this walk caps, that check strips and quotes — so the shared part ends where the
agreement ends.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path


# @cpt-begin:cpt-studio-algo-core-infra-armed-reversal:p1:inst-redact-home

def _home_pattern(home: str) -> str:
    """A regex matching ``home`` written with either separator, bounded by either.

    Three separate cross-platform holes, all reported as under-redaction:

    * the **boundary** accepted only the running platform's `os.sep`, so on Windows a path
      reported with forward slashes -- which plenty of tools emit, git among them -- had no
      accepted boundary after the prefix and was left whole;
    * the **prefix** was matched literally, so a home of ``C:\\Users\\x`` never matched
      text containing ``C:/Users/x`` at all;
    * the **case** was matched literally, so on Windows -- whose filesystem is
      case-insensitive -- a home of ``C:\\Users\\x`` never matched ``c:\\users\\x``, which
      tools emit freely. The `(?i)` prefix closes it, at the source so both sinks
      (`home_collapsed` and `decision_log._redact`) inherit it.

    Any one leaves a username in a file made to be shared, which is the single thing this
    module exists to prevent. Both separators are accepted in both positions.

    On POSIX this also collapses a literal backslash path, or a path that only case-matches
    the home, neither of which is that user's home there. That is over-redaction, not a leak,
    and this is the one direction in which erring is correct.
    """
    parts = re.split(r"[\\/]", home)
    return r"(?i)" + r"[\\/]".join(re.escape(part) for part in parts) + r"(?=[\\/]|$)"


def home_collapsed(value: object, *, logger: logging.Logger, subject: str) -> str:
    """``value`` as text with ``$HOME`` collapsed to ``~``, and never a username.

    `$HOME` is read **first**, because it is the precondition for both layers and the only
    way to tell whether the shared redactor actually did anything. `decision_log.capped_text`
    collapses `$HOME` by reading `Path.home()` itself and, when that raises, returns the
    value **unchanged without raising** — so a clean return is not proof of redaction, and
    trusting one is how a raw path reached the output in both copies of this.

    Three outcomes, each reported rather than silent:

    * `$HOME` unreadable — neither layer can redact and there is no prefix to learn, so
      nothing of the path is returned. Both original copies returned it truncated, which
      puts the username straight into the field this exists to keep it out of; returning
      the final component instead was the first repair and still leaked, because when the
      path reported is the home directory that component *is* the username. Raised in
      review.
    * the redactor unavailable — its exception is warned about and the collapse is done here.
    * the redactor ran — its answer is **checked**, not assumed, and collapsed here if it
      came back still carrying the prefix.

    `subject` names the calling module in the warnings, so a log line says which guard
    degraded. Callers cap, strip or quote the result themselves; this does not, because the
    two callers disagree about that and the disagreement is legitimate.
    """
    text = str(value)
    try:
        home = str(Path.home()).rstrip("/\\")
    except (OSError, RuntimeError) as exc:
        logger.warning("%s: the home directory could not be read, so no path component is "
                       "reported at all: %s", subject, type(exc).__name__)
        # No component at all. Returning the final one looked safe and was not: when the
        # path reported *is* the home directory, that component is the username, which is
        # the single thing this function exists to keep out. Without `$HOME` there is no
        # way to tell that case apart, so the safe answer is the only one available.
        # Over-redaction is the one direction in which erring here is correct.
        return "..."

    try:
        # Kept function-local, not only for the fail-safe below but because it closes a
        # `decision_log` <-> `redaction` cycle: `decision_log._redact` imports back into this
        # module (`_home_pattern`), so hoisting either import to module scope would make
        # whichever module loads first import a half-initialised other and raise at load.
        from .decision_log import capped_text  # pylint: disable=import-outside-toplevel
        reported = capped_text(text)
    except Exception as exc:  # pylint: disable=broad-except
        # Broad, and warned rather than swallowed. `capped_text` is another module's
        # function, so any exception from it would escape a leak guard through the guard's
        # own reporting path, and a leak guard that stops running without saying so is
        # worth a line.
        logger.warning("%s: the shared redactor is unavailable, so a path is reported "
                       "through a local fallback: %s", subject, type(exc).__name__)
        reported = text

    if home:
        # Substituted at any path boundary, not only at position 0 -- the same rule
        # `decision_log._redact` applies. A whole-string prefix test collapsed
        # `/home/x/p` but left `git error: /home/x/p not found` untouched, and this
        # branch runs exactly when the shared redactor is down, so it was the guard
        # leaking in the one case it exists for. Raised in review.
        try:
            reported = re.sub(_home_pattern(home), "~", reported)
        except Exception as exc:  # pylint: disable=broad-except
            # Never raise -- that is this function's stated contract -- and never leak. The
            # unguarded twin of `decision_log._redact`'s guarded call (which was fixed and this
            # one was not: the A6 asymmetry review flagged). If `_home_pattern` itself fails,
            # blank a value that could still carry the prefix rather than return it raw;
            # cross-separator and case-folded, matching `_home_pattern`'s own reach.
            logger.warning("%s: the home-redaction pattern is unavailable, so a value is "
                           "blanked rather than risk leaking a path: %s", subject,
                           type(exc).__name__)
            prefix = home.lower().replace("\\", "/")
            reported = "..." if prefix in reported.lower().replace("\\", "/") else reported
    return reported
# @cpt-end:cpt-studio-algo-core-infra-armed-reversal:p1:inst-redact-home
