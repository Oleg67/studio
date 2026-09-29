"""The last-resort `$HOME` guard, in a module that imports nothing.

Both `decision_log._redact` and `redaction.home_collapsed` fall back to this when the shared
redaction pattern is unavailable — `decision_log`'s copy runs precisely when
`from .redaction import _home_pattern` raises. That is why this module has **no imports at all**:
it takes part in no cycle, so both callers import it at module scope rather than inside the
handler for "an import failed". An import that happens at load time cannot be the thing that is
missing later; if it ever failed, the module would not load and there would be no fallback to
reach. The failure mode is removed rather than handled.

It was duplicated inline in both callers before, on the reasoning that sharing would give a
last-resort guard a dependency on the thing whose absence triggers it. True of `_home_pattern`
and `capped_text`; **not** true of this, which touches only `home` and the string. Raised in
review on the pull request that first proposed pinning the two copies with a parity test instead.
"""


# @cpt-begin:cpt-studio-algo-core-infra-armed-reversal:p1:inst-home-prefix-fallback
def blank_if_could_carry_home(value: str, home: str) -> str:
    """``"..."`` when ``value`` could still carry ``home``, otherwise ``value`` unchanged.

    Cross-separator and case-folded, the same holes `redaction._home_pattern` closes: a home of
    ``C:\\Users\\x`` must match text containing ``C:/Users/x``, and on a case-insensitive
    filesystem ``c:\\users\\x`` too. A plain same-case substring test let a forward-slash Windows
    path through raw.

    Blunter than `_home_pattern` on purpose: a plain substring test with no boundary lookahead,
    so a sibling like ``/home/alicesibling`` is blanked as well. That is over-redaction, and it
    is the one direction in which erring here is correct — this runs only when the precise
    matcher is already unavailable.

    An empty ``home`` means there is no prefix to look for, so nothing can be judged to carry it.
    """
    if not home:
        return value
    prefix = home.lower().replace("\\", "/")
    return "..." if prefix in value.lower().replace("\\", "/") else value
# @cpt-end:cpt-studio-algo-core-infra-armed-reversal:p1:inst-home-prefix-fallback
