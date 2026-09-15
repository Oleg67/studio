"""Severity as a property of a validation finding.

Before this module, a finding's severity was not data: it was decided by *which
list the call site appended to*, in about fourteen places across
``utils/constraints.py``, ``utils/toc.py``, ``commands/validate.py`` and
``commands/self_check.py``. That made severity impossible to configure, because
there was nothing to configure — only control flow.

Every finding is now stamped from ``DEFAULT_SEVERITY`` at build time. The table
reproduces today's routing exactly: the twelve codes currently sent to
``warnings`` default to ``warning``, every other code defaults to ``error``.
Stamping from a table rather than from a keyword default is what makes that
equivalence testable — a keyword default would silently agree with whatever the
call site already did.

The table is exhaustive over ``error_codes`` on purpose. A shorter table plus a
fallback would give every code *a* default while pinning only a handful, and a
rule shipped as ``warning`` when it should be ``error`` keeps every "fails on
bad input" test green, because those tests assert exit codes and message text
rather than the label itself.

@cpt-algo:cpt-studio-algo-traceability-validation-severity-policy:p1
"""

# @cpt-begin:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-imports
from typing import Dict, Optional

from . import error_codes as EC
# @cpt-end:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-imports

# @cpt-begin:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-vocabulary
# ---------------------------------------------------------------------------
# The severity vocabulary
# ---------------------------------------------------------------------------
# ``off`` is declared here but no code defaults to it yet: suppression arrives
# with the configuration layer. Declaring the full vocabulary now keeps the
# stamped field and the configured field one type rather than two.
#
# Named ``VALIDATION_SEVERITIES`` rather than ``SEVERITIES`` because
# ``utils/artifact_quality.py`` already exports a ``SEVERITIES`` for advisory
# findings (``info`` / ``warn``) — a different vocabulary for a different model,
# and one name for both would invite the wrong import.
ERROR = "error"
WARNING = "warning"
OFF = "off"

VALIDATION_SEVERITIES = (ERROR, WARNING, OFF)
# @cpt-end:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-vocabulary

# @cpt-begin:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-default-table
# ---------------------------------------------------------------------------
# Default severity per rule code — exhaustive over ``error_codes``
# ---------------------------------------------------------------------------
DEFAULT_SEVERITY: Dict[str, str] = {
    # --- warning by default: reproduces today's routing exactly -------------
    # Each entry below corresponds to a call site that appends to ``warnings``.
    # The CDSL missing-token trio is a deliberate, tracked backlog concession
    # (see ``_validate_cdsl_step`` in utils/constraints.py), not a judgement
    # that those rules are advisory in principle.
    EC.CDSL_MISSING_CHECKBOX:                   WARNING,
    EC.CDSL_MISSING_PHASE_TOKEN:                WARNING,
    EC.CDSL_MISSING_INST_ID:                    WARNING,
    EC.CDSL_INCOMPLETE_STEP_LINE:               WARNING,
    EC.TOC_HEADING_DUPLICATE:                   WARNING,
    EC.TOC_HEADING_DEPTH_JUMP:                  WARNING,
    EC.TOC_SECTION_TOO_LONG:                    WARNING,
    EC.TOC_MISSING_DESCRIPTION:                 WARNING,
    EC.TOC_STALE:                               WARNING,
    EC.REF_TARGET_NOT_IN_SCOPE:                 WARNING,
    EC.ID_NOT_REFERENCED_NO_SCOPE:              WARNING,
    EC.CODEBASE_ENTRY_EMPTY:                    WARNING,
    # The optional half of each template-placeholder pair; the required half is
    # an error below. Two codes rather than one code at two severities.
    EC.TEMPLATE_DEF_PLACEHOLDER_MISSING_OPTIONAL: WARNING,
    EC.TEMPLATE_REF_PLACEHOLDER_MISSING_OPTIONAL: WARNING,
    # Advisory today: a kind whose template/examples cannot be bound is reported
    # with status PASS and warning_count 1 by validate-kits.
    EC.KIT_TEMPLATE_BINDING_MISSING:            WARNING,

    # --- error by default ---------------------------------------------------
    EC.TEMPLATE_DEF_PLACEHOLDER_MISSING:        ERROR,
    EC.TEMPLATE_REF_PLACEHOLDER_MISSING:        ERROR,
    EC.TEMPLATE_ID_KIND_NO_TEMPLATE:            ERROR,
    EC.TEMPLATE_DEF_PLACEHOLDER_WRONG_HEADINGS: ERROR,
    EC.TEMPLATE_REF_PLACEHOLDER_WRONG_HEADINGS: ERROR,
    EC.TEMPLATE_READ_ERROR:                     ERROR,
    EC.CONSTRAINTS_INVALID:                     ERROR,
    EC.KIT_RESOURCE_PATH_NOT_FOUND:             ERROR,
    EC.KIT_MODEL_INVALID:                       ERROR,
    EC.KIT_PATH_NOT_ACCESSIBLE:                 ERROR,
    EC.KIT_BINDING_ERROR:                       ERROR,
    EC.REGISTRY_AUTODETECT_INVALID:             ERROR,
    EC.REGISTRY_AUTODETECT_FAILED:              ERROR,
    EC.CDSL_STEP_UNCHECKED:                     ERROR,
    EC.PARENT_UNCHECKED_ALL_DONE:               ERROR,
    EC.PARENT_CHECKED_NESTED_UNCHECKED:         ERROR,
    EC.REF_NO_DEFINITION:                       ERROR,
    EC.REF_DONE_DEF_NOT_DONE:                   ERROR,
    EC.DEF_DONE_REF_NOT_DONE:                   ERROR,
    EC.REF_TASK_DEF_NO_TASK:                    ERROR,
    EC.DUPLICATE_DEFINITION:                    ERROR,
    EC.HEADING_NUMBER_NOT_CONSECUTIVE:          ERROR,
    EC.ID_NOT_REFERENCED:                       ERROR,
    EC.MISSING_CONSTRAINTS:                     ERROR,
    EC.ID_SYSTEM_UNRECOGNIZED:                  ERROR,
    EC.ID_KIND_NOT_ALLOWED:                     ERROR,
    EC.REQUIRED_ID_KIND_MISSING:                ERROR,
    EC.TEMPLATE_DEF_KIND_NOT_IN_CONSTRAINTS:    ERROR,
    EC.TEMPLATE_REF_KIND_NOT_IN_CONSTRAINTS:    ERROR,
    EC.DEF_MISSING_TASK:                        ERROR,
    EC.DEF_PROHIBITED_TASK:                     ERROR,
    EC.DEF_MISSING_PRIORITY:                    ERROR,
    EC.DEF_PROHIBITED_PRIORITY:                 ERROR,
    EC.DEF_WRONG_HEADINGS:                      ERROR,
    EC.HEADING_MISSING:                         ERROR,
    EC.HEADING_PROHIBITS_MULTIPLE:              ERROR,
    EC.HEADING_REQUIRES_MULTIPLE:               ERROR,
    EC.HEADING_NUMBERING_MISMATCH:              ERROR,
    EC.REF_MISSING_FROM_KIND:                   ERROR,
    EC.REF_WRONG_HEADINGS:                      ERROR,
    EC.REF_MISSING_TASK_FOR_TRACKED:            ERROR,
    EC.REF_FROM_PROHIBITED_KIND:                ERROR,
    EC.REF_MISSING_TASK:                        ERROR,
    EC.REF_PROHIBITED_TASK:                     ERROR,
    EC.REF_MISSING_PRIORITY:                    ERROR,
    EC.REF_PROHIBITED_PRIORITY:                 ERROR,
    EC.MARKER_DUP_BEGIN:                        ERROR,
    EC.MARKER_END_NO_BEGIN:                     ERROR,
    EC.MARKER_EMPTY_BLOCK:                      ERROR,
    EC.MARKER_BEGIN_NO_END:                     ERROR,
    EC.MARKER_DUP_SCOPE:                        ERROR,
    EC.CODE_DOCS_ONLY:                          ERROR,
    EC.CODE_ORPHAN_REF:                         ERROR,
    EC.CODE_TASK_UNCHECKED:                     ERROR,
    EC.CODE_NO_MARKER:                          ERROR,
    EC.CODE_INST_MISSING:                       ERROR,
    EC.CODE_INST_ORPHAN:                        ERROR,
    EC.TOC_MISSING:                             ERROR,
    EC.TOC_ANCHOR_BROKEN:                       ERROR,
    EC.TOC_HEADING_NOT_IN_TOC:                  ERROR,
    EC.FILE_READ_ERROR:                         ERROR,
    EC.FILE_LOAD_ERROR:                         ERROR,
    EC.FILE_TOO_LARGE:                          ERROR,
    EC.CONTENT_LANGUAGE_VIOLATION:              ERROR,
    EC.CDSL_CODE_SYNTAX:                        ERROR,
    EC.CDSL_TYPE_ANNOTATION:                    ERROR,
    EC.CDSL_LANGUAGE_OPERATOR:                  ERROR,
    EC.CDSL_NOT_PLAIN_ENGLISH:                  ERROR,
    EC.CDSL_DUPLICATE_INST_ID:                  ERROR,
    EC.CDSL_PLACEHOLDER:                        ERROR,
}
# @cpt-end:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-default-table


def default_severity(code: Optional[str]) -> str:
    """Return the declared default severity for ``code``.

    An absent or unrecognised code resolves to ``error``. That direction is
    deliberate: an unknown rule must not become silently non-blocking, which is
    the failure mode this whole model exists to make impossible.
    """
    # @cpt-begin:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-resolve-default
    if code:
        declared = DEFAULT_SEVERITY.get(code)
        if declared is not None:
            return declared
    # @cpt-end:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-resolve-default
    # @cpt-begin:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-unknown-is-error
    return ERROR
    # @cpt-end:cpt-studio-algo-traceability-validation-severity-policy:p1:inst-severity-unknown-is-error
