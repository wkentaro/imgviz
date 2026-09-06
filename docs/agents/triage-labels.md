# Triage and PR Labels

This repository uses the following labels to route issue and pull-request work.

## Issue type

Every triaged issue carries exactly one type label:

| Label | Meaning |
| --- | --- |
| `type: bug` | Reporting a defect to fix |
| `type: feature` | Requesting a new capability or improvement |
| `type: task` | Other work, including maintenance, refactoring, documentation, and tests |

## Issue triage

Every triaged issue carries exactly one triage label. An issue with no triage
label is fresh work for the agent to route; `needs-triage` is reserved for a
maintainer decision.

| Role | Label | Meaning |
| --- | --- | --- |
| Maintainer evaluates | `needs-triage` | Maintainer needs to evaluate this issue |
| Reporter provides information | `needs-info` | Waiting on the reporter for more information; shared with PRs |
| Agent implements | `ready-for-agent` | Fully specified and ready for an AFK agent |
| Human implements | `ready-for-human` | Requires human implementation |
| Closed without action | `wontfix` | Will not be actioned |

## Pull-request state and verdicts

The native draft flag means that a PR is still being built or iterated. A
non-draft PR with no verdict is fresh work for the agent to finalize. After the
agent finalizes a PR, exactly one of these mutually exclusive agent verdicts is
used:

| Label | Meaning |
| --- | --- |
| `recommend-merge` | Agent finalized it and endorses review and merge |
| `recommend-close` | Agent recommends closing it; the maintainer reviews or closes |
| `recommend-triage` | Code is sound, but the maintainer must decide the product or scope question |

`needs-info` is shared between issues and PRs when waiting on an outside human.
`maintainer-approved` is a human-only verdict: a maintainer reviewed this exact
PR head and approves merging after required checks pass. An agent must apply it
only after explicit maintainer direction; it may coexist with an agent verdict.

Verdict labels record decisions; they never merge or close a PR. A new commit
makes every verdict stale, so its authority must remove the stale label and
renew the applicable verdict for the new head.
