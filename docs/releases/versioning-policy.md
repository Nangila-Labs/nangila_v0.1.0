# Versioning Policy

This document defines how the repository should be versioned and published on GitHub.

## Branches

- `main`: active development branch
- `release/vX.Y`: optional maintenance branch for a released minor line

Examples:
- `release/v0.1`
- `release/v0.2`

## Tags

Release tags use semantic versioning:

- `v0.1.0`
- `v0.1.1`
- `v0.2.0`

## Release Lines

### v0.1.x

Purpose:
- stabilize the DDP-first product surface
- keep the default install path non-CUDA-safe
- avoid broadening the support contract unless the support docs are updated

Allowed changes:
- bug fixes
- packaging fixes
- smoke-test and CI fixes
- docs clarifications
- narrow API cleanup

### v0.2.x

Purpose:
- continue development beyond the `v0.1` line without destabilizing `v0.1.x`

Likely areas:
- CUDA beta qualification
- broader DDP validation
- experimental feature iteration behind explicit boundaries

## GitHub Releases

Every GitHub release should:
- reference the matching `vX.Y.Z` tag
- summarize only user-visible changes
- state the supported surface clearly
- call out experimental areas separately

Use [`release-checklist.md`](/Users/craigchirara/nangila/docs/releases/release-checklist.md) before tagging.
Prefer the GitHub Actions workflow [`release.yml`](/Users/craigchirara/nangila/.github/workflows/release.yml) so the same `make release-check` validation runs before the tag is pushed.
