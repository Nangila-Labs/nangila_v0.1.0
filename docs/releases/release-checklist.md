# Nangila Release Checklist

Use this checklist before tagging any `v0.1.x` release.

## 1. Working Tree

- [ ] Confirm the branch contains only intended release changes.
- [ ] Review `git status` for unexpected tracked or untracked files.
- [ ] Confirm the release version and scope.

## 2. Tests

- [ ] Run `make release-check`
- [ ] If the helper target is unavailable, run `cargo test`
- [ ] If the helper target is unavailable, create or refresh a local virtual environment
- [ ] If the helper target is unavailable, run `maturin develop --release -F python`
- [ ] If the helper target is unavailable, run `python -m pytest -q`
- [ ] For CUDA pre-qualification, run `bash scripts/cuda_single_gpu_smoke.sh` on a Linux GPU node.
- [ ] For `v0.1` distributed qualification, run `CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 tests/integration/test_ddp_correctness.py` on one supported Linux + CUDA setup.

## 3. Docs

- [ ] README still matches the supported `v0.1` product surface.
- [ ] [`docs/releases/v0.1-support-matrix.md`](/Users/craigchirara/nangila/docs/releases/v0.1-support-matrix.md) is current.
- [ ] [`docs/releases/v0.1-release.md`](/Users/craigchirara/nangila/docs/releases/v0.1-release.md) is current.
- [ ] Experimental features are still labeled as experimental.

## 4. Changelog

- [ ] Add a new release section to `CHANGELOG.md` using [`docs/releases/changelog-template.md`](/Users/craigchirara/nangila/docs/releases/changelog-template.md).
- [ ] Summarize user-visible changes only.
- [ ] Include any support-matrix or compatibility changes.

## 5. CI

- [ ] Confirm the branch is green in GitHub Actions.
- [ ] Verify the `v0.1 CI` workflow still matches the documented local commands.

## 6. Tagging

- [ ] Create the release commit if needed.
- [ ] Confirm `CHANGELOG.md` contains the exact release header, for example `## v0.1.0 - YYYY-MM-DD`.
- [ ] Run the `v0.1 Release` GitHub Actions workflow with the intended version to validate, tag, and draft the release.
- [ ] If tagging manually instead, create an annotated tag, for example `git tag -a v0.1.0 -m "Nangila v0.1.0"`, then push the branch and tag.

## 7. Release Notes

- [ ] Publish release notes from the finalized changelog entry.
- [ ] State the supported surface clearly: DDP, source install, non-CUDA default build.
- [ ] Call out anything still experimental.
