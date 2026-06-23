# Phase 0 - Repo Scaffold

## Objective

Create a monorepo that can support Python model research and TypeScript agent/product work.

## Inputs

This Codex pack, especially project manifest and schemas.

## Outputs

Package directories, README, pyproject, package.json workspaces, basic tests.

## Acceptance criteria

`python -m pytest` passes for Python packages. `pnpm test` or equivalent passes for TS packages. Config files validate.

## Notes for Codex

Do not implement the full model in this phase. Build the skeleton that later phases can fill.
