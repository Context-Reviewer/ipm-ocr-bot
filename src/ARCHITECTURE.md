# Rebuild Skeleton

The active project now runs through the `ipm` package.

## Active runtime path

`src/main.py` -> `ipm.app.Application`

## Intended layers

1. `ipm.capture`
   - screenshot backends
   - desktop and ADB capture

2. `ipm.perception`
   - perception backend abstraction
   - legacy, OpenAI, and hybrid backends

3. `ipm.state`
   - typed game state objects

4. `ipm.tasks`
   - small deterministic tasks that operate on typed state

5. `ipm.scheduler`
   - task timing and orchestration

6. `ipm.runtime`
   - mutable runtime state only

## Current rebuild status

- The runtime skeleton is active.
- `planets` and `ores` are stub tasks.
- Old OCR-heavy modules remain in the repo for reference but are no longer the active path.
