# PROJECT CONTRACT — ipm-ocr-bot

This repo automates Idle Planet Miner running in BlueStacks 5 on Windows.

## Non-Negotiable Architecture Rules
1) Keyboard automation only.
   - All actions must be executed via key presses that trigger BlueStacks key-mapped taps/gestures.
   - Do not add mouse movement, mouse clicks, drag logic, or scroll wheel logic.

2) No OCR / no heavy vision.
   - Do not add OCR, OpenCV pipelines, template matching, or rectangle-driven UI parsing.
   - Any visual logic must be optional, minimal, and strictly limited to tiny “button enabled” signal sampling (small patches) and/or simple lap detection hashes.
   - If a feature can be done with deterministic key sequences, do that instead.

3) Keep it simple and modular.
   - Prefer small modules: one task per file.
   - Avoid frameworks, state machines, and over-abstraction.

4) Fail closed.
   - If the bot is uncertain, it should do nothing (or stop), not “guess.”

## Current Control Scheme (BlueStacks Keybindings)
### Panels
- Shift+1: Toggle Resources panel
- Shift+2: Toggle Production panel
- P: Open Planet menu (does NOT toggle close)

### Resources Tabs
- F1: Ores
- F2: Alloys
- F3: Items

### Production Tabs
- F4: Smelt
- F5: Craft

### Resource List / Selling
- 1–4: Select resource row 1–4
- Num5: Open sell / execute sell
- Num7: Set slider ~50%
- Num9: Set slider ~100%
- Num2: Scroll down (BlueStacks swipe)
- Num8: Scroll up (BlueStacks swipe)

### Planet Menu
- - : Previous planet (cycles forever)
- = : Next planet (cycles forever)
- Ctrl+1: Increase Mining
- Ctrl+2: Increase Ship Speed
- Ctrl+3: Increase Cargo
- Exit planet menu: Shift+1 then Shift+1 (open/close Resources)

## Runtime Behavior
- F9 toggles AFK automation on/off.
- F10 is emergency stop.
- Main loop runs small deterministic modules (planets, ores, etc.) with conservative delays.

## What the agent must NOT do
- Do not redesign the project around OCR/vision.
- Do not introduce complex UI detection or window capture frameworks.
- Do not refactor into a big architecture “because it’s cleaner.”
- Do not remove the manual override pattern (F9).

## Acceptable improvements
- Add new modules (alloys/items/production) using key sequences.
- Improve timing and stability.
- Optional: minimal pixel-sampling to skip unaffordable upgrades or detect a full planet loop.

## PR / Change Checklist (must pass before considering work done)

### Behavior + UX
- [ ] F9 toggles AFK on/off reliably.
- [ ] F10 emergency stop works immediately.
- [ ] Bot does not require mouse movement/clicks/scroll wheel at any time.
- [ ] Modules are deterministic: same inputs → same key sequence behavior.
- [ ] Bot fails closed: if something is uncertain, it stops or safely exits menus.

### Keybinding Contract
- [ ] No changes to the BlueStacks keybinding scheme without updating this file.
- [ ] Planet exit still works via: Shift+1 then Shift+1.
- [ ] Ores sell flow still uses: Row(1–4) → Num5 → Num9/Num7 → Num5.
- [ ] Scrolling still uses: Num8 (up), Num2 (down).

### Code Quality
- [ ] Changes are minimal; no large refactors unless explicitly requested.
- [ ] Each new capability is isolated in its own small module (e.g., `alloys.py`, `items.py`).
- [ ] Delays are conservative enough to avoid missed inputs.

### Dependencies
- [ ] No new heavy deps added (no OpenCV, no OCR libs).
- [ ] If any new dependency is added, it is justified and documented in the commit message.

### Smoke Test (manual)
- [ ] Start script, press F9, observe one full module pass without UI drift.
- [ ] Press F9 to stop, confirm no further inputs occur.
- [ ] Press F10 to emergency stop, confirm process exits immediately.
- [ ] Planets module returns to normal game view at the end (no planet menu stuck open).