# SolsBaby Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an inspectable dendritic medium in which local experience changes later somatic responses.

**Architecture:** Seeded morphology feeds a passive/active conductance cable.
A local slow shunt stores experience. A fixed probe and training-only ridge
decoder measure readout from one soma; the GUI and reports use the same model.

**Tech Stack:** Python, NumPy, Matplotlib, optional Tkinter; unittest.

**Spec:** `docs/superpowers/specs/2026-10-17-solsbaby-design.md`.

## Global Constraints

- Python >=3.10, NumPy >=1.24, Matplotlib >=3.6, CPU only.
- Dimensionless illustrative biophysics; no biological or speed superiority claim.
- Labels never enter the local writer; test episodes never fit a decoder.
- One somatic trace is the limited reader's only observation.
- Protocol is recorded before held-out measurements and failures remain visible.

## Review Focus

- Invalid seeds/dimensions or nonfinite parameters: reject before allocation.
- Repeated probes: retain slow state except declared decay, freeze writing.
- Very small signal with observation noise: don't normalize away the noise floor.
- GUI without Tk/display: give a headless demo command.
- Output directories with spaces: write JSON/figures without shell interpolation.

### Task 1: Generated cable and local writer

**Files:** `solsbaby/morphology.py`, `solsbaby/model.py`, `tests/test_medium.py`, `pyproject.toml`.

**Interfaces:** `Genome.grow() -> Morphology`; `Medium.initial(batch=1) -> State`;
`Medium.step(state, excitation, current=None, write=True) -> State`;
`Medium.transfer(tag, frequencies) -> complex array`.

- [ ] Write and run failing mathematical tests for the interfaces above.
- [ ] Implement seeded geometry, O(N) implicit tree solve, optional active
  conductance, exact bounded local tag update, and independent state copying/reset.
- [ ] Check dense-solve agreement, passive energy, conservation, history response,
  boundedness, malformed input, and display-pose invariance.
- [ ] Run `python -m unittest discover -s tests -v`; commit working medium.

### Task 2: Memory protocol and controls

**Files:** `solsbaby/experiment.py`, `protocols/v1.json`, `tests/test_experiment.py`.

**Consumes:** Task 1 morphology, state, and step interfaces.
**Produces:** `run_benchmark(protocol) -> dict`; `walkthrough(seed) -> dict`;
`RidgeReader.fit(x,y)` and `.predict(x)` with training-only preprocessing.

- [ ] Write and run failing tests for deterministic independent episodes,
  reader isolation, reset controls, read-only slow decay, and invalid protocols.
- [ ] Commit protocol, then implement batched pulse/write/wait/probe episodes,
  one-soma sampling, ridge reader, controls, metrics, and state oracle.
- [ ] Run the full suite and the declared multi-seed benchmark; retain failures.
- [ ] Record measurements and commit.

### Task 3: Runnable demo, reports, and delivery

**Files:** `solsbaby/gui.py`, `solsbaby/report.py`, `solsbaby/__main__.py`,
`README.md`, `docs/MECHANISM.md`, `.github/workflows/tests.yml`.

**Consumes:** Tasks 1 and 2; no alternative simulator in the UI.
**Produces:** GUI, CLI, JSON/PNG report, reproducible usage instructions.

- [ ] Add CLI integration tests covering missing display, invalid arguments,
  serialized results, and paths containing spaces; watch failures.
- [ ] Implement the visual teach/probe/reset flow and headless reporting.
- [ ] Run full suite, headless walkthrough, installation and GUI smoke checks.
- [ ] Review the complete branch, resolve important findings, verify again,
  and publish the tested implementation to the requested repository.
