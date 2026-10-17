# SolsBaby v1: experience changes the response

Antti requested a new implementation of the dendritic-medium idea, with design
choices delegated to the implementer. The deliverable is a runnable research
prototype in this repository, including a visual demo and honest measurements.

## Scope and alternatives

Use a small compartmental tree, local conductance plasticity, and one somatic
readout. A full biophysical cell would add calibration and dependency costs;
an unconstrained recurrent network would obscure the cable mechanism. This
version chooses the inspectable middle. CPU only; Python >=3.10; runtime
dependencies NumPy >=1.24 and Matplotlib >=3.6. Tkinter is needed only for GUI.
No downloads, pretrained networks, or GPU training.

## Medium

A seeded developmental recipe generates a rooted tree with four branches,
five segments each, and a soma. Lengths, diameters, and connectivity determine
positive axial conductances, capacitances, and leak. Display coordinates do
not independently affect electrical coupling. Units are dimensionless; this
is not a calibrated biological cell or a model of literal DNA.

The equations are C dv/dt = -L_ax v - (g_leak + g_memory*p)*v
+ g_input(t)*(E_exc-v) + g_active(v)*(E_exc-v) + I_external.
The optional active term is a voltage-dependent, NMDA-like conductance, not a
validated NMDA receptor model. Implicit tree solves keep the passive terms
stable. The local slow state p is written by depolarization above a threshold
and otherwise decays. It is bounded in [0,1] and increases shunting.
Labels never enter the writer. Fast voltage and slow conductance memory have
separate reset operations. Probe measurement freezes writing, while allowing
ordinary decay; this is an explicit experimental intervention.

## Decisive experiment

An episode exposes one branch to a jittered pulse train, waits, clears fast
voltage, and applies exactly the same branch-sweep probe to every history.
Only baseline-subtracted somatic voltage samples enter the limited reader.
Independent training/test episodes have independent nuisance variables and
observation noise. A ridge reader is fit on training data only. Report all
seeds, confusion matrices, time/space budgets, and a full-slow-state oracle.

Controls: matched frozen writer, erased slow state, and shuffled training
labels. A separate passive current-injection test checks superposition and
history-independent incremental response. Measure the saved slow state and
the frozen passive transfer function independently of the somatic decoder.

Commit the protocol before the first held-out benchmark. Default gate:
learned somatic accuracy mean >=0.80, minimum seed >=0.65; frozen and erased
controls mean <=0.37 for four classes. Report PASS/FAIL without silently
retuning. A positive result demonstrates the engineered mechanism on this
task, not superior storage, biological proof, or a transformer replacement.

## User flows

`python -m solsbaby gui`: teach one of four branches, inspect the tree's
voltage/slow-state colors, probe, compare with erased memory, clear voltage,
erase memory, change seed, and toggle the active conductance.

`python -m solsbaby benchmark --out runs/example`: save protocol, JSON,
episode metrics, and a plotted report. `python -m solsbaby demo --out ...`
saves a deterministic visual walkthrough without requiring a display.
Errors must be actionable; invalid numerical parameters fail before solving.

## Verification

Tests cover tree topology/reproducibility, positive parameters, constant-state
axial conservation, passive energy loss, the tree solver against dense solves,
linear superposition/history subtraction, bounded local plasticity, fast vs
slow reset, transfer sensitivity to electrical geometry but not display pose,
train/test isolation, controls, serialization, and CLI invalid inputs.
Inspect generated plots and smoke-test the GUI when a display is available.
