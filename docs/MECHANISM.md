# The mechanism and what the test means

## A medium implements a response operator

For a fixed passive cable discretization,

\[
C\dot v=-Gv+Bu,\quad y=Sv,\qquad
H(\omega)=S(G+i\omega C)^{-1}B.
\]

`morphology.py` generates a rooted tree from a seeded recipe. The resulting
axial Laplacian is symmetric, annihilates constant voltage, and conserves
axial current. Leak dissipates energy. Length and diameter influence
capacitance and conductance; drawing coordinates have no independent
electrical effect. This is a dimensionless illustrative model, not a
reconstructed neuron or a genome/connectome measurement.

## An incoming signal can also change the operator

The implemented model is

\[
C_i\dot v_i=-(L_{ax}v)_i
-(g_{L,i}+\gamma C_i p_i)v_i
+g_{in,i}(t)(E-v_i)+g_{a,i}(v)(E-v_i)+I_i(t).
\]

The input is normally an excitatory conductance. Its `g(E-v)` form changes
both the forcing and the coefficient on voltage. `current=` permits an
independent current-injection test of the exactly linear passive case.

The optional active term is a sigmoid voltage-dependent conductance on
nonproximal branch segments, with its resting activation subtracted. It
is evaluated at the previous voltage in each implicit step. It is an
NMDA-like demonstration of regenerative gain, not a receptor-level model.

Writing follows the explicit, engineered local rule

\[
\dot p_i=\eta[v_i-\theta]_+(1-p_i)-p_i/\tau_p.
\]

The frozen-rate update of this equation is integrated analytically, keeping
`p` in [0,1]. A positive tag increases a shunt. No class label, target,
classification gradient, or answer is passed to this rule. This is an
activity trace that alters conductance; it does not demonstrate a discovered
biological learning rule, predictive learning, or optimized task credit.

For voltages, each step solves an implicit linear tree system. All implicit
conductances are nonnegative. The Hines elimination uses O(N) arithmetic
per trajectory and step. Batched episodes share morphology. This algorithmic
fact is not a measured hardware-speed advantage.

## The probe is deliberately restricted

Each registered episode starts from rest. One branch receives three pulses
with independent amplitude/timing nuisance. Writing is allowed during this
experience. During washout, only slow decay continues. Then all fast voltage
is explicitly reset; slow tags remain. A common four-slot probe stimulates
the branch tips in a fixed order, regardless of the preceding context.

Probe and no-probe trajectories start from copies of the same state. Their
somatic difference is sampled every three steps. There is one physical
readout location, but 80 scalar measurements across time, and a known
spatial stimulation schedule. Local writing is frozen during both readings;
ordinary slow decay continues. The experiment is not a nondestructive-read
claim about real neurons. GUI comparisons preserve the live state by using
these simulated copies.

A ridge classifier is trained separately for each morphology seed using
only training episodes and their soma traces. Test episodes do not fit
normalization or weights. Observation noise is added to sampled traces
before the reader. Separate random streams generate train/test contexts
and observation noise. The oracle sees noiseless per-compartment slow tags.

This is interpolation among four familiar context classes. Decoder training
is supervised even though the medium's local writer receives no labels.

## What disappears when memory is erased?

Fixed linear passive dynamics retain transient voltage, but their
baseline-subtracted response to an added current probe is independent of
that previous voltage. A unit test checks this directly.

Here, slow state changes conductances. A probe response can therefore
depend on past input after voltage is gone. Erasing tags restores the
history-independent baseline; freezing the writer prevents their creation.
These are the registered controls. One label permutation per seed is
reported as a diagnostic; it is not a well-estimated permutation null.

Sihti's residue is a difference between states. For a diffusion step,
`r = x - x_next = dt*L*x`: it resembles the net-current-driven change,
with capacitance/leak accounted for in the cable analogy. The transmitted
somatic voltage is another quantity. SolsBaby does not assume a dendrite
stores all diffusion residues and forwards them intact.

## Reproducibility and limits

The committed `protocols/v1.json` preceded the first held-out run. The
canonical JSON SHA256 is saved with every result. Altering seeds, sample
counts, noise, or other protocol fields changes that hash. Original v1
measurements and predictions are retained in `results/v1/`.

One read simulates 240 probe and 240 blank integration steps. Persistent
slow state uses 168 bytes, plus voltage, static morphology, parameters,
reader weights, and computation. The four-way label can be stored in two
bits; this is an operator-memory demonstration, not compression.

Not tested: long sequences with interference, unseen context combinations,
input-distribution shifts, destructive reads, biological morphology fitting,
learning structural growth, competitive storage capacity, or language tasks.
The benchmark does not establish a benefit from the optional active term.
