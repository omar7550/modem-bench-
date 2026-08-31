# ModemBench: what this project is and how the code does it

This document assumes no prior knowledge of the project, radio, or the codebase. Each heading
links to the relevant Python files. A test fails the build if the document goes stale; §13 has
the details.

---

## 1. The one-sentence version

The benchmark hands a language model a raw recording of a radio transmission, tells it almost
nothing about how the transmission was made, and asks it to **write the software that decodes
it**. The recovered bits are then compared against the bits that were sent.

Everything else in the repository exists to make that comparison trustworthy.

---

## 2. Why this is a benchmark worth having

### The practical problem

Most coding benchmarks let the model check its own work: write a function, run the tests. A
signal processing engineer handed an unknown recording has no test suite, only measurements and
physics, and has to infer the structure of the problem before solving it.

Digital radio has useful properties for benchmarking this:

- **Objective.** Either the exact bits were recovered or they were not. No rubric, no judge
  model, no partial credit.
- **Not guessable.** The payload is random data protected by a checksum.
- **Hard.** Roughly eight distinct inferences must be chained, each useless without the others.
- **Generatable.** Fresh problems can be produced indefinitely, so memorization does not apply.

### Vocabulary

A radio transmitter takes bits, turns them into a wave, and sends it. A **receiver** does the
reverse. The recording given to the model is an **IQ capture**: a long list of numbers
describing that wave, analogous to an audio file where the "sound" is a data transmission.

The receiver has to work out four things, none of which are given:

| What | In plain terms | Why it's hard |
|---|---|---|
| **Symbol rate** | How fast the symbols arrive | Guess wrong by one step and you decode pure noise |
| **Carrier offset** | The transmitter's clock is slightly off from the receiver's | The whole signal slowly spins; undo it or nothing lines up |
| **Timing phase** | *Where within each symbol* to take your reading | Sample at the wrong instant and every bit is a coin flip |
| **Frame position** | Where in the recording the actual packet starts | Everything before it is silence and noise |

These parameters interact: none can usually be solved until the others are approximately
solved. That is what makes the task a test of reasoning rather than recall.

### The research question

Not "can a model write a receiver"; that is answered (§9). The question:

> **Does letting a model see how its last attempt scored, and try again, make it better?**

§7 covers how this is measured.

---

## 3. The shape of the whole system

```
   [ generator ]  makes a signal from secret parameters
         |
         |  iq.npy   (the recording, public)
         |  meta.json (a few harmless facts, public)
         |  private/ (the answer key; never shown to anyone)
         v
   [ agent harness ]  shows the model the recording via measuring tools
         |
         |  receiver.py  (the model's answer: a Python program)
         v
   [ AST gate ]  reads the code and refuses anything that tries to cheat
         |
         v
   [ kernel sandbox ]  runs it with no network, no filesystem, no escape
         |
         |  bits.npy  (what the model's program decoded)
         v
   [ evaluator ]  compares against the answer key -> did it get it exactly right?
         |
         v
   [ arms + analysis ]  many runs, two conditions, one statistical verdict
```

---

## 4. Making the problems: [`generator.py`](../src/modembench/generator.py)

Builds a capture from a random seed: picks the hidden parameters, generates a packet, shapes it
into a waveform, and writes three things: the recording, a small public metadata file, and a
**private** directory holding the true bits and the true parameters.

Two supporting pieces:

- [`framing.py`](../src/modembench/framing.py): the packet layout. A **sync word** (a known
  pattern marking the packet start), a length field, the payload, and a **CRC** (a checksum
  that catches errors).
- [`impairments.py`](../src/modembench/impairments.py): degrades the signal with noise,
  frequency drift, timing error, and multipath echo. This is the difficulty knob. A clean
  signal is trivially decodable; the calibration work in §9 is about finding a setting that is
  hard but solvable.

Rule enforced in code: the truth lives in `private/`, and no module the agent can reach may
name that path. the release history's isolation tests
parses every source file and fails the build if one does.

## 5. Knowing the problems are solvable: [`reference_rx.py`](../src/modembench/reference_rx.py)

A reference receiver, written using only the public information the model gets, must decode
every capture. If it fails, the capture is at fault, not the model.

- [`evaluator.py`](../src/modembench/evaluator.py): the scorer. Finds the sync word in the
  receiver's output, aligns to it, and compares bit by bit. Produces the headline
  **`packet_success`**: correct length, zero bit errors, CRC verified.
- [`gate.py`](../src/modembench/gate.py): a quality bar the reference receiver has to clear
  before a capture set is usable.
- [`diagnostics.py`](../src/modembench/diagnostics.py): tools for diagnosing *why* a decode
  failed rather than only *that* it did.

## 6. The security boundary: [`sandbox/`](../src/modembench/sandbox/)

The model writes arbitrary Python and it runs on the same machine as the answer key.

Two layers; only one is the boundary:

- [`sandbox/profile.py`](../src/modembench/sandbox/profile.py): the actual boundary. A macOS
  kernel-level sandbox policy: no network, no subprocesses, no writable filesystem outside one
  scratch directory, no read access to the private tree. Enforced by the operating system.
- [`sandbox/ast_gate.py`](../src/modembench/sandbox/ast_gate.py): defense in depth. Reads the
  submitted source *before running it* and rejects known escape idioms: `open`, `eval`,
  `__import__`, underscore attributes, NumPy's file-reading functions. This is explicitly not
  the boundary, since a static checker can be outwitted; it is a fast first rejection.
- [`sandbox/runner.py`](../src/modembench/sandbox/runner.py): drives execution and attributes
  faults. A receiver that crashed is the model's problem; the harness's own loader crashing is
  not, and must never be scored against the model.

Cost of the design: the AST gate rejects about 3 in 12 ordinary, innocent receiver idioms.
Measured in [`docs/tool-characterization.md`](tool-characterization.md); §7 covers how it is
accounted for.

## 7. Asking the model: [`agent/`](../src/modembench/agent/)

### What the model can see

- [`agent/tools.py`](../src/modembench/agent/tools.py): four measuring instruments. The model
  never sees the raw samples; it calls tools that measure the recording: frequency spectrum,
  autocorrelation, amplitude histogram, and a ranked list of likely symbol rates.
- [`agent/characterize.py`](../src/modembench/agent/characterize.py): measures instrument
  accuracy against truth over the whole development set. The results are given to the model,
  since an instrument with unknown error is not usable as an instrument. Example: the
  symbol-rate tool is top-1 correct on all 40 development captures, and the model is told so.

### The wall: [`agent/feedback.py`](../src/modembench/agent/feedback.py)

After a receiver runs, the model is told four things: whether the sync word was found, whether
the CRC passed, the bit error rate, and any error code. Nothing else.

The bit error rate is errors divided by payload length. Published exactly, continued-fraction
expansion recovers **the payload length** from a single value; this was verified working on a
real capture. The value is therefore snapped to the nearest 1/64 before the model sees it. That
grid was chosen so that no publishable value can reduce to a real payload length.

This function is the only path from the scorer to the model.

### Running one attempt: [`agent/harness.py`](../src/modembench/agent/harness.py)

The one-shot condition: measure, write a receiver, execute once. No feedback. This file also
owns the **outcome taxonomy**: the rules deciding whether a run counts as a success, a failure
the model earned, or an invalid run that is thrown away and re-run because the harness broke
rather than the model's code.

That distinction matters. An earlier version lumped six different crash causes together and
scored all of them against the model, which would have understated the model in every sweep run
against it.

### Running the loop: [`agent/iterative.py`](../src/modembench/agent/iterative.py)

The arm the thesis is about: write a receiver, execute it, read the four-field result, repair,
execute again. Three decisions in it that matter:

1. **Two rounds, described as two rounds, not "iteration".** The budget funds exactly two.
   Round 2 starts fresh (next point), so the model gets the *result* of its last attempt but
   not its own reasoning. What is measured is **single repair**, not an agent reflecting over
   time. The limits are recorded in the data as `MEASURED_CONSTRUCT`, a list of what a report
   may and may not claim from these numbers.
2. **Round 2 is a fresh context, not a conversation.** Carrying round-1 reasoning forward
   costs 2.76x the entire budget headroom. Round 2 carries the code, the result, the approach
   note and the round-1 measurements, not the reasoning. Consequence: this is not a
   conversational agent and cannot refer back to "what I tried last time".
3. **A broken round is re-run, not scored.** If the harness fails mid-loop, the round is
   re-issued and does not count against the model's two. Otherwise the loop's failure rate
   would be inflated simply by having more rounds in which the harness could break.

### Money and reproducibility

- [`agent/provider.py`](../src/modembench/agent/provider.py): talks to the model API, and can
  replay recorded responses instead, so the test suite runs offline.
- [`agent/subscription.py`](../src/modembench/agent/subscription.py): the same conversation
  served over the Claude CLI instead of the API, so the campaign can run on a subscription
  account. Each turn is one stateless CLI call carrying the whole serialized history; the
  harness cannot tell the two transports apart.
- [`agent/accounting.py`](../src/modembench/agent/accounting.py): costs every run under the
  real five-rate price table, on the run's own date. Prices changed mid-campaign; pricing
  everything at one date under-reported the sealed campaign by a third.

## 8. Comparing fairly: [`arms/`](../src/modembench/arms/)

Two conditions; the comparison between them is the result:

- **best-of-N** ([`arms/bestofn.py`](../src/modembench/arms/bestofn.py)): N blind attempts,
  none of which can see the others, then pick one.
- **iterative** ([`agent/iterative.py`](../src/modembench/agent/iterative.py)): one run of N
  rounds, each of which can see how the last one scored.

Both arms get the same budget. [`arms/budget.py`](../src/modembench/arms/budget.py) derives one
per-signal budget from the funding available, holds both arms to it, and hashes the entire
configuration so two runs that differ in any way cannot be silently compared.

Two rules:

- **The selector must be deployable.** best-of-N picks its answer using the CRC, which a real
  fielded receiver can check by itself. Picking with the bit error rate would be picking with
  the answer key, which measures an oracle nobody can deploy.
- **Under-spending is never an alarm.** Any measure of "how much budget did it use" moves
  toward firing as the loop gets better, because a working loop stops early on success. Three
  separate thresholds were written on that axis before the pattern was named.

[`arms/ledger.py`](../src/modembench/arms/ledger.py) tracks the money: what the campaign costs,
what is affordable, and therefore what N is. N is what the budget buys, not a chosen constant.

## 9. What is known so far

**Settled:** a model can write a working receiver from raw IQ. **9 of 10** development captures
decoded bit-exactly, first attempt, no feedback, through the real gate, sandbox and scorer.
Measured over the development split through the full gate, sandbox, and evaluator.

**A problem that creates:** at a 90% baseline, a *perfect* loop can only improve by 10 points,
against a 15-point threshold. The gate is arithmetically unpassable at the current difficulty;
this is a ceiling, not a statistics problem. Difficulty has to come down before the thesis is
testable. The hardest legal impairment setting reaches 82.5% (33/40), which is still not
enough. Established by a measured calibration campaign across every legal difficulty axis.

**Not yet measured:** whether feedback helps. The loop is built and tested; no paid run of it
has happened yet. It is gated on the difficulty reduction, since measuring a 10-point ceiling
against a 15-point threshold would waste the budget.

## 10. Keeping everyone honest

The integrity machinery is heavy for a project this size, because the failure mode of a
benchmark is a confident wrong answer.

- **Claims are written down before the data.** A pre-registration fixes every threshold in
  advance.
- **Routing is code, not prose.** [`conclusions.py`](../src/modembench/conclusions.py) decides
  which conclusion a result supports and *generates* the corresponding document section, so the
  document cannot drift from the rule.
- **Test captures are sealed.** [`splits.py`](../src/modembench/splits.py) and
  [`merkle.py`](../src/modembench/merkle.py) publish a cryptographic commitment to the hidden
  test set *before* anyone runs against it. Access is counted: there are exactly two authorized
  openings.
- **Records cannot leak.** [`records.py`](../src/modembench/records.py) is the single writer
  for anything that reaches disk. Sealed identity leaked into visible files **five times**
  during development. The durable fix moved the decision rather than the mechanism: sealedness
  is derived from the capture, never passed as a flag, so there is no boolean to get wrong.
- **The provenance trace is hash-chained.** Every round of every loop is one line in an
  append-only file, each carrying the hash of the previous line, with the final hash recorded
  in the run record. Editing, deleting or truncating any line is detectable. Without the chain,
  "append-only" is just a filesystem mode.

---

## 11. If you only open five files

| File | Why |
|---|---|
| [`agent/feedback.py`](../src/modembench/agent/feedback.py) | The only path from scorer to model |
| [`agent/iterative.py`](../src/modembench/agent/iterative.py) | The arm the thesis is about |
| [`sandbox/profile.py`](../src/modembench/sandbox/profile.py) | The boundary between model code and the answer key |
| [`evaluator.py`](../src/modembench/evaluator.py) | What "success" means, exactly |
| [`arms/budget.py`](../src/modembench/arms/budget.py) | Why the comparison is fair |

## 12. Every module, in one line each

**Signal chain**: [`generator.py`](../src/modembench/generator.py) makes captures ·
[`framing.py`](../src/modembench/framing.py) packet layout ·
[`impairments.py`](../src/modembench/impairments.py) the difficulty knob ·
[`reference_rx.py`](../src/modembench/reference_rx.py) the reference receiver ·
[`evaluator.py`](../src/modembench/evaluator.py) the scorer ·
[`evaluate.py`](../src/modembench/evaluate.py) the scoring entry point ·
[`gate.py`](../src/modembench/gate.py) the quality bar ·
[`diagnostics.py`](../src/modembench/diagnostics.py) failure analysis

**Sandbox**: [`sandbox/profile.py`](../src/modembench/sandbox/profile.py) the kernel boundary ·
[`sandbox/ast_gate.py`](../src/modembench/sandbox/ast_gate.py) the static pre-check ·
[`sandbox/runner.py`](../src/modembench/sandbox/runner.py) execution and fault attribution ·
[`sandbox/shim_template.py`](../src/modembench/sandbox/shim_template.py) the small trusted program that loads the recording, calls the model's `receive`, and writes the bits; the only trusted code inside the sandbox ·
[`sandbox/oracle_source.py`](../src/modembench/sandbox/oracle_source.py) a known-good receiver, to prove the sandbox can pass as well as fail

**Agent**: [`agent/harness.py`](../src/modembench/agent/harness.py) one-shot runs and the outcome taxonomy ·
[`agent/iterative.py`](../src/modembench/agent/iterative.py) the feedback loop ·
[`agent/tools.py`](../src/modembench/agent/tools.py) the four instruments ·
[`agent/characterize.py`](../src/modembench/agent/characterize.py) instrument error measurement ·
[`agent/feedback.py`](../src/modembench/agent/feedback.py) the feedback wall ·
[`agent/provider.py`](../src/modembench/agent/provider.py) API and replay ·
[`agent/accounting.py`](../src/modembench/agent/accounting.py) real pricing ·
[`agent/subscription.py`](../src/modembench/agent/subscription.py) the CLI transport

**Comparison**: [`arms/budget.py`](../src/modembench/arms/budget.py) the shared budget and the config hashes ·
[`gate_analysis.py`](../src/modembench/gate_analysis.py) the verdict machinery: paired bootstrap, McNemar, and the threshold rule; the conclusion is computed, not written ·
[`arms/bestofn.py`](../src/modembench/arms/bestofn.py) the blind-attempts arm ·
[`arms/ledger.py`](../src/modembench/arms/ledger.py) what the campaign costs

**Integrity**: [`records.py`](../src/modembench/records.py) the single writer ·
[`frozen.py`](../src/modembench/frozen.py) the freeze artifact: what the campaign is, written once before it runs and re-verified before any run spends ·
[`sealed.py`](../src/modembench/sealed.py) counted access to the test set ·
[`splits.py`](../src/modembench/splits.py) dataset splits and commitments ·
[`merkle.py`](../src/modembench/merkle.py) the commitment construction ·
[`conclusions.py`](../src/modembench/conclusions.py) result routing as code

**Entry point**: [`cli.py`](../src/modembench/cli.py) every command

---

## 13. How this document stays true

It is checked against the tree before release: every module under `src/modembench/` must be
named here, and every link must resolve. §9 in particular is the running answer to "what do
we actually know", and changes as results arrive.

*Last substantive update: 2026-08-24 (the iterative arm and the provenance trace).*
