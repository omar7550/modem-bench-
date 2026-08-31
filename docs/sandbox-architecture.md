# Receiver sandbox architecture

Receiver isolation is a two-layer boundary. The primary layer is the macOS kernel
sandbox applied by `/usr/bin/sandbox-exec`; the AST policy is defense in depth and
library-surface enforcement, not the containment mechanism. The runner fails closed
with `sandbox_unavailable` on non-macOS systems or whenever its policy preflight
cannot prove the boundary.

The per-run SBPL profile allows general reads needed to start CPython and then denies
the repository, home, global temporary roots, external volumes, the resolved capture,
and configured sealed roots. It re-allows only the virtual environment and the run's
scratch directory. Network, fork, signal, Mach/IPC, and non-interpreter execution have
no allowances. The rendered profile contains protected path names, so it is passed
inline with `sandbox-exec -p` and is never written into scratch.

Sealed denial is structural, not a caller obligation. `build_policy` resolves the sealed
roots itself and always places them in both the read-deny and the metadata-deny sets; the
`sealed_roots` parameter may only add further roots. Both the root this process is pointed
at and the repository's own configured root are denied, so an environment override naming a
decoy directory cannot leave the real store undenied. A policy file that cannot be loaded
refuses to build a policy at all: an unprovable sealed root means the boundary the profile
claims cannot be demonstrated. Sealed runs must also pass
`permitted_capture_parents=[sealed_root]`, since `build_policy` otherwise rejects any
capture outside `repo/captures`. Metadata on the sealed root is denied for the same reason
it is denied on the capture: `st_size` of a truth file is itself an oracle.

That profile denies the *receiver* the sealed root; it says nothing about the orchestrator,
which legitimately reads it. The orchestrator side is therefore gated too, in exactly one
place. Three audit rounds each found a new ungated reader (`run_receiver`, then
`sandbox-replay`, then `make-oracle-receiver` and `python -m modembench.evaluate`);
gating call sites one at a time does not converge. `sealed.read_private_artifact` is
now the only function in the package that opens anything under a capture's `private/`
directory: a capture inside any sealed root in play requires a live session token and the
read is logged against that session. an isolation check walks the AST of
every module under `src/modembench` and fails the build if another module reaches a private
path, so a fourth call site cannot be added ungated. There is no way to ask for an unlogged
sealed read. The log record carries a digest of the capture path rather than the path: the
log is a repository file, and a sealed `capture_id` must not land in one.

A gate is only as good as the correspondence between the path it checks and the path it
opens, and two defects used to break that correspondence. The check ran on a *resolved*
capture directory while the read opened an *unresolved* path, so a capture whose
`private/` was a symlink into the sealed store passed. The chokepoint now resolves
the artifact once, decides about that resolved object as well as the capture, opens it with
`O_NOFOLLOW`, and re-verifies the descriptor's `(st_dev, st_ino)` against what it
authorized, so a swap landing between check and open is refused rather than served. Second,
the chokepoint accepted `policy=` and `root=`, which let the caller move the boundary being
checked; the AST rules cannot see that, because a call that relocates the gate looks
exactly like one that respects it. Every function that can reach the gate decision now
derives it from process state alone, and the same AST test fails the build if one grows a
parameter capable of moving it.

A receiver source is subject to the same rule. `receiver_path` was arbitrary and its bytes
are retained in `runs/`, so passing a sealed capture's generated oracle, which inlines the
sync word, the payload length and every applied impairment as literals, copied sealed truth
into the repository with no session and no log record. A receiver resolving inside a sealed
root is now accepted only when it is that capture's own protected oracle, read through the
chokepoint like everything else.

The evaluator needed a mechanism rather than a rule. It runs out-of-process, and a
capability token cannot cross a process boundary without becoming forgeable, so
`run_receiver` performs the single authorized read itself and hands the child the manifest
and payload on **stdin**. `python -m modembench.evaluate <bits> <capture>/private`
therefore refuses a sealed capture outright (it used to print an `aligned_ber` for one,
with no session, no budget and no log entry) while a legitimate sealed run still evaluates.

`replay_run` is gated the same way and for the same reason: deciding whether a retained run
still reproduces means re-hashing the capture's private manifest and payload, so replaying a
sealed run is itself a sealed read. It takes a session token, requires one for a sealed
capture, logs the read, and forwards the token to the replayed run. Without that forward a
sealed run would be permanently unreplayable, breaking the reproducibility the sealed design
promises for the release runs the hr-250 disclosure depends on.

Integrity *verification* is outside this picture. `verify-split` compares stored
artifacts against the published Merkle root: it needs no seeds, no salt and no session,
consumes none of the sealed budget, writes nothing (no policy write, no log append, no
anchor update), and runs unchanged on a read-only checkout. Sealed operations that can
expose content are authorized and logged; content-free integrity checking is neither, by
design. It reaches captures through `sealed.verification_leaf`, which returns a leaf
pre-image (a seed and three digests, exactly what `leaves.json` publishes at disclosure)
and never a byte of manifest or payload; only `splits.py` may call it, pinned by the same
AST test. Because re-hashing files and folding them through a pinned tree does not depend on
the runtime, an environment delta is reported as `environment_changed` *information*
alongside a real verdict rather than instead of one; the `environment_changed` verdict
survives only for a commitment pinning a Merkle construction this build cannot recompute.
The access log's chain head is anchored in `data/sealed_log_anchor.json`, a
committed file that is deliberately *not* a `source_tree_provenance` input, so a sealed read
does not move the hash whose job is to expose a policy edit.

Sessions outlive the process that opened them. The release campaign is >=360 evaluations over
hours, and liveness held only in one process's memory meant a transient failure spent the
authorization and killed the campaign. Liveness is recorded in the ledger (`closed_at`), so
any process may re-enter an open session by `(run_name, split_id)`; only the minting frame,
or an explicit `close_sealed`, closes it. A crash still consumes the authorization
(`opened_at` is what a crash cannot unwrite); it simply no longer bricks the run.

**Recovering an abandoned session.** Re-entry is bounded by a lease
(`sealed.SESSION_LEASE_SECONDS`, 24h, measured from that session's newest log record).
Without one, a campaign that crashed and was never resumed left `opened_at` set and
`closed_at` null forever, and that entry is full sealed authority for anyone who runs the
tool. Past the lease, re-entry is refused and names the recovery:

    close_sealed(split_id, run_name)

which marks the ledger entry closed and appends a `closed` record. The authorization stays
consumed; a crash spends a session by design, and refunding one here would undo the
fail-closed ordering that makes `opened_at` durable before the token is yielded. Running
again means adding a new `authorized_runs` entry to `data/sealed_access.json`, a tracked,
provenance-bound edit, which is the visibility a second session should cost.

**The kill switch.** `MODEMBENCH_SEALED_MAX_OPENS=0` means no sealed access at all: no new
session, no re-entry into one another process left live, no sealed generation, read, write
or salt extraction. It used to mean only "no new sessions", which left re-entry open, since
re-entry consumes nothing and never met the budget check. `close_sealed` is still permitted
at zero, so the switch cannot strand the ledger in a state nobody is allowed to tidy up.

**Known limitation.** `policy_template_sha256` normalizes only the scratch directory and
its parent, so the sealed root's absolute path is baked into every sealed run's replay
hash. Relocating the sealed store therefore invalidates replay of all prior sealed runs.
The root is fixed for the project's life; `data/sealed_access.json` carries the same note.

Scratch begins with only `iq.npy`, `receiver.py`, and the trusted `_shim.py`.
Standard output, standard error, bits, and diagnostic status are execution products.
The receiver can write `status.json`, so it is diagnostic-only. Success instead
requires a zero child exit, parent-revalidated `bits.npy`, and confirmation that the
whole process group is gone. CPU and file-size limits are kernel limits. Memory is a
sampled aggregate process-group RSS guardrail because macOS does not hard-enforce
`RLIMIT_AS` or `RLIMIT_RSS`; classification close to the threshold is best-effort.

Agent-visible provenance lives in `runs/<run_id>/run.json` and contains only receiver
identity, gate/sandbox results, timings, and Tier-2 feedback. Protected details live in
`.orchestrator/run-internal.json`, including truth hashes, evaluator details, the
rendered policy, and any protected oracle locator, **for an unsealed capture only**. The
run root lives in the repository the agent works in, and a per-capture
`payload_sha256` there is an off-budget correctness oracle: an agent can test candidate
payloads without spending an evaluator call, corrupting the budget-matched comparison the
comparison gate depends on. That defect refuted an earlier revision, and the run record
reintroduced it. For a sealed capture neither file carries a truth hash, a `capture_id`, a
sealed path or the rendered profile; they carry an opaque `capture_ref`, and the real
identity is written to `<sealed root>/run-records/<run_id>.json`, which replay resolves with
a session token. The evaluator's internal block is redacted there for the same reason: it
reported `n_payload_bits`, the true payload length, verbatim into the repository, while the
sandbox profile already denies `file-read-metadata` on the truth roots specifically so a
receiver cannot learn that number from `payload.bin`'s size. `alignment_offset` and
`sync_hamming` go with it; `packet_success` stays, because it is one bit and the
budget-matched comparison is built on it. Ordinary source and validated bits are
retained by content; generated oracle source remains in the protected capture store and
is referenced rather than copied. Replay refuses changed source, truth, evaluator,
policy, shim, or runtime versions before classifying a rerun as `reproduced` or
`nondeterministic`.

`sandbox-exec` is Apple-deprecated but remains shipped and functional on the target
Darwin platform. This is a strong benchmark-integrity boundary for preventing an
agent-authored receiver from reaching ground truth; it is not VM-grade containment.
The shipped AST surface also keeps `ALLOW_STDLIB_MATH = False` pending resolution of
the locked-document `math`/`cmath` question.
