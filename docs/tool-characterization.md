# Tool characterization

Offline instrument characterization: every tool in `modembench.agent.tools` run over all 40 captures of `captures/dev-v1`, with each reported quantity compared against the closed-form value implied by that capture's manifest. This path reads protected truth; it never runs inside an agent run.

- tools policy: `modembench-tools-v1`
- tools sha256: `c290fb6f7bcbc55dbd40ecf1c41c045abdf9591cc79c9265b4780f2715e92f9f`
- settings: nfft=512, max_lag=128, bins=32

## Error by reported quantity

| tool | quantity | n | median error | p90 abs | max abs | median rel | p90 abs rel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `amplitude_histogram` | `rms_magnitude` | 40 | 1.775e-05 | 0.0002799 | 0.00131 | 0.0003451 | 0.004073 |
| `spectrum` | `beta_from_occupied_bandwidth` | 40 | 0.08527 | 0.2328 | 0.3233 | 0.247 | 1.02 |
| `spectrum` | `noise_floor_psd` | 40 | 2.025e-12 | 5.089e-11 | 3.467e-10 | 0.005127 | 0.01659 |
| `spectrum` | `occupied_bandwidth_hz` | 40 | 3143 | 9597 | 1.616e+04 | 0.06519 | 0.1908 |
| `spectrum` | `spectral_centroid_hz` | 40 | 6.471 | 143.9 | 195.3 | -0.0004225 | 0.1772 |
| `symbol_period_statistic` | `first_envelope_null_lag_samples` | 40 | -7.5 | 14 | 19 | -0.303 | 0.4091 |
| `symbol_period_statistic` | `first_null_lag_samples` | 40 | 0 | 2 | 4 | 0 | 0.07692 |
| `symbol_rate_candidates` | `top_symbol_rate_hz` | 40 | 0 | 0.03846 | 0.04545 | 0 | 1.1e-06 |

## Discrete accuracy

- `symbol_rate_candidates` top-1: **40/40** (100%); top-3: 40/40
- `symbol_period_statistic` first-null lag: exact 12/40, within +/-1 33/40, within +/-2 38/40, worst 4 samples


## Measured negative results

What the instrument does **not** resolve, measured rather than asserted. Both rows used to be claims in a tool description; a deleted claim with no checked-in replacement is a claim nobody can re-test.

- **Roll-off (`rrc_beta`) is not recoverable from the occupied bandwidth.** Over 40 captures the implied roll-off correlates with the true one at r = 0.2641, with mean absolute error 0.1265 against beta's own spread (sd 0.09329). Quoting the population mean scores 0.08182 — better than the measurement, so the measurement carries no usable information. `beta` is the one receiver-critical parameter no tool estimates, and the agent is told so.
- **The squared-envelope autocorrelation null is not the symbol period.** It equals the period on 0/40 captures and lands within +/-2 samples on 2/40, sitting instead at 0.6849 of the period with a 15% relative spread. It is reported to the agent as a shape cross-check, with this error published, and its row appears in the error table above.

## AST gate strictness

Policy `modembench-ast-v5-math1-dunderdef1-noframes-libnarrowed` rejects **3/12** ordinary receiver idioms (25%). Every rejection scores as an agent failure, so this bounds how much of a measured failure rate is gate strictness rather than task difficulty.

| idiom | accepted | rules fired | note |
| --- | :---: | --- | --- |
| `numpy_and_scipy_only` | yes | - | the intended shape: numpy plus scipy.signal, module-level function |
| `stdlib_math` | yes | - | `import math` for pi and sqrt: the single most natural import to reach for |
| `class_with_constructor` | yes | - | a class holding carrier-loop state; `__init__` is a dunder *definition*, not a dunder read, and is accepted from `dunderdef1` on |
| `underscore_prefixed_attribute` | **no** | `private_attribute` | `state._value`: any attribute beginning with an underscore is refused |
| `main_guard` | yes | - | `if __name__ == '__main__':` for local testing, left in the submission; a comparison operand, not an attribute read, so accepted from `dunderdef1` on |
| `module_level_underscore_helper` | yes | - | `def _rrc(...)` as a private module helper, then called normally |
| `scipy_signal_from_import` | yes | - | `from scipy.signal import resample_poly` |
| `numpy_fft_and_linalg` | yes | - | `np.fft` and `np.linalg` for coarse frequency and least squares |
| `cmath_for_phase` | yes | - | `import cmath` to unwrap a phase |
| `getattr_dispatch` | **no** | `builtin_banned` | `getattr(signal, name)` to pick a filter by name |
| `generator_and_comprehensions` | yes | - | plain Python control flow, closures and comprehensions |
| `numpy_save_debug` | **no** | `numpy_io_attribute` | `np.save` left behind from local debugging |

## Estimable quantities

What each tool narrows about the hidden manifest. A tool that hands over a hidden parameter is not disqualified by that; it is disqualified by not saying so.

| tool | manifest field | quantity | strength |
| --- | --- | --- | --- |
| `spectrum` | `impairments.cfo.applied_value` | spectral centroid | direct estimate of the carrier frequency offset; median error under 2 Hz, p90 under 160 Hz over the dev split |
| `spectrum` | `impairments.awgn.noise_variance_per_component` | noise-floor PSD = 2 * sigma^2 / sample_rate | STRONG and previously unlisted: the median of the smoothed PSD recovers the AWGN noise power to a median 0.5% relative error, so the noise variance is effectively published. Combined with the histogram's RMS it bounds the in-record SNR, though not the manifest's per-symbol Es/N0, which also depends on the unpublished burst duty cycle |
| `spectrum` | `waveform.rrc_beta` | occupied bandwidth / estimated symbol rate, minus one | NONE, measured. The earlier claim that this 'resolves beta once sps is known' is false: over the dev split the implied roll-off correlates with the true one at r = 0.25 and its mean absolute error (0.126) exceeds beta's own spread (sd 0.092), so the population mean is the better estimator. Kept in the table as a checked-in negative result |
| `symbol_period_statistic` | `waveform.sps` | first signal-autocorrelation null lag | coarse period estimate: exact on 12 of 40 dev captures, within +/-2 samples on 38 of 40. Weaker than symbol_rate_candidates on the same parameter |
| `symbol_period_statistic` | `waveform.sps` | first squared-envelope autocorrelation null lag | NONE as a period estimate, measured. It is a different statistic: the first envelope null equals the symbol period on 0 of 40 dev captures and lands within +/-2 samples on 2 of 40, sitting at roughly 0.69 of the period. Reported as a shape cross-check with the published error, not as a period |
| `amplitude_histogram` | `impairments.amplitude.applied_value` | reported RMS scale | the gain, up to the unknown noise power; derivable by the receiver from iq.npy unaided, so this narrows nothing the sandbox does not already have |
| `symbol_rate_candidates` | `waveform.sps` | ranked integer samples-per-symbol | SUBSTANTIAL: reduces a 31-way discrete choice to a ranked shortlist, top-1 correct on 40 of 40 dev captures; the tool a tools-ablation arm removes |
| `symbol_rate_candidates + the capture's own length` | `framing.payload_length_bytes` | upper bound only: floor(capture_samples / sps) symbols minus the framing overhead | BOUND, NOT A DETERMINATION. The earlier claim that capture length, sps and pulse_span_symbols 'determine the frame symbol count, hence the payload length' is false: the burst starts at an unpublished offset into the record (1738 to 19598 samples over the dev split), so capture_samples/sps bounds the frame symbol count from above and determines nothing. Measured over the dev split the bound leaves a median of 90 of the 97 possible payload lengths feasible and binds at all (excludes any length) on only 22 of 40 captures. Note the span is genuinely shared -- the published pulse_span_symbols equals the manifest's rrc_span_symbols, 12 on every capture -- so it is the offset, not the span, that breaks the determination. Recorded so the calibration neither mistakes this for model capability nor for a leak |
