# Anchoring the difficulty to real-world practice

Principle: rather than tuning difficulty to hit a statistical target band, ground every
difficulty parameter in what real receivers and real intercept scenarios face, and let the
difficulty be whatever realism produces.

## The scenario this benchmark is

Two kinds of receiver exist in practice, with different problems:

* **Cooperative** (a Zigbee chip, a pager, a satellite modem): built for the signal. It knows
  the symbol rate, the sync word and the framing from the published standard; its problems are
  noise, oscillator error and timing.
* **Non-cooperative** (spectrum monitoring, signals intelligence, cognitive radio): not built
  for the signal. It must estimate the symbol rate, find the burst, and infer the structure
  before it can decode a bit. The blind-estimation literature exists for this scenario;
  symbol-rate estimation is described as "a necessary condition for the signal decoding of
  blind receivers" in non-cooperative environments (military reconnaissance, radio monitoring,
  cognitive radio).

ModemBench is non-cooperative by construction: the brief is an LLM writing a receiver blind
from raw IQ with no waveform parameters. Its difficulty should therefore match the
non-cooperative scenario. "Industry standard" here means the working conditions of a blind
intercept receiver, with the solvability of a real link, not the conditions of a
standards-compliant chip.

## Parameter by parameter

### Signal-to-noise ratio: the frozen range brackets the physical cliff

Uncoded BPSK has a closed-form error rate: BER = ½·erfc(√(Eb/N0)). Packets are ~700–1,100
bits and success requires **zero** errors, so the probability a *perfect* receiver decodes a
packet cleanly is (1−BER)^bits:

| Eb/N0 | BER | P(zero errors in 900 bits) |
|---:|---:|---:|
| 6.0 dB | 2.4e-3 | 0.12 |
| 7.0 dB | 7.7e-4 | 0.50 |
| **7.8 dB** | **2.6e-4** | **0.79** |
| 8.4 dB | 1.0e-4 | 0.91 |
| 9.6 dB | 9.7e-6 | 0.99 |
| 12+ dB | <1e-8 | ~1.00 |

Practical uncoded links are engineered around this cliff: satellite telemetry modems target
~9–9.6 dB for 1e-5 BER plus 1–2 dB implementation margin; a measured AX.25/GFSK
implementation reaches BER < 1e-4 at ~6 dB Eb/N0 (FSK and PSK differ, but the order is the
point). The δ=0.70 range (7.8–17.8 dB Es/N0; Es=Eb for BPSK) starts at the cliff edge and
extends into the comfortable field-link region. A range lower than this measures channel
physics rather than receivers, and the upper half is where real uncoded links operate.
Verdict: δ=0.70 is the realistic setting, and also the hardest legal one. Keep it.

### Carrier frequency offset: narrowband intercept arithmetic

Standards put oscillator tolerance at ±20 ppm (802.11) to ±40 ppm per node / ±80 ppm system
(IEEE 802.15.4). The absolute offset that produces depends on the carrier frequency: 80 ppm
at 915 MHz is ~73 kHz. What matters to a demodulator is the offset **relative to the symbol
rate**: for a wideband signal (2 Mchip/s Zigbee) that is ~3.7%, but for a narrowband signal,
such as a 4,800-baud telemetry or paging channel at the same carrier, the same oscillators
produce an offset of multiples of the symbol rate. Narrowband, low-rate signals are the ones
a blind receiver most often faces, and large fractional CFO is their defining nuisance. The
δ=0.70 range of ±24% of the symbol rate sits inside that narrowband reality: harder than a
wideband standard link, far easier than the worst narrowband case. Verdict: realistic; keep.

### Burst placement: the monitored-spectrum geometry

A cooperative receiver is synchronized to its network; an intercept receiver records a band
and finds bursts wherever they land, with noise before and after. The pre-burst-axis
geometry (packet always inside the first 20k samples, recording always ending at the
packet's last sample) was a hidden disclosure no real capture provides: "everything from the
energy rise to the end" was a complete burst detector. The burst axis (random placement,
trailing noise) restores the real geometry. Preamble budget is comparable to practice: a
64-bit sync word against 802.15.4's 40 bits of preamble+SFD and POCSAG's 576-bit preamble.
Verdict: burst axis ON is the realistic geometry, independent of its measured (null) effect
on the success rate.

### Symbol-rate knowledge: estimation is receiver work in the blind scenario

A cooperative receiver reads its symbol rate out of the standard. A non-cooperative receiver
estimates it; that is what the blind symbol-rate estimation literature (cyclostationary
spectral correlation, autocorrelation methods, wavelet methods) is for, and those estimators
are components engineers build into blind receivers, not oracles handed to them. The
`symbol_rate_candidates` tool as shipped was closer to an oracle: measured top-1-correct on
40/40 dev captures, with the prompt asserting so. Withholding it does not deprive the agent
of anything a blind receiver has; the sandbox grants numpy and scipy.signal, so the agent
can (and the successful 27/40 receivers do) implement rate estimation from the
autocorrelation and spectrum inside `receiver.py`, where the literature puts it. Verdict:
withholding the rate tool is the realistic configuration for a non-cooperative benchmark.
The spectrum, autocorrelation and envelope tools stay; they are the classic instruments any
analyst has.

## The conclusion, and what it changes

The realism-anchored difficulty is the composition the calibration measured last:
burst ON + δ=0.70 + rate tool withheld → one-shot baseline 0.675 [0.52, 0.80]. Every
parameter now has a real-world anchor rather than a statistical target, a stronger
justification than the retired 0.25–0.40 band: the band was chosen to make an effect
detectable, while this setting is chosen because it is what the task is, with detectability
bought by sample size (80 signals; see `docs/n-r-rederivation.md` and the power table in
`docs/difficulty-calibration.md`).

One footnote the report must carry: part of the 0.675 is physics, not model failure.
Captures drawn near the 7.8 dB edge have a ceiling below 1.0 for any receiver. The
per-capture oracle ceiling is computable from the manifest SNR and should be reported beside
the model rate in the calibration section.

Sources:
- [IEEE 802.15.4 receiver sensitivity and packet structure](https://www.sciencedirect.com/topics/computer-science/receiver-sensitivity)
- [Theoretical and Practical Limits to Sensitivity in IEEE 802.15.4 Receivers](https://www.researchgate.net/publication/224311816_Theoretical_and_Practical_Limits_to_Sensitivity_in_IEEE_802154_Receivers)
- [NXP AN3251: Reference Oscillator Crystal Requirements (802.15.4 ±40 ppm)](https://www.nxp.com/docs/en/application-note/AN3251.pdf)
- [MATLAB berawgn: BER for uncoded data over AWGN](https://www.mathworks.com/help/comm/ref/berawgn.html)
- [BER calculation notes (Meghdadi)](https://www.unilim.fr/pages_perso/vahid/notes/ber_awgn.pdf)
- [FX.25 / AX.25 forward error correction background](https://en.wikipedia.org/wiki/FX.25_Forward_Error_Correction)
- [Blind symbol rate estimation using autocorrelation and zero crossing detection](https://www.researchgate.net/publication/261271634_Blind_symbol_rate_estimation_using_autocorrelation_and_zero_crossing_detection)
- [Blind estimation of carrier frequency and symbol rate from cyclic spectrum density](https://www.researchgate.net/publication/257723891_Blind_Estimation_of_Carrier_Frequency_and_Symbol_Rate_Based_on_Cyclic_Spectrum_Density)
- [Spectral-correlation blind modulation classification with symbol-rate estimation](https://globals.ieice.org/en_transactions/communications/10.1587/transcom.E96.B.1158/_p)
- [ExpressLRS: crystal oscillator frequency error in practice](https://www.expresslrs.org/hardware/crystal-frequency-error/)
