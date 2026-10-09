# Preregistered predictions and analysis plan

**Study**: Where does meditation sit on the axis of conscious level? A spectrally controlled EEG comparison with sleep and anaesthesia.
**Status**: draft — locked with `python main.py preregister` before any meditation dataset (ds003969, ds001787) is downloaded. The lock stores the SHA-256 of this file and a UTC timestamp in `predictions.lock`; the fetcher refuses meditation data without a valid lock.
**Scope of the lock**: everything below. Preprocessing and metric parameters may be adjusted only while passing gates G0–G1 on the sleep and anaesthesia datasets, and are frozen in `eeg/config.yaml` before locking. Any later change is reported as a deviation.

---

## 1. Data

| Role | Dataset | Contrast |
|---|---|---|
| Reference (anaesthesia) | Chennu et al. 2016, propofol, Cambridge repository | baseline vs moderate sedation |
| Reference (sleep) | ANPHY-Sleep (OSF r26fh) | wake vs N3 (REM, N2 reported descriptively) |
| Primary (meditation) | OpenNeuro ds003969 | meditation vs instructed mind-wandering, within subject |
| Replication (meditation) | OpenNeuro ds001787 | meditation vs mind-wandering defined by experience-sampling probes |

ANPHY sample rule (fixed before any ANPHY data were processed): OSF hosts 27 of the 29 described participants (82.5 GB). The primary analysis uses the **first 15 subjects in ascending OSF ID order** (EPCTL02 …), a download budget fixed in advance; all 27 are analysed as a sensitivity check if processed. Per subject and stage, up to 10 scored 30-s windows (random, seed 42) → 30 epochs of 10 s.

Exclusion: subjects with < 20 clean 10-s epochs in either condition of a contrast; channels interpolated if bad; subjects with > 20% bad channels excluded. Exclusions are reported per dataset.

## 2. Measures (frozen definitions in `eeg/config.yaml`)

- **LZc**: multichannel Lempel–Ziv complexity (Schartner et al. 2015), envelope binarisation, entropy normalisation, 19 common 10–20 channels, 10-s epochs, 1–40 Hz, 250 Hz.
- **Spectrum-independent LZc (LZc_si)**: (LZc − mean of surrogates) / SD of surrogates, 20 multivariate phase-randomised surrogates per epoch.
- **Aperiodic exponent**: robust log–log fit of Welch PSD, 1–40 Hz, peaks excluded.
- **Theta synchrony**: Kuramoto order parameter r across channels, 4–8 Hz, surface-Laplacian (CSD) data.
- **Theta algebraic connectivity**: second-smallest eigenvalue of the normalised Laplacian (BQI eq. 96) of the full weighted wPLI (4–8 Hz) matrix, no threshold. (Sparse proportional thresholds disconnect a 19-channel graph and force λ₂ = 0; detected at gate G0.) Mean wPLI reported alongside, since λ₂ also scales with overall coupling strength.

Per-subject values are means over epochs, with epoch counts equalised between the two conditions of a contrast by random subsampling (seed 42).

## 3. Hypotheses (derived from the BQI model; directions and benchmarks fixed here)

| ID | Prediction | Test | Supported if |
|---|---|---|---|
| H1 | LZc_si higher in meditation than mind-wandering | paired d_z, ds003969 | 95% bootstrap CI of d_z > 0 |
| H2 | Meditation shifts toward the "higher" end of the conscious-level axis | projection c (Sec. 5) | 95% bootstrap CI of mean c > 0 |
| H3 | Theta Kuramoto r higher in meditation | paired d_z | CI > 0 |
| H4 | Aperiodic exponent larger (steeper) in meditation ("lower temperature", BQI Cor. 12.1) | paired d_z | CI > 0 |
| H5 | Theta Fiedler value higher in meditation | paired d_z | CI > 0 |

Benchmarks quoted by the model (for discussion only, not tested as point values): PCI 0.623 vs 0.567 (d = 0.64); theta r 0.71 vs 0.42; Fiedler 0.31 vs 0.18.

**Registered tension**: H1 and H4 pull in opposite directions for raw LZc (steeper spectra lower raw LZc). H1 is therefore tested on LZc_si; raw LZc is reported alongside.

**Falsification**: a hypothesis is *not supported* if its CI includes 0; *contradicted* if the CI lies entirely on the opposite side; *equivalent to zero* if the paired TOST with bounds |d_z| < 0.2 is significant (alpha = 0.05).

## 4. Inference

- Primary: paired d_z with 10,000 subject-level bootstrap resamples (percentile CI). Linear mixed model `metric ~ condition + (1 | subject)` as confirmation.
- Multiplicity: Benjamini–Hochberg across H1, H3, H4, H5 (q = 0.05). H2 is a single test.
- Replication (ds001787): same tests; reported as replicated if the effect has the same sign and its CI excludes 0.
- Channel-level maps: sign-flip cluster permutation (5,000 permutations), exploratory.

## 5. Conscious-level axis (H2)

1. Metrics z-scored within each reference dataset across all subjects and conditions.
2. Per-subject difference vectors over the K = 5 measures of Sec. 2: D_sleep = wake − N3, D_prop = baseline − moderate.
3. Gate G2: cosine between mean D_sleep and mean D_prop must exceed 0.5; otherwise H2 is reported as exploratory with two separate axes.
4. Axis u = first right singular vector of the pooled reference differences (uncentred), sign such that the mean reference shift projects positively.
5. Meditation: D_med = meditation − mind-wandering (z-scored within ds003969), projection c = (D_med · u) / (mean reference shift along u).
6. Orthogonal residual of mean D_med tested against a sign-flip null (10,000 draws); a significant residual with c ≈ 0 is interpreted as a qualitatively different state rather than a "higher" one.

## 6. Exploratory (not confirmatory)

Group (3 traditions vs control), experience-dependence, gamma 60–110 Hz in ds003969, ACE/SCE, thresholded-graph metrics (E_glob, σ at densities 0.1–0.3), Φ_R (whole-minus-sum Φ_AR with MMI redundancy correction; Φ_WMS reported too), and the state-space model of Q1_pipeline.md Sec. 7.2 fitted to ds001787 probes.

## 7. Gates (enforced in code, results in `outputs/gates/`)

- G0: synthetic tool validation passes (Q1_pipeline.md Sec. 8).
- G1: LZc wake > N3 (ANPHY) and baseline > moderate (Chennu), both CIs excluding 0.
- G2: reference contrasts agree (Sec. 5, step 3).
- G3: state-space parameter recovery (exploratory model only).
