"""
Single source of truth for every experimental setting of the model paper
(Q1_pipeline_model.md). All five packages, the figure script, the dashboard
and the tests read these values; nothing is hard-coded elsewhere. Every
`run()` writes `config_snapshot.json` next to its results so each output
carries the settings it was produced with.

Changing a value here changes it everywhere; the pipeline then has to be rerun
(`python main.py model all`) so that gates, tables and figures stay consistent.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

SEED = 42                       # global seed; per-item seeds are deterministic functions of it
EPOCH_S = 10.0                  # EEG epoch length (s)
DT = EPOCH_S / 60.0             # epoch length in minutes (model rates are per minute)

# Nominal parameters = defaults of bqi.dmn_ode.StateSpaceDMN (kept there so the
# model class is self-contained); listed here for the snapshot and the paper.
NOMINAL = dict(k_inh=0.20, A0=0.60, sigma_A=0.10, k_int=0.081, k_d=0.034, sigma_Phi=0.05,
               alpha=2.34, beta=1.87, gamma0=0.0, r_A=0.10, r_Phi=0.10, r_Q=0.30, A_max=1.0)

KEY_PARAMS = ["k_inh", "A0", "alpha", "beta"]   # parameters reported in design/control analyses
TARGET_REL_ERROR = 0.20                           # "adequate design" = relative CRB / RMSE <= 20 %


@dataclass(frozen=True)
class ReferenceDesign:
    """ds001787-like experience-sampling design used as the common reference
    (identifiability table, CRB design cell, sensitivity output, control horizon)."""
    n_subjects: int = 12
    session_min: float = 45.0
    probe_every_min: float = 2.0
    r_obs: float = 0.10                # r_A = r_Phi
    config: str = "d"                  # rating + y_A + y_Phi

    @property
    def n_epochs(self) -> int:
        return int(round(self.session_min / DT))

    @property
    def probe_every(self) -> int:
        return max(1, int(round(self.probe_every_min / DT)))


REFERENCE = ReferenceDesign()


@dataclass(frozen=True)
class StructuralDesign:
    """Long, densely probed session for the Rothenberg rank analysis; gamma0 is
    set to a generic non-zero value so that no symmetry is hidden by 0."""
    session_min: float = 90.0
    probe_every_min: float = 1.0
    gamma0_generic: float = 0.3
    eig_tol: float = 1e-7


STRUCTURAL = StructuralDesign()


@dataclass(frozen=True)
class DesignGrid:
    """Full CRB grid (Sec 6) and the 24 representative cells for MLE recovery."""
    n_subjects: tuple = (6, 12, 24, 48)
    session_min: tuple = (20, 45, 90)
    probe_every_min: tuple = (0.5, 1.0, 2.0, 4.0)
    r_obs: tuple = (0.05, 0.10, 0.20)
    n_reps: int = 5                   # recoveries per representative cell
    n_starts: int = 2                 # L-BFGS-B restarts per fit
    init_jitter_log_sd: float = 0.3   # initial guess = truth * exp(N(0, sd))
    representative_cells: tuple = (
        (6, 20, 0.5, 0.05), (6, 20, 2.0, 0.10), (6, 45, 1.0, 0.10), (6, 90, 1.0, 0.05),
        (12, 20, 1.0, 0.10), (12, 20, 2.0, 0.20), (12, 45, 2.0, 0.10), (12, 45, 0.5, 0.05),
        (12, 90, 1.0, 0.10), (12, 90, 4.0, 0.20),
        (24, 20, 0.5, 0.05), (24, 20, 2.0, 0.10), (24, 45, 1.0, 0.10), (24, 45, 4.0, 0.20),
        (24, 90, 1.0, 0.05), (24, 90, 2.0, 0.10),
        (48, 20, 1.0, 0.10), (48, 20, 4.0, 0.20), (48, 45, 0.5, 0.05), (48, 45, 2.0, 0.10),
        (48, 90, 1.0, 0.05), (48, 90, 2.0, 0.10), (48, 90, 4.0, 0.20), (48, 90, 0.5, 0.05),
    )

    def cells(self) -> list[dict]:
        return [{"N": n, "session_min": s, "probe_every_min": p, "r_obs": r} for n, s, p, r in self.representative_cells]


GRID = DesignGrid()


@dataclass(frozen=True)
class ProbeSchedule:
    """Ds-optimality search over schedule families (Sec 6)."""
    power_exponents: tuple = tuple(round(x, 3) for x in [0.4 + i * (2.5 - 0.4) / 14 for i in range(15)])
    pair_gaps: tuple = (1, 2, 3, 6, 12)


SCHEDULE = ProbeSchedule()


@dataclass(frozen=True)
class Control:
    """LQG / bang-bang settings (Sec 7). Horizon = reference session."""
    target_qoc: float = 2.0
    u_max: float = 2.0
    b_control: float = 1.0
    Q_cost: float = 1.0
    R_cost: float = 0.1
    n_epochs: int = REFERENCE.n_epochs        # 270
    n_trials: int = 20                        # policy comparison
    n_trials_robustness: int = 10
    stationary_n_epochs: int = 4000           # closed-form check: long run
    stationary_burn_in: int = 1000
    analytic_tol: float = 0.20                # gate M2


CONTROL = Control()


@dataclass(frozen=True)
class Sensitivity:
    """Sobol/Saltelli settings (Sec 8)."""
    params: tuple = ("k_inh", "A0", "cv_A", "k_int", "k_d", "sigma_Phi", "alpha", "beta")
    factor_range: tuple = (0.25, 4.0)         # log-uniform multiplicative range
    cv_A_range: tuple = (0.05, 0.40)          # stationary CV of A instead of sigma_A
    n_base_fast: int = 16
    n_base_default: int = 64
    n_base_paper: int = 256
    sampler: str = "sobol_qmc"                # scrambled Sobol sequence (scipy.stats.qmc), n_base = power of 2
    # outputs analysed on log scale: they vary multiplicatively over the range, and on the
    # raw scale a few extreme samples dominate the variance (indices clip at 1, S1 << ST)
    log_outputs: tuple = ("qoc_variance", "crb_k_inh", "lqg_efficiency")
    n_boot: int = 1000                        # bootstrap resamples (rows) for index confidence intervals


SENSITIVITY = Sensitivity()


@dataclass(frozen=True)
class Gates:
    m0_residual_tol: float = 1e-6
    m0_p2_min_attenuation: float = 1e4
    m0_p2_max_amp_err: float = 0.01
    m0_p5_min_ks_p: float = 0.01
    m0_p5_thin_time_constants: float = 5.0
    m1_ratio_bounds: tuple = (0.5, 2.0)       # RMSE / CRB
    m1_min_fraction: float = 0.8
    m2_analytic_tol: float = CONTROL.analytic_tol


GATES = Gates()

# Theory checks (Sec 4): frequencies (rad/min) for the sinusoidal verification of P2
P2_OMEGAS = (0.05, 0.2, 1.0)
P2_STEP_XI = 2.0
P2_T_END_MIN = 600.0
P5_N_SAMPLES = 2000                           # thinned samples for the KS test, pooled over independent chains
P5_N_CHAINS = 5                               # seeds SEED, SEED+1, ... (a single path can drift for 100s of samples)

# Profile likelihood (Fig. 3): parameters and log-range around the truth
PROFILE_PARAMS = ("k_inh", "k_int", "k_d")
PROFILE_LOG_HALF_RANGE = 0.8
PROFILE_N_GRID = 9


def as_dict() -> dict:
    return {"seed": SEED, "epoch_s": EPOCH_S, "dt_min": DT, "nominal": NOMINAL, "key_params": KEY_PARAMS,
            "target_rel_error": TARGET_REL_ERROR, "reference": asdict(REFERENCE), "structural": asdict(STRUCTURAL),
            "grid": asdict(GRID), "schedule": asdict(SCHEDULE), "control": asdict(CONTROL),
            "sensitivity": asdict(SENSITIVITY), "gates": asdict(GATES),
            "theory": {"p2_omegas": P2_OMEGAS, "p2_step_xi": P2_STEP_XI, "p2_t_end_min": P2_T_END_MIN,
                       "p5_n_samples": P5_N_SAMPLES, "p5_n_chains": P5_N_CHAINS},
            "profiles": {"params": PROFILE_PARAMS, "log_half_range": PROFILE_LOG_HALF_RANGE, "n_grid": PROFILE_N_GRID}}


def snapshot(output_dir: str | None) -> str | None:
    """Write config_snapshot.json next to the results (idempotent)."""
    if not output_dir:
        return None
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(as_dict(), f, indent=2, default=str)
    return path


def check_nominal_matches_model() -> None:
    """Fail loudly if StateSpaceDMN defaults drift away from NOMINAL."""
    from bqi.dmn_ode import StateSpaceDMN
    m = StateSpaceDMN()
    bad = {k: (getattr(m, k), v) for k, v in NOMINAL.items() if abs(getattr(m, k) - v) > 1e-12}
    if bad:
        raise RuntimeError(f"model/config.NOMINAL khác StateSpaceDMN mặc định: {bad}")
