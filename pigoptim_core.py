# ============================================================
# pigoptim_core.py
# Logique pure du modele PigOptim V3 (portage Python du script R)
# Aucune dependance a Streamlit ici : uniquement numpy / pandas.
# Cela permet de tester et de reutiliser le moteur de simulation
# independamment de l'interface graphique.
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict, field
from typing import Optional


# ============================================================
# 1. PROFILS SANITAIRES (maladies)
# ============================================================
# Chaque profil contient des valeurs par defaut cliniques /
# epidemiologiques qui peuvent ensuite etre surchargees par
# l'utilisateur (parametrage complet demande).

DISEASE_PROFILES: dict[str, dict] = {
    "aucune": dict(
        label="Aucune",
        affected_fraction=0.00, onset=0, acute_duration=0,
        treatment_duration=0, withdrawal_duration=0,
        acute_adg=1.00, treatment_adg=1.00, recovery_adg=1.00, chronic_adg=1.00,
        acute_fcr=1.00, treatment_fcr=1.00, recovery_fcr=1.00, chronic_fcr=1.00,
        mortality_annual=0.000, incomplete_recovery=0.00, vet_cost=0.0,
    ),
    "mycoplasma": dict(
        label="Mycoplasma hyopneumoniae",
        affected_fraction=0.30, onset=0, acute_duration=21,
        treatment_duration=10, withdrawal_duration=7,
        acute_adg=0.82, treatment_adg=0.88, recovery_adg=0.94, chronic_adg=0.98,
        acute_fcr=1.15, treatment_fcr=1.11, recovery_fcr=1.06, chronic_fcr=1.02,
        mortality_annual=0.010, incomplete_recovery=0.10, vet_cost=3.0,
    ),
    "pcv2": dict(
        label="Circovirus porcin (PCV2)",
        affected_fraction=0.25, onset=0, acute_duration=28,
        treatment_duration=14, withdrawal_duration=7,
        acute_adg=0.68, treatment_adg=0.76, recovery_adg=0.88, chronic_adg=0.90,
        acute_fcr=1.32, treatment_fcr=1.24, recovery_fcr=1.16, chronic_fcr=1.10,
        mortality_annual=0.025, incomplete_recovery=0.25, vet_cost=4.0,
    ),
    "influenza": dict(
        label="Grippe porcine (Influenza)",
        affected_fraction=0.35, onset=0, acute_duration=14,
        treatment_duration=7, withdrawal_duration=5,
        acute_adg=0.78, treatment_adg=0.86, recovery_adg=0.92, chronic_adg=0.96,
        acute_fcr=1.20, treatment_fcr=1.14, recovery_fcr=1.08, chronic_fcr=1.04,
        mortality_annual=0.015, incomplete_recovery=0.12, vet_cost=2.5,
    ),
    "prrs": dict(
        label="SDRP / PRRS",
        affected_fraction=0.30, onset=0, acute_duration=28,
        treatment_duration=14, withdrawal_duration=7,
        acute_adg=0.70, treatment_adg=0.78, recovery_adg=0.86, chronic_adg=0.88,
        acute_fcr=1.35, treatment_fcr=1.27, recovery_fcr=1.18, chronic_fcr=1.12,
        mortality_annual=0.035, incomplete_recovery=0.30, vet_cost=6.0,
    ),
    "lawsonia": dict(
        label="Ileite (Lawsonia intracellularis)",
        affected_fraction=0.25, onset=0, acute_duration=21,
        treatment_duration=10, withdrawal_duration=7,
        acute_adg=0.80, treatment_adg=0.87, recovery_adg=0.93, chronic_adg=0.95,
        acute_fcr=1.18, treatment_fcr=1.12, recovery_fcr=1.07, chronic_fcr=1.03,
        mortality_annual=0.012, incomplete_recovery=0.15, vet_cost=3.0,
    ),
}


@dataclass
class DiseaseParams:
    disease_id: str
    label: str
    affected_fraction: float
    onset: float
    acute_duration: float
    treatment_duration: float
    withdrawal_duration: float
    acute_adg: float
    treatment_adg: float
    recovery_adg: float
    chronic_adg: float
    acute_fcr: float
    treatment_fcr: float
    recovery_fcr: float
    chronic_fcr: float
    mortality_annual: float
    incomplete_recovery: float
    vet_cost: float

    def validate(self):
        if not (0 <= self.affected_fraction <= 1):
            raise ValueError("La fraction d'animaux atteints doit etre entre 0 et 1.")
        for name in ("onset", "acute_duration", "treatment_duration", "withdrawal_duration"):
            if getattr(self, name) < 0:
                raise ValueError(f"'{name}' doit etre >= 0.")
        adg_vals = [self.acute_adg, self.treatment_adg, self.recovery_adg, self.chronic_adg]
        if any(v <= 0 or v > 1.5 for v in adg_vals):
            raise ValueError("Les multiplicateurs de GMQ doivent etre compris entre 0 (exclu) et 1.5.")
        fcr_vals = [self.acute_fcr, self.treatment_fcr, self.recovery_fcr, self.chronic_fcr]
        if any(v <= 0 or v > 3 for v in fcr_vals):
            raise ValueError("Les multiplicateurs d'IC doivent etre compris entre 0 (exclu) et 3.")
        if not (0 <= self.mortality_annual <= 1):
            raise ValueError("La mortalite annuelle liee a la maladie doit etre entre 0 et 1.")
        if not (0 <= self.incomplete_recovery <= 1):
            raise ValueError("La probabilite de recuperation incomplete doit etre entre 0 et 1.")
        if self.vet_cost < 0:
            raise ValueError("Le cout veterinaire doit etre >= 0.")
        return self


def resolve_disease_parameters(disease_id: str, overrides: Optional[dict] = None) -> DiseaseParams:
    """Construit les parametres sanitaires effectifs : profil par defaut,
    eventuellement remplace champ par champ par des valeurs saisies par
    l'utilisateur ('parametrage' de la partie maladie)."""
    if disease_id not in DISEASE_PROFILES:
        raise ValueError(f"Profil sanitaire inconnu : {disease_id}")

    base = dict(DISEASE_PROFILES[disease_id])
    if overrides:
        for k, v in overrides.items():
            if v is not None and k in base:
                base[k] = v

    if disease_id == "aucune":
        neutral = DISEASE_PROFILES["aucune"]
        base.update({k: v for k, v in neutral.items() if k != "label"})

    dp = DiseaseParams(disease_id=disease_id, **base)
    return dp.validate()


# ============================================================
# 2. MODELES DE CROISSANCE THEORIQUE
# ============================================================
# Deux modeles disponibles :
#   - "gompertz" (recommande) : vitesse de croissance deduite de
#     l'equation differentielle de Gompertz dW/dt = k.W.ln(Winf/W).
#     Plus realiste biologiquement qu'une droite : le GMQ potentiel
#     culmine autour de Winf/e puis decline en s'approchant du poids
#     a maturite, au lieu de decliner lineairement des le depart.
#   - "lineaire" : conserve l'ancienne approximation du script R
#     (pour comparaison / compatibilite).

@dataclass
class GrowthConfig:
    model: str = "gompertz"          # "gompertz" ou "lineaire"
    w_inf: float = 280.0             # poids a maturite (kg) - potentiel genetique
    k: float = 0.0080                # constante de croissance de Gompertz (1/jour)
    adg_min: float = 0.30            # borne basse realiste du GMQ (kg/j)
    adg_max: float = 1.35            # borne haute realiste du GMQ (kg/j)
    lin_intercept: float = 1.15
    lin_slope: float = 0.0045
    lin_wref: float = 50.0
    fcr_intercept: float = 1.45
    fcr_slope: float = 0.011
    fcr_wref: float = 25.0
    fcr_min: float = 1.30
    fcr_max: float = 5.50


def growth_adg_potential(weight: np.ndarray, cfg: GrowthConfig) -> np.ndarray:
    """GMQ potentiel (sans maladie ni chaleur) en fonction du poids courant."""
    w = np.asarray(weight, dtype=float)
    if cfg.model == "gompertz":
        w_safe = np.clip(w, 0.5, cfg.w_inf - 0.05)
        adg = cfg.k * w_safe * np.log(cfg.w_inf / w_safe)
    else:  # lineaire (historique V2)
        adg = cfg.lin_intercept - cfg.lin_slope * (w - cfg.lin_wref)
    return np.clip(adg, cfg.adg_min, cfg.adg_max)


def fcr_base(weight: np.ndarray, cfg: GrowthConfig) -> np.ndarray:
    w = np.asarray(weight, dtype=float)
    fcr = cfg.fcr_intercept + cfg.fcr_slope * (w - cfg.fcr_wref)
    return np.clip(fcr, cfg.fcr_min, cfg.fcr_max)


def dressing_percent(live_weight: np.ndarray) -> np.ndarray:
    w = np.asarray(live_weight, dtype=float)
    pct = 0.79 + 0.00015 * (w - 110)
    return np.clip(pct, 0.74, 0.83)


# ============================================================
# 3. ENTREE DES POIDS (base de donnees ou saisie directe)
# ============================================================

def weights_from_dataframe(df: pd.DataFrame, column: Optional[str] = None) -> np.ndarray:
    if column is None:
        candidates = [c for c in df.columns if str(c).strip().lower() in
                      ("weight", "liveweight", "live_weight", "poids")]
        if candidates:
            column = candidates[0]
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) != 1:
                raise ValueError(
                    "Le fichier doit contenir une colonne 'Weight'/'Poids', "
                    "ou une seule colonne numerique si elle n'est pas nommee."
                )
            column = numeric_cols[0]
    out = pd.to_numeric(df[column], errors="coerce").to_numpy()
    out = out[np.isfinite(out) & (out > 0)]
    if len(out) < 2:
        raise ValueError("Au moins deux poids positifs sont necessaires.")
    return out


def weights_from_text(text: str) -> np.ndarray:
    parts = [p.strip() for p in str(text).split(",")]
    out = pd.to_numeric(pd.Series(parts), errors="coerce").to_numpy()
    if len(out) < 2 or not np.all(np.isfinite(out)) or np.any(out <= 0):
        raise ValueError("Saisir au moins deux poids positifs separes par des virgules.")
    return out


def weights_truncated(mu: float, lo: float, hi: float, n: int, seed: int) -> np.ndarray:
    if not (0 < lo <= mu <= hi):
        raise ValueError("Il faut 0 < minimum <= moyenne <= maximum.")
    if n < 1:
        raise ValueError("Taille de lot invalide.")
    rng = np.random.default_rng(seed)
    sd_guess = max((hi - lo) / 4, 0.01)
    x = rng.normal(mu, sd_guess, size=n)
    x = np.clip(x, lo, hi)
    for _ in range(10):
        x = x + (mu - x.mean())
        x = np.clip(x, lo, hi)
    return x


def weights_mean_cv(mu: float, cv: float, n: int, seed: int) -> np.ndarray:
    if mu <= 0 or cv < 0:
        raise ValueError("Moyenne ou CV invalide.")
    rng = np.random.default_rng(seed)
    if cv == 0:
        return np.full(n, mu)
    sigma_log = np.sqrt(np.log(1 + cv ** 2))
    x = rng.lognormal(mean=np.log(mu) - 0.5 * sigma_log ** 2, sigma=sigma_log, size=n)
    return x / x.mean() * mu


# ============================================================
# 4. METEO ET STRESS THERMIQUE (THI)
# ============================================================

def calculate_thi(temp_c: np.ndarray, rh_percent: np.ndarray) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    rh = np.asarray(rh_percent, dtype=float)
    return (1.8 * t + 32) - ((0.55 - 0.0055 * rh) * (1.8 * t - 26))


def heat_response_from_thi(thi_values: np.ndarray) -> tuple[float, float]:
    thi_values = np.asarray(thi_values, dtype=float)
    thi_values = thi_values[np.isfinite(thi_values)]
    if len(thi_values) == 0:
        return 1.0, 1.0
    mild = np.clip(thi_values - 72, 0, 6)
    moderate = np.clip(thi_values - 78, 0, 6)
    severe = np.clip(thi_values - 84, 0, None)
    thermal_load = 0.015 * mild + 0.030 * moderate + 0.050 * severe
    mean_load = thermal_load.mean()
    adg_mult = max(0.65, 1 - mean_load)
    fcr_mult = min(1.35, 1 + 0.70 * mean_load)
    return adg_mult, fcr_mult


def heat_effect(weather: pd.DataFrame) -> Optional[dict]:
    """weather: DataFrame avec colonnes datetime, temp_c, rh_percent."""
    if weather is None or len(weather) == 0:
        return None
    w = weather.copy()
    w["THI"] = calculate_thi(w["temp_c"], w["rh_percent"])
    w["Date"] = pd.to_datetime(w["datetime"]).dt.date

    daily_rows = []
    for date_val, grp in w.groupby("Date"):
        adg_mult, fcr_mult = heat_response_from_thi(grp["THI"].to_numpy())
        daily_rows.append(dict(
            Date=date_val,
            MeanTHI=grp["THI"].mean(),
            MaximumTHI=grp["THI"].max(),
            HoursAbove72=int((grp["THI"] >= 72).sum()),
            HoursAbove84=int((grp["THI"] >= 84).sum()),
            adg=adg_mult,
            fcr=fcr_mult,
        ))
    daily = pd.DataFrame(daily_rows).sort_values("Date").reset_index(drop=True)

    mx = w["THI"].max()
    mean_thi = w["THI"].mean()
    bins = [-np.inf, 72, 75, 78, 84, np.inf]
    labels = ["normal", "vigilance", "alerte", "danger", "urgence"]
    heat_class = pd.cut([mx], bins=bins, labels=labels, right=False)[0]

    return dict(
        weather=w,
        daily=daily,
        mean=mean_thi,
        max=mx,
        above72=int((w["THI"] >= 72).sum()),
        above84=int((w["THI"] >= 84).sum()),
        klass=str(heat_class),
    )


def daily_heat_multipliers(heat: Optional[dict], days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Renvoie deux tableaux (meme longueur que 'days') de multiplicateurs
    GMQ / IC journaliers. Le dernier jour meteo disponible est reconduit
    au-dela de l'horizon meteo (comme dans le script R)."""
    if heat is None or heat["daily"] is None or len(heat["daily"]) == 0:
        return np.ones(len(days)), np.ones(len(days))
    daily = heat["daily"]
    n_weather = len(daily)
    adg_arr = np.empty(len(days))
    fcr_arr = np.empty(len(days))
    for idx, day in enumerate(days):
        widx = int(min(max(0, day), n_weather - 1))
        adg_arr[idx] = daily["adg"].iloc[widx]
        fcr_arr[idx] = daily["fcr"].iloc[widx]
    return adg_arr, fcr_arr


# ============================================================
# 5. MORTALITE
# ============================================================

def annual_to_daily_hazard(annual_rate: float) -> float:
    annual_rate = min(max(annual_rate, 0.0), 1.0)
    return 1 - (1 - annual_rate) ** (1 / 365.25)


# ============================================================
# 6. ETATS SANITAIRES (vectorise pig x jour)
# ============================================================
# Codes d'etat : 0=aucun 1=aigu 2=traitement 3=retrait 4=gueri 5=chronique

STATE_NONE, STATE_ACUTE, STATE_TREAT, STATE_WITHDRAW, STATE_RECOVERED, STATE_CHRONIC = range(6)


def compute_disease_states(days: np.ndarray, dp: DiseaseParams,
                            affected: np.ndarray, incomplete_recovery: np.ndarray) -> np.ndarray:
    n = len(affected)
    day_arr = days[None, :].astype(float)
    sick_end = dp.onset + dp.acute_duration
    treatment_end = sick_end + dp.treatment_duration
    withdrawal_end = treatment_end + dp.withdrawal_duration

    state = np.zeros((n, len(days)), dtype=int)
    acute_mask = (day_arr >= dp.onset) & (day_arr < sick_end)
    treat_mask = (day_arr >= sick_end) & (day_arr < treatment_end)
    withdraw_mask = (day_arr >= treatment_end) & (day_arr < withdrawal_end)
    post_mask = day_arr >= withdrawal_end

    state = np.where(acute_mask, STATE_ACUTE, state)
    state = np.where(treat_mask, STATE_TREAT, state)
    state = np.where(withdraw_mask, STATE_WITHDRAW, state)

    chronic_col = incomplete_recovery[:, None] & post_mask
    recovered_col = (~incomplete_recovery[:, None]) & post_mask
    state = np.where(chronic_col, STATE_CHRONIC, state)
    state = np.where(recovered_col, STATE_RECOVERED, state)

    state = np.where(affected[:, None], state, STATE_NONE)
    return state


def state_multiplier_arrays(state: np.ndarray, dp: DiseaseParams) -> tuple[np.ndarray, np.ndarray]:
    adg_lookup = np.array([1.0, dp.acute_adg, dp.treatment_adg, dp.recovery_adg,
                            dp.recovery_adg, dp.chronic_adg])
    fcr_lookup = np.array([1.0, dp.acute_fcr, dp.treatment_fcr, dp.recovery_fcr,
                            dp.recovery_fcr, dp.chronic_fcr])
    return adg_lookup[state], fcr_lookup[state]


STATE_LABELS = {
    STATE_NONE: "aucun", STATE_ACUTE: "aigu", STATE_TREAT: "traitement",
    STATE_WITHDRAW: "retrait", STATE_RECOVERED: "gueri", STATE_CHRONIC: "chronique",
}


# ============================================================
# 7. SIMULATION PRINCIPALE (vectorisee sur les porcs)
# ============================================================

@dataclass
class SimResult:
    growth: pd.DataFrame           # une ligne par porc x jour (scenario reel)
    baseline_daily: pd.DataFrame   # agrege : aucune degradation
    heat_daily: pd.DataFrame       # agrege : chaleur seule
    disease_daily: pd.DataFrame    # agrege : maladie seule
    daily: pd.DataFrame            # agrege economique / sanitaire, scenario reel
    best: pd.Series
    best_overall: pd.Series
    horizon: float
    heat: Optional[dict]
    disease_profile: DiseaseParams


def _simulate_weights(initial_weights: np.ndarray, days: np.ndarray, growth_cfg: GrowthConfig,
                       adg_state_mult: np.ndarray, heat_adg: np.ndarray,
                       death_day_disease: np.ndarray, background_death_day: np.ndarray,
                       remove_disease: bool, remove_heat: bool) -> np.ndarray:
    n = len(initial_weights)
    nd = len(days)
    W = np.empty((n, nd))
    W[:, 0] = initial_weights
    for j in range(1, nd):
        day = days[j - 1]
        w_prev = W[:, j - 1]
        base = growth_adg_potential(w_prev, growth_cfg)
        mult = np.ones(n) if remove_disease else adg_state_mult[:, j - 1]
        if not remove_heat:
            mult = mult * heat_adg[j - 1]
        adg = base * mult
        dead_disease = (not remove_disease) & np.isfinite(death_day_disease) & (day >= death_day_disease)
        dead_bg = np.isfinite(background_death_day) & (day >= background_death_day)
        dead = dead_disease | dead_bg
        W[:, j] = np.where(dead, w_prev, w_prev + adg)
    return W


def run_model(
    weights: np.ndarray,
    batch_size: int,
    age: float,
    horizon: float,
    seed: int,
    feed_price: float,
    carcass_price: float,
    fixed_cost_per_day: float,
    disease_params: DiseaseParams,
    background_mortality_annual: float,
    growth_cfg: GrowthConfig,
    heat: Optional[dict] = None,
) -> SimResult:

    rng = np.random.default_rng(seed)
    n = int(batch_size)

    initial_weights = rng.choice(weights, size=n, replace=True)

    dp = disease_params
    affected = np.zeros(n, dtype=bool)
    if dp.disease_id != "aucune" and dp.affected_fraction > 0:
        n_affected = int(round(n * dp.affected_fraction))
        n_affected = max(0, min(n, n_affected))
        if n_affected > 0:
            idx = rng.choice(n, size=n_affected, replace=False)
            affected[idx] = True

    incomplete_recovery = np.zeros(n, dtype=bool)
    if affected.any() and dp.disease_id != "aucune":
        incomplete_recovery[affected] = rng.random(affected.sum()) < dp.incomplete_recovery

    disease_hazard = annual_to_daily_hazard(dp.mortality_annual)
    background_hazard = annual_to_daily_hazard(background_mortality_annual)

    episode_days = max(0.0, dp.acute_duration + dp.treatment_duration + dp.withdrawal_duration)

    death_day_disease = np.full(n, np.inf)
    if disease_hazard > 0 and affected.any() and episode_days > 0:
        idx = np.where(affected)[0]
        offsets = rng.exponential(1 / disease_hazard, size=len(idx))
        dies = offsets <= episode_days
        death_day_disease[idx[dies]] = dp.onset + np.ceil(offsets[dies])

    background_death_day = np.full(n, np.inf)
    if background_hazard > 0:
        background_death_day = np.ceil(rng.exponential(1 / background_hazard, size=n))

    n_days = int(max(60, horizon + 30)) + 1
    days = np.arange(n_days)

    heat_adg_arr, heat_fcr_arr = daily_heat_multipliers(heat, days)

    state = compute_disease_states(days, dp, affected, incomplete_recovery)
    adg_state_mult, fcr_state_mult = state_multiplier_arrays(state, dp)

    # -------- scenario reel (maladie + chaleur) --------
    W_real = _simulate_weights(initial_weights, days, growth_cfg, adg_state_mult, heat_adg_arr,
                                death_day_disease, background_death_day,
                                remove_disease=False, remove_heat=(heat is None))

    # -------- feed / etat / mortalite jour par jour (scenario reel) --------
    feed_incr = np.zeros_like(W_real)
    for j in range(1, n_days):
        day = days[j - 1]
        w_prev = W_real[:, j - 1]
        fcr_mult = fcr_state_mult[:, j - 1] * (heat_fcr_arr[j - 1] if heat is not None else 1.0)
        fcr = fcr_base(w_prev, growth_cfg) * fcr_mult
        base = growth_adg_potential(w_prev, growth_cfg)
        adg_mult = adg_state_mult[:, j - 1] * (heat_adg_arr[j - 1] if heat is not None else 1.0)
        adg = base * adg_mult
        dead_disease = np.isfinite(death_day_disease) & (day >= death_day_disease)
        dead_bg = np.isfinite(background_death_day) & (day >= background_death_day)
        dead = dead_disease | dead_bg
        feed_incr[:, j] = np.where(dead, 0.0, adg * fcr)

    cum_feed = np.cumsum(feed_incr, axis=1)

    disease_dead_mask = np.isfinite(death_day_disease)[:, None] & (days[None, :] >= death_day_disease[:, None])
    background_dead_mask = np.isfinite(background_death_day)[:, None] & (days[None, :] >= background_death_day[:, None])
    alive_mask = ~(disease_dead_mask | background_dead_mask)

    eligible_mask = alive_mask & np.isin(state, [STATE_NONE, STATE_RECOVERED])
    disease_active_mask = np.isin(state, [STATE_ACUTE, STATE_TREAT])
    treatment_mask = state == STATE_TREAT
    withdrawal_mask = state == STATE_WITHDRAW
    chronic_mask = state == STATE_CHRONIC

    vet_cost_mask = affected[:, None] & (days[None, :] == dp.onset)
    vet_cost_arr = np.where(vet_cost_mask, dp.vet_cost, 0.0)

    dress_pct = dressing_percent(W_real)
    carcass = W_real * dress_pct
    revenue = np.where(eligible_mask, carcass * carcass_price, 0.0)
    profit = revenue - cum_feed * feed_price - days[None, :] * fixed_cost_per_day - vet_cost_arr

    pig_ids = np.repeat(np.arange(1, n + 1), n_days)
    day_grid = np.tile(days, n)
    growth = pd.DataFrame({
        "PigID": pig_ids,
        "Day": day_grid,
        "Age": age + day_grid,
        "LiveWeight": W_real.ravel(),
        "CumFeed": cum_feed.ravel(),
        "Affected": np.repeat(affected, n_days),
        "DiseaseState": [STATE_LABELS[s] for s in state.ravel()],
        "Eligible": eligible_mask.ravel(),
        "Disease": disease_active_mask.ravel(),
        "Treatment": treatment_mask.ravel(),
        "Withdrawal": withdrawal_mask.ravel(),
        "Chronic": chronic_mask.ravel(),
        "DiseaseDeath": disease_dead_mask.ravel(),
        "BackgroundDeath": background_dead_mask.ravel(),
        "Alive": alive_mask.ravel(),
        "VetCost": vet_cost_arr.ravel(),
        "Revenue": revenue.ravel(),
        "Profit": profit.ravel(),
    })

    # -------- scenarios contrefactuels --------
    W_baseline = _simulate_weights(initial_weights, days, growth_cfg, adg_state_mult, heat_adg_arr,
                                    death_day_disease, background_death_day,
                                    remove_disease=True, remove_heat=True)
    W_heat_only = _simulate_weights(initial_weights, days, growth_cfg, adg_state_mult, heat_adg_arr,
                                     death_day_disease, background_death_day,
                                     remove_disease=True, remove_heat=(heat is None))
    W_disease_only = _simulate_weights(initial_weights, days, growth_cfg, adg_state_mult, heat_adg_arr,
                                        death_day_disease, background_death_day,
                                        remove_disease=False, remove_heat=True)

    def _aggregate(W, prefix):
        df = pd.DataFrame({
            "Day": day_grid, "Age": age + day_grid, "W": W.ravel(),
        })
        agg = df.groupby(["Day", "Age"]).agg(
            **{f"{prefix}MeanWeight": ("W", "mean"),
               f"{prefix}P05Weight": ("W", lambda s: np.percentile(s, 5)),
               f"{prefix}P95Weight": ("W", lambda s: np.percentile(s, 95))}
        ).reset_index()
        return agg

    baseline_daily = _aggregate(W_baseline, "Baseline")
    heat_daily = _aggregate(W_heat_only, "HeatOnly")
    disease_daily = _aggregate(W_disease_only, "DiseaseOnly")

    daily = growth.groupby(["Day", "Age"]).agg(
        BatchProfit=("Profit", "sum"),
        Marketable=("Eligible", "sum"),
        Diseased=("Disease", "sum"),
        Treatment=("Treatment", "sum"),
        Withdrawal=("Withdrawal", "sum"),
        Chronic=("Chronic", "sum"),
        DiseaseDeaths=("DiseaseDeath", "sum"),
        BackgroundDeaths=("BackgroundDeath", "sum"),
        VetCost=("VetCost", "sum"),
        MeanWeight=("LiveWeight", "mean"),
    ).reset_index()
    daily["Deaths"] = daily["DiseaseDeaths"] + daily["BackgroundDeaths"]

    valid = daily[daily["Marketable"] > 0]
    if len(valid) == 0:
        raise ValueError("Aucun porc eligible dans l'horizon de simulation.")

    valid_horizon = valid[valid["Day"] <= horizon]
    if len(valid_horizon) > 0:
        best = valid_horizon.sort_values(["BatchProfit", "Day"], ascending=[False, True]).iloc[0]
    else:
        best = valid.sort_values(["BatchProfit", "Day"], ascending=[False, True]).iloc[0]
    best_overall = valid.sort_values(["BatchProfit", "Day"], ascending=[False, True]).iloc[0]

    return SimResult(
        growth=growth, baseline_daily=baseline_daily, heat_daily=heat_daily,
        disease_daily=disease_daily, daily=daily, best=best, best_overall=best_overall,
        horizon=horizon, heat=heat, disease_profile=dp,
    )


# ============================================================
# 8. COURBES DE REFERENCE (litterature / guides commerciaux)
# ============================================================
# Reprises telles quelles depuis le script R d'origine (donnees
# chiffrees fournies par l'utilisateur, non re-extraites d'un texte).

REFERENCE_POINTS = pd.DataFrame({
    "Source": (["Etude porcine commerciale 2023"] * 5 + ["Genesus 2022"] * 7 +
               ["PIC 410"] * 2 + ["PIC 337"] * 14),
    "SourceType": (["Etude reference NCBI"] * 5 + ["Guide genetique commercial"] * 7 +
                   ["Guide commercial PIC"] * 2 + ["Guide commercial PIC"] * 14),
    "Year": ([2023] * 5 + [2022] * 7 + [2021] * 2 + [2019] * 14),
    "Age": ([111, 126, 143, 165, 191] + [65, 82, 104, 124, 140, 151, 158] + [71, 162] +
            [21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112]),
    "Weight": ([48.95, 60.45, 74.95, 95.20, 118.50] + [27, 43, 64, 82, 100, 109, 123] +
               [28, 120] + [5.4, 6.6, 8.6, 11.3, 14.5, 18.8, 23.6, 28.6, 34.1, 39.9,
                            46.0, 52.4, 59.0, 65.7]),
    "SourceURL": (["https://pmc.ncbi.nlm.nih.gov/articles/PMC10199096/"] * 5 +
                  ["https://genesus.com/wp-content/uploads/2022/06/Grow-Finisher-Feeding-Guidelines-2022-Digital.pdf"] * 7 +
                  ["https://www.picrsa.co.za/growth-curve/"] * 2 +
                  ["https://www.picrsa.co.za/growth-curve/"] * 14),
})


def build_reference_curves(age_min: int = 20, age_max: int = 195) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref_age = np.arange(age_min, age_max + 1)
    curves = []
    for source in REFERENCE_POINTS["Source"].unique():
        sub = REFERENCE_POINTS[REFERENCE_POINTS["Source"] == source].sort_values("Age")
        if len(sub) < 2:
            continue
        keep = (ref_age >= sub["Age"].min()) & (ref_age <= sub["Age"].max())
        if not keep.any():
            continue
        w = np.interp(ref_age[keep], sub["Age"], sub["Weight"])
        curves.append(pd.DataFrame({"Age": ref_age[keep], "Weight": w, "Source": source}))
    reference_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(columns=["Age", "Weight", "Source"])

    agg = reference_curves.groupby("Age").agg(Weight=("Weight", "mean"), Sources=("Weight", "count")).reset_index()
    aggregated_reference = agg[agg["Sources"] >= 2]
    return reference_curves, aggregated_reference


# ============================================================
# 9. MONITORING (comparaison poids observes vs attendus)
# ============================================================

def monitor_weights(growth: pd.DataFrame, observed_weights: np.ndarray, monitoring_age: float) -> dict:
    stats = growth.groupby("Age")["LiveWeight"].agg(["mean", "std"]).reset_index()
    stats.columns = ["Age", "E", "S"]
    stats["S"] = stats["S"].clip(lower=0.1)

    if monitoring_age < stats["Age"].min() or monitoring_age > stats["Age"].max():
        raise ValueError("L'age de monitoring est hors de la plage simulee.")

    expected_mean = float(np.interp(monitoring_age, stats["Age"], stats["E"]))
    expected_sd = float(np.interp(monitoring_age, stats["Age"], stats["S"]))
    observed_mean = float(np.mean(observed_weights))
    z = (observed_mean - expected_mean) / expected_sd
    status = "A surveiller" if abs(z) >= 2 else "Dans la plage attendue"

    return dict(
        age=monitoring_age, observed_mean=observed_mean,
        expected_mean=expected_mean, expected_sd=expected_sd, z=z, status=status,
    )
