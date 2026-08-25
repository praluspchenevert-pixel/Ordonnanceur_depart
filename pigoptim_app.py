# ============================================================
# pigoptim_app.py
# Interface Streamlit pour le modele PigOptim V3 (portage Python
# du modele R "pigoptim V2").
#
# Lancer avec :  streamlit run pigoptim_app.py
# ============================================================

import io
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

import pigoptim_core as core

st.set_page_config(page_title="PigOptim V3", layout="wide")

# ------------------------------------------------------------
# Etat de session
# ------------------------------------------------------------
if "weather_df" not in st.session_state:
    st.session_state.weather_df = None
if "result" not in st.session_state:
    st.session_state.result = None
if "monitor_result" not in st.session_state:
    st.session_state.monitor_result = None


# ------------------------------------------------------------
# Bloc de saisie des poids (reutilisable : lot principal + monitoring)
# ------------------------------------------------------------
def weight_input_block(key_prefix: str, n_for_random: int, seed: int, default_mean: float = 110.0):
    mode = st.selectbox(
        "Saisie des poids",
        ["Fichier mesure (base de donnees)", "Poids separes par des virgules",
         "Min/moyenne/max estimes", "Moyenne seule"],
        key=f"{key_prefix}_mode",
    )

    if mode == "Fichier mesure (base de donnees)":
        f = st.file_uploader("Fichier CSV/Excel de poids", type=["csv", "xlsx", "xls"], key=f"{key_prefix}_file")
        if f is None:
            return None
        try:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
            return core.weights_from_dataframe(df)
        except Exception as e:
            st.error(f"Erreur de lecture du fichier : {e}")
            return None

    if mode == "Poids separes par des virgules":
        text = st.text_input("Poids (kg), separes par des virgules", "95,101,108,114,121", key=f"{key_prefix}_text")
        try:
            return core.weights_from_text(text)
        except Exception as e:
            st.error(str(e))
            return None

    if mode == "Min/moyenne/max estimes":
        c1, c2, c3 = st.columns(3)
        mu = c1.number_input("Moyenne (kg)", value=default_mean, key=f"{key_prefix}_mu")
        lo = c2.number_input("Minimum (kg)", value=default_mean - 20, key=f"{key_prefix}_lo")
        hi = c3.number_input("Maximum (kg)", value=default_mean + 20, key=f"{key_prefix}_hi")
        try:
            return core.weights_truncated(mu, lo, hi, n_for_random, seed)
        except Exception as e:
            st.error(str(e))
            return None

    # Moyenne seule
    c1, c2 = st.columns(2)
    mean = c1.number_input("Moyenne (kg)", value=default_mean, key=f"{key_prefix}_mean")
    cv = c2.number_input("CV suppose", value=0.10, min_value=0.0, step=0.01, key=f"{key_prefix}_cv")
    try:
        return core.weights_mean_cv(mean, cv, n_for_random, seed)
    except Exception as e:
        st.error(str(e))
        return None


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("PigOptim V3")
st.sidebar.caption("Portage Python — croissance theorique retravaillee et maladies parametrables")

st.sidebar.header("1. Lot de porcs")
batch_size = st.sidebar.number_input("Nombre de porcs", value=200, min_value=1, step=1)
age = st.sidebar.number_input("Age actuel (jours)", value=110, min_value=1, step=1)
seed = st.sidebar.number_input("Graine aleatoire (seed)", value=123, min_value=1, step=1)

with st.sidebar.expander("Poids initiaux", expanded=True):
    weights = weight_input_block("main", int(batch_size), int(seed))

st.sidebar.header("2. Economie")
feed_price = st.sidebar.number_input("Prix aliment (€/kg)", value=0.34, min_value=0.0, step=0.01)
carcass_price = st.sidebar.number_input("Prix carcasse (€/kg)", value=1.55, min_value=0.0, step=0.01)
fixed_cost = st.sidebar.number_input("Cout fixe (€/porc/jour)", value=0.015, min_value=0.0, step=0.001, format="%.3f")
horizon = st.sidebar.number_input("Horizon de decision (jours)", value=30, min_value=1, step=1)

st.sidebar.header("3. Croissance theorique")
growth_model = st.sidebar.selectbox(
    "Modele de croissance", ["Gompertz (recommande)", "Lineaire (historique V2)"],
)
if growth_model.startswith("Gompertz"):
    c1, c2 = st.sidebar.columns(2)
    w_inf = c1.number_input("Poids a maturite (kg)", value=280.0, min_value=50.0, step=5.0)
    k_gompertz = c2.number_input("Constante k", value=0.0080, min_value=0.0001, step=0.0005, format="%.4f")
    c3, c4 = st.sidebar.columns(2)
    adg_min = c3.number_input("GMQ min (kg/j)", value=0.30, min_value=0.01, step=0.05)
    adg_max = c4.number_input("GMQ max (kg/j)", value=1.35, min_value=0.1, step=0.05)
    growth_cfg = core.GrowthConfig(model="gompertz", w_inf=w_inf, k=k_gompertz, adg_min=adg_min, adg_max=adg_max)
else:
    c1, c2 = st.sidebar.columns(2)
    lin_intercept = c1.number_input("GMQ a poids de reference", value=1.15, step=0.05)
    lin_slope = c2.number_input("Pente GMQ (par kg)", value=0.0045, step=0.0005, format="%.4f")
    growth_cfg = core.GrowthConfig(model="lineaire", lin_intercept=lin_intercept, lin_slope=lin_slope)

with st.sidebar.expander("Indice de consommation (IC / FCR) de base"):
    fcr_intercept = st.number_input("IC au poids de reference", value=growth_cfg.fcr_intercept, step=0.01)
    fcr_slope = st.number_input("Pente IC (par kg)", value=growth_cfg.fcr_slope, step=0.001, format="%.3f")
    growth_cfg.fcr_intercept = fcr_intercept
    growth_cfg.fcr_slope = fcr_slope

st.sidebar.header("4. Defi sanitaire (maladie)")
disease_options = {v["label"]: k for k, v in core.DISEASE_PROFILES.items()}
disease_label = st.sidebar.selectbox("Maladie", list(disease_options.keys()))
disease_id = disease_options[disease_label]
customize = st.sidebar.checkbox("Personnaliser les parametres de la maladie", value=False)

overrides = None
if customize:
    profile = core.DISEASE_PROFILES[disease_id]
    st.sidebar.markdown("**Epidemiologie**")
    overrides = {}
    overrides["affected_fraction"] = st.sidebar.number_input(
        "Fraction d'animaux atteints", value=profile["affected_fraction"], min_value=0.0, max_value=1.0,
        step=0.01, key=f"ov_af_{disease_id}")
    overrides["onset"] = st.sidebar.number_input(
        "Jour d'apparition", value=float(profile["onset"]), min_value=0.0, step=1.0, key=f"ov_on_{disease_id}")
    overrides["acute_duration"] = st.sidebar.number_input(
        "Duree phase aigue (j)", value=float(profile["acute_duration"]), min_value=0.0, step=1.0, key=f"ov_ad_{disease_id}")
    overrides["treatment_duration"] = st.sidebar.number_input(
        "Duree traitement (j)", value=float(profile["treatment_duration"]), min_value=0.0, step=1.0, key=f"ov_td_{disease_id}")
    overrides["withdrawal_duration"] = st.sidebar.number_input(
        "Duree retrait/convalescence (j)", value=float(profile["withdrawal_duration"]), min_value=0.0, step=1.0, key=f"ov_wd_{disease_id}")

    st.sidebar.markdown("**Effets zootechniques**")
    overrides["acute_adg"] = st.sidebar.number_input(
        "Multiplicateur GMQ - phase aigue", value=profile["acute_adg"], min_value=0.1, max_value=1.5, step=0.01, key=f"ov_aadg_{disease_id}")
    overrides["treatment_adg"] = st.sidebar.number_input(
        "Multiplicateur GMQ - traitement", value=profile["treatment_adg"], min_value=0.1, max_value=1.5, step=0.01, key=f"ov_tadg_{disease_id}")
    overrides["recovery_adg"] = st.sidebar.number_input(
        "Multiplicateur GMQ - convalescence", value=profile["recovery_adg"], min_value=0.1, max_value=1.5, step=0.01, key=f"ov_radg_{disease_id}")
    overrides["chronic_adg"] = st.sidebar.number_input(
        "Multiplicateur GMQ - chronique", value=profile["chronic_adg"], min_value=0.1, max_value=1.5, step=0.01, key=f"ov_cadg_{disease_id}")
    overrides["acute_fcr"] = st.sidebar.number_input(
        "Multiplicateur IC - phase aigue", value=profile["acute_fcr"], min_value=0.5, max_value=3.0, step=0.01, key=f"ov_afcr_{disease_id}")
    overrides["treatment_fcr"] = st.sidebar.number_input(
        "Multiplicateur IC - traitement", value=profile["treatment_fcr"], min_value=0.5, max_value=3.0, step=0.01, key=f"ov_tfcr_{disease_id}")
    overrides["recovery_fcr"] = st.sidebar.number_input(
        "Multiplicateur IC - convalescence", value=profile["recovery_fcr"], min_value=0.5, max_value=3.0, step=0.01, key=f"ov_rfcr_{disease_id}")
    overrides["chronic_fcr"] = st.sidebar.number_input(
        "Multiplicateur IC - chronique", value=profile["chronic_fcr"], min_value=0.5, max_value=3.0, step=0.01, key=f"ov_cfcr_{disease_id}")

    st.sidebar.markdown("**Consequences sanitaires**")
    overrides["mortality_annual"] = st.sidebar.number_input(
        "Mortalite annuelle liee a la maladie", value=profile["mortality_annual"], min_value=0.0, max_value=1.0,
        step=0.001, format="%.3f", key=f"ov_mort_{disease_id}")
    overrides["incomplete_recovery"] = st.sidebar.number_input(
        "Probabilite de recuperation incomplete", value=profile["incomplete_recovery"], min_value=0.0, max_value=1.0,
        step=0.01, key=f"ov_ir_{disease_id}")
    overrides["vet_cost"] = st.sidebar.number_input(
        "Cout veterinaire (€/porc atteint)", value=profile["vet_cost"], min_value=0.0, step=0.1, key=f"ov_vc_{disease_id}")

background_mortality = st.sidebar.number_input(
    "Mortalite de fond annuelle (hors maladie)", value=0.02, min_value=0.0, max_value=1.0, step=0.001, format="%.3f")

st.sidebar.header("5. Meteo / coup de chaleur")
weather_mode = st.sidebar.selectbox("Source meteo", ["Aucune", "Open-Meteo (en ligne)", "Fichier CSV", "Saisie manuelle"])

lat = lon = None
if weather_mode in ("Open-Meteo (en ligne)",):
    c1, c2 = st.sidebar.columns(2)
    lat = c1.number_input("Latitude", value=47.59, format="%.4f")
    lon = c2.number_input("Longitude", value=0.25, format="%.4f")
    weather_days = st.sidebar.number_input("Jours de prevision", value=7, min_value=1, max_value=16, step=1)
    if st.sidebar.button("Recuperer la meteo"):
        try:
            days_clamped = int(max(1, min(16, weather_days)))
            url = (
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                "&hourly=temperature_2m,relative_humidity_2m"
                f"&forecast_days={days_clamped}&timezone=UTC"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            hourly = data.get("hourly")
            if not hourly or "time" not in hourly:
                raise ValueError("Reponse Open-Meteo incomplete.")
            wdf = pd.DataFrame({
                "datetime": pd.to_datetime(hourly["time"]),
                "temp_c": hourly["temperature_2m"],
                "rh_percent": hourly["relative_humidity_2m"],
            })
            st.session_state.weather_df = wdf
            st.sidebar.success(f"{len(wdf)} observations horaires recuperees.")
        except Exception as e:
            st.sidebar.error(f"Echec de la recuperation meteo : {e}")

elif weather_mode == "Fichier CSV":
    wf = st.sidebar.file_uploader("CSV meteo (datetime, temp_c, rh_percent)", type=["csv"], key="weather_csv")
    if wf is not None:
        try:
            wdf = pd.read_csv(wf)
            needed = {"datetime", "temp_c", "rh_percent"}
            if not needed.issubset(wdf.columns):
                raise ValueError(f"Colonnes requises : {', '.join(needed)}")
            wdf["datetime"] = pd.to_datetime(wdf["datetime"])
            wdf["temp_c"] = pd.to_numeric(wdf["temp_c"], errors="coerce")
            wdf["rh_percent"] = pd.to_numeric(wdf["rh_percent"], errors="coerce").clip(0, 100)
            wdf = wdf.dropna(subset=["datetime", "temp_c", "rh_percent"]).sort_values("datetime")
            if len(wdf) == 0:
                raise ValueError("Aucune observation valide dans le fichier.")
            st.session_state.weather_df = wdf
        except Exception as e:
            st.sidebar.error(f"Erreur fichier meteo : {e}")

elif weather_mode == "Saisie manuelle":
    c1, c2 = st.sidebar.columns(2)
    temp_manual = c1.number_input("Temperature (°C)", value=30.0)
    rh_manual = c2.number_input("Humidite relative (%)", value=60.0, min_value=0.0, max_value=100.0)
    now = pd.Timestamp.now(tz=timezone.utc).tz_localize(None)
    wdf = pd.DataFrame({
        "datetime": pd.date_range(now, periods=24, freq="h"),
        "temp_c": temp_manual,
        "rh_percent": rh_manual,
    })
    st.session_state.weather_df = wdf

else:
    st.session_state.weather_df = None

run_clicked = st.sidebar.button("Lancer la simulation", type="primary")


# ------------------------------------------------------------
# EXECUTION DU MODELE
# ------------------------------------------------------------
st.title("Ordonnanceur — modele porc charcutier ça te plait mathieur ?")

if run_clicked:
    if weights is None or len(weights) < 2:
        st.error("Veuillez fournir au moins deux poids initiaux valides avant de lancer la simulation.")
    else:
        try:
            dp = core.resolve_disease_parameters(disease_id, overrides)
            heat = core.heat_effect(st.session_state.weather_df) if weather_mode != "Aucune" else None
            with st.spinner("Simulation en cours..."):
                result = core.run_model(
                    weights=weights, batch_size=int(batch_size), age=float(age), horizon=float(horizon),
                    seed=int(seed), feed_price=float(feed_price), carcass_price=float(carcass_price),
                    fixed_cost_per_day=float(fixed_cost), disease_params=dp,
                    background_mortality_annual=float(background_mortality),
                    growth_cfg=growth_cfg, heat=heat,
                )
            st.session_state.result = result
            st.success("Simulation terminee.")
        except Exception as e:
            st.error(f"Erreur de simulation : {e}")

result = st.session_state.result

tabs = st.tabs([
    "Expedition", "Croissance", "Donnees de croissance", "Reference",
    "Sante / economie", "Chaleur", "Monitoring", "Notes",''
])

# ---------------- Expedition ----------------
with tabs[0]:
    if result is None:
        st.info("Configurez le lot dans la barre laterale puis cliquez sur *Lancer la simulation*.")
    else:
        b = result.best
        g0 = result.growth[result.growth["Day"] == 0]
        gb = result.growth[result.growth["Day"] == b["Day"]]
        delta = gb["LiveWeight"].mean() - g0["LiveWeight"].mean()
        adg_est = delta / b["Day"] if b["Day"] > 0 else np.nan
        fcr_est = gb["CumFeed"].mean() / delta if delta > 0 else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Meilleur jour d'expedition", f"Jour {int(b['Day'])} (age {int(b['Age'])} j)")
        c2.metric("Profit du lot", f"{b['BatchProfit']:.0f} €")
        c3.metric("GMQ estime du lot", f"{adg_est:.3f} kg/j" if np.isfinite(adg_est) else "n/a")
        c4.metric("IC estime du lot", f"{fcr_est:.2f}" if np.isfinite(fcr_est) else "n/a")

        if result.best_overall["Day"] > result.horizon:
            st.warning("L'optimum economique global pourrait se situer au-dela de l'horizon defini par l'utilisateur.")
        st.caption("Les porcs malades, en traitement ou en periode de retrait sont exclus des animaux commercialisables.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.daily["Day"], y=result.daily["BatchProfit"], mode="lines", name="Profit du lot"))
        fig.add_vline(x=result.horizon, line_dash="dash", annotation_text="Horizon")
        fig.update_layout(xaxis_title="Jour de simulation", yaxis_title="Profit du lot (€)", height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(result.daily, use_container_width=True)

# ---------------- Croissance ----------------
with tabs[1]:
    if result is None:
        st.info("Aucun resultat disponible.")
    else:
        actual = result.growth.groupby("Age")["LiveWeight"].agg(
            ["mean", lambda s: np.percentile(s, 5), lambda s: np.percentile(s, 95)]
        )
        actual.columns = ["Mean", "P05", "P95"]
        actual = actual.reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([actual["Age"], actual["Age"][::-1]]),
            y=pd.concat([actual["P95"], actual["P05"][::-1]]),
            fill="toself", fillcolor="rgba(5,112,176,0.15)", line=dict(color="rgba(0,0,0,0)"),
            name="Intervalle P05-P95 (reel)", showlegend=True,
        ))
        fig.add_trace(go.Scatter(x=result.baseline_daily["Age"], y=result.baseline_daily["BaselineMeanWeight"],
                                  name="Aucune degradation", line=dict(color="#238B45", width=3)))
        fig.add_trace(go.Scatter(x=result.heat_daily["Age"], y=result.heat_daily["HeatOnlyMeanWeight"],
                                  name="Chaleur seule", line=dict(color="#E66101", width=3, dash="dash")))
        fig.add_trace(go.Scatter(x=result.disease_daily["Age"], y=result.disease_daily["DiseaseOnlyMeanWeight"],
                                  name="Maladie seule", line=dict(color="#7570B3", width=3, dash="dot")))
        fig.add_trace(go.Scatter(x=actual["Age"], y=actual["Mean"],
                                  name="Maladie + chaleur (reel)", line=dict(color="#0570B0", width=4)))
        fig.update_layout(title="Quatre scenarios theoriques de croissance", xaxis_title="Age (jours)",
                           yaxis_title="Poids vif (kg)", height=520)
        st.plotly_chart(fig, use_container_width=True)

        best_day_df = result.growth[result.growth["Day"] == result.best["Day"]]
        fig2 = go.Figure(go.Histogram(x=best_day_df["LiveWeight"], nbinsx=24, marker_color="#2C7FB8"))
        fig2.update_layout(title="Distribution des poids au jour d'expedition retenu",
                            xaxis_title="Poids vif (kg)", yaxis_title="Nombre de porcs", height=420)
        st.plotly_chart(fig2, use_container_width=True)

# ---------------- Donnees de croissance ----------------
with tabs[2]:
    if result is None:
        st.info("Aucun resultat disponible.")
    else:
        st.markdown(
            "**Aucune degradation** = ni maladie ni chaleur. **Chaleur seule** = chaleur sans maladie. "
            "**Maladie seule** = maladie sans chaleur. **Maladie + chaleur** = scenario reellement simule."
        )
        merged = (
            result.disease_daily
            .merge(result.heat_daily, on=["Day", "Age"], how="outer")
            .merge(result.baseline_daily, on=["Day", "Age"], how="outer")
            .sort_values("Day")
        )
        merged["DiseaseLoss"] = merged["BaselineMeanWeight"] - merged["DiseaseOnlyMeanWeight"]
        merged["HeatLoss"] = merged["DiseaseOnlyMeanWeight"] - merged["HeatOnlyMeanWeight"]
        st.dataframe(merged.round(2), use_container_width=True)
        csv_bytes = result.growth.to_csv(index=False).encode("utf-8")
        st.download_button("Telecharger les donnees de croissance (CSV)", data=csv_bytes,
                            file_name=f"pigoptim_growth_{datetime.now().date()}.csv", mime="text/csv")

# ---------------- Reference ----------------
with tabs[3]:
    st.markdown(
        "Courbes assemblees a partir de points de reference publies ou de guides commerciaux. "
        "La moyenne agregee n'est calculee que la ou au moins deux sources se recouvrent. "
        "Ce ne sont pas une meta-analyse formelle ni une prediction propre a une exploitation."
    )
    ref_curves, agg_ref = core.build_reference_curves()
    fig = go.Figure()
    for src in ref_curves["Source"].unique():
        sub = ref_curves[ref_curves["Source"] == src]
        fig.add_trace(go.Scatter(x=sub["Age"], y=sub["Weight"], name=src, mode="lines"))
    if len(agg_ref):
        fig.add_trace(go.Scatter(x=agg_ref["Age"], y=agg_ref["Weight"], name="Reference agregee",
                                  line=dict(color="black", width=4)))
    fig.update_layout(title="Courbes de croissance de reference (porcs charcutiers)",
                       xaxis_title="Age (jours)", yaxis_title="Poids vif (kg)", height=500)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(core.REFERENCE_POINTS, use_container_width=True)
    for src, grp in core.REFERENCE_POINTS.groupby("Source"):
        row = grp.iloc[0]
        st.markdown(f"**{row['Source']}** ({row['SourceType']}, {row['Year']}) — [{row['SourceURL']}]({row['SourceURL']})")

# ---------------- Sante / economie ----------------
with tabs[4]:
    if result is None:
        st.info("Aucun resultat disponible.")
    else:
        dp = result.disease_profile
        prof_df = pd.DataFrame({
            "Parametre": ["Maladie", "Fraction atteinte", "Apparition (j)", "Duree aigue (j)",
                          "Duree traitement (j)", "Duree retrait (j)", "GMQ aigu", "GMQ traitement",
                          "GMQ convalescence", "GMQ chronique", "IC aigu", "IC traitement",
                          "IC convalescence", "IC chronique", "Mortalite annuelle", "Recuperation incomplete",
                          "Cout veterinaire (€/porc atteint)"],
            "Valeur": [dp.label, dp.affected_fraction, dp.onset, dp.acute_duration, dp.treatment_duration,
                       dp.withdrawal_duration, dp.acute_adg, dp.treatment_adg, dp.recovery_adg, dp.chronic_adg,
                       dp.acute_fcr, dp.treatment_fcr, dp.recovery_fcr, dp.chronic_fcr, dp.mortality_annual,
                       dp.incomplete_recovery, dp.vet_cost],
        })
        st.dataframe(prof_df, use_container_width=True, hide_index=True)

        st.dataframe(result.daily[["Day", "Age", "Diseased", "Treatment", "Withdrawal", "Chronic",
                                    "DiseaseDeaths", "BackgroundDeaths", "Deaths", "VetCost", "BatchProfit"]],
                     use_container_width=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.daily["Day"], y=result.daily["Diseased"], name="Malades", line=dict(color="#D73027")))
        fig.add_trace(go.Scatter(x=result.daily["Day"], y=result.daily["Withdrawal"], name="Retrait", line=dict(color="#FDAE61")))
        fig.add_trace(go.Scatter(x=result.daily["Day"], y=result.daily["Chronic"], name="Chroniques", line=dict(color="#756BB1")))
        fig.update_layout(xaxis_title="Jour de simulation", yaxis_title="Nombre de porcs", height=420)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Chaleur ----------------
with tabs[5]:
    if result is None or result.heat is None:
        st.info("Aucune donnee meteo utilisee pour cette simulation.")
    else:
        h = result.heat
        st.dataframe(pd.DataFrame({
            "THI moyen": [round(h["mean"], 1)], "THI maximum": [round(h["max"], 1)],
            "Heures >=72": [h["above72"]], "Heures >=84": [h["above84"]], "Classe": [h["klass"]],
        }), use_container_width=True, hide_index=True)

        daily = h["daily"]
        fig = go.Figure()
        for ymin, ymax, color in [(-100, 72, "#D9F0D3"), (72, 78, "#FEE08B"),
                                   (78, 84, "#FDAE61"), (84, 120, "#D73027")]:
            fig.add_hrect(y0=ymin, y1=ymax, fillcolor=color, opacity=0.25, line_width=0)
        fig.add_trace(go.Scatter(x=daily["Date"], y=daily["MeanTHI"], mode="lines+markers", name="THI moyen quotidien"))
        for yv in (72, 78, 84):
            fig.add_hline(y=yv, line_dash="dash", line_color="grey")
        fig.update_layout(xaxis_title="Date", yaxis_title="THI moyen quotidien", height=500)
        st.plotly_chart(fig, use_container_width=True)

# ---------------- Monitoring ----------------
with tabs[6]:
    st.subheader("Comparer des poids mesures a la courbe attendue")
    if result is None:
        st.info("Lancez d'abord une simulation pour disposer d'une courbe de reference.")
    else:
        mon_age = st.number_input("Age au moment de la pesee (jours)", value=float(age), min_value=0.0, step=1.0)
        mon_weights = weight_input_block("mon", int(batch_size), int(seed) + 10, default_mean=float(age))
        if st.button("Evaluer"):
            if mon_weights is None or len(mon_weights) < 1:
                st.error("Veuillez fournir au moins un poids mesure.")
            else:
                try:
                    mon = core.monitor_weights(result.growth, mon_weights, mon_age)
                    st.session_state.monitor_result = mon
                except Exception as e:
                    st.error(str(e))

        mon = st.session_state.monitor_result
        if mon is not None:
            color = "🔴" if mon["status"] == "A surveiller" else "🟢"
            st.markdown(f"### {color} {mon['status']}")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Poids moyen observe", f"{mon['observed_mean']:.1f} kg")
            c2.metric("Poids moyen attendu", f"{mon['expected_mean']:.1f} kg")
            c3.metric("Ecart-type attendu", f"{mon['expected_sd']:.1f} kg")
            c4.metric("Score Z", f"{mon['z']:.2f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=result.growth["Age"], y=result.growth["LiveWeight"], mode="markers",
                                      marker=dict(color="grey", opacity=0.10), name="Population simulee"))
            fig.add_trace(go.Scatter(x=[mon["age"]], y=[mon["observed_mean"]], mode="markers",
                                      marker=dict(color="red", size=14), name="Observation"))
            fig.update_layout(xaxis_title="Age (jours)", yaxis_title="Poids vif (kg)", height=450)
            st.plotly_chart(fig, use_container_width=True)

# ---------------- Notes ----------------
with tabs[7]:
    st.markdown("""
**Notes du modele PigOptim V3**

- Quatre scenarios theoriques de croissance sont produits a chaque simulation :
  aucune degradation, chaleur seule, maladie seule, et maladie + chaleur (scenario reellement simule).
- La croissance theorique peut etre modelisee soit par une equation de **Gompertz**
  (`dW/dt = k.W.ln(Winf/W)`, recommandee car le potentiel de GMQ culmine puis decline
  progressivement en s'approchant du poids a maturite genetique), soit par l'ancienne
  approximation **lineaire** du modele V2, au choix dans la barre laterale.
- Les maladies utilisent des etats successifs : aigu, traitement, retrait, gueri /
  chronique (recuperation incomplete). Tous les parametres epidemiologiques et
  zootechniques de chaque profil sanitaire peuvent etre entierement personnalises.
- La mortalite liee a la maladie n'est appliquee que pendant l'episode sanitaire
  (phase aigue + traitement + retrait). La mortalite de fond est saisie annuellement
  puis convertie en risque journalier.
- Les effets de la chaleur sont calcules a partir du THI horaire puis appliques
  jour par jour (et non avec un multiplicateur unique moyenne sur toute la prevision).
- Le THI est un indice de depistage : il ne represente pas explicitement la vitesse
  d'air, la chaleur radiante, le rafraichissement du sol, la densite, l'acces a
  l'eau, le genotype, l'acclimatation ou la reponse physiologique individuelle.
- Les courbes de reference sont illustratives et ne constituent pas une prediction
  propre a une exploitation donnee.
- Les coefficients de maladie et de chaleur necessitent une calibration sur des
  donnees d'elevage ou d'essai avant toute utilisation pour une decision economique.
""")
