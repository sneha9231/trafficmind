from __future__ import annotations

import warnings
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


APP_TITLE = "TrafficMind"
APP_SUBTITLE = "Incident Response & Traffic Operations Platform"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "priority_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
INCIDENT_EVENTS_PATH = BASE_DIR / "data" / "incident_events.csv"

MODEL_NUMERIC_COLUMNS = [
    "avg_speed_kmph", "density_veh_per_km", "avg_wait_time_s", "occupancy_pct",
    "flow_veh_per_hr", "queue_length_veh", "avg_accel_ms2", "heading_deg",
    "signal_state_num", "incident_num", "temp_c", "visibility_km",
    "rain_intensity_mmph", "channel_busy_ratio_pct", "msg_rate_hz",
    "avg_comm_delay_ms", "rssi_dbm", "packet_loss_pct", "speed_density_ratio",
    "congestion_pressure", "wireless_congestion_intensity",
    "throughput_per_queued_vehicle", "acceleration_directionality",
    "weather_factor", "attendance",
]

SIMILARITY_INPUT_COLUMNS = ["event_type", "attendance", "location_type", "weather_factor", "density_veh_per_km", "queue_length_veh"]

C = {
    "bg":          "#0E1117",
    "surface":     "#161B22",
    "surface2":    "#1C2333",
    "border":      "#30363D",
    "text":        "#C9D1D9",
    "muted":       "#8B949E",
    "accent":      "#1F6FEB",
    "accent_lt":   "#0D47A1",
    "danger":      "#F85149",
    "warn":        "#D29922",
    "ok":          "#3FB950",
    "info":        "#58A6FF",
    "purple":      "#A371F7",
    "gradient1":   "#1F6FEB",
    "gradient2":   "#A371F7",
}

STATUS_COLOURS = {
    "Critical":            C["danger"],
    "Needs Attention":     C["warn"],
    "Operationally Ready": C["ok"],
    "Well Prepared":       C["info"],
}


def ensure_sklearn_pickle_compatibility() -> None:
    try:
        import sklearn.compose._column_transformer as m
        if not hasattr(m, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            m._RemainderColsList = _RemainderColsList
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def load_prediction_artifacts():
    ensure_sklearn_pickle_compatibility()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    model        = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_enc    = joblib.load(LABEL_ENCODER_PATH)
    feat_cols    = joblib.load(FEATURE_COLUMNS_PATH)
    return model, preprocessor, label_enc, feat_cols


@st.cache_data(show_spinner=False)
def load_incident_data() -> pd.DataFrame:
    df = pd.read_csv(INCIDENT_EVENTS_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_bool(v):
    if pd.isna(v):
        return False
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_incident_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["start_datetime", "resolved_datetime", "end_datetime"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce", utc=True)
    if "requires_road_closure" in out.columns:
        out["requires_road_closure"] = out["requires_road_closure"].apply(parse_bool)
    for col in ["latitude", "longitude"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def apply_styles() -> None:
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

      html, body, [class*="css"] {{
        font-family: 'Inter', system-ui, sans-serif;
        color: {C['text']};
      }}
      .stApp {{ background: {C['bg']}; }}
      .block-container {{ padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1220px; }}

      [data-testid="stSidebar"] {{
        background: {C['surface']};
        border-right: 1px solid {C['border']};
      }}
      [data-testid="stSidebar"] .block-container {{ padding-top: 1.5rem; }}

      .nav-item {{
        display: block; padding: 0.55rem 0.9rem; border-radius: 6px;
        font-size: 0.875rem; font-weight: 500; color: {C['muted']};
        text-decoration: none; cursor: pointer;
        transition: background 0.12s, color 0.12s; margin-bottom: 2px;
      }}
      .nav-item:hover {{ background: {C['accent_lt']}; color: {C['accent']}; }}
      .nav-item.active {{ background: {C['accent_lt']}; color: {C['accent']}; font-weight: 600; }}

      .page-title {{
        font-size: 1.55rem; font-weight: 800; color: {C['text']};
        margin: 0 0 0.2rem 0; letter-spacing: -0.02em;
      }}
      .page-sub {{
        font-size: 0.88rem; color: {C['muted']}; margin: 0 0 1.75rem 0; line-height: 1.5;
      }}

      .card {{
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border: 1px solid {C['border']}; border-radius: 12px;
        padding: 1.25rem 1.4rem; margin-bottom: 1rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
      }}
      .card:hover {{ transform: translateY(-2px); box-shadow: 0 8px 25px rgba(0,0,0,0.3); }}

      .kpi-label {{
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: {C['muted']}; margin-bottom: 0.3rem;
      }}
      .kpi-value {{
        font-size: 1.75rem; font-weight: 700; color: {C['text']};
        line-height: 1; font-family: 'IBM Plex Mono', monospace;
      }}
      .kpi-note {{ font-size: 0.75rem; color: {C['muted']}; margin-top: 0.25rem; }}

      .badge {{
        display: inline-block; padding: 0.22rem 0.65rem; border-radius: 999px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
      }}
      .section-rule {{ border: none; border-top: 1px solid {C['border']}; margin: 1.75rem 0 1.5rem; }}

      [data-testid="stMetric"] {{
        background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(12px);
        border: 1px solid {C['border']}; border-radius: 12px; padding: 0.9rem 1rem;
      }}
      [data-testid="stMetricLabel"] {{ color: {C['muted']}; font-size: 0.78rem; }}
      [data-testid="stMetricValue"] {{ color: {C['text']}; font-weight: 700; }}

      .js-plotly-plot .plotly {{ border-radius: 8px; }}
      #MainMenu, footer {{ visibility: hidden; }}

      [data-testid="stTabs"] button[role="tab"] {{
        font-family: 'Inter', sans-serif; font-size: 0.85rem; font-weight: 500;
      }}

      @keyframes pulse {{ 0%, 100% {{ opacity: 1; }} 50% {{ opacity: 0.6; }} }}
      .pulse {{ animation: pulse 2s ease-in-out infinite; }}

      .accent-bar {{ height: 3px; background: {C['accent']}; border-radius: 2px; margin-bottom: 1.5rem; }}

      .sitrep {{
        background: rgba(22, 27, 34, 0.8); border: 1px solid {C['border']};
        border-left: 4px solid {C['accent']}; border-radius: 0 12px 12px 0;
        padding: 1.25rem 1.5rem; margin: 1rem 0;
        font-size: 0.88rem; line-height: 1.7; color: {C['text']};
      }}
      .sitrep strong {{ color: {C['info']}; }}
      .sitrep .sitrep-title {{
        font-size: 0.72rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 0.1em; color: {C['purple']}; margin-bottom: 0.6rem;
      }}

      .learning-card {{
        background: rgba(22, 27, 34, 0.7); backdrop-filter: blur(12px);
        border: 1px solid {C['border']}; border-radius: 12px;
        padding: 1.1rem 1.3rem; margin-bottom: 0.75rem;
        transition: border-color 0.2s ease;
      }}
      .learning-card:hover {{ border-color: {C['accent']}; }}
      .learning-title {{ font-weight: 600; font-size: 0.92rem; color: {C['text']}; }}
      .learning-detail {{ font-size: 0.82rem; color: {C['muted']}; margin-top: 0.3rem; line-height: 1.5; }}

      .hero-stat {{ text-align: center; padding: 1.5rem 1rem; }}
      .hero-stat-value {{
        font-size: 2.5rem; font-weight: 800;
        font-family: 'IBM Plex Mono', monospace; color: {C['info']};
      }}
      .hero-stat-label {{
        font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; color: {C['muted']}; margin-top: 0.3rem;
      }}
    </style>
    """, unsafe_allow_html=True)


def render_sidebar(active: str) -> str:
    with st.sidebar:
        st.markdown(f"""
        <div style="padding: 0 0.5rem 1.5rem; border-bottom: 1px solid {C['border']}; margin-bottom: 1rem;">
          <div style="display: flex; align-items: center; gap: 0.5rem;">
            <div style="width: 32px; height: 32px; border-radius: 8px;
                        background: {C['accent']};
                        display: flex; align-items: center; justify-content: center;
                        font-size: 1rem; font-weight: 800; color: #fff;">T</div>
            <div>
              <div style="font-size: 1.05rem; font-weight: 700; color: {C['text']}; letter-spacing: -0.01em;">TrafficMind</div>
              <div style="font-size: 0.68rem; color: {C['muted']}; margin-top: 1px;">Bengaluru Traffic Police</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        pages = {
            "Dashboard":               "Operations overview",
            "Incident Assessment":     "Assess an incident",
            "Severity Assessment":     "Priority & readiness",
            "Response Plan":           "Deployment plan",
            "Post-Event Learning":     "Lessons & insights",
            "Historical Intelligence": "Historical context",
            "Hotspot Map":             "Geographic clusters",
        }

        selected = active
        for name, tip in pages.items():
            is_active = (name == active)
            if st.button(name, key=f"nav_{name}", use_container_width=True,
                         type="primary" if is_active else "secondary"):
                selected = name

        st.markdown("<hr style='border:none;border-top:1px solid #30363D;margin:1.25rem 0 1rem'>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.7rem;color:{C['muted']};line-height:1.6'>Decision support for traffic police operations.</div>", unsafe_allow_html=True)

    return selected


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown("<div class='accent-bar'></div>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding-top:0.2rem'><p class='page-title'>{title}</p></div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<p class='page-sub'>{subtitle}</p>", unsafe_allow_html=True)


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (label, value, note) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="card" style="margin-bottom:0">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value">{value}</div>
              {"<div class='kpi-note'>" + note + "</div>" if note else ""}
            </div>
            """, unsafe_allow_html=True)


def map_planning_to_model_inputs(p: dict) -> dict:
    corridor_to_location = {
        "Non-corridor": "Urban", "CBD": "Urban",
        "ORR East 1": "Highway", "ORR East 2": "Highway",
        "ORR West 1": "Highway", "ORR North 1": "Highway",
        "Tumkur Road": "Highway", "Bellary Road 2": "Highway",
    }
    return {
        "event_type": p["event_type"],
        "event_cause": p["event_cause"],
        "expected_attendance": float(p.get("expected_attendance", 0.0)),
        "event_duration_hours": float(p.get("event_duration_hours", 0.0)),
        "requires_road_closure": bool(p["requires_road_closure"]),
        "veh_type": p["veh_type"],
        "police_station": p["police_station"],
        "authenticated": bool(p["authenticated"]),
        "corridor": p.get("corridor", "Unknown"),
        "location_type": corridor_to_location.get(p.get("corridor", "Non-corridor"), "Urban"),
    }


def build_input_frame(user_inputs: dict, raw_columns: list) -> pd.DataFrame:
    row = {c: 0 for c in raw_columns}
    row.update({
        "event_type":            user_inputs["event_type"],
        "event_cause":           user_inputs["event_cause"],
        "expected_attendance":   float(user_inputs.get("expected_attendance", 0.0)),
        "event_duration_hours":  float(user_inputs.get("event_duration_hours", 0.0)),
        "requires_road_closure": bool(user_inputs["requires_road_closure"]),
        "veh_type":              user_inputs["veh_type"],
        "police_station":        user_inputs["police_station"],
        "authenticated":         bool(user_inputs["authenticated"]),
        "corridor":              user_inputs.get("corridor", "Unknown"),
        "location_type":         user_inputs.get("location_type", "Urban"),
    })
    return pd.DataFrame([row], columns=raw_columns)


def predict_congestion(model, preprocessor, label_encoder, feature_columns, user_inputs):
    input_frame      = build_input_frame(user_inputs, list(getattr(preprocessor, "feature_names_in_", feature_columns)))
    cat_features     = preprocessor.transform(input_frame)
    cat_feature_names= list(preprocessor.get_feature_names_out())
    cat_df           = pd.DataFrame(cat_features, columns=cat_feature_names, index=input_frame.index)
    numeric_cols     = [c for c in MODEL_NUMERIC_COLUMNS if c in input_frame.columns]
    numeric_df       = input_frame[numeric_cols].reset_index(drop=True) if numeric_cols else pd.DataFrame(index=input_frame.index)
    model_input      = pd.concat([cat_df.reset_index(drop=True), numeric_df], axis=1)
    model_input      = model_input[[c for c in feature_columns if c in model_input.columns]]
    model_input      = model_input.reindex(columns=feature_columns, fill_value=0)
    prediction_raw   = model.predict(model_input.to_numpy(dtype=float))[0]
    proba            = model.predict_proba(model_input.to_numpy(dtype=float))[0] if hasattr(model, "predict_proba") else None
    try:
        prediction_label = label_encoder.inverse_transform([prediction_raw])[0]
    except Exception:
        prediction_label = str(prediction_raw)
    confidence = float(proba.max() * 100) if proba is not None else 0.0
    return input_frame, prediction_label, confidence, proba


def compute_readiness_score(prediction_label, risk_score, officers, barricades, diversions):
    cat_score    = {"High": 35, "Low": 80}.get(prediction_label, 50)
    risk_pen     = min(float(risk_score) * 0.35, 40.0)
    deploy_bonus = min(officers * 1.6 + barricades * 2.8 + diversions * 4.5, 35.0)
    score = max(0.0, min(100.0, cat_score - risk_pen + deploy_bonus))
    if score <= 40:   status = "Critical"
    elif score <= 60: status = "Needs Attention"
    elif score <= 80: status = "Operationally Ready"
    else:             status = "Well Prepared"
    return score, status


def compute_event_impact_score(prediction_label: str, attendance: float, duration_hours: float, closure_rate: float) -> tuple[int, str]:
    base               = {"High": 40, "Low": 10}.get(prediction_label, 20)
    attendance_factor  = min(attendance / 200_000.0 * 35.0, 35.0)
    duration_factor    = min(duration_hours / 24.0 * 15.0, 15.0)
    closure_factor     = min(closure_rate / 100.0 * 30.0, 30.0)
    score = int(round(min(100.0, base + attendance_factor + duration_factor + closure_factor)))
    if score <= 30:   category = "Low Impact"
    elif score <= 55: category = "Medium Impact"
    elif score <= 75: category = "High Impact"
    else:             category = "Critical Impact"
    return score, category


def recommend_resources_for_impact(score: int) -> dict:
    if score <= 30:   officers, barricades, diversions = 2, 1, 0
    elif score <= 55: officers, barricades, diversions = 5, 3, 1
    elif score <= 75: officers, barricades, diversions = 10, 5, 2
    else:             officers, barricades, diversions = 15, 7, 3
    return {"officers": officers, "barricades": barricades, "diversions": diversions}


def generate_situation_report(planning_inputs, prediction_label, confidence,
                               impact_score, impact_category, readiness, status, recommendation) -> str:
    event      = planning_inputs.get("event_type", "Unknown")
    cause      = planning_inputs.get("event_cause", "Unknown")
    attendance = int(planning_inputs.get("expected_attendance", 0))
    duration   = float(planning_inputs.get("event_duration_hours", 0))
    corridor   = planning_inputs.get("corridor", "Unknown")
    station    = planning_inputs.get("police_station", "Unknown")
    closure    = "Yes" if planning_inputs.get("requires_road_closure") else "No"
    severity_word = "HIGH PRIORITY" if prediction_label == "High" else "ROUTINE"
    urgency       = "Immediate deployment recommended." if prediction_label == "High" else "Standard deployment timeline acceptable."

    return f"""<div class="sitrep">
<div class="sitrep-title">Situation Report (SITREP)</div>
<strong>Incident Type:</strong> {event} ({cause})<br>
<strong>Location:</strong> {corridor} corridor · {station} jurisdiction<br>
<strong>Expected Impact:</strong> {attendance:,} persons affected · {duration:.1f} hour duration · Road closure: {closure}<br><br>
<strong>CLASSIFICATION: {severity_word}</strong> (Model confidence: {confidence:.0f}%)<br>
<strong>Impact Assessment:</strong> {impact_score}/100 — {impact_category}<br>
<strong>Operational Readiness:</strong> {readiness:.0f}/100 — {status}<br><br>
<strong>Recommended Deployment:</strong><br>
&nbsp;&nbsp;• <strong>{recommendation['officers']}</strong> police officers for crowd management &amp; signal control<br>
&nbsp;&nbsp;• <strong>{recommendation['barricades']}</strong> barricade units for pedestrian/vehicle flow separation<br>
&nbsp;&nbsp;• <strong>{recommendation['diversions']}</strong> diversion routes to redirect through-traffic<br><br>
<strong>Action Required:</strong> {urgency}<br>
<span style="font-size:0.75rem;color:{C['muted']}">Report generated at {datetime.now().strftime('%d %b %Y, %H:%M')} IST</span>
</div>"""


# ── pages ────────────────────────────────────────────────────────────────────

def page_dashboard(incident_df: pd.DataFrame):
    page_header("Operations Dashboard", "Real-time overview of Bengaluru traffic incident intelligence.")

    df = incident_df.copy()
    total_incidents  = len(df)
    high_priority    = len(df[df["priority"] == "High"]) if "priority" in df.columns else 0
    unique_zones     = df["zone"].nunique() if "zone" in df.columns else 0
    closure_pct      = 0.0
    if "requires_road_closure" in df.columns:
        closure_pct = df["requires_road_closure"].apply(parse_bool).mean() * 100

    avg_resolve = None
    if "start_datetime" in df.columns and "resolved_datetime" in df.columns:
        tdf = df.copy()
        tdf["start_datetime"]    = pd.to_datetime(tdf["start_datetime"], errors="coerce", utc=True)
        tdf["resolved_datetime"] = pd.to_datetime(tdf["resolved_datetime"], errors="coerce", utc=True)
        tdf = tdf.dropna(subset=["start_datetime", "resolved_datetime"])
        tdf["res_hrs"] = (tdf["resolved_datetime"] - tdf["start_datetime"]).dt.total_seconds() / 3600
        tdf = tdf[(tdf["res_hrs"] >= 0) & (tdf["res_hrs"] < 200)]
        if not tdf.empty:
            avg_resolve = float(tdf["res_hrs"].mean())

    resolve_str = f"{avg_resolve:.1f}h" if avg_resolve is not None else "N/A"

    st.markdown(f"""
    <div style="background:{C['surface']};border:1px solid {C['border']};border-radius:16px;padding:0.5rem 1rem;margin-bottom:1.5rem;">
      <div style="display:flex;justify-content:space-around;flex-wrap:wrap;">
        <div class="hero-stat"><div class="hero-stat-value">{total_incidents:,}</div><div class="hero-stat-label">Total Incidents</div></div>
        <div class="hero-stat"><div class="hero-stat-value" style="color:{C['danger']};">{high_priority:,}</div><div class="hero-stat-label">High Priority</div></div>
        <div class="hero-stat"><div class="hero-stat-value">{unique_zones}</div><div class="hero-stat-label">Active Zones</div></div>
        <div class="hero-stat"><div class="hero-stat-value">{closure_pct:.1f}%</div><div class="hero-stat-label">Closure Rate</div></div>
        <div class="hero-stat"><div class="hero-stat-value">{resolve_str}</div><div class="hero-stat-label">Avg Resolution</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if "event_type" in df.columns:
            type_counts = df["event_type"].value_counts().reset_index()
            type_counts.columns = ["Event Type", "Count"]
            fig = px.pie(type_counts.head(8), values="Count", names="Event Type",
                         color_discrete_sequence=px.colors.qualitative.Set2, hole=0.45)
            fig.update_layout(title="Incident Distribution by Type",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              height=340, margin={"t":50,"b":20,"l":20,"r":20},
                              legend=dict(font=dict(color=C["muted"])))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "priority" in df.columns:
            prio_counts = df["priority"].value_counts().reset_index()
            prio_counts.columns = ["Priority", "Count"]
            color_map = {"High": C["danger"], "Low": C["ok"], "Medium": C["warn"]}
            fig = px.bar(prio_counts, x="Priority", y="Count", color="Priority", color_discrete_map=color_map)
            fig.update_layout(title="Incidents by Priority Level",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              height=340, margin={"t":50,"b":20,"l":40,"r":20}, showlegend=False,
                              xaxis=dict(gridcolor=C["border"]), yaxis=dict(gridcolor=C["border"]))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown(f"**Incident Temporal Heatmap** — *When do incidents peak?*")

    if "start_datetime" in df.columns:
        tdf2 = df.copy()
        tdf2["start_datetime"] = pd.to_datetime(tdf2["start_datetime"], errors="coerce", utc=True)
        tdf2 = tdf2.dropna(subset=["start_datetime"])
        tdf2["hour"]     = tdf2["start_datetime"].dt.hour
        tdf2["day_name"] = tdf2["start_datetime"].dt.day_name()
        day_order        = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        heatmap_data     = tdf2.groupby(["day_name","hour"]).size().reset_index(name="incidents")
        heatmap_pivot    = heatmap_data.pivot(index="day_name", columns="hour", values="incidents").fillna(0)
        heatmap_pivot    = heatmap_pivot.reindex(day_order)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=[f"{h:02d}:00" for h in range(24)],
            y=day_order,
            colorscale="Viridis",
            hovertemplate="Day: %{y}<br>Hour: %{x}<br>Incidents: %{z}<extra></extra>",
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"family":"Inter,sans-serif","size":11,"color":C["text"]},
                          height=280, margin={"t":20,"b":40,"l":100,"r":20},
                          xaxis=dict(title="Hour of Day"), yaxis=dict(title="", autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if "zone" in df.columns:
            zone_df = df["zone"].value_counts().head(8).reset_index()
            zone_df.columns = ["Zone", "Incidents"]
            fig = px.bar(zone_df, x="Incidents", y="Zone", orientation="h",
                         color_discrete_sequence=[C["accent"]])
            fig.update_layout(title="Top Zones by Incident Count",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              height=300, margin={"t":50,"b":20,"l":20,"r":20},
                              xaxis=dict(gridcolor=C["border"]), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)

    with col4:
        if "event_cause" in df.columns:
            cause_df = df["event_cause"].value_counts().head(8).reset_index()
            cause_df.columns = ["Cause", "Incidents"]
            fig = px.bar(cause_df, x="Incidents", y="Cause", orientation="h",
                         color_discrete_sequence=[C["purple"]])
            fig.update_layout(title="Top Incident Causes",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              height=300, margin={"t":50,"b":20,"l":20,"r":20},
                              xaxis=dict(gridcolor=C["border"]), yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)


def page_plan_event() -> dict:
    page_header("Incident Assessment", "Enter incident details to assess priority and recommended response.")

    with st.form("event_form"):
        c1, c2 = st.columns(2)
        with c1:
            event_type = st.selectbox("Event type",
                ["Accident","Concert","Construction","Other","Parade","Sporting Event"])
            event_cause = st.selectbox("Event cause",
                ["accident","vehicle_breakdown","construction","public_event","protest","procession","tree_fall","water_logging","others"])
            expected_attendance = st.number_input("Expected attendance",
                min_value=0, max_value=200_000, value=25_000, step=500,
                help="Estimated number of people attending or affected")
            requires_road_closure = st.selectbox("Requires road closure", [False, True])
        with c2:
            veh_type = st.selectbox("Vehicle type",
                ["auto","bmtc_bus","heavy_vehicle","ksrtc_bus","lcv","others","private_bus","private_car","taxi","truck","nan"])
            corridor = st.selectbox("Corridor",
                ["Non-corridor","CBD","ORR East 1","ORR East 2","ORR West 1","ORR North 1","Tumkur Road","Bellary Road 2"])
            event_duration_hours = st.number_input("Event duration (hours)",
                min_value=0.5, max_value=48.0, value=2.0, step=0.5,
                help="Expected duration in hours")
            police_station = st.selectbox("Police station",
                ["Adugodi","Ashok Nagar","Banashankari","Banaswadi","Basavanagudi","Bellandur",
                 "Byatarayanapura","Chamarajpet","Chikkabanavara","Chikkajala","City Market",
                 "Cubbon Park","Devanahalli Airport","Electronic City","HAL Old Airport",
                 "HSR Layout","Halasur","Halasuru Gate","Hebbala","Hennuru","High ground",
                 "Hulimavu","J.P. Nagar","Jalahalli","Jayanagara","Jeevanbheemanagar",
                 "Jnanabharathi","K.G. Halli","K.R. Pura","K.S. Layout","Kamakshipalya",
                 "Kengeri","Kodigehalli","Madiwala","Magadi Road","Mahadevapura","Malleshwaram",
                 "Mico Layout","No Police Station","Peenya","Pulikeshinagar(F.Town)","R.T. Nagar",
                 "Rajajinagar","Sadashivanagar","Sheshadripuram","Shivajinagar","Thalagattapura",
                 "Upparpet","V.V.Puram (C.Pet)","Vijayanagara","Whitefield","Wilson Garden",
                 "Yelahanka","Yeshwanthpura"], index=0)
            authenticated = st.selectbox("Authenticated", [False, True])

        submitted = st.form_submit_button("Assess Incident", type="primary", use_container_width=True)

    if submitted or st.session_state.get("planning_inputs"):
        inputs = {
            "event_type": event_type, "event_cause": event_cause,
            "expected_attendance": expected_attendance, "event_duration_hours": event_duration_hours,
            "requires_road_closure": requires_road_closure, "veh_type": veh_type,
            "corridor": corridor, "police_station": police_station, "authenticated": authenticated,
        }
        st.session_state.planning_inputs = inputs
        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("**Incident Summary**")
        kpi_row([
            ("Event type",  event_type,                  ""),
            ("Attendance",  f"{expected_attendance:,}",  "people"),
            ("Duration",    f"{event_duration_hours:.1f} hrs", ""),
            ("Road closure",str(requires_road_closure),  ""),
            ("Vehicle",     veh_type,                    ""),
        ])
        if submitted:
            st.success("Assessment ready — navigate to **Severity Assessment** to see the priority decision.")

    return st.session_state.get("planning_inputs", {})


def page_risk_forecast(model, preprocessor, label_encoder, feature_columns, planning_inputs: dict):
    page_header("Incident Severity Assessment",
                "Predicted incident priority, event impact score, and operational readiness.")

    if not planning_inputs:
        st.info("Complete the Incident Assessment first, then return here.")
        return

    user_inputs = map_planning_to_model_inputs(planning_inputs)
    officers    = int(st.session_state.get("officers", 0))
    barricades  = int(st.session_state.get("barricades", 0))
    diversions  = int(st.session_state.get("diversions", 0))

    _, prediction_label, confidence, proba = predict_congestion(
        model, preprocessor, label_encoder, feature_columns, user_inputs)
    st.session_state.last_prediction = prediction_label

    incident_df  = normalize_incident_dataframe(load_incident_data())
    closure_rate = 0.0
    if "event_type" in incident_df.columns and "requires_road_closure" in incident_df.columns:
        type_df = incident_df[incident_df["event_type"] == user_inputs["event_type"]]
        base_df = type_df if not type_df.empty else incident_df
        closure_rate = float(base_df["requires_road_closure"].astype(bool).mean() * 100)

    impact_score, impact_category = compute_event_impact_score(
        prediction_label,
        user_inputs.get("expected_attendance", 0.0),
        user_inputs.get("event_duration_hours", 0.0),
        closure_rate,
    )
    recommendation = recommend_resources_for_impact(impact_score)
    st.session_state.recommended_officers   = recommendation["officers"]
    st.session_state.recommended_barricades = recommendation["barricades"]
    st.session_state.recommended_diversions = recommendation["diversions"]

    priority_score = {"High": 80, "Low": 30}.get(prediction_label, 50)
    adjusted_risk  = max(0.0, min(100.0, priority_score - officers - barricades * 2 - diversions * 4))
    improvement    = ((priority_score - adjusted_risk) / priority_score * 100) if priority_score > 0 else 0.0
    readiness, status = compute_readiness_score(prediction_label, adjusted_risk, officers, barricades, diversions)

    badge_bg           = STATUS_COLOURS.get(status, C["muted"])
    priority_badge_bg  = C["danger"] if prediction_label == "High" else C["ok"]
    pulse_class        = "pulse" if status == "Critical" else ""
    status_icon        = {"Critical":"⚠️","Needs Attention":"⚠️","Operationally Ready":"✅","Well Prepared":"✅"}.get(status,"")

    st.markdown(f"""
    <div style="background:{badge_bg};border-radius:16px;padding:2rem 2.5rem 1.75rem;margin-bottom:1.5rem;
                color:#fff;border:1px solid {badge_bg}44;box-shadow:0 8px 32px {badge_bg}33;">
      <div style="font-size:0.72rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;opacity:0.85;margin-bottom:0.5rem;">
        Traffic Readiness Score
      </div>
      <div style="display:flex;align-items:flex-end;gap:1.5rem;flex-wrap:wrap;">
        <div style="font-size:5rem;font-weight:800;line-height:1;font-family:'IBM Plex Mono',monospace;">
          {readiness:.0f}<span style="font-size:2rem;opacity:0.6;">/100</span>
        </div>
        <div style="padding-bottom:0.6rem;">
          <div style="display:inline-block;background:{priority_badge_bg};color:#08111f;padding:0.28rem 0.8rem;
                      border-radius:999px;font-weight:700;font-size:0.85rem;margin-bottom:0.5rem;">
            Incident Priority: {prediction_label}
          </div><br/>
          <div class="{pulse_class}" style="font-size:1.4rem;font-weight:600;">{status_icon} {status}</div>
          <div style="font-size:0.85rem;opacity:0.8;margin-top:0.2rem;">
            {prediction_label} priority &nbsp;·&nbsp; {confidence:.0f}% model confidence
          </div>
        </div>
      </div>
      <div style="margin-top:1.25rem;font-size:0.8rem;opacity:0.65;">
        Adjust officers, barricades, and diversions in Response Plan to see this score update.
      </div>
      <div style="margin-top:0.8rem;font-size:0.88rem;opacity:0.9;">
        Recommended: <strong>{recommendation['officers']} officers</strong>,
        <strong>{recommendation['barricades']} barricades</strong>,
        <strong>{recommendation['diversions']} diversions</strong> — based on {impact_category}.
      </div>
    </div>
    """, unsafe_allow_html=True)

    kpi_row([
        ("Base priority",     f"{priority_score}",        "before any deployment"),
        ("Adjusted priority", f"{adjusted_risk:.0f}",     "after current deployment"),
        ("Event impact",      f"{impact_score}",          impact_category),
        ("Closure rate",      f"{closure_rate:.1f}%",     "historical event type rate"),
    ])

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    sitrep = generate_situation_report(
        planning_inputs, prediction_label, confidence,
        impact_score, impact_category, readiness, status, recommendation)
    st.markdown(sitrep, unsafe_allow_html=True)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=readiness,
            number={"suffix":"/100","font":{"size":36,"color":C["text"]}},
            title={"text":"Readiness Score","font":{"size":14,"color":C["muted"]}},
            gauge={
                "axis":{"range":[0,100],"tickwidth":1,"tickcolor":C["border"]},
                "bar":{"color":badge_bg,"thickness":0.25},
                "bgcolor":C["surface"],"bordercolor":C["border"],
                "steps":[
                    {"range":[0,40],"color":"#3D1F1F"},{"range":[40,60],"color":"#3D351F"},
                    {"range":[60,80],"color":"#1F3D20"},{"range":[80,100],"color":"#1F2D3D"},
                ],
                "threshold":{"line":{"color":badge_bg,"width":3},"value":readiness},
            },
        ))
        gauge.update_layout(height=300, margin={"t":60,"b":20,"l":20,"r":20},
                            paper_bgcolor="rgba(0,0,0,0)", font={"family":"Inter,sans-serif"})
        st.plotly_chart(gauge, use_container_width=True)

    with col2:
        bar_fig = go.Figure(data=[go.Bar(
            x=["Before Deployment","After Deployment"],
            y=[priority_score, adjusted_risk],
            marker_color=[C["warn"], C["ok"]],
            text=[f"{priority_score}", f"{adjusted_risk:.0f}"],
            textposition="outside", textfont={"color":C["text"]}, width=[0.4,0.4],
        )])
        bar_fig.update_layout(
            title={"text":"Priority Reduction via Deployment","font":{"color":C["text"]}},
            yaxis=dict(range=[0,110],title="Priority Score",gridcolor=C["border"],color=C["muted"]),
            xaxis=dict(color=C["muted"]), height=300, margin={"t":60,"b":20,"l":40,"r":20},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family":"Inter,sans-serif","size":12})
        st.plotly_chart(bar_fig, use_container_width=True)

    if proba is not None:
        st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
        st.markdown("**Probability Breakdown**")
        labels = [
            label_encoder.inverse_transform([lbl])[0] if not isinstance(lbl, str) else lbl
            for lbl in list(getattr(model, "classes_", range(len(proba))))
        ]
        prob_df = pd.DataFrame({
            "Priority": labels,
            "Probability": [round(float(p)*100, 1) for p in proba],
        }).sort_values("Probability", ascending=False)
        prob_fig = px.bar(prob_df, x="Priority", y="Probability", color="Priority",
                          color_discrete_sequence=[C["accent"],C["ok"],C["warn"],C["danger"]])
        prob_fig.update_layout(
            showlegend=False, yaxis_title="Probability (%)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
            height=280, margin={"t":20,"b":30,"l":40,"r":20},
            xaxis=dict(color=C["muted"]), yaxis=dict(gridcolor=C["border"],color=C["muted"]))
        st.plotly_chart(prob_fig, use_container_width=True)


def page_deploy_resources(planning_inputs: dict):
    page_header("Response Plan", "Configure deployment resources. Readiness score updates in real time.")

    if not planning_inputs:
        st.info("Complete the Incident Assessment first, then return here.")
        return

    recommended = {
        "officers":   st.session_state.get("recommended_officers", 2),
        "barricades": st.session_state.get("recommended_barricades", 1),
        "diversions": st.session_state.get("recommended_diversions", 0),
    }

    st.markdown(f"""
    <div style="background:{C['surface']};border:1px solid {C['border']};border-radius:12px;
                padding:1rem 1.3rem;margin-bottom:1.25rem;">
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
                  color:{C['purple']};margin-bottom:0.4rem;">Recommended Deployment</div>
      <div style="font-size:0.95rem;color:{C['text']};">
        <strong>{recommended['officers']}</strong> officers &nbsp;·&nbsp;
        <strong>{recommended['barricades']}</strong> barricades &nbsp;·&nbsp;
        <strong>{recommended['diversions']}</strong> diversions
      </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"<div style='font-size:0.8rem;color:{C['muted']};margin-bottom:0.3rem;'>Each officer: <strong style=\"color:{C['text']}\">-1 pt</strong> priority, ₹3,000</div>", unsafe_allow_html=True)
        st.session_state.officers = st.slider("Police officers", 0, 20, int(st.session_state.get("officers", recommended["officers"])), 1)
    with c2:
        st.markdown(f"<div style='font-size:0.8rem;color:{C['muted']};margin-bottom:0.3rem;'>Each barricade: <strong style=\"color:{C['text']}\">-2 pts</strong> priority, ₹5,000</div>", unsafe_allow_html=True)
        st.session_state.barricades = st.slider("Barricades", 0, 10, int(st.session_state.get("barricades", recommended["barricades"])), 1)
    with c3:
        st.markdown(f"<div style='font-size:0.8rem;color:{C['muted']};margin-bottom:0.3rem;'>Each diversion: <strong style=\"color:{C['text']}\">-4 pts</strong> priority, ₹8,000</div>", unsafe_allow_html=True)
        st.session_state.diversions = st.slider("Diversion routes", 0, 5, int(st.session_state.get("diversions", recommended["diversions"])), 1)

    officers   = int(st.session_state.officers)
    barricades = int(st.session_state.barricades)
    diversions = int(st.session_state.diversions)
    total_cost     = officers * 3000 + barricades * 5000 + diversions * 8000
    priority_score = {"High": 80, "Low": 30}.get(st.session_state.get("last_prediction", "Low"), 50)
    adj_risk       = max(0.0, min(100.0, priority_score - officers - barricades * 2 - diversions * 4))
    reduction      = priority_score - adj_risk

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    kpi_row([
        ("Total Cost",         f"₹{total_cost:,}",                               "deployment estimate"),
        ("Priority Reduction", f"{reduction:.0f} pts",                            "vs no deployment"),
        ("Adjusted Priority",  f"{adj_risk:.0f}",                                 "after deployment"),
        ("Cost Efficiency",    f"{(reduction/total_cost*1000):.1f}" if total_cost > 0 else "—",
                               "pts/₹1000" if total_cost > 0 else ""),
    ])

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("**Resource Allocation Breakdown**")

    col1, col2 = st.columns(2)
    with col1:
        resource_data = pd.DataFrame({
            "Resource":    ["Officers","Barricades","Diversions"],
            "Deployed":    [officers, barricades, diversions],
            "Recommended": [recommended["officers"], recommended["barricades"], recommended["diversions"]],
        })
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Recommended", x=resource_data["Resource"], y=resource_data["Recommended"],
                             marker_color=C["accent"], opacity=0.4,
                             text=resource_data["Recommended"], textposition="outside",
                             textfont={"color":C["muted"]}))
        fig.add_trace(go.Bar(name="Deployed", x=resource_data["Resource"], y=resource_data["Deployed"],
                             marker_color=C["accent"],
                             text=resource_data["Deployed"], textposition="outside",
                             textfont={"color":C["text"]}))
        fig.update_layout(barmode="overlay", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                          height=300, margin={"t":20,"b":30,"l":30,"r":20},
                          legend=dict(font=dict(color=C["muted"])),
                          xaxis=dict(color=C["muted"]), yaxis=dict(gridcolor=C["border"],color=C["muted"]))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cost_data = pd.DataFrame({
            "Resource": ["Officers","Barricades","Diversions"],
            "Cost":     [officers*3000, barricades*5000, diversions*8000],
        })
        cost_data = cost_data[cost_data["Cost"] > 0]
        if not cost_data.empty:
            fig = px.pie(cost_data, values="Cost", names="Resource",
                         color_discrete_sequence=[C["accent"],C["purple"],C["ok"]], hole=0.5)
            fig.update_layout(title={"text":"Cost Distribution","font":{"size":13,"color":C["text"]}},
                              paper_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              height=300, margin={"t":50,"b":20,"l":20,"r":20},
                              legend=dict(font=dict(color=C["muted"])))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Deploy resources to see cost distribution.")

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("**Resource Deployment Guide**")
    resources = pd.DataFrame({
        "Resource":           ["Police Officer","Barricade","Diversion Route"],
        "Priority Reduction": ["-1 per unit","-2 per unit","-4 per unit"],
        "Cost":               ["₹3,000","₹5,000","₹8,000"],
        "Best Used For":      [
            "Crowd management, signal control, VIP escort",
            "Separating pedestrian/vehicle flows, access control",
            "Redirecting through-traffic, reducing bottlenecks",
        ],
        "Effectiveness": ["Low","Medium","High"],
    })
    st.dataframe(resources, use_container_width=True, hide_index=True)


def page_post_event_learning(incident_df: pd.DataFrame, planning_inputs: dict):
    page_header("Post-Event Learning System",
                "Extract lessons from historical incidents to improve future event management. Closes the feedback loop.")

    df = incident_df.copy()

    for col in ["event_type","event_cause","zone","junction","priority","police_station","corridor","status"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)

    df["requires_road_closure"] = df.get("requires_road_closure", pd.Series(False, index=df.index)).fillna(False)
    if "start_datetime" in df.columns:
        df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    if "resolved_datetime" in df.columns:
        df["resolved_datetime"] = pd.to_datetime(df["resolved_datetime"], errors="coerce", utc=True)

    has_resolution = False
    if "start_datetime" in df.columns and "resolved_datetime" in df.columns:
        df["resolution_hours"] = (df["resolved_datetime"] - df["start_datetime"]).dt.total_seconds() / 3600
        df.loc[df["resolution_hours"] < 0,   "resolution_hours"] = np.nan
        df.loc[df["resolution_hours"] > 200,  "resolution_hours"] = np.nan
        has_resolution = df["resolution_hours"].notna().sum() > 0

    total        = len(df)
    high_count   = len(df[df["priority"] == "High"])
    high_pct     = (high_count / total * 100) if total > 0 else 0
    closure_rate = df["requires_road_closure"].apply(parse_bool).mean() * 100
    top_zone     = df["zone"].value_counts().index[0] if not df["zone"].value_counts().empty else "N/A"
    top_cause    = df["event_cause"].value_counts().index[0] if "event_cause" in df.columns else "N/A"

    st.markdown(f"""
    <div style="background:{C['surface']};border:1px solid {C['border']};border-radius:16px;
                padding:1.5rem 2rem;margin-bottom:1.5rem;">
      <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
                  color:{C['purple']};margin-bottom:1rem;">
        Key Learnings from {total:,} Historical Incidents
      </div>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1rem;">
        <div class="learning-card">
          <span class="learning-title">{high_pct:.1f}% High Priority</span>
          <div class="learning-detail">{high_count:,} of {total:,} incidents required urgent response</div>
        </div>
        <div class="learning-card">
          <span class="learning-title">{closure_rate:.1f}% Road Closures</span>
          <div class="learning-detail">Plan closures proactively for high-attendance events</div>
        </div>
        <div class="learning-card">
          <span class="learning-title">Hotspot: {top_zone}</span>
          <div class="learning-detail">Most incident-prone zone — pre-deploy resources here</div>
        </div>
        <div class="learning-card">
          <span class="learning-title">Top Cause: {top_cause}</span>
          <div class="learning-detail">Most frequent incident trigger in the dataset</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs(["Pattern Analysis","Resolution Insights","Lessons"])

    with tabs[0]:
        st.markdown("**What types of events cause the most disruption?**")
        col1, col2 = st.columns(2)
        with col1:
            type_prio = df.groupby(["event_type","priority"]).size().reset_index(name="count")
            fig = px.bar(type_prio, x="event_type", y="count", color="priority", barmode="stack",
                         color_discrete_map={"High":C["danger"],"Low":C["ok"],"Medium":C["warn"],"Unknown":C["muted"]},
                         labels={"event_type":"Event Type","count":"Incidents"})
            fig.update_layout(title={"text":"Event Type × Priority","font":{"color":C["text"]}},
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              height=350, margin={"t":50,"b":30,"l":40,"r":20},
                              legend=dict(font=dict(color=C["muted"]),title_font=dict(color=C["muted"])),
                              xaxis=dict(gridcolor=C["border"],color=C["muted"]),
                              yaxis=dict(gridcolor=C["border"],color=C["muted"]))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            cause_prio  = df.groupby(["event_cause","priority"]).size().reset_index(name="count")
            top_causes  = df["event_cause"].value_counts().head(8).index.tolist()
            cause_prio  = cause_prio[cause_prio["event_cause"].isin(top_causes)]
            fig = px.bar(cause_prio, x="event_cause", y="count", color="priority", barmode="stack",
                         color_discrete_map={"High":C["danger"],"Low":C["ok"],"Medium":C["warn"],"Unknown":C["muted"]},
                         labels={"event_cause":"Cause","count":"Incidents"})
            fig.update_layout(title={"text":"Cause × Priority","font":{"color":C["text"]}},
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              height=350, margin={"t":50,"b":30,"l":40,"r":20},
                              legend=dict(font=dict(color=C["muted"]),title_font=dict(color=C["muted"])),
                              xaxis=dict(gridcolor=C["border"],color=C["muted"]),
                              yaxis=dict(gridcolor=C["border"],color=C["muted"]))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Road closure likelihood by event type — plan diversions proactively**")
        closure_by_type = df.groupby("event_type").agg(
            total=("requires_road_closure","size"),
            closures=("requires_road_closure", lambda x: x.apply(parse_bool).sum()),
        ).reset_index()
        closure_by_type["rate"] = (closure_by_type["closures"] / closure_by_type["total"] * 100).round(1)
        closure_by_type = closure_by_type.sort_values("rate", ascending=True)
        fig = px.bar(closure_by_type, y="event_type", x="rate", orientation="h",
                     color="rate", color_continuous_scale=[[0,C["ok"]],[0.5,C["warn"]],[1,C["danger"]]],
                     labels={"event_type":"","rate":"Closure Rate (%)"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                          height=280, margin={"t":20,"b":30,"l":20,"r":20}, showlegend=False,
                          xaxis=dict(gridcolor=C["border"],color=C["muted"]),
                          yaxis=dict(color=C["muted"]))
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if has_resolution:
            st.markdown("**How quickly are different incident types resolved?**")
            res_by_type = df.dropna(subset=["resolution_hours"]).groupby("event_type")["resolution_hours"]\
                           .agg(["mean","median","count"]).reset_index()
            res_by_type.columns = ["Event Type","Mean (hrs)","Median (hrs)","Count"]
            res_by_type = res_by_type.sort_values("Mean (hrs)", ascending=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(name="Mean", y=res_by_type["Event Type"], x=res_by_type["Mean (hrs)"],
                                 orientation="h", marker_color=C["warn"],
                                 text=res_by_type["Mean (hrs)"].round(1), textposition="outside",
                                 textfont={"color":C["text"]}))
            fig.add_trace(go.Bar(name="Median", y=res_by_type["Event Type"], x=res_by_type["Median (hrs)"],
                                 orientation="h", marker_color=C["accent"],
                                 text=res_by_type["Median (hrs)"].round(1), textposition="outside",
                                 textfont={"color":C["text"]}))
            fig.update_layout(title={"text":"Resolution Time by Event Type","font":{"color":C["text"]}},
                              barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              height=350, margin={"t":50,"b":30,"l":20,"r":60},
                              legend=dict(font=dict(color=C["muted"])),
                              xaxis=dict(title="Hours",gridcolor=C["border"],color=C["muted"]),
                              yaxis=dict(color=C["muted"]))
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**Slowest responding zones — where to improve?**")
            res_by_zone = df.dropna(subset=["resolution_hours"]).groupby("zone")["resolution_hours"]\
                           .agg(["mean","count"]).reset_index()
            res_by_zone.columns = ["Zone","Mean Resolution (hrs)","Incidents"]
            res_by_zone = res_by_zone[res_by_zone["Incidents"] >= 3]\
                           .sort_values("Mean Resolution (hrs)", ascending=False).head(10)
            if not res_by_zone.empty:
                fig = px.bar(res_by_zone, x="Zone", y="Mean Resolution (hrs)",
                             color="Mean Resolution (hrs)",
                             color_continuous_scale=[[0,C["ok"]],[0.5,C["warn"]],[1,C["danger"]]],
                             text=res_by_zone["Mean Resolution (hrs)"].round(1))
                fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                                  height=300, margin={"t":20,"b":30,"l":40,"r":20}, showlegend=False,
                                  xaxis=dict(gridcolor=C["border"],color=C["muted"]),
                                  yaxis=dict(gridcolor=C["border"],color=C["muted"]))
                fig.update_traces(textposition="outside", textfont={"color":C["text"]})
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Resolution time data is not available in the dataset.")

    with tabs[2]:
        st.markdown("**Lessons for Future Events**")
        lessons = []

        if "start_datetime" in df.columns:
            hourly = df.dropna(subset=["start_datetime"]).copy()
            hourly["hour"] = hourly["start_datetime"].dt.hour
            if not hourly.empty:
                peak_hour = hourly["hour"].value_counts().index[0]
                lessons.append({
                    "title":  f"Peak Incident Hour: {peak_hour:02d}:00",
                    "detail": f"Deploy additional officers between {peak_hour:02d}:00–{(peak_hour+2)%24:02d}:00 when incident probability is highest.",
                    "color":  C["warn"],
                })

        high_df = df[df["priority"] == "High"]
        if not high_df.empty:
            top_high_cause = high_df["event_cause"].value_counts().index[0]
            count_hc = high_df[high_df["event_cause"] == top_high_cause].shape[0]
            lessons.append({
                "title":  f"Most Dangerous Cause: {top_high_cause}",
                "detail": f"{top_high_cause} accounts for {count_hc} high-priority incidents. Pre-position response teams.",
                "color":  C["danger"],
            })

        zone_closure = df.groupby("zone").agg(
            total=("requires_road_closure","size"),
            closures=("requires_road_closure", lambda x: x.apply(parse_bool).sum()),
        ).reset_index()
        zone_closure["rate"] = zone_closure["closures"] / zone_closure["total"] * 100
        zone_closure = zone_closure[zone_closure["total"] >= 5].sort_values("rate", ascending=False)
        if not zone_closure.empty:
            wz = zone_closure.iloc[0]
            lessons.append({
                "title":  f"Highest Closure Zone: {wz['zone']}",
                "detail": f"{wz['rate']:.0f}% closure rate across {int(wz['total'])} incidents. Pre-plan diversion routes.",
                "color":  C["accent"],
            })

        if "veh_type" in df.columns and not high_df.empty:
            veh_high = high_df["veh_type"].value_counts()
            if not veh_high.empty:
                top_veh = veh_high.index[0]
                lessons.append({
                    "title":  f"Vehicle Risk Factor: {top_veh}",
                    "detail": f"{top_veh} is involved in {veh_high.iloc[0]} high-priority incidents. Focus enforcement here.",
                    "color":  C["purple"],
                })

        if "corridor" in df.columns and not high_df.empty:
            corr_high = high_df["corridor"].value_counts()
            if not corr_high.empty:
                top_corr = corr_high.index[0]
                lessons.append({
                    "title":  f"Riskiest Corridor: {top_corr}",
                    "detail": f"{corr_high.iloc[0]} high-priority incidents on {top_corr}. Consider permanent augmentation.",
                    "color":  C["ok"],
                })

        if "start_datetime" in df.columns:
            dow = df.dropna(subset=["start_datetime"]).copy()
            dow["day"] = dow["start_datetime"].dt.day_name()
            if not dow.empty:
                peak_day = dow["day"].value_counts().index[0]
                lessons.append({
                    "title":  f"Peak Day: {peak_day}",
                    "detail": f"Highest incident volume on {peak_day}s. Schedule extra patrol shifts.",
                    "color":  C["info"],
                })

        for lesson in lessons:
            st.markdown(f"""
            <div class="learning-card" style="border-left:3px solid {lesson['color']};">
              <span class="learning-title">{lesson['title']}</span>
              <div class="learning-detail">{lesson['detail']}</div>
            </div>
            """, unsafe_allow_html=True)

        if planning_inputs:
            st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
            st.markdown("**Lessons Specific to Your Current Incident**")
            current_type  = planning_inputs.get("event_type", "Unknown")
            current_cause = planning_inputs.get("event_cause", "Unknown")
            same_type     = df[df["event_type"]  == current_type]
            same_cause    = df[df["event_cause"] == current_cause]
            type_high_pct  = (same_type["priority"]  == "High").mean() * 100 if not same_type.empty  else 0
            cause_high_pct = (same_cause["priority"] == "High").mean() * 100 if not same_cause.empty else 0

            col1, col2 = st.columns(2)
            with col1:
                color = C["danger"] if type_high_pct > 50 else C["warn"] if type_high_pct > 25 else C["ok"]
                st.markdown(f"""
                <div class="learning-card" style="border-left:3px solid {color};">
                  <span class="learning-title">{current_type} Events</span>
                  <div class="learning-detail">
                    {len(same_type)} historical incidents found<br>
                    {type_high_pct:.1f}% were high priority<br>
                    {"Historically risky — deploy extra resources" if type_high_pct > 40 else "Manageable with standard deployment"}
                  </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                color = C["danger"] if cause_high_pct > 50 else C["warn"] if cause_high_pct > 25 else C["ok"]
                st.markdown(f"""
                <div class="learning-card" style="border-left:3px solid {color};">
                  <span class="learning-title">{current_cause} Cause</span>
                  <div class="learning-detail">
                    {len(same_cause)} historical incidents with this cause<br>
                    {cause_high_pct:.1f}% were high priority<br>
                    {"High-risk cause — escalate response" if cause_high_pct > 40 else "Standard response appropriate"}
                  </div>
                </div>
                """, unsafe_allow_html=True)


def page_past_events(incident_df: pd.DataFrame, planning_inputs: dict):
    page_header("Historical Intelligence",
                "Patterns from historical incidents that inform this planning decision.")

    df = incident_df.copy()
    for col in ["event_type","event_cause","zone","junction","priority","location_type"]:
        if col not in df.columns:
            df[col] = "Unknown"
        df[col] = df[col].fillna("Unknown").astype(str)
    if "requires_road_closure" not in df.columns:
        df["requires_road_closure"] = False

    current_event_type = planning_inputs.get("event_type", "Accident") if planning_inputs else "Accident"
    current_cause      = planning_inputs.get("event_cause", "accident") if planning_inputs else "accident"
    current_location   = planning_inputs.get("location_type", "Urban") if planning_inputs else "Urban"
    current_closure    = bool(planning_inputs.get("requires_road_closure", False)) if planning_inputs else False

    closure_rate = 0.0
    type_df = df[df["event_type"] == current_event_type]
    if not type_df.empty:
        closure_rate = float(type_df["requires_road_closure"].apply(parse_bool).mean() * 100)

    avg_resolution = None
    if "start_datetime" in df.columns and "resolved_datetime" in df.columns:
        timeframe = df.copy()
        timeframe["start_datetime"]    = pd.to_datetime(timeframe["start_datetime"], errors="coerce", utc=True)
        timeframe["resolved_datetime"] = pd.to_datetime(timeframe["resolved_datetime"], errors="coerce", utc=True)
        timeframe = timeframe.dropna(subset=["start_datetime","resolved_datetime"])
        timeframe["resolution_hours"] = (timeframe["resolved_datetime"] - timeframe["start_datetime"]).dt.total_seconds() / 3600
        timeframe = timeframe[timeframe["resolution_hours"] >= 0]
        if not timeframe.empty:
            avg_resolution = float(timeframe["resolution_hours"].mean())

    kpi_row([
        ("Profiled event type",    current_event_type,  ""),
        ("Current location type",  current_location,    ""),
        ("Road closure likelihood",f"{closure_rate:.1f}%", "same event type"),
        ("Average resolve time",   f"{avg_resolution:.1f} hrs" if avg_resolution is not None else "N/A", "historical average"),
    ])

    tabs = st.tabs(["Comparable Incidents","Location Risk","Closure & Resolution"])

    with tabs[0]:
        st.markdown("**Most relevant historical incidents for this plan**")
        st.caption("Matching by event type, cause, location and closure pattern.")

        match_df = df.copy()
        for col in ["event_type","event_cause","zone","junction","priority"]:
            if col in match_df.columns:
                mask = (
                    match_df[col].notna() &
                    (match_df[col].astype(str).str.strip().str.lower() != "unknown") &
                    (match_df[col].astype(str).str.strip().str.lower() != "nan") &
                    (match_df[col].astype(str).str.strip() != "")
                )
                match_df = match_df[mask]

        match_df["match_score"] = (
            np.where(match_df["event_type"]           == current_event_type, 4, 0)
            + np.where(match_df["event_cause"]        == current_cause,      2, 0)
            + np.where(match_df["location_type"]      == current_location,   1, 0)
            + np.where(match_df["requires_road_closure"].apply(parse_bool)
                       == current_closure,                                    1, 0)
        )
        match_df = match_df.sort_values(["match_score","priority"], ascending=[False,False]).head(5).reset_index(drop=True)

        if match_df.empty:
            st.info("Not enough historical data to show comparable incidents.")
        else:
            match_df["Similarity"] = ((match_df["match_score"] / 8) * 100).round(0).astype(int).astype(str) + "%"
            display_cols = [c for c in ["event_type","event_cause","zone","junction","priority","requires_road_closure","Similarity"] if c in match_df.columns]
            rename_map   = {"event_type":"Event type","event_cause":"Cause","zone":"Zone",
                            "junction":"Junction","priority":"Priority","requires_road_closure":"Road closure"}
            st.dataframe(match_df[display_cols].rename(columns=rename_map),
                         use_container_width=True, hide_index=True)
            if match_df.iloc[0]["match_score"] >= 6:
                st.success("Top comparable incidents are a strong match for your current plan.")
            else:
                st.warning("Historical coverage is limited; review closure and location risk closely.")

    with tabs[1]:
        st.markdown("**Location risk within the same area**")
        st.caption("Where similar incidents cluster for the selected location type.")

        location_df = df[df["location_type"] == current_location].copy()
        for col in ["zone","junction"]:
            if col in location_df.columns:
                location_df = location_df[
                    location_df[col].notna() &
                    (location_df[col].astype(str).str.strip().str.lower() != "unknown") &
                    (location_df[col].astype(str).str.strip().str.lower() != "nan") &
                    (location_df[col].astype(str).str.strip() != "")
                ]

        if location_df.empty:
            st.info("No historical incidents recorded for this location type.")
        else:
            zone_counts = location_df["zone"].value_counts().rename_axis("zone").reset_index(name="Incidents").head(8)
            fig = px.bar(zone_counts, x="zone", y="Incidents", color_discrete_sequence=[C["accent"]],
                         labels={"zone":"Zone"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              margin={"t":20,"b":30,"l":40,"r":20}, height=320,
                              xaxis=dict(gridcolor=C["border"],color=C["muted"]),
                              yaxis=dict(gridcolor=C["border"],color=C["muted"]))
            st.plotly_chart(fig, use_container_width=True)

            top_junctions = location_df["junction"].value_counts().rename_axis("junction").reset_index(name="Incidents").head(6)
            st.markdown("**Top junctions in this location type**")
            st.dataframe(top_junctions.rename(columns={"junction":"Junction"}),
                         use_container_width=True, hide_index=True)

    with tabs[2]:
        st.markdown("**Closure and resolution patterns**")
        st.caption("Use these metrics to decide whether to pre-plan diversion or extra support.")

        closure_summary = df.groupby("event_type", dropna=False).agg(
            Total=("requires_road_closure","size"),
            Closures=("requires_road_closure", lambda x: x.apply(parse_bool).sum()),
        ).reset_index()
        closure_summary["Closure rate"] = (closure_summary["Closures"] / closure_summary["Total"] * 100).round(1)
        closure_summary = closure_summary.sort_values("Closure rate", ascending=False)

        if not closure_summary.empty:
            fig = px.bar(closure_summary.head(8), x="event_type", y="Closure rate",
                         color="Closure rate",
                         color_continuous_scale=[[0,C["ok"]],[0.5,C["warn"]],[1,C["danger"]]],
                         labels={"event_type":"Event type","Closure rate":"Closure rate (%)"})
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font={"family":"Inter,sans-serif","size":12,"color":C["text"]},
                              margin={"t":20,"b":30,"l":40,"r":20}, height=320, showlegend=False,
                              xaxis=dict(gridcolor=C["border"],color=C["muted"]),
                              yaxis=dict(gridcolor=C["border"],color=C["muted"]))
            st.plotly_chart(fig, use_container_width=True)

            sel = closure_summary[closure_summary["event_type"] == current_event_type]
            if not sel.empty:
                rate      = float(sel.iloc[0]["Closure rate"])
                total_inc = int(sel.iloc[0]["Total"])
                color     = C["danger"] if rate > 50 else C["warn"] if rate > 25 else C["ok"]
                st.markdown(f"""
                <div class="card" style="border-left:4px solid {color}">
                  <div class="kpi-label">Road closure likelihood for {current_event_type}</div>
                  <div class="kpi-value">{rate:.1f}%</div>
                  <div class="kpi-note">Based on {total_inc} historical incidents of this type</div>
                </div>
                """, unsafe_allow_html=True)

        if avg_resolution is not None:
            st.markdown("**Historical resolution time**")
            st.markdown(f"Average time to resolve incidents: **{avg_resolution:.1f} hours**.")
            if avg_resolution > 8:
                st.warning("Expect a longer response timeline for complex incidents.")
            else:
                st.info("Most incidents resolve within a standard operational window.")


def page_hotspot_map(incident_df: pd.DataFrame):
    page_header("Incident Hotspot Map",
                "Where incidents cluster across Bengaluru. Filter by event type or zone to focus deployment.")

    map_df = incident_df.dropna(subset=["latitude","longitude"]).copy()
    if map_df.empty:
        st.warning("No valid coordinates found in the incident records.")
        return

    map_df = map_df[
        map_df["latitude"].notna() & map_df["longitude"].notna() &
        (map_df["latitude"] != 0) & (map_df["longitude"] != 0)
    ]
    for col in ["zone","junction"]:
        if col in map_df.columns:
            map_df = map_df[
                map_df[col].notna() &
                (map_df[col] != "") & (map_df[col] != "nan") & (map_df[col] != "Unknown")
            ]

    if map_df.empty:
        st.warning("No valid coordinates or zones found after filtering.")
        return

    map_df["priority"]   = map_df["priority"].fillna("Unknown")
    map_df["event_type"] = map_df["event_type"].fillna("Unknown")
    map_df["zone"]       = map_df["zone"].fillna("Unknown")

    fc1, fc2 = st.columns(2)
    with fc1:
        all_types = sorted(map_df["event_type"].unique().tolist())
        sel_types = st.multiselect("Filter by event type", all_types, default=all_types, key="map_types")
    with fc2:
        all_zones = sorted(map_df["zone"].unique().tolist())
        sel_zones = st.multiselect("Filter by zone", all_zones, default=all_zones, key="map_zones")

    filtered = map_df[map_df["event_type"].isin(sel_types) & map_df["zone"].isin(sel_zones)]
    if filtered.empty:
        st.warning("No incidents match the selected filters.")
        return

    kpi_row([
        ("Incidents shown",  f"{len(filtered):,}",                           "after filters"),
        ("Zones covered",    f"{filtered['zone'].nunique()}",                 ""),
        ("Most common type", filtered["event_type"].value_counts().index[0], ""),
        ("Top zone",         filtered["zone"].value_counts().index[0],       "by incident count"),
    ])

    hover_cols = [c for c in ["zone","junction","event_cause","police_station","status","priority"] if c in filtered.columns]
    fig = px.scatter_mapbox(
        filtered, lat="latitude", lon="longitude", color="priority",
        size_max=14, hover_name="event_type", hover_data=hover_cols,
        zoom=10, height=580,
        color_discrete_map={
            "Critical":C["danger"],"High":C["warn"],"Medium":C["accent"],"Low":C["ok"],
            "P1":C["danger"],"P2":C["warn"],"P3":C["accent"],"P4":C["ok"],"Unknown":C["muted"],
        },
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0},
                      legend_title_text="Priority", font={"family":"Inter,sans-serif"})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='section-rule'>", unsafe_allow_html=True)
    st.markdown("**Top 10 Incident Clusters**")
    hotspot = (
        filtered.groupby(["zone","junction"], dropna=True)
        .size().reset_index(name="Incidents")
        .sort_values("Incidents", ascending=False).head(10)
    )
    st.dataframe(hotspot.rename(columns={"zone":"Zone","junction":"Junction"}),
                 use_container_width=True, hide_index=True)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="TrafficMind | Incident Response & Traffic Operations Platform",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_styles()

    try:
        model, preprocessor, label_encoder, feature_columns = load_prediction_artifacts()
    except Exception as exc:
        st.error("Could not load model files. Check that models/ directory is present.")
        st.exception(exc)
        st.stop()

    incident_df = normalize_incident_dataframe(load_incident_data())

    for k, v in [("page","Dashboard"),("officers",0),("barricades",0),("diversions",0),("planning_inputs",{})]:
        if k not in st.session_state:
            st.session_state[k] = v

    new_page = render_sidebar(st.session_state.page)
    if new_page != st.session_state.page:
        st.session_state.page = new_page
        st.rerun()

    page = st.session_state.page

    if page == "Dashboard":
        page_dashboard(incident_df)
    elif page == "Incident Assessment":
        page_plan_event()
    elif page == "Severity Assessment":
        page_risk_forecast(model, preprocessor, label_encoder, feature_columns,
                           st.session_state.planning_inputs)
    elif page == "Response Plan":
        page_deploy_resources(st.session_state.planning_inputs)
    elif page == "Post-Event Learning":
        page_post_event_learning(incident_df, st.session_state.planning_inputs)
    elif page == "Historical Intelligence":
        page_past_events(incident_df, st.session_state.planning_inputs)
    elif page == "Hotspot Map":
        page_hotspot_map(incident_df)


if __name__ == "__main__":
    main()
