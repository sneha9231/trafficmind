from __future__ import annotations

import warnings
from pathlib import Path

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


APP_TITLE = "TrafficMind Operational Dashboard"
APP_ICON = "🚦"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "traffic_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
FEATURE_COLUMNS_PATH = MODEL_DIR / "feature_columns.pkl"
INCIDENT_EVENTS_PATH = BASE_DIR / "data" / "incident_events.csv"

RISK_SCORE_MAP = {
    "Free-flow": 20,
    "Moderate": 50,
    "Heavy": 75,
    "Gridlock": 95,
}

MODEL_NUMERIC_COLUMNS = [
    "avg_speed_kmph",
    "density_veh_per_km",
    "avg_wait_time_s",
    "occupancy_pct",
    "flow_veh_per_hr",
    "queue_length_veh",
    "avg_accel_ms2",
    "heading_deg",
    "signal_state_num",
    "incident_num",
    "temp_c",
    "visibility_km",
    "rain_intensity_mmph",
    "channel_busy_ratio_pct",
    "msg_rate_hz",
    "avg_comm_delay_ms",
    "rssi_dbm",
    "packet_loss_pct",
    "speed_density_ratio",
    "congestion_pressure",
    "wireless_congestion_intensity",
    "throughput_per_queued_vehicle",
    "acceleration_directionality",
    "weather_factor",
    "attendance",
]

SIMILARITY_INPUT_COLUMNS = [
    "event_type",
    "attendance",
    "location_type",
    "weather_factor",
    "density_veh_per_km",
    "queue_length_veh",
]


def ensure_sklearn_pickle_compatibility() -> None:
    try:
        import sklearn.compose._column_transformer as column_transformer_module

        if not hasattr(column_transformer_module, "_RemainderColsList"):

            class _RemainderColsList(list):
                pass

            column_transformer_module._RemainderColsList = _RemainderColsList
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def load_prediction_artifacts():
    ensure_sklearn_pickle_compatibility()
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    return model, preprocessor, label_encoder, feature_columns


@st.cache_data(show_spinner=False)
def load_incident_data() -> pd.DataFrame:
    df = pd.read_csv(INCIDENT_EVENTS_PATH)
    df.columns = [c.strip() for c in df.columns]
    return df


def parse_bool(value: object) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def normalize_incident_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "start_datetime" in normalized.columns:
        normalized["start_datetime"] = pd.to_datetime(normalized["start_datetime"], errors="coerce", utc=True)
    if "resolved_datetime" in normalized.columns:
        normalized["resolved_datetime"] = pd.to_datetime(normalized["resolved_datetime"], errors="coerce", utc=True)
    if "end_datetime" in normalized.columns:
        normalized["end_datetime"] = pd.to_datetime(normalized["end_datetime"], errors="coerce", utc=True)
    if "requires_road_closure" in normalized.columns:
        normalized["requires_road_closure"] = normalized["requires_road_closure"].apply(parse_bool)
    for col in ["latitude", "longitude"]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized


@st.cache_resource(show_spinner=False)
def load_similarity_engine():
    incident_df = normalize_incident_dataframe(load_incident_data())
    similarity_df = incident_df.copy()
    similarity_df["event_type"] = similarity_df["event_type"].fillna("Unknown")
    similarity_df["zone"] = similarity_df["zone"].fillna("Unknown")
    similarity_df["junction"] = similarity_df["junction"].fillna("Unknown")
    if "event_cause" in similarity_df.columns:
        similarity_df["event_cause"] = similarity_df["event_cause"].fillna("Unknown")
    else:
        similarity_df["event_cause"] = "Unknown"
    if "priority" in similarity_df.columns:
        similarity_df["priority"] = similarity_df["priority"].fillna("Unknown")
    else:
        similarity_df["priority"] = "Unknown"
    if "requires_road_closure" not in similarity_df.columns:
        similarity_df["requires_road_closure"] = False

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ["event_type", "zone", "junction", "priority"],
            ),
            (
                "num",
                StandardScaler(),
                ["latitude", "longitude"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    nn = NearestNeighbors(n_neighbors=5, metric="euclidean")
    pipeline = Pipeline([("preprocessor", preprocessor), ("neighbors", nn)])

    fitted_matrix = pipeline.named_steps["preprocessor"].fit_transform(
        similarity_df[["event_type", "zone", "junction", "priority", "latitude", "longitude"]].fillna(0)
    )
    pipeline.named_steps["neighbors"].fit(fitted_matrix)
    return similarity_df, pipeline


def set_page_config() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #07111f 0%, #0b1628 40%, #0e1b31 100%);
                color: #e5eefb;
            }
            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2rem;
            }
            [data-testid="stMetric"] {
                background: rgba(255,255,255,0.04);
                border: 1px solid rgba(255,255,255,0.08);
                padding: 0.75rem 0.9rem;
                border-radius: 14px;
            }
            [data-testid="stMetricLabel"] {
                color: #9fb6d0;
            }
            [data-testid="stMetricValue"] {
                color: #f5f9ff;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Operational intelligence for Bengaluru Traffic Police")


def compute_readiness_score(
    prediction_label: str,
    risk_score: float,
    attendance: float,
    weather_condition: str,
    officers: int,
    barricades: int,
    diversions: int,
) -> tuple[float, str]:
    category_score = {
        "Free-flow": 100,
        "Moderate": 75,
        "Heavy": 45,
        "Gridlock": 20,
    }.get(prediction_label, 50)

    weather_penalty = {
        "Clear": 0,
        "Cloudy": 4,
        "Rain": 10,
        "Heavy Rain": 16,
        "Storm": 20,
    }.get(weather_condition, 6)

    demand_penalty = min(attendance / 6000.0, 25.0)
    risk_penalty = min(float(risk_score) * 0.45, 45.0)
    deployment_bonus = min(officers * 1.6 + barricades * 2.8 + diversions * 4.5, 35.0)

    readiness = category_score - risk_penalty - weather_penalty - demand_penalty + deployment_bonus
    readiness = max(0.0, min(100.0, readiness))

    if readiness <= 40:
        status = "Critical"
    elif readiness <= 60:
        status = "Needs Attention"
    elif readiness <= 80:
        status = "Operationally Ready"
    else:
        status = "Well Prepared"

    return readiness, status


def render_event_details_tab() -> dict[str, object]:
    st.subheader("Event Details")
    st.write("Start here with the event plan. The rest of the workflow updates from these selections.")

    col1, col2 = st.columns(2)
    with col1:
        event_type = st.selectbox(
            "Event Type",
            ["Accident", "Concert", "Construction", "Other", "Parade", "Sporting Event"],
            index=0,
        )
        expected_attendance = st.number_input(
            "Expected Attendance",
            min_value=0,
            max_value=1_000_000,
            value=25000,
            step=500,
        )
        weather_condition = st.selectbox(
            "Weather Condition",
            ["Clear", "Cloudy", "Rain", "Heavy Rain", "Storm"],
            index=0,
        )
    with col2:
        zone = st.selectbox(
            "Zone",
            ["Central", "East", "West", "North", "South", "Outer Ring Road", "Peripheral"],
            index=0,
        )
        time_of_day = st.selectbox(
            "Time of Day",
            ["Morning Peak", "Midday", "Evening Peak", "Night"],
            index=0,
        )

    planning_inputs = {
        "event_type": event_type,
        "expected_attendance": expected_attendance,
        "zone": zone,
        "weather_condition": weather_condition,
        "time_of_day": time_of_day,
    }
    st.session_state.weather_condition = weather_condition

    st.markdown("#### Planning Summary")
    summary_cols = st.columns(5)
    summary_cols[0].metric("Event Type", event_type)
    summary_cols[1].metric("Attendance", f"{expected_attendance:,}")
    summary_cols[2].metric("Zone", zone)
    summary_cols[3].metric("Weather", weather_condition)
    summary_cols[4].metric("Time", time_of_day)

    return planning_inputs


def build_input_frame(user_inputs: dict[str, object], raw_columns: list[str]) -> pd.DataFrame:
    row = {column: 0 for column in raw_columns}
    row.update(
        {
            "road_segment_id": user_inputs.get("road_segment_id", "S001"),
            "avg_speed_kmph": user_inputs.get("avg_speed_kmph", 45.0),
            "density_veh_per_km": float(user_inputs["density_veh_per_km"]),
            "avg_wait_time_s": user_inputs.get("avg_wait_time_s", 30.0),
            "occupancy_pct": float(user_inputs["occupancy_pct"]),
            "flow_veh_per_hr": float(user_inputs["flow_veh_per_hr"]),
            "queue_length_veh": float(user_inputs["queue_length_veh"]),
            "avg_accel_ms2": user_inputs.get("avg_accel_ms2", 0.0),
            "heading_deg": user_inputs.get("heading_deg", 0.0),
            "signal_state_num": user_inputs.get("signal_state_num", 0.0),
            "incident_num": user_inputs.get("incident_num", 0.0),
            "temp_c": user_inputs.get("temp_c", 28.0),
            "visibility_km": user_inputs.get("visibility_km", 10.0),
            "rain_intensity_mmph": user_inputs.get("rain_intensity_mmph", 0.0),
            "channel_busy_ratio_pct": user_inputs.get("channel_busy_ratio_pct", 0.0),
            "msg_rate_hz": user_inputs.get("msg_rate_hz", 0.0),
            "avg_comm_delay_ms": user_inputs.get("avg_comm_delay_ms", 0.0),
            "rssi_dbm": user_inputs.get("rssi_dbm", -70.0),
            "packet_loss_pct": user_inputs.get("packet_loss_pct", 0.0),
            "speed_density_ratio": user_inputs.get("speed_density_ratio", 0.0),
            "congestion_pressure": user_inputs.get("congestion_pressure", 0.0),
            "wireless_congestion_intensity": user_inputs.get("wireless_congestion_intensity", 0.0),
            "throughput_per_queued_vehicle": user_inputs.get("throughput_per_queued_vehicle", 0.0),
            "acceleration_directionality": user_inputs.get("acceleration_directionality", 0.0),
            "weather_factor": float(user_inputs["weather_factor"]),
            "event_type": user_inputs["event_type"],
            "attendance": float(user_inputs["attendance"]),
            "location_type": user_inputs["location_type"],
        }
    )
    return pd.DataFrame([row], columns=raw_columns)


def map_planning_to_model_inputs(planning_inputs: dict[str, object]) -> dict[str, object]:
    event_type = planning_inputs["event_type"]
    zone = planning_inputs["zone"]
    weather_condition = planning_inputs["weather_condition"]
    time_of_day = planning_inputs["time_of_day"]
    attendance = float(planning_inputs["expected_attendance"])

    zone_to_location = {
        "Central": "Urban",
        "East": "Urban",
        "West": "Suburban",
        "North": "Suburban",
        "South": "Urban",
        "Outer Ring Road": "Highway",
        "Peripheral": "Rural",
    }
    weather_to_factor = {
        "Clear": 10.0,
        "Cloudy": 35.0,
        "Rain": 75.0,
        "Heavy Rain": 90.0,
        "Storm": 100.0,
    }
    time_to_profile = {
        "Morning Peak": {"avg_speed_kmph": 28.0, "avg_wait_time_s": 55.0, "flow_veh_per_hr": 3400.0, "queue_length_veh": 110.0, "occupancy_pct": 72.0},
        "Midday": {"avg_speed_kmph": 42.0, "avg_wait_time_s": 28.0, "flow_veh_per_hr": 2200.0, "queue_length_veh": 60.0, "occupancy_pct": 52.0},
        "Evening Peak": {"avg_speed_kmph": 22.0, "avg_wait_time_s": 70.0, "flow_veh_per_hr": 3900.0, "queue_length_veh": 145.0, "occupancy_pct": 80.0},
        "Night": {"avg_speed_kmph": 48.0, "avg_wait_time_s": 18.0, "flow_veh_per_hr": 1200.0, "queue_length_veh": 25.0, "occupancy_pct": 35.0},
    }
    event_profile = {
        "Accident": {"density_veh_per_km": 95.0, "avg_wait_time_s": 80.0, "flow_veh_per_hr": 1800.0, "queue_length_veh": 160.0},
        "Concert": {"density_veh_per_km": 115.0, "avg_wait_time_s": 60.0, "flow_veh_per_hr": 2600.0, "queue_length_veh": 140.0},
        "Construction": {"density_veh_per_km": 85.0, "avg_wait_time_s": 75.0, "flow_veh_per_hr": 1600.0, "queue_length_veh": 120.0},
        "Other": {"density_veh_per_km": 70.0, "avg_wait_time_s": 40.0, "flow_veh_per_hr": 1800.0, "queue_length_veh": 70.0},
        "Parade": {"density_veh_per_km": 125.0, "avg_wait_time_s": 90.0, "flow_veh_per_hr": 1100.0, "queue_length_veh": 180.0},
        "Sporting Event": {"density_veh_per_km": 110.0, "avg_wait_time_s": 65.0, "flow_veh_per_hr": 2400.0, "queue_length_veh": 130.0},
    }

    profile = time_to_profile.get(time_of_day, time_to_profile["Midday"]).copy()
    profile.update(event_profile.get(event_type, event_profile["Other"]))

    density_multiplier = 1.0 + min(attendance / 100000.0, 1.25)
    profile["density_veh_per_km"] = round(profile["density_veh_per_km"] * density_multiplier, 2)
    profile["flow_veh_per_hr"] = round(profile["flow_veh_per_hr"] * (0.85 + min(attendance / 200000.0, 0.45)), 2)
    profile["queue_length_veh"] = round(profile["queue_length_veh"] * (0.9 + min(attendance / 150000.0, 0.55)), 2)
    profile["occupancy_pct"] = round(profile["occupancy_pct"] * (0.95 + min(attendance / 250000.0, 0.4)), 2)

    return {
        "event_type": event_type,
        "attendance": attendance,
        "location_type": zone_to_location.get(zone, "Urban"),
        "weather_factor": weather_to_factor.get(weather_condition, 35.0),
        "density_veh_per_km": profile["density_veh_per_km"],
        "occupancy_pct": profile["occupancy_pct"],
        "flow_veh_per_hr": profile["flow_veh_per_hr"],
        "queue_length_veh": profile["queue_length_veh"],
        "avg_speed_kmph": profile["avg_speed_kmph"],
        "avg_wait_time_s": profile["avg_wait_time_s"],
        "avg_accel_ms2": 0.0,
        "heading_deg": 0.0,
        "signal_state_num": 0.0,
        "incident_num": 0.0,
        "temp_c": 28.0 if weather_condition in {"Clear", "Cloudy"} else 24.0,
        "visibility_km": 10.0 if weather_condition == "Clear" else 6.0 if weather_condition == "Cloudy" else 3.0,
        "rain_intensity_mmph": 0.0 if weather_condition == "Clear" else 1.5 if weather_condition == "Cloudy" else 8.0,
        "channel_busy_ratio_pct": 0.0,
        "msg_rate_hz": 0.0,
        "avg_comm_delay_ms": 0.0,
        "rssi_dbm": -70.0,
        "packet_loss_pct": 0.0,
        "speed_density_ratio": round(profile["avg_speed_kmph"] / max(profile["density_veh_per_km"], 1.0), 4),
        "congestion_pressure": round(profile["density_veh_per_km"] / max(profile["flow_veh_per_hr"], 1.0), 4),
        "wireless_congestion_intensity": 0.0,
        "throughput_per_queued_vehicle": round(profile["flow_veh_per_hr"] / max(profile["queue_length_veh"], 1.0), 4),
        "acceleration_directionality": 0.0,
        "road_segment_id": {
            "Central": "S001",
            "East": "S050",
            "West": "S120",
            "North": "S220",
            "South": "S320",
            "Outer Ring Road": "S420",
            "Peripheral": "S500",
        }.get(zone, "S001"),
    }


def predict_congestion(model, preprocessor, label_encoder, feature_columns, user_inputs):
    input_frame = build_input_frame(user_inputs, list(getattr(preprocessor, "feature_names_in_", feature_columns)))
    cat_features = preprocessor.transform(input_frame)
    cat_feature_names = list(preprocessor.get_feature_names_out())
    cat_df = pd.DataFrame(cat_features, columns=cat_feature_names, index=input_frame.index)
    numeric_df = input_frame[MODEL_NUMERIC_COLUMNS].reset_index(drop=True)
    model_input = pd.concat([cat_df.reset_index(drop=True), numeric_df], axis=1)
    model_input = model_input[[c for c in feature_columns if c in model_input.columns]]
    model_input = model_input.reindex(columns=feature_columns, fill_value=0)
    prediction_raw = model.predict(model_input.to_numpy(dtype=float))[0]
    proba = model.predict_proba(model_input.to_numpy(dtype=float))[0] if hasattr(model, "predict_proba") else None
    try:
        prediction_label = label_encoder.inverse_transform([prediction_raw])[0]
    except Exception:
        prediction_label = str(prediction_raw)
    confidence = float(proba.max() * 100) if proba is not None else 0.0
    return input_frame, prediction_label, confidence, proba


def render_prediction_tab(model, preprocessor, label_encoder, feature_columns, user_inputs, intervention_values):
    st.subheader("Traffic Risk Assessment")
    input_frame, prediction_label, confidence, proba = predict_congestion(
        model, preprocessor, label_encoder, feature_columns, user_inputs
    )

    risk_score = RISK_SCORE_MAP.get(prediction_label, 0)
    original_risk = float(risk_score)
    officers = intervention_values["officers"]
    barricades = intervention_values["barricades"]
    diversions = intervention_values["diversions"]
    adjusted_risk = max(0.0, min(100.0, original_risk - officers - (barricades * 2) - (diversions * 4)))
    improvement_pct = ((original_risk - adjusted_risk) / original_risk * 100.0) if original_risk > 0 else 0.0
    readiness_score, readiness_status = compute_readiness_score(
        prediction_label,
        adjusted_risk,
        float(user_inputs["attendance"]),
        st.session_state.get("weather_condition", "Clear"),
        officers,
        barricades,
        diversions,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Congestion Label", prediction_label)
    with col2:
        st.metric("Confidence", f"{confidence:.2f}%")
    with col3:
        st.metric("Risk Score", f"{adjusted_risk:.0f}")

    readiness_col1, readiness_col2 = st.columns([2, 1])
    with readiness_col1:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(18,35,58,0.95), rgba(9,18,33,0.98));
                border: 1px solid rgba(120, 180, 255, 0.18);
                border-radius: 20px;
                padding: 1.2rem 1.4rem;
                margin: 0.5rem 0 1rem 0;
                box-shadow: 0 12px 30px rgba(0,0,0,0.28);
            ">
                <div style="font-size: 0.9rem; letter-spacing: 0.08em; text-transform: uppercase; color: #8fb3d9;">Traffic Readiness Score</div>
                <div style="font-size: 3.2rem; font-weight: 700; color: #f5f9ff; line-height: 1.1;">{readiness_score:.0f}</div>
                <div style="font-size: 1.05rem; color: #cfe3ff;">{readiness_status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with readiness_col2:
        badge_color = {
            "Critical": "#ef4444",
            "Needs Attention": "#f59e0b",
            "Operationally Ready": "#22c55e",
            "Well Prepared": "#38bdf8",
        }[readiness_status]
        st.markdown(
            f"""
            <div style="
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                margin-top: 0.5rem;
            ">
                <div style="
                    display:inline-block;
                    background:{badge_color};
                    color:#0b1220;
                    padding:0.4rem 0.8rem;
                    border-radius:999px;
                    font-weight:700;
                    margin-bottom:0.75rem;
                ">{readiness_status}</div>
                <div style="color:#cfe3ff; line-height:1.5;">
                    The readiness score recomputes automatically from the latest prediction, weather, attendance, and deployment plan.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=readiness_score,
            number={"suffix": "/100"},
            title={"text": "Traffic Readiness Score"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#38bdf8"},
                "steps": [
                    {"range": [0, 40], "color": "#7f1d1d"},
                    {"range": [40, 60], "color": "#92400e"},
                    {"range": [60, 80], "color": "#14532d"},
                    {"range": [80, 100], "color": "#0f766e"},
                ],
                "threshold": {"line": {"color": "#38bdf8", "width": 4}, "value": readiness_score},
            },
        )
    )
    gauge.update_layout(height=330, margin={"t": 45, "b": 20, "l": 20, "r": 20})
    st.plotly_chart(gauge, use_container_width=True)

    st.markdown("### Intervention Simulator")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.metric("Original Risk", f"{original_risk:.1f}")
    with k2:
        st.metric("Adjusted Risk", f"{adjusted_risk:.1f}")
    with k3:
        st.metric("Improvement %", f"{improvement_pct:.1f}%")

    bar_fig = go.Figure(
        data=[
            go.Bar(
                x=["Before", "After"],
                y=[original_risk, adjusted_risk],
                marker_color=["#d97706", "#22c55e"],
                text=[f"{original_risk:.1f}", f"{adjusted_risk:.1f}"],
                textposition="auto",
            )
        ]
    )
    bar_fig.update_layout(
        title="Before vs After Risk",
        yaxis=dict(range=[0, 100], title="Risk Score"),
        xaxis_title="Scenario",
        height=360,
        margin={"t": 50, "b": 20, "l": 20, "r": 20},
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("### Cost Optimization")
    total_cost = (officers * 3000) + (barricades * 5000) + (diversions * 8000)
    risk_reduction = original_risk - adjusted_risk
    cost_efficiency = (risk_reduction / total_cost) if total_cost > 0 else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Cost", f"₹{total_cost:,.0f}")
    c2.metric("Risk Reduction", f"{risk_reduction:.1f}")
    c3.metric("Cost Efficiency", f"{cost_efficiency:.6f}")

    st.markdown("#### Prepared Input")
    st.dataframe(input_frame, use_container_width=True)

    with st.expander("Prediction Probabilities", expanded=False):
        if proba is not None:
            labels = [
                label_encoder.inverse_transform([label])[0]
                if hasattr(label_encoder, "inverse_transform") and not isinstance(label, str)
                else str(label)
                for label in list(getattr(model, "classes_", range(len(proba))))
            ]
            st.dataframe(
                pd.DataFrame(
                    {"Congestion Label": labels, "Probability %": [round(float(p) * 100, 2) for p in proba]}
                ).sort_values("Probability %", ascending=False),
                use_container_width=True,
            )


def render_incident_intelligence(incident_df: pd.DataFrame):
    st.subheader("Historical Incident Intelligence")
    total_incidents = len(incident_df)
    top_causes = incident_df["event_cause"].fillna("Unknown").value_counts().head(5).rename_axis("event_cause").reset_index(name="count")
    top_zones = incident_df["zone"].fillna("Unknown").value_counts().head(5).rename_axis("zone").reset_index(name="count")
    closure_pct = incident_df["requires_road_closure"].fillna(False).mean() * 100 if "requires_road_closure" in incident_df else 0
    top_junctions = incident_df["junction"].fillna("Unknown").value_counts().head(5).rename_axis("junction").reset_index(name="count")

    a, b, c, d = st.columns(4)
    a.metric("Total Incidents", f"{total_incidents:,}")
    b.metric("% Road Closure", f"{closure_pct:.1f}%")
    c.metric("Top Cause", top_causes.index[0] if len(top_causes) else "N/A")
    d.metric("Top Zone", top_zones.index[0] if len(top_zones) else "N/A")

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig = px.bar(
            top_causes,
            x="event_cause",
            y="count",
            title="Top Event Causes",
            labels={"event_cause": "Event Cause", "count": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)
    with chart_col2:
        fig = px.bar(
            top_zones,
            x="zone",
            y="count",
            title="Top Affected Zones",
            labels={"zone": "Zone", "count": "Count"},
        )
        st.plotly_chart(fig, use_container_width=True)

    junction_fig = px.bar(
        top_junctions,
        x="junction",
        y="count",
        title="Most Affected Junctions",
        labels={"junction": "Junction", "count": "Count"},
    )
    st.plotly_chart(junction_fig, use_container_width=True)

    closure_fig = px.pie(
        incident_df.assign(
            closure_state=incident_df["requires_road_closure"].map({True: "Requires Closure", False: "No Closure"})
        ),
        names="closure_state",
        title="Road Closure Requirement Share",
    )
    st.plotly_chart(closure_fig, use_container_width=True)

    st.markdown("#### Police Deployment Insights")
    insights = []
    if len(top_zones):
        insights.append(f"Zone {top_zones.index[0]} experiences the highest incident volume.")
    if len(top_junctions):
        insights.append(f"Most incidents occur near {top_junctions.index[0]}.")
    closure_zone = (
        incident_df.groupby("zone")["requires_road_closure"].mean().sort_values(ascending=False).head(1)
        if "requires_road_closure" in incident_df.columns
        else None
    )
    if closure_zone is not None and len(closure_zone):
        insights.append(
            f"Incidents in {closure_zone.index[0]} frequently require road closures."
        )
    for item in insights:
        st.write(f"- {item}")


def render_zone_intelligence(incident_df: pd.DataFrame):
    st.subheader("Bengaluru Zone Intelligence")

    zone_df = incident_df.copy()
    zone_df["zone"] = zone_df["zone"].fillna("Unknown")
    zone_counts = zone_df["zone"].value_counts().rename_axis("zone").reset_index(name="incident_count")
    total_zones_impacted = int(zone_df["zone"].nunique(dropna=True))
    most_active_zone = zone_counts.iloc[0]["zone"] if not zone_counts.empty else "N/A"

    time_col = None
    for candidate in ["start_datetime", "created_date", "modified_datetime"]:
        if candidate in zone_df.columns:
            zone_df[candidate] = pd.to_datetime(zone_df[candidate], errors="coerce", utc=True)
            if zone_df[candidate].notna().any():
                time_col = candidate
                break

    if time_col is not None:
        recent_cutoff = zone_df[time_col].max() - pd.Timedelta(days=30)
        prior_cutoff = zone_df[time_col].max() - pd.Timedelta(days=60)
        recent = zone_df[zone_df[time_col] >= recent_cutoff].groupby("zone").size()
        prior_window = zone_df[(zone_df[time_col] >= prior_cutoff) & (zone_df[time_col] < recent_cutoff)].groupby("zone").size()
        growth_df = (
            pd.concat([recent.rename("recent"), prior_window.rename("prior")], axis=1)
            .fillna(0)
            .reset_index()
        )
        growth_df["growth"] = growth_df["recent"] - growth_df["prior"]
        zone_with_highest_growth = (
            growth_df.sort_values(["growth", "recent"], ascending=[False, False]).iloc[0]["zone"]
            if not growth_df.empty
            else "N/A"
        )
    else:
        growth_df = zone_counts.copy()
        growth_df["growth"] = 0
        zone_with_highest_growth = "N/A"

    zone_metrics = st.columns(3)
    zone_metrics[0].metric("Most Active Zone", most_active_zone)
    zone_metrics[1].metric("Total Zones Impacted", f"{total_zones_impacted}")
    zone_metrics[2].metric("Zone with Highest Incident Growth", zone_with_highest_growth)

    top_zones = zone_counts.head(10)
    col1, col2 = st.columns(2)
    with col1:
        bar_fig = px.bar(
            top_zones,
            x="zone",
            y="incident_count",
            title="Top Affected Zones",
            labels={"zone": "Zone", "incident_count": "Incident Count"},
        )
        st.plotly_chart(bar_fig, use_container_width=True)
    with col2:
        pie_fig = px.pie(
            top_zones,
            names="zone",
            values="incident_count",
            title="Percentage Distribution by Zone",
            hole=0.45,
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    growth_view = growth_df.sort_values("growth", ascending=False).head(10) if "growth" in growth_df.columns else top_zones.assign(growth=0)
    growth_fig = px.bar(
        growth_view,
        x="zone",
        y="growth",
        title="Zone Incident Growth",
        labels={"zone": "Zone", "growth": "Growth in Incidents"},
        color="growth",
        color_continuous_scale="Reds",
    )
    st.plotly_chart(growth_fig, use_container_width=True)

    st.markdown("#### Zone Planning Table")
    planning_table = top_zones.copy()
    planning_table["share_%"] = (planning_table["incident_count"] / planning_table["incident_count"].sum() * 100).round(2)
    st.dataframe(planning_table.rename(columns={"zone": "Zone", "incident_count": "Incident Count", "share_%": "Share %"}), use_container_width=True)


def render_junction_risk_ranking(incident_df: pd.DataFrame):
    st.subheader("High Risk Junction Intelligence")
    st.caption("Recurring junctions with high incident frequency and weighted operational risk.")

    junction_df = incident_df.copy()
    junction_df["junction"] = junction_df["junction"].fillna("Unknown")
    junction_df["priority"] = junction_df["priority"].fillna("Unknown").astype(str)

    priority_weights = {
        "Low": 1.0,
        "Medium": 2.0,
        "High": 3.0,
        "Critical": 4.0,
        "P1": 4.0,
        "P2": 3.0,
        "P3": 2.0,
        "P4": 1.0,
        "Unknown": 1.0,
    }

    junction_df["priority_weight"] = junction_df["priority"].map(priority_weights).fillna(1.0)
    ranking_df = (
        junction_df.groupby("junction", dropna=False)
        .agg(
            incident_count=("junction", "size"),
            priority_weighted_risk_score=("priority_weight", "sum"),
        )
        .reset_index()
    )
    ranking_df["priority_weighted_risk_score"] = ranking_df["priority_weighted_risk_score"].round(2)
    ranking_df = ranking_df.sort_values(
        ["priority_weighted_risk_score", "incident_count"], ascending=[False, False]
    ).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1

    top_10 = ranking_df.head(10)
    if top_10.empty:
        st.warning("No junction data available.")
        return

    metrics = st.columns(3)
    metrics[0].metric("Top Risk Junction", top_10.iloc[0]["junction"])
    metrics[1].metric("Most Recurring Junction Incidents", f"{int(top_10.iloc[0]['incident_count'])}")
    metrics[2].metric("High-Risk Junctions Shown", f"{len(top_10)}")

    chart_df = top_10.sort_values("priority_weighted_risk_score", ascending=True)
    chart_fig = px.bar(
        chart_df,
        x="priority_weighted_risk_score",
        y="junction",
        orientation="h",
        title="Top 10 High-Risk Junctions",
        labels={
            "junction": "Junction",
            "priority_weighted_risk_score": "Priority-Weighted Risk Score",
        },
        color="incident_count",
        color_continuous_scale="Reds",
        text="incident_count",
    )
    chart_fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=500,
        margin={"t": 60, "b": 20, "l": 20, "r": 20},
    )
    st.plotly_chart(chart_fig, use_container_width=True)

    st.dataframe(
        top_10.rename(
            columns={
                "rank": "Rank",
                "junction": "Junction",
                "incident_count": "Incident Count",
                "priority_weighted_risk_score": "Priority-Weighted Risk Score",
            }
        )[["Rank", "Junction", "Incident Count", "Priority-Weighted Risk Score"]],
        use_container_width=True,
    )


def render_road_closure_probability(incident_df: pd.DataFrame):
    st.subheader("Road Closure Probability")
    st.caption("Supports pre-event traffic planning by estimating closure likelihood from historical incidents.")

    closure_df = incident_df.copy()
    closure_df["event_type"] = closure_df["event_type"].fillna("Unknown")
    closure_df["event_cause"] = closure_df["event_cause"].fillna("Unknown")
    closure_df["requires_road_closure"] = closure_df["requires_road_closure"].fillna(False).astype(bool)

    closure_summary = (
        closure_df.groupby(["event_type", "event_cause"], dropna=False)
        .agg(
            total_incidents=("event_cause", "size"),
            road_closure_required=("requires_road_closure", "sum"),
        )
        .reset_index()
    )
    closure_summary["road_closure_probability_%"] = (
        closure_summary["road_closure_required"] / closure_summary["total_incidents"] * 100
    ).round(2)
    closure_summary = closure_summary.sort_values(
        ["road_closure_probability_%", "total_incidents"], ascending=[False, False]
    )

    event_options = sorted(closure_df["event_type"].dropna().unique().tolist())
    selected_event_type = st.selectbox("Select Event Type", event_options, key="closure_event_type_selector")

    selected_view = closure_summary[closure_summary["event_type"] == selected_event_type].copy()
    if selected_view.empty:
        st.info("No historical closure data found for the selected event type.")
    else:
        st.metric(
            "Historical Road Closure Probability",
            f"{selected_view['road_closure_probability_%'].mean():.2f}%",
        )

    table_fig_df = closure_summary.rename(
        columns={
            "event_type": "Event Type",
            "event_cause": "Event Cause",
            "total_incidents": "Total Incidents",
            "road_closure_required": "Road Closure Required",
            "road_closure_probability_%": "Road Closure Probability (%)",
        }
    )
    st.dataframe(
        table_fig_df[[
            "Event Type",
            "Event Cause",
            "Total Incidents",
            "Road Closure Required",
            "Road Closure Probability (%)",
        ]],
        use_container_width=True,
    )

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        event_type_summary = (
            closure_df.groupby("event_type", dropna=False)
            .agg(
                total_incidents=("event_type", "size"),
                road_closure_required=("requires_road_closure", "sum"),
            )
            .reset_index()
        )
        event_type_summary["road_closure_probability_%"] = (
            event_type_summary["road_closure_required"] / event_type_summary["total_incidents"] * 100
        ).round(2)
        fig = px.bar(
            event_type_summary.sort_values("road_closure_probability_%", ascending=False),
            x="event_type",
            y="road_closure_probability_%",
            title="Road Closure Probability by Event Type",
            labels={"event_type": "Event Type", "road_closure_probability_%": "Probability (%)"},
            color="road_closure_probability_%",
            color_continuous_scale="OrRd",
        )
        st.plotly_chart(fig, use_container_width=True)
    with chart_col2:
        cause_summary = (
            closure_df.groupby("event_cause", dropna=False)
            .agg(
                total_incidents=("event_cause", "size"),
                road_closure_required=("requires_road_closure", "sum"),
            )
            .reset_index()
        )
        cause_summary["road_closure_probability_%"] = (
            cause_summary["road_closure_required"] / cause_summary["total_incidents"] * 100
        ).round(2)
        top_cause_summary = cause_summary.sort_values("road_closure_probability_%", ascending=False).head(10)
        fig = px.bar(
            top_cause_summary,
            x="event_cause",
            y="road_closure_probability_%",
            title="Top Causes by Road Closure Probability",
            labels={"event_cause": "Event Cause", "road_closure_probability_%": "Probability (%)"},
            color="road_closure_probability_%",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig, use_container_width=True)

    if not selected_view.empty:
        selected_chart = px.bar(
            selected_view.sort_values("road_closure_probability_%", ascending=False),
            x="event_cause",
            y="road_closure_probability_%",
            title=f"Historical Road Closure Probability for {selected_event_type}",
            labels={"event_cause": "Event Cause", "road_closure_probability_%": "Probability (%)"},
        )
        st.plotly_chart(selected_chart, use_container_width=True)


def render_historical_deployment_benchmark(incident_df: pd.DataFrame):
    st.subheader("Historical Deployment Benchmark")
    st.caption("Operational reference for planning manpower and control measures before deployment.")

    benchmark_df = incident_df.copy()
    benchmark_df["event_type"] = benchmark_df["event_type"].fillna("Unknown")
    benchmark_df["zone"] = benchmark_df["zone"].fillna("Unknown")
    benchmark_df["requires_road_closure"] = benchmark_df["requires_road_closure"].fillna(False).astype(bool)

    event_options = sorted(benchmark_df["event_type"].unique().tolist())
    selected_event_type = st.selectbox("Select Event Type for Benchmark", event_options, key="benchmark_event_type_selector")

    similar_incidents = benchmark_df[benchmark_df["event_type"] == selected_event_type].copy()
    if similar_incidents.empty:
        st.info("No historical incidents available for the selected event type.")
        return

    frequency = len(similar_incidents)
    affected_zones = similar_incidents["zone"].value_counts().head(5)
    road_closure_rate = similar_incidents["requires_road_closure"].mean() * 100
    top_zone = affected_zones.index[0] if not affected_zones.empty else "N/A"

    insight_card = f"Events of this type historically impact {top_zone} and require road closure in {road_closure_rate:.0f}% of cases."
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 1rem; color: #cfe3ff; line-height: 1.6;">
                {insight_card}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics = st.columns(3)
    metrics[0].metric("Historical Frequency", f"{frequency}")
    metrics[1].metric("Most Affected Zone", top_zone)
    metrics[2].metric("Road Closure Rate", f"{road_closure_rate:.1f}%")

    zone_fig = px.bar(
        affected_zones.rename_axis("zone").reset_index(name="incident_count"),
        x="zone",
        y="incident_count",
        title=f"Affected Zones for {selected_event_type}",
        labels={"zone": "Zone", "incident_count": "Incidents"},
        orientation="v",
        color="incident_count",
        color_continuous_scale="Blues",
    )
    zone_fig.update_layout(showlegend=False)
    st.plotly_chart(zone_fig, use_container_width=True)

    closure_fig = px.pie(
        similar_incidents.assign(
            closure_state=similar_incidents["requires_road_closure"].map({True: "Requires Closure", False: "No Closure"})
        ),
        names="closure_state",
        title=f"Road Closure Split for {selected_event_type}",
    )
    st.plotly_chart(closure_fig, use_container_width=True)

    benchmark_table = similar_incidents[
        [
            col
            for col in ["event_type", "event_cause", "zone", "junction", "priority", "requires_road_closure"]
            if col in similar_incidents.columns
        ]
    ].copy()
    benchmark_table = benchmark_table.rename(
        columns={
            "event_type": "Event Type",
            "event_cause": "Event Cause",
            "zone": "Zone",
            "junction": "Junction",
            "priority": "Priority",
            "requires_road_closure": "Road Closure Required",
        }
    )
    st.dataframe(benchmark_table.head(10), use_container_width=True)


def render_incident_timeline_analysis(incident_df: pd.DataFrame):
    st.subheader("Incident Timeline Analysis")
    st.caption("Measures incident duration and resolution performance for operational traffic management.")

    timeline_df = incident_df.copy()
    for col in ["start_datetime", "end_datetime", "resolved_datetime"]:
        if col in timeline_df.columns:
            timeline_df[col] = pd.to_datetime(timeline_df[col], errors="coerce", utc=True)

    if "start_datetime" not in timeline_df.columns:
        st.info("No start time data available for timeline analysis.")
        return

    timeline_df = timeline_df.dropna(subset=["start_datetime"])
    if timeline_df.empty:
        st.warning("No valid timeline records found.")
        return

    if "end_datetime" in timeline_df.columns:
        timeline_df["incident_duration_hours"] = (
            (timeline_df["end_datetime"].fillna(timeline_df["resolved_datetime"]) - timeline_df["start_datetime"])
            .dt.total_seconds()
            / 3600.0
        )
    else:
        timeline_df["incident_duration_hours"] = (
            (timeline_df["resolved_datetime"] - timeline_df["start_datetime"]).dt.total_seconds() / 3600.0
        )

    if "resolved_datetime" in timeline_df.columns:
        timeline_df["resolution_duration_hours"] = (
            (timeline_df["resolved_datetime"] - timeline_df["start_datetime"]).dt.total_seconds() / 3600.0
        )
    else:
        timeline_df["resolution_duration_hours"] = pd.NA

    timeline_df = timeline_df[
        timeline_df["incident_duration_hours"].notna() & (timeline_df["incident_duration_hours"] >= 0)
    ].copy()
    if "resolution_duration_hours" in timeline_df.columns:
        timeline_df = timeline_df[
            timeline_df["resolution_duration_hours"].isna()
            | (timeline_df["resolution_duration_hours"] >= 0)
        ]

    if timeline_df.empty:
        st.warning("No valid duration records available.")
        return

    avg_incident_duration = timeline_df["incident_duration_hours"].mean()
    avg_resolution_duration = (
        timeline_df["resolution_duration_hours"].dropna().mean()
        if timeline_df["resolution_duration_hours"].notna().any()
        else 0.0
    )
    fastest_resolution = (
        timeline_df["resolution_duration_hours"].dropna().min()
        if timeline_df["resolution_duration_hours"].notna().any()
        else 0.0
    )
    slowest_resolution = (
        timeline_df["resolution_duration_hours"].dropna().max()
        if timeline_df["resolution_duration_hours"].notna().any()
        else 0.0
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Avg Incident Duration", f"{avg_incident_duration:.2f} hrs")
    k2.metric("Avg Resolution Duration", f"{avg_resolution_duration:.2f} hrs")
    k3.metric("Fastest Resolution", f"{fastest_resolution:.2f} hrs")
    k4.metric("Slowest Resolution", f"{slowest_resolution:.2f} hrs")

    timeline_df["date"] = timeline_df["start_datetime"].dt.date
    timeline_by_day = timeline_df.groupby("date").agg(
        incident_count=("incident_duration_hours", "size"),
        avg_incident_duration_hours=("incident_duration_hours", "mean"),
        avg_resolution_duration_hours=("resolution_duration_hours", "mean"),
    ).reset_index()

    by_event_type = (
        timeline_df.groupby("event_type", dropna=False)["incident_duration_hours"]
        .mean()
        .reset_index()
        .sort_values("incident_duration_hours", ascending=False)
    )
    by_zone = (
        timeline_df.groupby("zone", dropna=False)["incident_duration_hours"]
        .mean()
        .reset_index()
        .sort_values("incident_duration_hours", ascending=False)
    )

    col1, col2 = st.columns(2)
    with col1:
        line_fig = px.line(
            timeline_by_day,
            x="date",
            y=["incident_count", "avg_incident_duration_hours"],
            title="Incident Activity Over Time",
            labels={"value": "Value", "date": "Date"},
        )
        st.plotly_chart(line_fig, use_container_width=True)
    with col2:
        resolution_fig = px.line(
            timeline_by_day,
            x="date",
            y="avg_resolution_duration_hours",
            title="Average Resolution Duration Over Time",
            labels={"avg_resolution_duration_hours": "Hours", "date": "Date"},
        )
        st.plotly_chart(resolution_fig, use_container_width=True)

    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        incident_dist_fig = px.histogram(
            timeline_df,
            x="incident_duration_hours",
            nbins=30,
            title="Incident Duration Distribution",
            labels={"incident_duration_hours": "Incident Duration (hrs)"},
        )
        st.plotly_chart(incident_dist_fig, use_container_width=True)
    with dist_col2:
        resolution_dist_fig = px.histogram(
            timeline_df.dropna(subset=["resolution_duration_hours"]),
            x="resolution_duration_hours",
            nbins=30,
            title="Resolution Duration Distribution",
            labels={"resolution_duration_hours": "Resolution Duration (hrs)"},
        )
        st.plotly_chart(resolution_dist_fig, use_container_width=True)

    type_fig = px.bar(
        by_event_type.head(10),
        x="event_type",
        y="incident_duration_hours",
        title="Average Duration by Event Type",
        labels={"event_type": "Event Type", "incident_duration_hours": "Average Duration (hrs)"},
    )
    st.plotly_chart(type_fig, use_container_width=True)

    zone_fig = px.bar(
        by_zone.head(10),
        x="zone",
        y="incident_duration_hours",
        title="Average Duration by Zone",
        labels={"zone": "Zone", "incident_duration_hours": "Average Duration (hrs)"},
    )
    st.plotly_chart(zone_fig, use_container_width=True)

    insight_zone = by_zone.iloc[0]["zone"] if not by_zone.empty else "N/A"
    insight_type = by_event_type.iloc[0]["event_type"] if not by_event_type.empty else "N/A"
    st.markdown("#### Operational Insights")
    st.write(f"- {insight_zone} shows the longest average incident duration, so it may need faster response coverage.")
    st.write(f"- {insight_type} incidents show the longest average durations and may require pre-positioned resources.")


def render_hotspot_map(incident_df: pd.DataFrame):
    st.subheader("Traffic Incident Hotspots")
    map_df = incident_df.dropna(subset=["latitude", "longitude"]).copy()
    if map_df.empty:
        st.warning("No valid incident coordinates available for mapping.")
        return

    map_df["priority"] = map_df["priority"].fillna("Unknown")
    map_df["event_type"] = map_df["event_type"].fillna("Unknown")

    map_fig = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="priority" if "priority" in map_df.columns else "event_type",
        hover_name="event_type",
        hover_data=["zone", "junction", "event_cause", "police_station", "status"],
        zoom=10,
        height=700,
        color_discrete_sequence=px.colors.qualitative.Safe,
        title="Traffic Incident Hotspots",
    )
    map_fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    st.plotly_chart(map_fig, use_container_width=True)

    hotspot_counts = (
        map_df.groupby(["zone", "junction"], dropna=False)
        .size()
        .reset_index(name="incident_count")
        .sort_values("incident_count", ascending=False)
        .head(10)
    )
    st.dataframe(hotspot_counts, use_container_width=True)


def render_memory_engine(incident_df: pd.DataFrame, current_event_type: str, current_location_type: str):
    st.subheader("Traffic Memory Engine")
    st.write(
        "Current systems treat every event as new.\n\n"
        "TrafficMind remembers previous events and retrieves similar historical scenarios to support planning decisions."
    )

    memory_df = incident_df.copy()
    memory_df["event_type"] = memory_df["event_type"].fillna("Unknown")
    memory_df["zone"] = memory_df["zone"].fillna("Unknown")
    memory_df["junction"] = memory_df["junction"].fillna("Unknown")
    memory_df["event_cause"] = memory_df["event_cause"].fillna("Unknown")
    memory_df["priority"] = memory_df["priority"].fillna("Unknown")

    lookup_df = memory_df[["event_type", "event_cause", "zone", "junction", "priority", "requires_road_closure"]].copy()
    lookup_df["priority"] = lookup_df["priority"].astype(str)

    location_proxy = {
        "Highway": "corridor",
        "Rural": "peripheral",
        "Suburban": "suburban",
        "Urban": "urban",
    }.get(current_location_type, "urban")
    lookup_df["location_proxy"] = np.where(
        lookup_df["zone"].str.contains("zone", case=False, na=False),
        "urban",
        "corridor",
    )
    similarity_features = pd.DataFrame(
        {
            "event_type": lookup_df["event_type"],
            "location_proxy": lookup_df["location_proxy"],
        }
    )

    encoder = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["event_type", "location_proxy"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    nn = NearestNeighbors(n_neighbors=5, metric="cosine")
    pipeline = Pipeline([("preprocessor", encoder), ("neighbors", nn)])
    X = pipeline.named_steps["preprocessor"].fit_transform(similarity_features)
    pipeline.named_steps["neighbors"].fit(X)

    query = pd.DataFrame([{"event_type": current_event_type, "location_proxy": location_proxy}])
    q = pipeline.named_steps["preprocessor"].transform(query)
    distances, indices = pipeline.named_steps["neighbors"].kneighbors(q, n_neighbors=5)

    result = lookup_df.iloc[indices[0]].copy().reset_index(drop=True)
    result["Similarity %"] = (1 - distances[0]) * 100
    result["Similarity %"] = result["Similarity %"].clip(lower=0).round(2)

    st.dataframe(
        result.rename(
            columns={
                "event_type": "Event Type",
                "event_cause": "Event Cause",
                "zone": "Zone",
                "junction": "Junction",
                "priority": "Priority",
                "requires_road_closure": "Road Closure Required",
            }
        )[["Event Type", "Event Cause", "Zone", "Junction", "Priority", "Road Closure Required", "Similarity %"]],
        use_container_width=True,
    )


def render_resolution_insights(incident_df: pd.DataFrame):
    st.subheader("Incident Resolution Insights")
    df = incident_df.copy()
    if "start_datetime" not in df.columns or "resolved_datetime" not in df.columns:
        st.info("Datetime fields not available for resolution analysis.")
        return

    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df["resolved_datetime"] = pd.to_datetime(df["resolved_datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["start_datetime", "resolved_datetime"])
    if df.empty:
        st.warning("No resolvable incidents found.")
        return

    df["resolution_time_hours"] = (df["resolved_datetime"] - df["start_datetime"]).dt.total_seconds() / 3600.0
    df = df[df["resolution_time_hours"].notna() & (df["resolution_time_hours"] >= 0)]
    if df.empty:
        st.warning("No valid resolution durations available.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Average Resolution Time", f"{df['resolution_time_hours'].mean():.2f} hrs")
    fastest = df.loc[df["resolution_time_hours"].idxmin()]
    slowest = df.loc[df["resolution_time_hours"].idxmax()]
    c2.metric("Fastest Resolved", f"{fastest['resolution_time_hours']:.2f} hrs")
    c3.metric("Slowest Resolved", f"{slowest['resolution_time_hours']:.2f} hrs")

    by_type = (
        df.groupby("event_type", dropna=False)["resolution_time_hours"]
        .mean()
        .reset_index()
        .sort_values("resolution_time_hours", ascending=False)
    )
    fig = px.bar(
        by_type.head(10),
        x="event_type",
        y="resolution_time_hours",
        title="Resolution Time by Event Type",
        labels={"event_type": "Event Type", "resolution_time_hours": "Average Resolution Time (hrs)"},
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    set_page_config()
    apply_styles()
    render_header()

    try:
        model, preprocessor, label_encoder, feature_columns = load_prediction_artifacts()
    except Exception as exc:
        st.error("Unable to load model artifacts.")
        st.exception(exc)
        st.stop()

    incident_df = normalize_incident_dataframe(load_incident_data())

    tab_event, tab_risk, tab_deploy, tab_memory, tab_intel = st.tabs(
        [
            "Event Details",
            "Traffic Risk Assessment",
            "Resource Deployment Simulator",
            "Traffic Memory Engine",
            "Incident Intelligence",
        ]
    )

    with tab_event:
        planning_inputs = render_event_details_tab()

    user_inputs = map_planning_to_model_inputs(planning_inputs)

    if "officers" not in st.session_state:
        st.session_state.officers = 0
    if "barricades" not in st.session_state:
        st.session_state.barricades = 0
    if "diversions" not in st.session_state:
        st.session_state.diversions = 0

    intervention_values = {
        "officers": int(st.session_state.officers),
        "barricades": int(st.session_state.barricades),
        "diversions": int(st.session_state.diversions),
    }

    with tab_risk:
        render_prediction_tab(model, preprocessor, label_encoder, feature_columns, user_inputs, intervention_values)

    with tab_deploy:
        st.subheader("Resource Deployment Simulator")
        st.info("Use these sliders to instantly update the live risk gauge in the Traffic Risk Assessment tab.")
        st.session_state.officers = st.slider("officers", min_value=0, max_value=20, value=st.session_state.officers, step=1)
        st.session_state.barricades = st.slider("barricades", min_value=0, max_value=10, value=st.session_state.barricades, step=1)
        st.session_state.diversions = st.slider("diversions", min_value=0, max_value=5, value=st.session_state.diversions, step=1)
        st.write("The intervention simulation now drives the live risk score above.")

    with tab_memory:
        render_memory_engine(incident_df, planning_inputs["event_type"], user_inputs["location_type"])

    with tab_intel:
        render_incident_intelligence(incident_df)
        render_zone_intelligence(incident_df)
        render_junction_risk_ranking(incident_df)
        render_road_closure_probability(incident_df)
        render_historical_deployment_benchmark(incident_df)
        render_incident_timeline_analysis(incident_df)
        render_resolution_insights(incident_df)
        render_hotspot_map(incident_df)


if __name__ == "__main__":
    main()
