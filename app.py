import streamlit as st
import joblib
import pandas as pd
from datetime import date

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bike Sharing Demand Forecast",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px;}
        .app-header {
            background: linear-gradient(90deg, #0f2a43 0%, #1d4e79 100%);
            padding: 1.6rem 2rem; border-radius: 10px; color: #ffffff; margin-bottom: 1.5rem;
        }
        .app-header h1 {margin: 0; font-size: 1.9rem; font-weight: 700; color: #ffffff;}
        .app-header p {margin: .35rem 0 0 0; font-size: .98rem; color: #d6e4f0;}
        .section-title {
            font-size: 1.05rem; font-weight: 600; color: #1d4e79;
            border-bottom: 2px solid #1d4e79; padding-bottom: .3rem; margin: 1rem 0 .8rem 0;
        }
        .result-card {
            background: #f4f8fc; border-left: 6px solid #1d4e79; border-radius: 8px;
            padding: 1.4rem 1.6rem; margin-top: .5rem;
        }
        .result-card .label {font-size: .85rem; color: #4a5b6b; text-transform: uppercase; letter-spacing: .08em;}
        .result-card .value {font-size: 2.8rem; font-weight: 700; color: #0f2a43; line-height: 1.1;}
        .result-card .unit {font-size: .95rem; color: #4a5b6b;}
        .footer {text-align: center; color: #7a8896; font-size: .8rem; margin-top: 2.5rem;}
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            background-color: #1d4e79; color: #ffffff; border: none; border-radius: 6px;
            font-weight: 600; padding: .6rem 1.2rem;
        }
        div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
            background-color: #0f2a43; color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Load model artefacts (cached)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = joblib.load("best_xgb_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns


try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Required model file not found: {e.filename}. Place best_xgb_model.pkl, "
        "scaler.pkl and feature_columns.pkl in the same folder as app.py."
    )
    st.stop()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
NUMERICAL_COLS = ["temp", "atemp", "hum", "windspeed"]
WEEKDAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def get_day_period(hour: int) -> str:
    """Same bins used in training: [-1,5,11,16,20,23]."""
    if hour <= 5:
        return "Night"
    if hour <= 11:
        return "Morning"
    if hour <= 16:
        return "Afternoon"
    if hour <= 20:
        return "Evening"
    return "Late Night"


def build_input(selected_date, hour, season, weather, holiday, workingday,
                temp, atemp, hum, windspeed) -> pd.DataFrame:
    # Dataset convention: Sunday = 0 ... Saturday = 6
    weekday = (selected_date.weekday() + 1) % 7
    period = get_day_period(hour)

    row = {col: 0 for col in feature_columns}
    row.update({
        "yr": selected_date.year,
        # "yr": selected_date.year - 2011,
        "mnth": selected_date.month,
        "hr": hour,
        "holiday": 1 if holiday == "Yes" else 0,
        "workingday": 1 if workingday == "Yes" else 0,
        "temp": temp,
        "atemp": atemp,
        "hum": hum,
        "windspeed": windspeed,
        "day": selected_date.day,
        "dayofyear": selected_date.timetuple().tm_yday,
    })

    # One-hot encoding (drop_first=True in training -> reference categories stay 0)
    for key in (f"season_{season}", f"day_period_{period}",
                f"weathersit_{weather}", f"weekday_{weekday}"):
        if key in row:
            row[key] = 1

    df = pd.DataFrame([row], columns=feature_columns)
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])
    return df


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## About")
    st.write(
        "This application estimates hourly bike rental demand for a bike sharing "
        "system using a trained XGBoost regression model."
    )
    st.markdown("### Model Summary")
    st.markdown(
        "- **Algorithm:** XGBoost Regressor\n"
        f"- **Input features:** {len(feature_columns)}\n"
        "- **Target:** Total rentals per hour"
    )
    st.markdown("### How to Use")
    st.markdown(
        "1. Select the date and hour.\n"
        "2. Specify the calendar and weather conditions.\n"
        "3. Click **Generate Forecast**."
    )
    st.caption("Note: weather values are normalized (0–1), as in the source dataset.")

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <h1>🚲 Bike Sharing Demand Forecast</h1>
        <p>Predictive analytics for hourly bike rental demand</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Input form
# ----------------------------------------------------------------------------
with st.form("prediction_form"):
    st.markdown('<div class="section-title">Date &amp; Time</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        selected_date = st.date_input(
            "Date",
            value=date(2012, 1, 1),
            min_value=date(2011, 1, 1),
            max_value=date(2012, 12, 31),
            help="The model was trained on data from 2011 and 2012.",
        )
    with c2:
        hour = st.slider("Hour of Day (24h)", 0, 23, 12)

    st.markdown('<div class="section-title">Calendar &amp; Conditions</div>', unsafe_allow_html=True)
    c3, c4, c5, c6 = st.columns(4)
    with c3:
        season = st.selectbox("Season", ["spring", "summer", "fall", "winter"],
                              format_func=str.capitalize)
    with c4:
        weather = st.selectbox("Weather Condition", ["Clear", "Mist", "Light Snow", "Heavy Rain"])
    with c5:
        holiday = st.selectbox("Public Holiday", ["No", "Yes"])
    with c6:
        workingday = st.selectbox("Working Day", ["Yes", "No"])

    st.markdown('<div class="section-title">Weather Measurements (Normalized)</div>',
                unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        temp = st.slider("Temperature", 0.0, 1.0, 0.5, 0.01)
        hum = st.slider("Humidity", 0.0, 1.0, 0.5, 0.01)
    with c8:
        atemp = st.slider("Feels-Like Temperature", 0.0, 1.0, 0.5, 0.01)
        windspeed = st.slider("Wind Speed", 0.0, 1.0, 0.2, 0.01)

    submit = st.form_submit_button("Generate Forecast", use_container_width=True)

# ----------------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------------
if submit:
    input_df = build_input(selected_date, hour, season, weather, holiday,
                           workingday, temp, atemp, hum, windspeed)
    prediction = max(0, round(float(model.predict(input_df)[0])))

    st.markdown('<div class="section-title">Forecast Result</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Estimated Rentals</div>
                <div class="value">{prediction:,}</div>
                <div class="unit">bikes in the selected hour</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        weekday_name = WEEKDAY_NAMES[(selected_date.weekday() + 1) % 7]
        summary = pd.DataFrame(
            {
                "Parameter": ["Date", "Day of Week", "Hour", "Time Period", "Season",
                              "Weather", "Holiday", "Working Day"],
                "Value": [
                    selected_date.strftime("%d %b %Y"), weekday_name, f"{hour:02d}:00",
                    get_day_period(hour), season.capitalize(), weather, holiday, workingday,
                ],
            }
        )
        st.dataframe(summary, hide_index=True, use_container_width=True)

    with st.expander("View processed model input"):
        st.dataframe(input_df, use_container_width=True)

st.markdown(
    '<div class="footer">Bike Sharing Demand Forecast · Powered by XGBoost &amp; Streamlit</div>',
    unsafe_allow_html=True,
)