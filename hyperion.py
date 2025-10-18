import pandas as pd
import numpy as np
import streamlit as st
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import warnings
from urllib.parse import quote_plus
import io

#
warnings.filterwarnings("ignore")

NASA_TAP_BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


BASE_FEATURES = [
    'pl_orbper', 'pl_rade', 'pl_masse', 'pl_dens', 'pl_orbeccen',
    'pl_insol', 'pl_eqt', 'st_teff', 'st_mass', 'st_rad',
    'st_met', 'pl_tranmid', 'pl_trandur', 'pl_trandep'
]


COL_MAP = {
    'pl_orbper': 'koi_period', 'pl_rade': 'koi_prad', 'pl_masse': 'koi_smass',
    'pl_dens': 'koi_srho', 'pl_orbeccen': 'koi_eccen', 'pl_insol': 'koi_insol',
    'pl_eqt': 'koi_teq', 'st_teff': 'koi_steff', 'st_mass': 'koi_smass',
    'st_rad': 'koi_srad', 'st_met': 'koi_smass', 
    'pl_tranmid': 'koi_time0bk', 'pl_trandur': 'koi_duration', 'pl_trandep': 'koi_depth'
}

# --- NEW: Map technical variable names to user-friendly display names ---
DISPLAY_MAP = {
    'pl_orbper': 'Orbital Period (days)',
    'pl_rade': 'Planet Radius (R⊕)',
    'pl_masse': 'Planet Mass (M⊕)',
    'pl_dens': 'Planet Density (g/cm³)',
    'pl_orbeccen': 'Orbital Eccentricity',
    'pl_insol': 'Insolation (S⊕)',
    'pl_eqt': 'Equilibrium Temp (K)',
    'st_teff': 'Star Temp (K)',
    'st_mass': 'Star Mass (M☉)',
    'st_rad': 'Star Radius (R☉)',
    'pl_trandur': 'Transit Duration (hrs)',
    'pl_trandep': 'Transit Depth (Frac)',
    'pl_tranmid': 'Transit Midpoint (JD)',
    'st_met': 'Star Metallicity',
    'pl_rs_ratio': 'Radius/Stellar Ratio', 
    'fp_flag': 'Hard FP Flag' 
}


def check_habitable_zone(st_teff, pl_insol):
    """Calculates Habitable Zone boundaries (Kopparapu et al., 2013 simplified)
        and checks if the planet's insulation is within the conservative HZ."""
    S_eff_inner = 1.107 
    S_eff_outer = 0.356 
    
    T_star = st_teff
    if T_star < 4000:
        return "Unknown (Cool Star)", False
        
    T_corr = (T_star - 5780) / 5780
    a = [1.7779, 1.2588, 0.5350, 0.1706] 
    b = [2.1580, 1.4140, 0.6030, 0.2224] 

    S_inner = S_eff_inner * (1 + a[0]*T_corr + a[1]*T_corr**2 + a[2]*T_corr**3 + a[3]*T_corr**4)
    S_outer = S_eff_outer * (1 + b[0]*T_corr + b[1]*T_corr**2 + b[2]*T_corr**3 + b[3]*T_corr**4)
    
    is_in_hz = (pl_insol >= S_outer) and (pl_insol <= S_inner)
    
    if is_in_hz:
        status = f"✅ In HZ ({S_outer:.2f} < S < {S_inner:.2f})"
    elif pl_insol > S_inner:
        status = f"🔥 Too Hot (Inner Boundary: {S_inner:.2f})"
    else:
        status = f"❄️ Too Cold (Outer Boundary: {S_outer:.2f})"
        
    return status, is_in_hz


def generate_training_data(n_samples=500, label=1):
    np.random.seed(42)
    
    data = pd.DataFrame({
        'pl_orbper': np.random.uniform(1.0, 100.0, n_samples),
        'pl_rade': np.random.uniform(0.5, 10.0, n_samples),
        'pl_masse': np.random.uniform(1.0, 300.0, n_samples),
        'pl_dens': np.random.uniform(1.0, 10.0, n_samples),
        'pl_orbeccen': np.random.uniform(0.0, 0.3, n_samples),
        'pl_insol': np.random.uniform(0.1, 10.0, n_samples),
        'pl_eqt': np.random.uniform(200.0, 1000.0, n_samples),
        'st_teff': np.random.uniform(4000.0, 6500.0, n_samples),
        'st_mass': np.random.uniform(0.5, 1.5, n_samples),
        'st_rad': np.random.uniform(0.5, 1.5, n_samples),
        'st_met': np.random.uniform(-0.5, 0.5, n_samples),
        'pl_tranmid': np.random.uniform(2455000.0, 2456000.0, n_samples),
        'pl_trandur': np.random.uniform(0.1, 10.0, n_samples),
        'pl_trandep': np.random.uniform(0.001, 0.05, n_samples)
    })
    data['label'] = label
    return data[BASE_FEATURES + ['label']]


def generate_synthetic_true_negatives(n_stn_samples=2000):
    """Generates synthetic True Negative (FP) data to aggressively train against Eclipsing Binaries."""
    np.random.seed(43) 
    stn_df = pd.DataFrame(index=range(n_stn_samples), columns=BASE_FEATURES)
    
    
    stn_df['pl_rade'] = np.random.uniform(15.0, 50.0, n_stn_samples) 
   
    stn_df['pl_trandep'] = np.random.uniform(0.05, 0.50, n_stn_samples) 
    
    stn_df['pl_orbper'] = np.random.uniform(0.5, 5.0, n_stn_samples) 

    
    median_values = {
        'pl_dens': 5.51, 'pl_orbeccen': 0.016, 'pl_insol': 1.0, 
        'pl_eqt': 288.0, 'st_teff': 5778.0, 'st_mass': 1.0, 
        'st_rad': 1.0, 'st_met': 0.0, 'pl_masse': 1000.0, 
        'pl_trandur': 3.0, 'pl_tranmid': 2455000.0
    }
    
    for col in BASE_FEATURES:
        if stn_df[col].isnull().any():
            fill_val = median_values.get(col, 1.0)
            random_fill_series = pd.Series(
                np.random.normal(fill_val, abs(fill_val) * 0.2 + 0.1, n_stn_samples),
                index=stn_df.index
            )
            stn_df[col] = stn_df[col].combine_first(random_fill_series)

    
    stn_df['st_rad'] = np.random.uniform(0.8, 1.5, n_stn_samples)
    stn_df['pl_masse'] = np.random.uniform(500.0, 10000.0, n_stn_samples) 
    
    stn_df['label'] = 0
    return stn_df[BASE_FEATURES + ['label']].dropna(subset=BASE_FEATURES)


@st.cache_data(ttl=24*3600)
def load_nasa_data(name_col='pl_name'):
    st.info("📡 Fetching data from NASA Exoplanet Archive...")

    def fetch_data(query, name_col_api, label_val, status_msg):
        encoded_query = quote_plus(query)
        url = f"{NASA_TAP_BASE_URL}?query={encoded_query}&format=csv"
        try:
            response = requests.get(url, timeout=45) # Increased timeout
            if response.status_code == 200:
                df = pd.read_csv(io.StringIO(response.text), na_values=['', ' '])
                if df.empty:
                    st.warning(f"NASA API returned empty data for {status_msg}.")
                    return None
                
                # Standardize columns
                df = df.rename(columns={'pl_mass': 'pl_masse', name_col_api: 'pl_name'})
                
                # Apply column mapping for Kepler data
                if 'koi_period' in df.columns:
                    inverse_map = {v: k for k, v in COL_MAP.items()}
                    df = df.rename(columns={k: v for k, v in inverse_map.items() if k in df.columns})
                    
                df = df[df.columns.intersection(['pl_name'] + BASE_FEATURES)]
                df['label'] = label_val
                st.success(f"✅ Loaded {len(df)} {status_msg}.")
                return df
            else:
                st.warning(f"NASA API returned status {response.status_code} for {status_msg}.")
                return None
        except Exception as e:
            st.warning(f"Could not fetch data due to exception: {e}")
            return None

    
    ps_cols = ','.join([name_col] + BASE_FEATURES)
    query_confirmed = f"SELECT {ps_cols} FROM ps WHERE tran_flag=1 AND default_flag=1"
    df_confirmed = fetch_data(query_confirmed, name_col, 1, "Confirmed Planets")
    
    if df_confirmed is None or df_confirmed.empty:
        df_confirmed = generate_training_data(n_samples=500, label=1)
        df_confirmed['pl_name'] = 'Fallback_Conf_' + df_confirmed.index.astype(str)
        st.warning("Using synthetic Confirmed data as a fallback.")

    
    koi_cols_used = [COL_MAP[c] for c in BASE_FEATURES]
    query_fp = f"SELECT kepoi_name, {','.join(koi_cols_used)} FROM cumulative WHERE koi_disposition='FALSE POSITIVE'"
    df_fp = fetch_data(query_fp, 'kepoi_name', 0, "Kepler False Positives")
    
    query_candidate = f"SELECT kepoi_name, {','.join(koi_cols_used)} FROM cumulative WHERE koi_disposition='CANDIDATE'"
    df_candidate = fetch_data(query_candidate, 'kepoi_name', 0, "Kepler Candidates")
    
    if df_fp is None and df_candidate is None:
        df_koi = generate_training_data(n_samples=1000, label=0)
        df_koi['pl_name'] = 'Fallback_Cand_' + df_koi.index.astype(str)
        st.warning("Using synthetic Candidate data as a fallback.")
    elif df_fp is not None and df_candidate is not None:
          df_koi = pd.concat([df_fp, df_candidate], ignore_index=True)
    elif df_fp is not None:
          df_koi = df_fp
    else:
          df_koi = df_candidate
        
    
    df_stn = generate_synthetic_true_negatives(n_stn_samples=2000)
    df_stn['pl_name'] = 'STN_' + df_stn.index.astype(str)
    st.info(f"✨ Injected {len(df_stn)} Synthetic True Negative examples to reduce bias.")

    
    all_data = pd.concat([df_confirmed, df_koi, df_stn], ignore_index=True, sort=False)
    common_cols = list(set(BASE_FEATURES) & set(all_data.columns))
    
    X_full = all_data[[c for c in ['pl_name'] + common_cols + ['label'] if c in all_data.columns]].copy()
    X_full = X_full.replace([np.inf, -np.inf], np.nan)
    
    for col in common_cols:
          X_full[col] = X_full[col].fillna(X_full[col].median())
    
    
    X_full['pl_rs_ratio'] = X_full['pl_rade'] / (X_full['st_rad'] * 109.2) 
    
    
    extreme_radius_threshold = 15.0 # R_earth
    deep_depth_threshold = 0.05       # 5% depth (Fractional)
    short_period_threshold = 10.0    # days
    
    X_full['fp_flag'] = 0
    X_full.loc[(X_full['pl_rade'] > extreme_radius_threshold) & 
               (X_full['pl_trandep'] > deep_depth_threshold) &
               (X_full['pl_orbper'] < short_period_threshold), 'fp_flag'] = 1
    
    X_full = X_full.dropna()
    
    
    X_train_df = X_full.drop('pl_name', axis=1)
    
    y = X_train_df['label'].values
    X = X_train_df.drop('label', axis=1)
    
    return X, y, X_full.drop('label', axis=1).rename(columns={'pl_name': 'Name'}) 


def classify_planet(radius):
    if radius < 1.25: return " Rocky Planet (Earth-like)"
    elif radius < 2.0: return " Super-Earth"
    elif radius < 4.0: return " Mini-Neptune"
    elif radius < 10.0: return " Neptune-like"
    else: return " Gas Giant/Stellar Impostor"

def calculate_habitability(radius, temp):
    EARTH_TEMP = 288
    EARTH_RADIUS = 1.0
    
    radius = radius if radius is not None else EARTH_RADIUS
    temp = temp if temp is not None else EARTH_TEMP
    
    
    esi_radius = 1 - abs((radius - EARTH_RADIUS) / (radius + EARTH_RADIUS))
    esi_temp = 1 - abs((temp - EARTH_TEMP) / (temp + EARTH_TEMP))
    esi_radius = max(0, esi_radius)
    esi_temp = max(0, esi_temp)
    esi = ((esi_radius ** 0.57) * (esi_temp ** 0.70)) ** 0.5
    
    return min(esi * 100, 100), esi


st.set_page_config(page_title="Hyperion Exoplanet AI", page_icon="🌌", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 25%, #0f0c29 50%, #1a1a3e 75%, #0a0e27 100%);
        background-attachment: fixed;
        color: white;
    }
    h1, h2, h3, p, div, label { color: white !important; }
    .stButton>button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stSpinner > div > div { color: #667eea !important; }
    .stMetric > div:first-child { color: #ccc; }
    </style>
    """, unsafe_allow_html=True
)

st.title("🌌 Hyperion - Advanced Exoplanet Prediction and Analysis using AI")
st.caption("Exoplanet Prediction and Scientific Analysis Platform for researchers and common people : NASA Space Apps Challenge 2025")


st.markdown("---")
with st.container(border=True):
    st.subheader(" What is Hyperion?")
    st.markdown("""
        Hyperion is an AI-powered exoplanet research and citizen science platform. It combines ensemble machine learning models (Logistic Regression + Random Forest) to classify exoplanet candidates and assess their likelihood of being a confirmed exoplanet.

It enables users to:

Upload or input parameters (e.g., orbital period, transit depth, radius, stellar temperature) and get real-time classification.

Search and compare with the NASA Exoplanet Archive of confirmed planets.

Perform scientific analysis using ensemble predictions, confusion matrix, and uncertainty quantification.

Visualize and interpret planetary properties using interactive Plotly charts.
    """)
st.markdown("---")

with st.spinner('🔭 Loading, cleaning, synthesizing data, and engineering features...'):
    X, y, X_database = load_nasa_data()
    
FINAL_FEATURES = X.columns.tolist()

# ML Pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

with st.spinner('🤖 Training Ensemble Models...'):
    logreg = LogisticRegression(max_iter=5000, C=0.5, random_state=42, class_weight='balanced') 
    logreg.fit(X_train_scaled, y_train)
    
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42, class_weight='balanced') 
    rf.fit(X_train_scaled, y_train)
    
    gb = GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42)
    gb.fit(X_train_scaled, y_train)
    
    

st.success("✅ Models ready and trained with synthetic FP suppression!")


def run_analysis(input_data_df, name="Candidate"):
    input_scaled = scaler.transform(input_data_df[FINAL_FEATURES])

    prob_lr = logreg.predict_proba(input_scaled)[0, 1]
    prob_rf = rf.predict_proba(input_scaled)[0, 1]
    prob_gb = gb.predict_proba(input_scaled)[0, 1]
    
    ensemble_prob = (prob_lr + prob_rf + prob_gb) / 3
    ensemble_pred = 1 if ensemble_prob >= 0.5 else 0

    return {
        'ensemble_prob': ensemble_prob,
        'ensemble_pred': ensemble_pred,
        'probs': {'LR': prob_lr, 'RF': prob_rf, 'GB': prob_gb},
        'scaled_data': input_scaled,
        'original_data': input_data_df.iloc[0].to_dict()
    }


st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([" New Candidate Analysis", " Database Search", " Model Stats", " Data Visuals"])


with tab1:
    st.header(" New Candidate Parameters")
    
    user_inputs = {}
    
    
    col_planet, col_star_transit = st.columns(2)
    
    with col_planet:
        st.markdown("#### Planet & Orbit")
        col1a, col1b = st.columns(2)
        with col1a:
           
            user_inputs['pl_orbper'] = st.number_input(DISPLAY_MAP['pl_orbper'], 0.01, value=365.25, key="t1_orbper")
            user_inputs['pl_rade'] = st.number_input(DISPLAY_MAP['pl_rade'], 0.01, value=1.0, key="t1_rade")
            user_inputs['pl_masse'] = st.number_input(DISPLAY_MAP['pl_masse'], 0.01, value=1.0, key="t1_masse")
        with col1b:
            
            user_inputs['pl_dens'] = st.number_input(DISPLAY_MAP['pl_dens'], 0.01, value=5.51, key="t1_dens")
            user_inputs['pl_orbeccen'] = st.number_input(DISPLAY_MAP['pl_orbeccen'], 0.0, 0.99, value=0.016, key="t1_eccen")
            user_inputs['pl_eqt'] = st.number_input(DISPLAY_MAP['pl_eqt'], 50.0, value=288.0, key="t1_eqt")

    with col_star_transit:
        st.markdown("#### Star & Transit")
        col2a, col2b = st.columns(2)
        with col2a:
          
            user_inputs['st_teff'] = st.number_input(DISPLAY_MAP['st_teff'], 2000.0, value=5778.0, key="t1_teff")
            user_inputs['st_mass'] = st.number_input(DISPLAY_MAP['st_mass'], 0.1, value=1.0, key="t1_mass")
            user_inputs['st_rad'] = st.number_input(DISPLAY_MAP['st_rad'], 0.1, value=1.0, key="t1_rad")
        with col2b:
           
            user_inputs['pl_trandep'] = st.number_input(DISPLAY_MAP['pl_trandep'], 0.0001, value=0.01, format="%.4f", key="t1_dep", help="The fractional drop in starlight (e.g., 0.01 for a 1% drop).")
            user_inputs['pl_trandur'] = st.number_input(DISPLAY_MAP['pl_trandur'], 0.1, value=3.0, key="t1_dur")
            user_inputs['pl_insol'] = st.number_input(DISPLAY_MAP['pl_insol'], 0.01, value=1.0, key="t1_insol")
            
            user_inputs['pl_tranmid'] = 2455000.0 
            user_inputs['st_met'] = 0.0 

    st.markdown("---")
    

    comparison_options = X_database['Name'].unique().tolist()
    comparison_planet_name = st.selectbox(
        "Compare against (Optional):",
        ['(None)'] + comparison_options,
        key="comp_select"
    )
    
    if st.button(" Analyze Candidate", use_container_width=True, type="primary"):
        
        
        rade_val = user_inputs.get('pl_rade', 1.0)
        orbper_val = user_inputs.get('pl_orbper', 365.25)
        st_rad_val = user_inputs.get('st_rad', 1.0)
        trandep_val = user_inputs.get('pl_trandep', 0.01)
        
        
        pl_rs_ratio_val = rade_val / (st_rad_val * 109.2)
        
        
        extreme_radius_threshold = 15.0 
        deep_depth_threshold = 0.05
        short_period_threshold = 10.0
        fp_flag_val = 1 if (rade_val > extreme_radius_threshold and 
                             trandep_val > deep_depth_threshold and 
                             orbper_val < short_period_threshold) else 0

        full_input_data = {'Name': 'USER_CANDIDATE'}
        for col in BASE_FEATURES:
            full_input_data[col] = user_inputs[col]
        full_input_data['pl_rs_ratio'] = pl_rs_ratio_val
        full_input_data['fp_flag'] = fp_flag_val

        manual_df = pd.DataFrame([full_input_data], columns=['Name'] + FINAL_FEATURES)
        
        
        analysis_manual = run_analysis(manual_df.drop('Name', axis=1), name='USER_CANDIDATE')
        
        st.subheader("Analysis Results")
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            if analysis_manual['ensemble_pred'] == 1:
                st.success(f"### 🎉 EXOPLANET CONFIRMED!")
            else:
                st.error(f"### ❌ NOT AN EXOPLANET")
        with pred_col2:
            st.metric("Ensemble Confidence", f"{analysis_manual['ensemble_prob']:.1%}")

        
        st.markdown("#### Input Parameters")
        
        display_df = pd.DataFrame([full_input_data]).drop(columns=['Name']).T.rename(columns={0: "Value"}).rename(DISPLAY_MAP)
        st.dataframe(display_df.applymap(lambda x: f'{x:.4f}' if isinstance(x, (float, np.float64)) else x), 
                     use_container_width=True, height=350)

       
        st.markdown("#### Habitability Check (ESI & HZ)")
        hab_score, esi = calculate_habitability(rade_val, user_inputs.get('pl_eqt'))
        hz_status, in_hz = check_habitable_zone(user_inputs.get('st_teff'), user_inputs.get('pl_insol'))
        
        hab_col1, hab_col2, hab_col3 = st.columns(3)
        with hab_col1: st.metric("Planet Type", classify_planet(rade_val))
        with hab_col2: st.metric("ESI Score", f"{esi:.3f}")
        with hab_col3: st.metric("Habitable Zone (ZOH)", hz_status)
        
        
        
        if comparison_planet_name != '(None)':
            st.markdown("---")
            st.header(f"↔️ Comparison: Your Candidate vs {comparison_planet_name}")
            
            comparison_df = X_database[X_database['Name'] == comparison_planet_name].copy().drop('Name', axis=1)
            comparison_df = comparison_df.reset_index(drop=True)
            
            analysis_comp = run_analysis(comparison_df, name=comparison_planet_name)
            
            
            comp_df = pd.DataFrame({
                'Feature': ['Detection', 'Radius (R⊕)', 'Period (days)', 'Transit Depth (Frac)', 'Habitable Zone'],
                'Your Candidate': [
                    "Confirmed" if analysis_manual['ensemble_pred'] else "Not Confirmed",
                    f"{rade_val:.2f}", 
                    f"{orbper_val:.2f}",
                    f"{trandep_val:.4f}",
                    hz_status
                ],
                comparison_planet_name: [
                    "Confirmed" if analysis_comp['ensemble_pred'] else "Not Confirmed",
                    f"{comparison_df['pl_rade'].iloc[0]:.2f}", 
                    f"{comparison_df['pl_orbper'].iloc[0]:.2f}",
                    f"{comparison_df['pl_trandep'].iloc[0]:.4f}",
                    check_habitable_zone(comparison_df['st_teff'].iloc[0], comparison_df['pl_insol'].iloc[0])[0]
                ]
            }).set_index('Feature')
            
            st.dataframe(comp_df, use_container_width=True)



with tab2:
    st.header("NASA Database Search")
    
    search_options = X_database['Name'].unique().tolist()
    selected_planet_name = st.selectbox(
        "Select a planet from the database to analyze:",
        ['Select a Planet'] + search_options
    )
    
    if selected_planet_name != 'Select a Planet':
        
        planet_df = X_database[X_database['Name'] == selected_planet_name].copy().drop('Name', axis=1)
        planet_df = planet_df.reset_index(drop=True)
        
        # Run Analysis
        analysis_db = run_analysis(planet_df, name=selected_planet_name)
        
        st.subheader(f"AI Analysis for {selected_planet_name}")
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            if analysis_db['ensemble_pred'] == 1:
                st.success(f"### EXOPLANET CONFIRMED!")
            else:
                st.error(f"### NOT AN EXOPLANET")
        with pred_col2:
            st.metric("Ensemble Confidence", f"{analysis_db['ensemble_prob']:.1%}")

        # Habitability/ZOH Check
        st.markdown("#### Habitability & Key Parameters")
        data = analysis_db['original_data']
        rade = data.get('pl_rade', 1.0)
        eqt = data.get('pl_eqt', 288.0)
        teff = data.get('st_teff', 5778.0)
        insol = data.get('pl_insol', 1.0)

        hab_score, esi = calculate_habitability(rade, eqt)
        hz_status, in_hz = check_habitable_zone(teff, insol)
        
        col_type, col_esi, col_hz = st.columns(3)
        with col_type: st.metric("Planet Type", classify_planet(rade))
        with col_esi: st.metric("ESI Score", f"{esi:.3f}")
        with col_hz: st.metric("Habitable Zone (ZOH)", hz_status)
        
        st.markdown("---")
        st.markdown("#### Full Data Table")
        
        st.dataframe(planet_df.T.rename(columns={0: "Value"}).rename(DISPLAY_MAP).applymap(lambda x: f'{x:.4f}' if isinstance(x, (float, np.float64)) else x), 
                     use_container_width=True)



with tab3:
    st.header("Model Performance Summary (on Test Set)")
    
    
    y_pred_logreg = logreg.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test_scaled)
    y_pred_gb = gb.predict(X_test_scaled)
    
    
    acc_logreg = accuracy_score(y_test, y_pred_logreg)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    acc_gb = accuracy_score(y_test, y_pred_gb)
    
    st.subheader("Individual Model Accuracies")
    col_ind1, col_ind2, col_ind3 = st.columns(3)
    with col_ind1:
        st.metric("Logistic Regression", f"{acc_logreg:.1%}")
    with col_ind2:
        st.metric("Random Forest", f"{acc_rf:.1%}")
    with col_ind3:
        st.metric("Gradient Boosting", f"{acc_gb:.1%}")

    st.markdown("---")
    
    
    y_pred_ensemble = ((y_pred_logreg + y_pred_rf + y_pred_gb) >= 2).astype(int)
    
    st.subheader("Ensemble Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_ensemble):.1%}")
    with col2:
        st.metric("Precision (FP Suppression)", f"{precision_score(y_test, y_pred_ensemble):.1%}")
    with col3:
        st.metric("Recall (Detection Rate)", f"{recall_score(y_test, y_pred_ensemble):.1%}")
    with col4:
        st.metric("F1 Score", f"{f1_score(y_test, y_pred_ensemble):.1%}")

    st.markdown("---")
    st.subheader("Confusion Matrix (Plotly)")
    
    cm = confusion_matrix(y_test, y_pred_ensemble)
    tn, fp, fn, tp = cm.ravel()
    
    
    z = [[tn, fp], [fn, tp]]
    x = ['Predicted Negative (0)', 'Predicted Positive (1)']
    y_labels = ['Actual Negative (0)', 'Actual Positive (1)'] 

    fig = ff.create_annotated_heatmap(z, x=x, y=y_labels, colorscale='Viridis')
    
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 14
        
    fig.update_layout(title='Ensemble Model Confusion Matrix',
                      xaxis=dict(title='Predicted Label'),
                      yaxis=dict(title='Actual Label', autorange="reversed"),
                      plot_bgcolor='rgba(0,0,0,0)', 
                      paper_bgcolor='rgba(0,0,0,0)', 
                      font_color="white")
    st.plotly_chart(fig, use_container_width=True)

    
    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        st.metric("True Positives (TP)", tp)
    with cm_col2:
        st.metric("True Negatives (TN)", tn)
        st.metric("False Positives (FP)", fp)
        st.metric("False Negatives (FN)", fn)
    
    st.markdown("---")
    
    st.subheader("Random Forest Feature Importance")
    importances = rf.feature_importances_
    # Map column names to display names
    feature_names = [DISPLAY_MAP.get(col, col) for col in X.columns]
    
    fig_mat, ax = plt.subplots(figsize=(10, 6))
    sorted_idx = np.argsort(importances)
    ax.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color='skyblue')
    ax.set_xlabel('Importance', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    fig_mat.patch.set_facecolor('none')
    ax.set_facecolor('none')
    st.pyplot(fig_mat)



with tab4:
    st.header("Data Visualizations")
    
    X_plot = X_database.copy()
    X_plot['Classification'] = 'Candidate/FP (y=0)'
    X_plot.loc[X_plot['Name'].str.contains('Conf_'), 'Classification'] = 'Confirmed (y=1)'
    X_plot.loc[X_plot['Name'].str.startswith('STN'), 'Classification'] = 'Synthetic Negatives (y=0)'

    st.subheader("Radius vs. Orbital Period (R-P Diagram)")
    fig_rp = px.scatter(X_plot, x='pl_orbper', y='pl_rade', color='Classification', 
                         log_x=True, size='pl_rade', 
                         hover_data=['Name', 'pl_eqt', 'st_teff'],
                         labels={'pl_orbper': 'Orbital Period (days)', 'pl_rade': 'Planet Radius (R⊕)'},
                         color_discrete_map={'Confirmed (y=1)': '#636EFA', 'Candidate/FP (y=0)': '#EF553B', 'Synthetic Negatives (y=0)': '#00CC96'})
    
    fig_rp.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_rp, use_container_width=True)

    st.subheader("Temperature vs. Insolation")
    fig_teq = px.scatter(X_plot, x='pl_eqt', y='pl_insol', color='Classification', 
                          log_y=True, 
                          hover_data=['Name', 'pl_rade', 'st_teff'],
                          labels={'pl_eqt': 'Equilibrium Temperature (K)', 'pl_insol': 'Insolation (S⊕)'},
                          color_discrete_map={'Confirmed (y=1)': '#636EFA', 'Candidate/FP (y=0)': '#EF553B', 'Synthetic Negatives (y=0)': '#00CC96'})
    
    
    fig_teq.add_hrect(y0=0.3, y1=1.5, line_width=0, fillcolor="green", opacity=0.1, annotation_text="Conceptual HZ (S~0.3 to 1.5)", annotation_position="top right")

    fig_teq.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_teq, use_container_width=True)
