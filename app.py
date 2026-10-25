import streamlit as st
import pandas as pd
from patent_lense import load_data, compare_years, score_phrases, validate

# ---- Page Config ----
st.set_page_config(page_title="Patent Lens", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

# ---- Custom CSS for Premium Look ----
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Gradient Title */
    .gradient-text {
        background: -webkit-linear-gradient(45deg, #1E3A8A, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .sub-text {
        font-size: 18px;
        color: #64748B;
        margin-top: 0px;
        margin-bottom: 30px;
    }

    /* Metrics Cards */
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-top: 4px solid #3B82F6;
        margin-bottom: 20px;
    }
    .metric-card h3 { margin: 0; color: #1E293B; font-size: 16px;}
    .metric-card h1 { margin: 10px 0 0 0; color: #3B82F6; font-size: 32px;}
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown('<p class="gradient-text">🔍 Patent Lens</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Discover which technologies are growing or dying based on patent data.</p>', unsafe_allow_html=True)

# ---- 1. File Source (Sidebar) ----
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2941/2941916.png", width=60)
    st.header("Data Source")
    
    uploaded = st.file_uploader("Upload a patent CSV", type="csv")
    
    if uploaded is not None:
        df = load_data(uploaded)
        st.success("✅ File uploaded successfully!")
    else:
        st.info("ℹ️ Using default dataset (Indian Patents).")
        df = load_data("archive/patent.csv")

    # Show validation warnings
    warnings = validate(df)
    if warnings:
        st.divider()
        st.subheader("⚠️ Data Warnings")
        for w in warnings:
            st.warning(w)

# ---- 2. Controls ----
st.subheader("⚙️ Analysis Settings")
years = sorted(df["year"].unique())

# Only show controls if there are enough years
if len(years) < 2:
    st.error("The uploaded dataset does not have at least two different years. Cannot analyze trends.")
else:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        base_year = st.selectbox("Base Year", years, index=0)
    with col2:
        current_year = st.selectbox("Current Year", years, index=len(years) - 1)
    with col3:
        st.write("Options")
        use_trigrams = st.checkbox("Include 3-word phrases (Trigrams)", value=True)

    st.write("") # Spacer

    # ---- 3. Analyze Button ----
    if st.button("🚀 Analyze Trends", use_container_width=True):
        if base_year == current_year:
            st.error("Base year and current year must be different.")
        else:
            with st.spinner("Analyzing N-grams and calculating Bayesian scores..."):
                result = compare_years(df, base_year, current_year, min_count=1, use_trigrams=use_trigrams)
                result = score_phrases(result)

                if result.empty:
                    st.warning("No trends found. Try lowering the minimum count or checking the dataset.")
                else:
                    # Top Metrics
                    total_analyzed = len(result)
                    up_trends = len(result[result["trend"] == "Up"])
                    down_trends = len(result[result["trend"] == "Down"])
                    
                    st.write("")
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.markdown(f'<div class="metric-card"><h3>Total Phrases Analyzed</h3><h1>{total_analyzed}</h1></div>', unsafe_allow_html=True)
                    with m2:
                        st.markdown(f'<div class="metric-card" style="border-top-color: #10B981;"><h3>Growing Tech</h3><h1 style="color:#10B981;">{up_trends}</h1></div>', unsafe_allow_html=True)
                    with m3:
                        st.markdown(f'<div class="metric-card" style="border-top-color: #EF4444;"><h3>Declining Tech</h3><h1 style="color:#EF4444;">{down_trends}</h1></div>', unsafe_allow_html=True)

                    st.divider()

                    # Layout for charts
                    top_up = result[result["trend"] == "Up"].head(10)
                    top_down = result[result["trend"] == "Down"].head(10)
                    
                    col_left, col_right = st.columns(2)

                    with col_left:
                        st.subheader("📈 Top Growing")
                        if not top_up.empty:
                            st.dataframe(
                                top_up[["phrase", "base_count", "current_count", "growth_ratio", "score"]].style.format({"score": "{:.1f}"}),
                                hide_index=True,
                                use_container_width=True
                            )
                            st.bar_chart(top_up.set_index("phrase")["score"], color="#10B981")
                        else:
                            st.write("No growing phrases found.")

                    with col_right:
                        st.subheader("📉 Top Declining")
                        if not top_down.empty:
                            st.dataframe(
                                top_down[["phrase", "base_count", "current_count", "growth_ratio", "score"]].style.format({"score": "{:.1f}"}),
                                hide_index=True,
                                use_container_width=True
                            )
                            st.bar_chart(top_down.set_index("phrase")["score"], color="#EF4444")
                        else:
                            st.write("No declining phrases found.")

                    # Raw data expander
                    st.divider()
                    with st.expander("🔍 View All Analyzed Phrases"):
                        st.dataframe(result, use_container_width=True)