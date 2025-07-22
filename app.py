import streamlit as st
from pages import analyzeDataset, analyzeAd, analyzeDemographic

# --- Page Configuration (Unified for the entire app) ---
st.set_page_config(
    page_title="BiasGuard: Ethical AI Guardrail for Marketing Automation",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Main Application Entry Point ---
def main_app():
    st.title("🛡️ BiasGuard: Ethical AI Guardrail for Marketing Automation")
    st.markdown("""
    _Ensuring Trust, Security, and Fairness in Your Automated Marketing Campaigns._
    """)

    st.markdown("---")

    st.sidebar.header("BiasGuard Modules")
    page_selection = st.sidebar.radio(
        "Select a module to analyze:",
        ("Target Audience/Dataset Bias", "Image Advertisement Analysis", "Ad Demographic Analyzer")
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "BiasGuard helps marketing teams identify and mitigate biases, "
        "and ensure security compliance in their AI-generated content and audience targeting."
    )

    if page_selection == "Target Audience/Dataset Bias":
        analyzeDataset.run_app()
    elif page_selection == "Image Advertisement Analysis":
        analyzeAd.run_app()
    elif page_selection == "Ad Demographic Analyzer":
        analyzeDemographic.run_app()

if __name__ == "__main__":
    main_app()