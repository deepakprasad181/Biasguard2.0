import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import parse_csv, detect_sensitive_parameters, analyze_data_match, \
    create_age_chart, create_gender_chart, create_top_values_chart, \
    generate_insights, generate_text_report, generate_pdf_report_dataset # Renamed to avoid conflict

def run_app():
    """Runs the Target Audience/Dataset Bias module."""
    st.header("📊 Audience Segmentation & Targeting Analysis")
    
    tab1 = st.tabs(["Targeting Match Analysis"])[0]
    
    with tab1:
        st.markdown("""
        Upload candidate data CSV to analyze how well they match your target audience.
        """)
        
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], key="csv_uploader")
        
        if uploaded_file is not None:
            with st.spinner("Parsing CSV file..."):
                df = parse_csv(uploaded_file)
                
            if df is not None: # Proceed only if df is not None (parsing was successful)
                st.success("CSV file uploaded successfully!")
                
                with st.expander("View Dataset (First 10 rows)"):
                    st.dataframe(df.head(10))
                
                # Detect sensitive parameters
                sensitive_params = detect_sensitive_parameters(df)
                
                st.subheader("Target Audience Criteria")
                
                # Create two columns for input controls
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Gender Targeting**")
                    target_gender = st.radio(
                        "Select target gender:",
                        ["both", "male", "female"],
                        index=0,
                        format_func=lambda x: "Both" if x == "both" else x.capitalize(),
                        key="gender_radio",
                        horizontal=True
                    )
                
                with col2:
                    st.markdown("**Age Range Targeting**")
                    age_min, age_max = st.slider(
                        "Select age range:",
                        min_value=0,
                        max_value=100,
                        value=(18, 65),
                        key="age_slider"
                    )
                
                if st.button("Analyze Data Match", key="analyze_match_btn"):
                    with st.spinner("Analyzing data match..."):
                        # Perform the analysis
                        analysis_results = analyze_data_match(df, sensitive_params, target_gender, age_min, age_max)
                        
                        # Store results in session state
                        st.session_state['analysis_results'] = analysis_results
                        st.session_state['sensitive_params'] = sensitive_params
                        st.session_state['target_gender'] = target_gender
                        st.session_state['age_min'] = age_min
                        st.session_state['age_max'] = age_max
                        
                        # Show analysis results
                        st.subheader("Targeting Match Analysis Results")
                        
                        # Overall Match Score with gauge chart
                        st.markdown("### Overall Match Score")
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Determine color based on score
                            if analysis_results['match_score'] >= 80:
                                color = "green"
                                icon = "✅"
                            elif analysis_results['match_score'] >= 50:
                                color = "orange"
                                icon = "⚠️"
                            else:
                                color = "red"
                                icon = "❌"
                            
                            st.markdown(f"""
                            <div style="text-align: center;">
                                <h1 style="color: {color}; font-size: 3rem;">{analysis_results['match_score']}% {icon}</h1>
                                <p style="font-size: 1.2rem;">Match Score</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Create a gauge chart
                            fig = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=analysis_results['match_score'],
                                domain={'x': [0, 1], 'y': [0, 1]},
                                gauge={
                                    'axis': {'range': [0, 100]},
                                    'bar': {'color': color},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "gray"},
                                        {'range': [80, 100], 'color': "darkgray"}],
                                    'threshold': {
                                        'line': {'color': "black", 'width': 4},
                                        'thickness': 0.75,
                                        'value': analysis_results['match_score']}
                                }
                            ))
                            fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), height=350) # Consistent height
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Score description
                        if analysis_results['match_score'] >= 80:
                            st.success("""
                            **Excellent match!** The dataset aligns very well with your target audience.
                            """)
                        elif analysis_results['match_score'] >= 50:
                            st.warning("""
                            **Moderate match.** Consider adjusting your targeting or finding a more suitable dataset.
                            """)
                        else:
                            st.error("""
                            **Poor match.** This dataset does not align well with your target audience.
                            """)
                        
                        st.markdown("---")
                        
                        # Key Metrics
                        st.subheader("Key Metrics")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Total Records",
                                f"{analysis_results['total_records']:,}",
                                help="Total number of records in the dataset"
                            )
                        
                        with col2:
                            st.metric(
                                "Gender Match Rate",
                                f"{analysis_results['gender_match']:.1f}%",
                                help="Percentage of records matching gender criteria"
                            )
                        
                        with col3:
                            st.metric(
                                "Age Match Rate",
                                f"{analysis_results['age_match']:.1f}%",
                                help="Percentage of records matching age criteria"
                            )
                        
                        st.markdown("---")
                        
                        # Detailed Insights
                        st.subheader("Detailed Insights")
                        insights = generate_insights(
                            analysis_results,
                            sensitive_params,
                            target_gender,
                            age_min,
                            age_max
                        )
                        
                        for insight in insights:
                            # Determine the appropriate container based on insight type
                            if insight['type'] == 'positive':
                                container = st.success
                            elif insight['type'] == 'warning':
                                container = st.warning
                            elif insight['type'] == 'negative':
                                container = st.error
                            else:  # neutral
                                container = st.info
                            
                            container(f"**{insight['title']}**\n\n{insight['data']}\n\n{insight.get('description', '')}")
                        
                        st.markdown("---")
                        
                        # Data Visualizations
                        st.subheader("Data Visualizations")
                        
                        # Create tabs for different visualizations
                        viz_tabs = st.tabs(["Demographics", "Parameter Distributions"])
                        
                        with viz_tabs[0]:  # Demographics tab
                            # Age distribution if available
                            if 'age' in sensitive_params:
                                st.markdown("### Age Distribution")
                                age_fig = create_age_chart(df, sensitive_params['age'])
                                if age_fig:
                                    st.plotly_chart(age_fig, use_container_width=True)
                                else:
                                    st.info("No age distribution chart could be generated.")
                            else:
                                st.info("No age data column found in the dataset to display age distribution.")
                            
                            # Gender distribution if available
                            if 'gender' in analysis_results['parameter_stats'] and analysis_results['parameter_stats']['gender']:
                                st.markdown("### Gender Distribution")
                                gender_fig = create_gender_chart(analysis_results['parameter_stats']['gender'])
                                if gender_fig:
                                    st.plotly_chart(gender_fig, use_container_width=True)
                                else:
                                    st.info("No gender distribution chart could be generated.")
                            else:
                                st.info("No gender data column found in the dataset to display gender distribution.")
                        
                        with viz_tabs[1]:  # Parameter Distributions tab
                            # Show charts for other parameters (up to 3)
                            chart_count = 0
                            has_other_params = False
                            for param, col in sensitive_params.items():
                                if param not in ['age', 'gender'] and col in df.columns: # Check if column actually exists in df
                                    has_other_params = True
                                    if chart_count < 3: # Limit to 3 charts for other parameters
                                        st.markdown(f"### {col} Distribution")
                                        # Pass the full DataFrame to get top values
                                        param_fig = create_top_values_chart(col, df, col) 
                                        if param_fig:
                                            st.plotly_chart(param_fig, use_container_width=True)
                                            chart_count += 1
                                        else:
                                            st.warning(f"Could not generate distribution chart for {col}")
                            
                            if not has_other_params:
                                st.info("No additional relevant parameter distributions to display (e.g., location, occupation).")
                            elif chart_count == 0:
                                st.info("No charts could be generated for other parameters.")
                        
                        # Add download buttons for reports
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            text_report_data = generate_text_report(analysis_results, sensitive_params, target_gender, age_min, age_max)
                            if text_report_data:
                                st.download_button(
                                    label="Download Text Report",
                                    data=text_report_data,
                                    file_name="targeting_analysis_report.txt",
                                    mime="text/plain",
                                    help="Download a text summary of the analysis results",
                                    key="text_report_btn"
                                )
                            else:
                                st.warning("Text report could not be generated.")
                        
                        with col2:
                            # Generate PDF report when button is clicked
                            pdf_buffer = generate_pdf_report_dataset(analysis_results, sensitive_params, target_gender, age_min, age_max)
                            if pdf_buffer:
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_buffer,
                                    file_name="targeting_analysis_report.pdf",
                                    mime="application/pdf",
                                    help="Download a detailed PDF report of the analysis",
                                    key="pdf_report_btn"
                                )
                            else:
                                st.warning("PDF report could not be generated.")