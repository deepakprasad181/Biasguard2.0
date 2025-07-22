import streamlit as st
import io
import re
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from collections import Counter
import json
from utils import (
    DEMOGRAPHIC_GROUPS, ALL_DEMOGRAPHICS, validate_image_file, 
    validate_and_convert_image, analyze_image_for_demographic_appeal_via_caption,
    analyze_hashtags_appeal, combine_results_appeal, generate_demographic_pdf_report,
    image_captioner, load_text_model_appeal, get_popular_hashtags_for_demographics,
    get_ollama_insights, check_ollama_server # Import new functions
)

def analyze_combined_demographics(image_results, text_results, image_weight=0.5, text_weight=0.5):
    """
    Enhanced combination of image and text demographic analysis.
    """
    total_weight = image_weight + text_weight
    if total_weight == 0:
        return {}
    
    norm_image_weight = image_weight / total_weight
    norm_text_weight = text_weight / total_weight
    
    combined = {dem: 0.0 for dem in ALL_DEMOGRAPHICS} 
    
    for dem in ALL_DEMOGRAPHICS:
        image_score = image_results.get(dem, 0.0) * norm_image_weight
        text_score = text_results.get(dem, 0.0) * norm_text_weight
        combined[dem] = image_score + text_score
    
    max_score_raw = max(combined.values()) if combined else 0
    threshold = max_score_raw * 0.05 
    filtered_combined = {k: v for k, v in combined.items() if v >= threshold}
    
    total = sum(filtered_combined.values())
    if total > 0:
        normalized = {k: v/total for k, v in filtered_combined.items()}
    else:
        normalized = filtered_combined
    
    return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))

def get_top_demographics(combined_results, image_results, text_results, image_weight, text_weight, n=5):
    """Get top n demographics with analysis of why they scored high"""
    top_demos = sorted(combined_results.items(), key=lambda x: x[1], reverse=True)[:n]
    
    analysis = []
    for demo, score in top_demos:
        reasons = []
        weighted_image_score = image_results.get(demo, 0) * image_weight
        weighted_text_score = text_results.get(demo, 0) * text_weight

        if weighted_image_score > weighted_text_score * 1.2:
            reasons.append("strong visual alignment")
        elif weighted_text_score > weighted_image_score * 1.2:
            reasons.append("hashtag/text relevance")
        else:
            reasons.append("balanced visual and text appeal")
        
        for group_name, demos_in_group in DEMOGRAPHIC_GROUPS.items():
            if demo in demos_in_group:
                if "Age Groups" == group_name:
                    if "kids" in demo or "adolescents" in demo or "young adults" in demo:
                        reasons.append("youthful content")
                    elif "elderly" in demo:
                        reasons.append("mature content")
                elif "Gender & Sexuality" == group_name:
                    if "LGBTQ+" in demo:
                        reasons.append("inclusive representation")
                elif "Socio-economic Status" == group_name:
                    if "luxury" in demo or "high income" in demo:
                        reasons.append("premium positioning")
                elif "Interests & Hobbies" == group_name:
                    if "fitness" in demo:
                        reasons.append("active lifestyle focus")
                    elif "tech-savvy" in demo:
                        reasons.append("innovative/digital focus")
                break
        
        analysis.append((demo, score, ", ".join(reasons)))
    
    return analysis

def run_app():
    """Runs the Ad Demographic Analyzer module."""
    st.title("📊 Ad Demographic Analyzer")
    st.markdown("Upload an image to analyze which demographics it appeals to most. This module now uses the LLaVA model for enhanced strategic recommendations.")
    st.markdown("---")

    if not check_ollama_server():
        st.warning("Ollama server not found. LLaVA-powered insights will be unavailable. Please start Ollama to enable this feature.")

    text_classifier_for_appeal = load_text_model_appeal()

    if image_captioner is None:
        st.error("Image captioning model is not loaded. Image-based demographic analysis will not function.")
    if text_classifier_for_appeal is None:
        st.error("Text classification model (for appeal) is not loaded. Hashtag and caption-based demographic analysis will not function.")

    col1, col2 = st.columns(2)

    uploaded_file = None
    uploaded_image_bytes = None
    image_caption = ""

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an ad image",
            type=["png", "jpg", "jpeg", "webp"],
            help="Supported formats: PNG, JPG, JPEG, WEBP",
            key="appeal_image_uploader"
        )

        if uploaded_file is not None:
            try:
                is_valid, validation_msg = validate_image_file(uploaded_file.getvalue(), uploaded_file.name)
                if not is_valid:
                    st.error(f"Image validation failed: {validation_msg}")
                    uploaded_file = None
                else:
                    success, converted_image_bytes, conversion_error = validate_and_convert_image(uploaded_file.getvalue(), uploaded_file.name)
                    if not success:
                        st.error(f"Image conversion failed: {conversion_error}")
                        uploaded_file = None
                    else:
                        uploaded_image_bytes = converted_image_bytes
                        image = Image.open(io.BytesIO(uploaded_image_bytes))
                        st.image(image, caption="Uploaded Image", use_container_width=True)
                        
                        if image_captioner:
                            if 'image_caption' not in st.session_state or st.session_state.get('uploaded_file_id') != uploaded_file.file_id:
                                with st.spinner("Generating image caption..."):
                                    caption_results = image_captioner(image)
                                    if caption_results and caption_results[0] and 'generated_text' in caption_results[0]:
                                        image_caption = caption_results[0]['generated_text']
                                        st.session_state.image_caption = image_caption
                                        st.session_state.uploaded_file_id = uploaded_file.file_id
                                    else:
                                        st.session_state.image_caption = ""
                                        st.session_state.uploaded_file_id = None
                            else:
                                image_caption = st.session_state.image_caption
                        else:
                            st.session_state.image_caption = ""

            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
                uploaded_file = None

    with col2:
        st.text_area(
            "Enter relevant hashtags (optional, for context)",
            value="",
            help="This input is for display purposes and context. The primary analysis is driven by the image.",
            height=150,
            key="appeal_hashtags_input"
        )

        st.markdown("**Analysis Weighting:**")
        image_weight = st.slider(
            "Weight for Image Appeal",
            min_value=0.0, max_value=1.0, value=0.6, step=0.1,
            help="How much to prioritize visual content analysis in overall appeal.",
            key="appeal_image_weight"
        )
        text_weight = st.slider(
            "Weight for Hashtag/Caption Appeal",
            min_value=0.0, max_value=1.0, value=0.4, step=0.1,
            help="How much to prioritize text (caption) analysis in overall appeal.",
            key="appeal_text_weight"
        )

    st.markdown("---")

    if st.button("Analyze Ad Demographics", key="analyze_btn"):
        if uploaded_file is None:
            st.warning("Please upload an image to analyze ad demographics.")
            return

        # FIX: Update spinner message for better user feedback
        with st.spinner("Analyzing... LLaVA is processing the image and can take up to 5 minutes."):
            image_results = {}
            caption_text_results = {}
            ollama_demographic_insights = "Not generated."
            
            if uploaded_image_bytes is not None and image_captioner is not None and text_classifier_for_appeal is not None:
                current_image_caption = st.session_state.get('image_caption', '')
                if current_image_caption:
                    image_results = analyze_image_for_demographic_appeal_via_caption(uploaded_image_bytes, image_captioner, text_classifier_for_appeal)
                    caption_text_results = analyze_hashtags_appeal(current_image_caption, text_classifier_for_appeal)
                else:
                    st.warning("Image caption not available for analysis. Please ensure image is valid.")
            
            combined_results = analyze_combined_demographics(
                image_results, 
                caption_text_results, 
                image_weight, 
                text_weight
            )

            if not combined_results:
                st.error("Heuristic analysis failed. Please check your inputs and try again.")
                return
            
            top_5_demos = list(combined_results.keys())[:5]
            
            prompt_for_llava = f"""
            You are a senior marketing strategist. I will provide you with an image and pre-computed demographic appeal data.
            Your task is to generate 'Actionable Insights & Recommendations' for a marketing campaign based on this ad.

            **Pre-computed Analysis Data:**
            - Ad Image Caption: "{st.session_state.get('image_caption', 'N/A')}"
            - Top 5 Appealing Demographics (Heuristic): {json.dumps(top_5_demos)}

            **Your Instructions:**
            1. Directly analyze the provided image to understand its visual language, mood, and the subjects depicted.
            2. Synthesize your visual analysis with the pre-computed list of appealing demographics.
            3. Explain WHY the ad likely appeals to these groups. What specific elements (visual or thematic) are driving this appeal?
            4. Provide creative and strategic recommendations. Suggest how to:
               a) Strengthen the appeal to the primary demographic.
               b) Broaden the appeal to other relevant groups without losing focus.
            5. Structure your response clearly under a single heading: "Actionable Insights & Recommendations". Do not use markdown formatting like bold or italics.
            """
            ollama_demographic_insights = get_ollama_insights(uploaded_image_bytes, prompt_for_llava)

        st.success("Analysis Complete!")
        st.markdown("## 🎉 Analysis Results")
        st.markdown("---")

        st.subheader("🎯 Actionable Insights & Recommendations (from LLaVA)")
        st.markdown(f"> {ollama_demographic_insights.replace('.', '. ')}")
        st.markdown("---")

        with st.expander("Show Detailed Heuristic-Based Analysis"):
            st.subheader("📊 Key Takeaways (Heuristic-Based)")
            top_appealing_demographics = [
                (k, v, reason) for k, v, reason in get_top_demographics(combined_results, image_results, caption_text_results, image_weight, text_weight, n=5) if v > 0.005
            ]
            st.session_state.top_appealing_demographics = top_appealing_demographics

            if top_appealing_demographics:
                cols_metrics = st.columns(3)
                for i, (demo, score, reason) in enumerate(top_appealing_demographics[:3]):
                    with cols_metrics[i]:
                        st.metric(label=f"Top {i+1} Demographic", value=demo, delta=f"{score*100:.1f}% Appeal")
                        st.caption(f"Reason: {reason}")
            else:
                st.info("No prominent individual demographics found from heuristic analysis.")
            st.markdown("---")

            st.subheader("Detailed Appeal by Demographic Group")
            group_names = list(DEMOGRAPHIC_GROUPS.keys())
            for i in range(0, len(group_names), 2):
                cols_charts = st.columns(2)
                for j in range(2):
                    if i + j < len(group_names):
                        group_name = group_names[i + j]
                        demographics_list = DEMOGRAPHIC_GROUPS[group_name]

                        group_specific_scores = {}
                        for dem in demographics_list:
                            if combined_results.get(dem, 0) > 0.001:
                                group_specific_scores[dem] = combined_results.get(dem, 0)

                        total_group_score = sum(group_specific_scores.values())
                        if total_group_score > 0:
                            normalized_group_scores = {k: v / total_group_score for k, v in group_specific_scores.items()}
                        else:
                            normalized_group_scores = {}

                        with cols_charts[j]:
                            if normalized_group_scores:
                                st.markdown(f"**Appeal within {group_name}**")
                                fig_pie, ax_pie = plt.subplots(figsize=(6, 6)) 
                                labels = [f"{k}" for k, v in normalized_group_scores.items()]
                                sizes = [v for v in normalized_group_scores.values()]

                                def autopct_format(pct):
                                    return f'{pct:.1f}%' if pct > 0.5 else ''

                                wedges, texts, autotexts = ax_pie.pie(
                                    sizes, labels=labels, autopct=autopct_format, startangle=90, pctdistance=0.85,
                                    wedgeprops=dict(width=0.4), textprops=dict(color="black")
                                )
                                for autotext in autotexts:
                                    autotext.set_color('white')
                                for text in texts:
                                    text.set_fontsize(8)

                                ax_pie.set_title(f'Distribution within {group_name}', fontsize=12)
                                st.pyplot(fig_pie, use_container_width=True)
                                plt.close(fig_pie)
                            else:
                                st.info(f"No significant appeal detected within the **{group_name}** group.")
                st.markdown("---")

            st.subheader("Top Individual Demographics Appeal (Overall)")
            top_individual_demographics = {k: v for k, v in combined_results.items() if v > 0.005}
            if top_individual_demographics:
                top_individual_demographics_sorted = dict(sorted(top_individual_demographics.items(), key=lambda item: item[1], reverse=True)[:10])

                fig_bar, ax_bar = plt.subplots(figsize=(10, 6)) 
                demographics_labels = list(top_individual_demographics_sorted.keys())
                scores = list(top_individual_demographics_sorted.values())

                bars = ax_bar.barh(demographics_labels, scores, color='lightcoral')
                ax_bar.set_xlabel('Appeal Score (Normalized)', fontsize=12)
                ax_bar.set_title('Overall Highest Appealing Individual Demographics', fontsize=16)
                ax_bar.invert_yaxis()

                for bar in bars:
                    width = bar.get_width()
                    ax_bar.text(width, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%',
                                ha='left', va='center', fontsize=9, color='black')
                
                ax_bar.xaxis.set_tick_params(labelbottom=True)
                plt.tight_layout()
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar)
            else:
                st.info("No significant individual demographic appeal detected across all categories.")
            st.markdown("---")
        
        if combined_results:
            pdf_buffer = generate_demographic_pdf_report(
                combined_results,
                st.session_state.get('image_caption', 'No caption generated.'),
                st.session_state.get('top_appealing_demographics', []),
                uploaded_image_bytes,
                ollama_insights=ollama_demographic_insights
            )
            
            if pdf_buffer:
                st.download_button(
                    label="Download Full Report (PDF)",
                    data=pdf_buffer,
                    file_name=f"demographic_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf"
                )

if __name__ == "__main__":
    run_app()
