import streamlit as st
import io
from PIL import Image
from datetime import datetime
import json
from utils import (
    validate_image_file, validate_and_convert_image,
    extract_text_from_image, analyze_text_for_bias,
    analyze_image_for_visual_context, check_for_pii, check_for_harmful_content,
    calculate_overall_bias_score, create_gauge_chart, generate_image_analysis_pdf_report,
    get_ollama_insights, check_ollama_server
)

def run_app():
    """Runs the Image Advertisement Analysis module."""
    st.header("名ｸImage Advertisement Analysis (OCR & Bias)")
    st.markdown("""
    Upload an image of an advertisement (e.g., flyer, social media ad) to extract its text
    and analyze it for biases and security concerns. This module now uses the LLaVA model
    for enhanced insights and recommendations.
    """)

    if not check_ollama_server():
        st.warning("Ollama server not found. LLaVA-powered insights will be unavailable. Please start Ollama to enable this feature.")

    uploaded_file = st.file_uploader(
        "Choose an image file...", type=["png", "jpg", "jpeg", "webp", "tiff", "bmp", "gif"], key="image_uploader"
    )

    if uploaded_file is not None:
        try:
            is_valid, validation_msg = validate_image_file(uploaded_file.getvalue(), uploaded_file.name)
            if not is_valid:
                st.error(f"Image validation failed: {validation_msg}")
                st.stop()
            
            success, converted_image_bytes, conversion_error = validate_and_convert_image(uploaded_file.getvalue(), uploaded_file.name)
            if not success:
                st.error(f"Image conversion failed: {conversion_error}")
                st.stop()
            
            st.image(converted_image_bytes, caption='Uploaded Image', use_container_width=True)
            st.write("")

            if st.button("Analyze Image Ad", key="analyze_image_btn"):
                # FIX: Update spinner message for better user feedback
                with st.spinner("Analyzing... LLaVA is processing the image and can take up to 5 minutes."):
                    extracted_text = ""
                    text_bias_results = {'is_biased': False, 'bias_categories': {}, 'suggestions': [], 'bias_score': 0, 'highlighted_text': ""}
                    visual_analysis_results = {'generated_caption': 'N/A', 'visual_flags': [], 'bias_categories': {}, 'is_visually_biased': False, 'visual_bias_score': 0}
                    pii_results = {'has_pii': False, 'detected_items': []}
                    harmful_results = {'has_harmful_content': False, 'detected_items': []}
                    ollama_ad_insights = "Not generated."

                    try:
                        extracted_text = extract_text_from_image(converted_image_bytes)
                        if extracted_text:
                            text_bias_results = analyze_text_for_bias(extracted_text)
                            pii_results = check_for_pii(extracted_text)
                            harmful_results = check_for_harmful_content(extracted_text)
                        else:
                            st.warning("No text extracted from the image. Skipping text-based analysis (bias, PII, harmful content).")
                    except Exception as e:
                        st.error(f"Failed to perform text extraction and related analysis due to an unexpected error: {e}.")
                    
                    try:
                        from utils import image_captioner 
                        if image_captioner is not None:
                            visual_analysis_results = analyze_image_for_visual_context(converted_image_bytes, extracted_text)
                        else:
                            st.warning("Image captioning model not available, skipping visual context analysis.")
                    except Exception as e:
                        st.error(f"Failed to perform visual context analysis due to an unexpected error: {e}.")
                    
                    compliance_risk = pii_results['has_pii'] or harmful_results['has_harmful_content']
                    overall_bias_score = calculate_overall_bias_score(
                        text_bias_results['bias_score'],
                        visual_analysis_results['visual_bias_score']
                    )

                    bias_categories_summary = {k: f"{len(v)} flag(s)" for k, v in text_bias_results['bias_categories'].items()}
                    visual_bias_summary = {k: f"{len(v)} flag(s)" for k, v in visual_analysis_results['bias_categories'].items()}
                    
                    prompt_for_llava = f"""
                    You are an expert in ethical advertising and bias detection. I will provide you with an image and pre-computed analysis data.
                    Your task is to generate actionable 'Insights & Recommendations' based on a holistic view of the image and the provided data.

                    **Pre-computed Analysis Data:**
                    - Extracted Text from Image: "{extracted_text[:500]}"
                    - Textual Bias Score (0-10): {text_bias_results['bias_score']}
                    - Detected Textual Bias Categories: {json.dumps(bias_categories_summary)}
                    - Visual Bias Score (0-10): {visual_analysis_results['visual_bias_score']}
                    - Detected Visual Bias Categories (from caption): {json.dumps(visual_bias_summary)}
                    - PII Detected: {pii_results['has_pii']}
                    - Harmful Content Detected: {harmful_results['has_harmful_content']}

                    **Your Instructions:**
                    1. Analyze the image directly for subtle cues that the automated analysis might have missed.
                    2. Synthesize the pre-computed data with your own visual analysis.
                    3. Provide actionable recommendations to mitigate any detected biases, improve inclusivity, and enhance the ad's overall ethical standing and effectiveness.
                    4. Structure your response clearly under a single heading: "Insights & Recommendations". Do not use markdown formatting like bold or italics.
                    """
                    ollama_ad_insights = get_ollama_insights(converted_image_bytes, prompt_for_llava)

                st.subheader("Overall Ad Status:")
                overall_ad_status_text = ""
                if compliance_risk or overall_bias_score > 6:
                    st.error("閥 Urgent Action (High Risk)")
                    overall_ad_status_text = "Urgent Action (High Risk): This image ad has significant issues (PII, harmful content, or high bias) that require immediate attention."
                elif overall_bias_score > 3:
                    st.warning("泯 Review Needed (Moderate Risk)")
                    overall_ad_status_text = "Review Needed (Moderate Risk): This image ad has potential issues that should be reviewed for ethical and compliance standards."
                else:
                    st.success("泙 Compliant (Low Risk)")
                    overall_ad_status_text = "Compliant (Low Risk): This image ad appears to be relatively free of significant biases or security risks."
                st.markdown("---")

                st.subheader("Combined Bias Severity:")
                st.progress(overall_bias_score / 10, text=f"Combined Bias Score: {overall_bias_score}/10")
                st.markdown("---")
                
                st.subheader("庁 Insights & Recommendations (from LLaVA):")
                st.markdown(f"> {ollama_ad_insights.replace('.', '. ')}")
                st.markdown("---")


                st.subheader("Extracted Text (with potential highlights):")
                if extracted_text:
                    st.markdown(text_bias_results['highlighted_text'], unsafe_allow_html=True)
                else:
                    st.warning("No text could be extracted from the image or an error occurred.")
                st.markdown("---")

                with st.expander("Show Detailed Heuristic-Based Analysis"):
                    st.markdown("##### Textual Bias Analysis (from OCR):")
                    if text_bias_results['is_biased']:
                        st.error("Potential Textual Bias Detected!")
                        for category, flags_data in text_bias_results['bias_categories'].items():
                            if flags_data:
                                st.markdown(f"**- {category.replace('_', ' ').title()} Bias:**")
                                for flag_item in flags_data:
                                    st.warning(f"  - {flag_item['message']}")

                        text_bias_severity_scores = {
                            category.replace('_', ' ').title(): sum(item['score'] for item in flags_data)
                            for category, flags_data in text_bias_results['bias_categories'].items()
                        }
                        if text_bias_severity_scores:
                            st.subheader("Textual Bias Breakdown by Severity:")
                            cols = st.columns(2)
                            col_idx = 0
                            for bias_type, score in text_bias_severity_scores.items():
                                with cols[col_idx % 2]:
                                    st.plotly_chart(create_gauge_chart(bias_type, score), use_container_width=True)
                                col_idx += 1
                        st.markdown("---")

                    else:
                        st.success("No significant textual bias detected.")
                        st.markdown("---")

                    st.markdown("##### Visual Context Analysis (from Image Captioning):")
                    if visual_analysis_results['is_visually_biased']:
                        st.error("Potential Visual Bias Detected!")
                        for category, flags_data in visual_analysis_results['bias_categories'].items():
                            if flags_data:
                                st.markdown(f"**- {category.replace('_', ' ').title()} Bias:**")
                                for flag_item in flags_data:
                                    st.warning(f"  - {flag_item['message']}")
                        st.write("Review image for stereotypical depictions based on the generated caption and overall context.")

                        visual_bias_severity_scores = {
                            category.replace('_', ' ').title(): sum(item['score'] for item in flags_data)
                            for category, flags_data in visual_analysis_results['bias_categories'].items()
                        }
                        if visual_bias_severity_scores:
                            st.subheader("Visual Bias Breakdown by Severity:")
                            cols = st.columns(2)
                            col_idx = 0
                            for bias_type, score in visual_bias_severity_scores.items():
                                with cols[col_idx % 2]:
                                    st.plotly_chart(create_gauge_chart(bias_type, score), use_container_width=True)
                                col_idx += 1
                        st.markdown("---")

                    else:
                        st.success("No significant visual stereotypical context detected based on analysis of the caption.")
                        st.markdown("---")
                    
                    st.markdown("##### Security Suggestions:")
                    if pii_results['has_pii']:
                        st.error(f"- PII Detected! Found: {', '.join(pii_results['detected_items'])}. This information should be redacted.")
                        st.info("  - Action: Remove or redact sensitive personal information before public display.")
                    else:
                        st.success("- No Personal Identifiable Information (PII) found.")

                    if harmful_results['has_harmful_content']:
                        st.error(f"- Harmful Content Detected! Found: {', '.join(harmful_results['detected_items'])}. Review for offensive or manipulative language.")
                        st.info("  - Action: Remove or rephrase offensive/manipulative language to maintain a positive brand image and ethical standards.")
                    else:
                        st.success("- No direct harmful content detected by keyword check.")
                    st.markdown("---")

                pdf_report_buffer = generate_image_analysis_pdf_report(
                    uploaded_file.name,
                    extracted_text,
                    text_bias_results,
                    visual_analysis_results,
                    pii_results,
                    harmful_results,
                    overall_ad_status_text,
                    overall_bias_score,
                    converted_image_bytes,
                    ollama_insights=ollama_ad_insights
                )
                if pdf_report_buffer:
                    st.download_button(
                        label="Download Report (PDF)",
                        data=pdf_report_buffer,
                        file_name=f"image_ad_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        help="Download a PDF report of the image advertisement analysis."
                    )
                else:
                    st.warning("PDF report could not be generated due to an error.")

        except Exception as e:
            st.error(f"An unexpected error occurred during image processing: {str(e)}")
            st.stop()

    else:
        st.info("Upload an image to start the analysis.")