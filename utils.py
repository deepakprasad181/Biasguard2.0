import streamlit as st
import easyocr
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import re
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch._dynamo
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
from fpdf import FPDF
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models, transforms
from collections import Counter
import json
import tempfile
import random
import requests
import base64

# Suppress PyTorch errors for cleaner output in Streamlit
torch._dynamo.config.suppress_errors = True

# --- Ollama LLaVA Integration ---

OLLAMA_API_URL = "http://localhost:11434/api/generate"

@st.cache_data(show_spinner=False)
def check_ollama_server():
    """Checks if the Ollama server is running."""
    try:
        response = requests.get("http://localhost:11434", timeout=2)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

def get_ollama_insights(image_bytes, prompt_text):
    """
    Gets insights from the Ollama LLaVA model.

    Args:
        image_bytes (bytes): The image file in bytes.
        prompt_text (str): The text prompt to guide the model's analysis.

    Returns:
        str: The generated insights from the model, or an error message.
    """
    if not check_ollama_server():
        return "Ollama server not detected. Please ensure Ollama is running on your local machine to use this feature."
    
    try:
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        payload = {
            "model": "llava",
            "prompt": prompt_text,
            "stream": False,
            "images": [image_b64]
        }

        # Increased timeout from 120 to 300 seconds (5 minutes)
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()

        response_data = response.json()
        return response_data.get('response', "No response content from LLaVA.").strip()

    except requests.exceptions.Timeout:
        return "Ollama Timed Out: The LLaVA model took too long to respond (more than 5 minutes). This can happen with complex images on CPU. Please try again or use a simpler image."
    except requests.ConnectionError:
        return "Connection Error: Could not connect to the Ollama server at http://localhost:11434. Please ensure it's running."
    except requests.exceptions.RequestException as e:
        return f"An error occurred while querying the LLaVA model: {e}"
    except Exception as e:
        return f"An unexpected error occurred during LLaVA analysis: {e}"


# --- Common Model Initializations ---
@st.cache_resource
def get_easyocr_reader():
    """Initializes and caches the EasyOCR reader."""
    try:
        return easyocr.Reader(['en'], gpu=False)
    except Exception as e:
        st.error(f"Error initializing EasyOCR: {e}. Text extraction from images will not be available.")
        return None

reader = get_easyocr_reader()

@st.cache_resource
def get_image_captioning_pipeline():
    """Initializes and caches the image captioning pipeline."""
    try:
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", trust_remote_code=False)
    except Exception as e:
        st.error(f"Could not load image captioning model: {e}. Visual context analysis will be limited.")
        return None

image_captioner = get_image_captioning_pipeline()

@st.cache_resource
def get_text_bias_classifier():
    """Initializes and caches the text bias classifier pipeline."""
    try:
        tokenizer = AutoTokenizer.from_pretrained("cirimus/modernbert-large-bias-type-classifier")
        model = AutoModelForSequenceClassification.from_pretrained("cirimus/modernbert-large-bias-type-classifier")
        return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, trust_remote_code=False)
    except Exception as e:
        st.error(f"Could not load text bias classifier model: {e}. Falling back to keyword-based bias detection for text analysis.")
        return None

text_bias_classifier = get_text_bias_classifier()

@st.cache_resource
def load_image_model_appeal():
    """Loads and caches a generic image model (e.g., ResNet) if needed for other purposes."""
    try:
        model = models.resnet50(pretrained=True)
        model = torch.nn.Sequential(*(list(model.children())[:-1])) # Remove the final classification layer
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading ResNet model for appeal.py (if needed): {e}")
        return None

@st.cache_resource
def load_text_model_appeal():
    """Loads and caches the zero-shot classification model for text analysis (for appeal.py)."""
    try:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", multi_label=True)
        return classifier
    except Exception as e:
        st.error(f"Error loading zero-shot classification model for appeal.py: {e}")
        return None

# --- Image Validation and Conversion Functions ---
def validate_and_convert_image(uploaded_file_bytes, file_name):
    """Validates an image file and converts it to a standardized PNG format if needed."""
    try:
        with Image.open(io.BytesIO(uploaded_file_bytes)) as img:
            img.verify()
        
        with Image.open(io.BytesIO(uploaded_file_bytes)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            png_buffer = io.BytesIO()
            img.save(png_buffer, format='PNG')
            png_bytes = png_buffer.getvalue()
            
            return True, png_bytes, None
            
    except UnidentifiedImageError:
        return False, None, "The uploaded file is not a recognized image format or is corrupted."
    except Exception as e:
        return False, None, f"An unexpected error occurred during image processing: {str(e)}"

def validate_image_file(uploaded_file_bytes, file_name):
    """Validates an image file for format and corruption, with strict extension matching."""
    try:
        with Image.open(io.BytesIO(uploaded_file_bytes)) as image:
            image.verify()
        
        _, ext = os.path.splitext(file_name)
        pil_format_lower = image.format.lower() if image.format else ""
        ext_lower = ext.lower()

        strict_format_to_extensions = {
            'jpeg': ['.jpg', '.jpeg'],
            'png': ['.png'],
            'webp': ['.webp'],
            'tiff': ['.tif', '.tiff'],
            'bmp': ['.bmp'],
            'gif': ['.gif']
        }

        if pil_format_lower in strict_format_to_extensions:
            if ext_lower in strict_format_to_extensions[pil_format_lower]:
                return True, None
            else:
                expected_exts = " or ".join(strict_format_to_extensions[pil_format_lower])
                return False, f"File format mismatch: Image content is detected as '{pil_format_lower.upper()}', but the file extension is '{ext}'. Expected {expected_exts}."
        else:
            return False, f"Unsupported or unknown image format detected ('{pil_format_lower.upper()}' for extension '{ext}'). Please upload a common image type (PNG, JPG, JPEG, WebP, TIFF, BMP, GIF) with a matching extension."

    except UnidentifiedImageError:
        return False, "The uploaded file is not a recognized image format or is corrupted. Please upload a valid image (e.g., PNG, JPG, JPEG, WebP, TIFF, BMP, GIF)."
    except Exception as e:
        return False, f"An unexpected error occurred during image validation: {e}. Please try another image or contact support."

# --- Audience Targeting Analysis Functions ---
def parse_csv(uploaded_file):
    """Parse CSV file into a pandas DataFrame"""
    try:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8-sig"))
        df = pd.read_csv(stringio)
        if df.empty:
            st.warning("The uploaded CSV file is empty or contains no data rows.")
            return None
        return df
    except UnicodeDecodeError:
        st.error("Error: Could not decode the CSV file. Please ensure it's a valid UTF-8 encoded CSV.")
        return None
    except pd.errors.EmptyDataError:
        st.error("Error: The CSV file is empty.")
        return None
    except Exception as e:
        st.error(f"Error parsing CSV file: {e}. Please ensure it's a well-formatted CSV.")
        return None

def detect_sensitive_parameters(df):
    """Detect sensitive parameters in the dataset"""
    parameter_mappings = {
        'age': ['age', 'years', 'yrs', 'year'],
        'gender': ['gender', 'sex', 'male/female', 'm/f'],
        'location': ['location', 'city', 'state', 'country', 'region', 'address'],
        'religion': ['religion', 'faith', 'belief', 'denomination'],
        'race': ['race', 'ethnicity', 'ethnic'],
        'disability': ['disability', 'disabled', 'handicap'],
        'occupation': ['occupation', 'job', 'profession', 'work'],
        'income': ['income', 'salary', 'wage', 'earnings'],
        'education': ['education', 'degree', 'qualification'],
        'marital': ['marital', 'maritalstatus', 'relationshipstatus']
    }
    
    sensitive_params = {}
    for param_type, patterns in parameter_mappings.items():
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in patterns):
                sensitive_params[param_type] = col
                break
    
    for col in df.columns:
        if col not in sensitive_params.values() and col not in parameter_mappings:
            sensitive_params[col] = col
    
    return sensitive_params

def analyze_data_match(df, sensitive_params, target_gender, age_min, age_max):
    """Analyze how well the data matches the target audience"""
    results = {
        'total_records': len(df),
        'matched_records': 0,
        'gender_match': 0,
        'age_match': 0,
        'match_score': 0,
        'parameter_stats': {}
    }
    
    if df.empty:
        st.warning("Cannot analyze an empty dataset.")
        return results

    if 'age' in sensitive_params:
        results['parameter_stats']['age'] = {'stats': {'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0, 'avg': 0}}
    if 'gender' in sensitive_params:
        results['parameter_stats']['gender'] = {}
    
    gender_col = sensitive_params.get('gender')
    age_col = sensitive_params.get('age')
    
    gender_match_count = 0
    age_match_count = 0
    both_match_count = 0
    
    for _, row in df.iterrows():
        gender_match = True
        if gender_col and gender_col in row:
            try:
                gender_val = str(row[gender_col]).lower().strip()
                gender_match = target_gender == 'both' or gender_val == target_gender
            except Exception as e:
                st.warning(f"Could not process gender value '{row[gender_col]}': {e}")
                gender_match = False
        
        age_match = True
        if age_col and age_col in row:
            try:
                age = int(float(row[age_col]))
                age_match = age_min <= age <= age_max
                
                if 'age' in results['parameter_stats']:
                    stats = results['parameter_stats']['age']['stats']
                    stats['min'] = min(stats['min'], age)
                    stats['max'] = max(stats['max'], age)
                    stats['sum'] += age
                    stats['count'] += 1
            except (ValueError, TypeError):
                age_match = False
                st.warning(f"Could not convert age value '{row[age_col]}' to a number. Skipping for age match.")
        
        if gender_match:
            gender_match_count += 1
            if gender_col and gender_col in row:
                try:
                    gender_val = str(row[gender_col]).lower().strip()
                    results['parameter_stats']['gender'][gender_val] = results['parameter_stats']['gender'].get(gender_val, 0) + 1
                except Exception as e:
                    st.warning(f"Error updating gender stats for '{row[gender_col]}': {e}")
        
        if age_match:
            age_match_count += 1
        
        if gender_match and age_match:
            both_match_count += 1
    
    results['matched_records'] = both_match_count
    results['gender_match'] = (gender_match_count / results['total_records']) * 100 if results['total_records'] > 0 else 0
    results['age_match'] = (age_match_count / results['total_records']) * 100 if results['total_records'] > 0 else 0
    results['match_score'] = round((results['gender_match'] * 0.4) + (results['age_match'] * 0.6))
    
    if 'age' in results['parameter_stats'] and results['parameter_stats']['age']['stats']['count'] > 0:
        stats = results['parameter_stats']['age']['stats']
        stats['avg'] = round(stats['sum'] / stats['count'])
    
    return results

def create_age_chart(df, age_col):
    """Create age distribution chart"""
    if age_col not in df.columns:
        st.info(f"Age column '{age_col}' not found in the dataset.")
        return None
    
    try:
        df_copy = df.copy()
        df_copy[age_col] = pd.to_numeric(df_copy[age_col], errors='coerce').dropna()

        if df_copy[age_col].empty:
            st.warning(f"No valid numeric age data found in column '{age_col}' to create a chart.")
            return None

        df_copy['age_group'] = (df_copy[age_col] // 10) * 10
        age_groups_labels = [f"{i}-{i+9}" for i in range(0, 101, 10)]
        df_copy['age_group_label'] = df_copy['age_group'].apply(lambda x: f"{int(x)}-{int(x+9)}")
        df_copy['age_group_label'] = pd.Categorical(df_copy['age_group_label'], categories=age_groups_labels, ordered=True)

        age_counts = df_copy['age_group_label'].value_counts().reset_index()
        age_counts.columns = ['Age Group', 'Count']
        age_counts = age_counts.sort_values('Age Group')
        
        fig = px.bar(age_counts, x='Age Group', y='Count', title='Age Distribution')
        fig.update_layout(height=350)
        return fig
    except Exception as e:
        st.warning(f"Could not create age chart due to an error: {e}")
        return None

def create_gender_chart(gender_stats):
    """Create gender distribution chart"""
    if not gender_stats:
        st.info("No gender statistics available to create a chart.")
        return None
    
    genders = list(gender_stats.keys())
    counts = list(gender_stats.values())
    
    if not genders or sum(counts) == 0:
        st.warning("No valid gender data to display in chart.")
        return None
        
    try:
        fig = px.pie(names=genders, values=counts, title='Gender Distribution')
        fig.update_layout(height=350)
        return fig
    except Exception as e:
        st.warning(f"Could not create gender chart due to an error: {e}")
        return None

def create_top_values_chart(param_name, df, col_name):
    """Create chart for top values of a parameter"""
    if col_name not in df.columns:
        st.info(f"Column '{col_name}' not found in the dataset for parameter '{param_name}'.")
        return None
    
    try:
        param_counts = df[col_name].astype(str).value_counts().reset_index()
        param_counts.columns = [col_name, 'Count']
        param_counts = param_counts.head(5) # Get top 5
        
        if param_counts.empty:
            st.warning(f"No valid data in column '{col_name}' to create a chart for '{param_name}'.")
            return None

        fig = px.bar(param_counts, x=col_name, y='Count', title=f'Top {param_name} Values')
        fig.update_layout(xaxis_title=param_name, yaxis_title='Count', height=350)
        return fig
    except Exception as e:
        st.warning(f"Could not create top values chart for {param_name} due to an error: {e}")
        return None

def generate_insights(analysis_results, sensitive_params, target_gender, age_min, age_max):
    """Generate insights based on analysis results"""
    insights = []

    match_percentage = (analysis_results['matched_records'] / analysis_results['total_records']) * 100 if analysis_results['total_records'] > 0 else 0
    insights.append({
        'type': 'positive' if analysis_results['match_score'] >= 80 else 'warning' if analysis_results['match_score'] >= 50 else 'negative',
        'title': 'Target Audience Match',
        'data': f"{analysis_results['matched_records']} of {analysis_results['total_records']} records ({match_percentage:.1f}%) match both your gender and age criteria",
        'description': 'Excellent alignment with your target audience' if analysis_results['match_score'] >= 80 else \
                       'Moderate alignment - some adjustments may be needed' if analysis_results['match_score'] >= 50 else \
                       'Poor alignment - consider different targeting parameters or dataset'
    })

    if 'gender' in sensitive_params:
        insights.append({
            'type': 'positive' if analysis_results['gender_match'] >= 80 else 'warning' if analysis_results['gender_match'] >= 50 else 'negative',
            'title': 'Gender Match',
            'data': f"{analysis_results['gender_match']:.1f}% match with your gender target",
            'description': 'You are targeting both genders' if target_gender == 'both' else \
                           'Strong gender match' if analysis_results['gender_match'] >= 80 else \
                           'Moderate gender match' if analysis_results['gender_match'] >= 50 else \
                           'Weak gender match'
        })

    if 'age' in sensitive_params:
        insights.append({
            'type': 'positive' if analysis_results['age_match'] >= 80 else 'warning' if analysis_results['age_match'] >= 50 else 'negative',
            'title': 'Age Range Match',
            'data': f"{analysis_results['age_match']:.1f}% match with your age range ({age_min}-{age_max})",
            'description': 'Strong age range match' if analysis_results['age_match'] >= 80 else \
                           'Moderate age range match' if analysis_results['age_match'] >= 50 else \
                           'Weak age range match'
        })
    
    if 'age' in sensitive_params and analysis_results['parameter_stats'].get('age', {}).get('stats', {}).get('count', 0) > 0:
        age_stats = analysis_results['parameter_stats']['age']['stats']
        insights.append({
            'type': 'info',
            'title': 'Overall Age Distribution',
            'data': f"Min Age: {age_stats['min']}, Max Age: {age_stats['max']}, Avg Age: {age_stats['avg']}",
            'description': 'Summary of age data in your dataset'
        })

    if 'gender' in sensitive_params and analysis_results['parameter_stats'].get('gender'):
        gender_dist = ", ".join([f"{k.capitalize()}: {v}" for k, v in analysis_results['parameter_stats']['gender'].items()])
        insights.append({
            'type': 'info',
            'title': 'Overall Gender Distribution',
            'data': gender_dist,
            'description': 'Breakdown of gender representation in your dataset'
        })

    # Add insights for other sensitive parameters if they exist and have meaningful data
    for param_type, col_name in sensitive_params.items():
        if param_type not in ['age', 'gender']: # Already handled
            param_stats_key = f'top_{param_type}_values'
            if param_stats_key in analysis_results['parameter_stats']:
                top_values_data = analysis_results['parameter_stats'][param_stats_key]
                if top_values_data:
                    top_str = ", ".join([f"{k}: {v}" for k, v in top_values_data.items()])
                    insights.append({
                        'type': 'info',
                        'title': f'Top {param_type.capitalize()} Values',
                        'data': top_str,
                        'description': f'Most frequent {param_type} values in your dataset.'
                    })
                else:
                    insights.append({
                        'type': 'info',
                        'title': f'Top {param_type.capitalize()} Values',
                        'data': 'No specific data found or too diverse to summarize.',
                        'description': f'Could not determine top values for {param_type}.'
                    })
            else:
                insights.append({
                    'type': 'info',
                    'title': f'{param_type.capitalize()} Data',
                    'data': f"'{col_name}' column found for {param_type}.",
                    'description': f'Data related to {param_type} is present in your dataset.'
                })


    return insights

def generate_text_report(analysis_results, sensitive_params, target_gender, age_min, age_max):
    """Generate a text report of the analysis results"""
    report = []
    
    try:
        report.append("=" * 60)
        report.append("TARGETING ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("TARGET AUDIENCE CRITERIA")
        report.append("-" * 60)
        report.append(f"Gender: {'Both' if target_gender == 'both' else target_gender.capitalize()}")
        report.append(f"Age Range: {age_min}-{age_max}")
        report.append("")
        
        report.append("SUMMARY METRICS")
        report.append("-" * 60)
        report.append(f"Total Records: {analysis_results['total_records']:,}")
        report.append(f"Matched Records: {analysis_results['matched_records']:,}")
        report.append(f"Overall Match Score: {analysis_results['match_score']}%")
        report.append(f"Gender Match Rate: {analysis_results['gender_match']:.1f}%")
        report.append(f"Age Match Rate: {analysis_results['age_match']:.1f}%")
        report.append("")
        
        report.append("DETAILED INSIGHTS")
        report.append("-" * 60)
        insights = generate_insights(analysis_results, sensitive_params, target_gender, age_min, age_max)
        for insight in insights:
            report.append(f"{insight['title'].upper()}")
            report.append(f"  - {insight['data']}")
            if 'description' in insight:
                report.append(f"  - {insight['description']}")
            report.append("")
        
        report.append("PARAMETER STATISTICS")
        report.append("-" * 60)
        for param, stats in analysis_results['parameter_stats'].items():
            report.append(f"{param.upper()}")
            if param == 'age' and 'stats' in stats and stats['stats']['count'] > 0:
                report.append(f"  - Min: {stats['stats']['min']}")
                report.append(f"  - Max: {stats['stats']['max']}")
                report.append(f"  - Avg: {stats['stats']['avg']}")
            elif isinstance(stats, dict) and stats:
                for value, count in stats.items():
                    report.append(f"  - {value}: {count}")
            else:
                report.append(f"  - No data for {param}")
            report.append("")
        
        return "\n".join(report)
    except Exception as e:
        st.error(f"Error generating text report: {e}")
        return "Error generating report."

def clean_text_for_pdf(text):
    """
    Ensures text is suitable for FPDF's latin-1 encoding.
    Replaces common problematic Unicode characters and handles encoding errors.
    Also breaks long words to prevent FPDF layout issues.
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Replace common problematic Unicode characters with standard ASCII equivalents
    text = text.replace('\u2013', '-')  # en dash
    text = text.replace('\u2014', '--') # em dash
    text = text.replace('\u2018', "'").replace('\u2019', "'") # smart single quotes
    text = text.replace('\u201C', '"').replace('\u201D', '"') # smart double quotes
    text = text.replace('\u2026', '...') # ellipsis
    text = text.replace('\u00A0', ' ')  # non-breaking space
    text = text.replace('\u200B', '')  # zero width space
    text = text.replace('\uFEFF', '')  # byte order mark (BOM)

    # Break long words: insert a space every 70 characters for sequences of non-whitespace
    # This helps FPDF with word wrapping and prevents "Not enough horizontal space" errors
    long_word_pattern = r'(\S{70})(?=\S)' # Match 70 non-whitespace chars, if followed by another non-whitespace
    text = re.sub(long_word_pattern, r'\1 ', text)

    # Attempt to encode and decode to filter out unmappable characters
    # 'replace' will put '?' for unmappable chars, which is generally safe for PDF
    return text.encode('latin-1', errors='replace').decode('latin-1')


def generate_pdf_report_dataset(analysis_results, sensitive_params, target_gender, age_min, age_max):
    """Generate a PDF report of the analysis results for dataset."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    try:
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt=clean_text_for_pdf("TARGETING ANALYSIS REPORT"), ln=1, align='C')
        pdf.set_font("Arial", '', 10)
        pdf.cell(200, 10, txt=clean_text_for_pdf(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=1, align='C')
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=clean_text_for_pdf("Target Audience Criteria"), ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, txt=clean_text_for_pdf(f"Gender: {'Both' if target_gender == 'both' else target_gender.capitalize()}"), ln=1)
        pdf.cell(200, 10, txt=clean_text_for_pdf(f"Age Range: {age_min}-{age_max}"), ln=1)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=clean_text_for_pdf("Summary Metrics"), ln=1)
        pdf.set_font("Arial", '', 12)
        
        col_widths = [60, 60, 60]
        row_height = 10
        
        pdf.set_fill_color(200, 220, 255)
        pdf.cell(col_widths[0], row_height, clean_text_for_pdf("Metric"), border=1, fill=True)
        pdf.cell(col_widths[1], row_height, clean_text_for_pdf("Value"), border=1, fill=True)
        pdf.cell(col_widths[2], row_height, clean_text_for_pdf("Description"), border=1, fill=True, ln=1)
        
        metrics = [
            ("Total Records", f"{analysis_results['total_records']:,}", "Total number of records"),
            ("Matched Records", f"{analysis_results['matched_records']:,}", "Records matching both gender and age"),
            ("Match Score", f"{analysis_results['match_score']}%", "Overall match score (0-100)"),
            ("Gender Match", f"{analysis_results['gender_match']:.1f}%", "Percentage matching gender"),
            ("Age Match", f"{analysis_results['age_match']:.1f}%", "Percentage matching age range")
        ]
        
        pdf.set_fill_color(245, 245, 245)
        fill = False
        for metric in metrics:
            pdf.cell(col_widths[0], row_height, clean_text_for_pdf(metric[0]), border=1, fill=fill)
            pdf.cell(col_widths[1], row_height, clean_text_for_pdf(metric[1]), border=1, fill=fill)
            pdf.cell(col_widths[2], row_height, clean_text_for_pdf(metric[2]), border=1, fill=True, ln=1)
            fill = not fill
        
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=clean_text_for_pdf("Detailed Insights"), ln=1)
        pdf.set_font("Arial", '', 12)
        
        insights = generate_insights(analysis_results, sensitive_params, target_gender, age_min, age_max)
        for insight in insights:
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(200, 10, txt=clean_text_for_pdf(insight['title']), ln=1)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf(insight['data']))
            if 'description' in insight:
                pdf.set_font("Arial", 'I', 10)
                pdf.multi_cell(0, 10, txt=clean_text_for_pdf(insight['description']))
            pdf.ln(5)
        
        from io import BytesIO
        buffer = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        buffer.write(pdf_bytes)
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF report: {e}")
        return None

def extract_text_from_image(image_bytes):
    """Extracts text from an image using EasyOCR"""
    if reader is None:
        return ""
    try:
        with Image.open(io.BytesIO(image_bytes)).convert("RGB") as image:
            image_np = np.array(image)
            results = reader.readtext(image_np, detail=0)
            return " ".join(results)
    except UnidentifiedImageError:
        return ""
    except Exception as e:
        st.error(f"An error occurred during text extraction (OCR): {e}. This image might be corrupted or in an unexpected format for OCR processing.")
        return ""

def analyze_image_for_visual_context(image_bytes, extracted_text=""):
    """Analyzes image visual context for potential biases"""
    visual_bias_categories = {
        "racial": [], "religious": [], "gender": [], "age": [],
        "nationality": [], "sexuality": [], "socioeconomic": [],
        "educational": [], "disability": [], "political": [],
        "physical": [], "general_visual_context_flag": []
    }
    visual_bias_score = 0
    generated_caption = ""
    all_visual_flags = []

    if image_captioner is None:
        return {
            "generated_caption": "Image captioning model not available.",
            "visual_flags": ["Visual analysis skipped due to model loading issue."],
            "bias_categories": {},
            "is_visually_biased": False,
            "visual_bias_score": 0
        }

    try:
        with Image.open(io.BytesIO(image_bytes)).convert("RGB") as image:
            caption_results = image_captioner(image)
            if caption_results and caption_results[0] and 'generated_text' in caption_results[0]:
                generated_caption = caption_results[0]['generated_text']
                st.write(f"**Generated Image Caption:** `{generated_caption}`")

                caption_bias_analysis = analyze_text_for_bias(generated_caption, is_caption=True)

                for category, flags_data in caption_bias_analysis['bias_categories'].items():
                    if flags_data:
                        for flag_item in flags_data:
                            prefixed_message = f"(Caption-derived) {flag_item['message']}"
                            flag_score = flag_item['score']
                            
                            if category not in visual_bias_categories:
                                visual_bias_categories[category] = []

                            visual_bias_categories[category].append({'message': prefixed_message, 'score': flag_score})
                            all_visual_flags.append(prefixed_message)

                visual_bias_score += caption_bias_analysis['bias_score']

                caption_lower = generated_caption.lower()
                text_lower = extracted_text.lower()

                if ('man' in caption_lower or 'person' in caption_lower or 'people' in caption_lower) and \
                   ('suit' in caption_lower) and \
                   ('struggling' in text_lower or 'money' in text_lower or 'financial freedom' in text_lower):
                    flag_message = "Potential racial/socio-economic stereotype implied by contrasting individuals in a financial narrative. **Requires Human Review.**"
                    flag_score = 5
                    if flag_message not in [f['message'] for f in visual_bias_categories.get("racial", [])]:
                        visual_bias_categories["racial"].append({'message': flag_message, 'score': flag_score})
                        visual_bias_categories["socioeconomic"].append({'message': flag_message, 'score': flag_score})
                        all_visual_flags.append(flag_message)
                        visual_bias_score += flag_score

                if ('woman' in caption_lower or 'female' in caption_lower) and \
                   ('car' in caption_lower or 'vehicle' in caption_lower) and \
                   ('safety' in text_lower or 'safe' in text_lower or 'protection' in text_lower or 'confidence starts with feeling safe' in text_lower) and \
                   ('especially for women' in text_lower or 'for women' in text_lower):
                    flag_message = "Potential benevolent sexism/vulnerability stereotype (e.g., implying women need special safety). **Requires Human Review.**"
                    flag_score = 4
                    if flag_message not in [f['message'] for f in visual_bias_categories.get("gender", [])]:
                        visual_bias_categories["gender"].append({'message': flag_message, 'score': flag_score})
                        all_visual_flags.append(flag_message)
                        visual_bias_score += flag_score

                active_sport_keywords = ['skateboard', 'skateboarding', 'sport', 'active', 'run', 'jump', 'play']
                if ('person' in caption_lower or 'man' in caption_lower or 'people' in caption_lower) and \
                   any(keyword in text_lower for keyword in active_sport_keywords) and \
                   ('sitting' in caption_lower or 'seated' in caption_lower or 'wheelchair' in caption_lower or 'bench' in caption_lower):
                    flag_message = "Potential ableism/exclusion in active sport context (e.g., person in wheelchair with active sport ad). **Requires Human Review.**"
                    flag_score = 4
                    if flag_message not in [f['message'] for f in visual_bias_categories.get("disability", [])]:
                        visual_bias_categories["disability"].append({'message': flag_message, 'score': flag_score})
                        all_visual_flags.append(flag_message)
                        visual_bias_score += flag_score

            else:
                st.warning("Could not generate a descriptive caption for the image.")

    except UnidentifiedImageError:
        st.error("Error: The uploaded file is not a recognized image format or is corrupted for visual analysis.")
        all_visual_flags.append("Error during visual analysis: Unrecognized image format or corrupted file.")
    except Exception as e:
        st.error(f"An error occurred during visual context analysis: {e}. This image might be corrupted or in an unexpected format for the image captioning model.")
        all_visual_flags.append(f"Error during visual analysis: {e}")

    filtered_bias_categories = {k: v for k, v in visual_bias_categories.items() if v}

    return {
        "generated_caption": generated_caption,
        "visual_flags": all_visual_flags,
        "bias_categories": filtered_bias_categories,
        "is_visually_biased": len(all_visual_flags) > 0,
        "visual_bias_score": min(visual_bias_score, 10)
    }

def analyze_text_for_bias(text, is_caption=False):
    """Analyzes text for potential bias using NLP model and heuristics"""
    bias_categories = {
        "racial": [], "religious": [], "gender": [], "age": [],
        "nationality": [], "sexuality": [], "socioeconomic": [],
        "educational": [], "disability": [], "political": [],
        "physical": [], "general_heuristic_flag": []
    }
    suggestions = []
    bias_score = 0
    all_flags_messages = []
    text_lower = text.lower()
    
    highlighted_text = text

    if text_bias_classifier and text.strip():
        try:
            classifier_results = text_bias_classifier(text)
            if classifier_results and classifier_results[0]:
                for result in classifier_results[0]:
                    label = result['label']
                    score = result['score']
                    if score > 0.5:
                        flag_message = f"{label.replace('_', ' ').title()} bias detected with confidence score - {score:.2f}."
                        if label not in bias_categories:
                            bias_categories[label] = []
                        bias_categories[label].append({'message': flag_message, 'score': score * 3})
                        all_flags_messages.append(flag_message)
                        bias_score += int(score * 3)

                        if label in ["gender", "age", "racial", "religious", "nationality", "sexuality", "socioeconomic", "disability", "physical"]:
                            suggestions.append(f"Review for '{label.replace('_', ' ').title()}' bias. Ensure inclusive language and representation.")
                        elif label in ["educational", "political"]:
                            suggestions.append(f"Review for '{label.replace('_', ' ').title()}' bias. Ensure neutrality or appropriate context.")

        except Exception as e:
            if is_caption:
                st.warning(f"Error running text bias classifier on caption '{text[:50]}...': {e}. Falling back to keyword analysis for this segment.")
            else:
                st.warning(f"Error running text bias classifier on '{text[:50]}...': {e}. Falling back to keyword analysis for this segment.")
    else:
        if text.strip():
            st.info("Text bias classifier not available. Using keyword-based detection only.")

    def apply_highlight_and_flag(original_text, keywords, category, suggestion_text, score_add):
        nonlocal highlighted_text, bias_score
        flags_added_count = 0
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, original_text.lower()):
                if f"<mark>{keyword}</mark>" not in highlighted_text.lower():
                    highlighted_text = re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", highlighted_text, flags=re.IGNORECASE)
                
                flag_message = f"Keyword-based '{category.replace('_', ' ').title()}' bias detected: '{keyword}'."
                if flag_message not in all_flags_messages:
                    if category not in bias_categories:
                        bias_categories[category] = []
                    bias_categories[category].append({'message': flag_message, 'score': score_add})
                    all_flags_messages.append(flag_message)
                    suggestions.append(suggestion_text)
                    bias_score += score_add
                    flags_added_count += 1
        return flags_added_count

    gender_centric_phrases = ['he or she', 'him or her', 'manpower', 'workman', 'cameraman', 'foreman', 'stewardess', 'actress', 'women know best']
    apply_highlight_and_flag(text, gender_centric_phrases, "gender", "Consider using gender-neutral terms like 'human resources', 'workforce', 'camera operator', 'supervisor', 'flight attendant', 'actor', or avoid gender-specific claims.", 1)

    age_stereotypes_old = ['slowed down', 'fragile', 'past their prime', 'tech-illiterate', 'elderly (when used negatively)']
    apply_highlight_and_flag(text, age_stereotypes_old, "age", "Avoid ageist stereotypes. Focus on capabilities and value for all ages.", 2)
    age_stereotypes_young = ['inexperienced', 'entitled', 'attention span of a gnat', 'millennial (when used negatively)', 'gen z (when used negatively)']
    apply_highlight_and_flag(text, age_stereotypes_young, "age", "Avoid ageist stereotypes. Focus on potential and value for all ages.", 2)

    if (('woman' in text_lower or 'female' in text_lower) and ('homemaker' in text_lower or 'nurturing' in text_lower or 'delicate' in text_lower)) or \
       (('man' in text_lower or 'male' in text_lower) and ('breadwinner' in text_lower or 'dominant' in text_lower or 'strongest' in text_lower)):
        flag_message = "Reinforces traditional gender roles or attributes."
        flag_score = 2
        if flag_message not in all_flags_messages:
            bias_categories["gender"].append({'message': flag_message, 'score': flag_score})
            all_flags_messages.append(flag_message)
            suggestions.append("Avoid reinforcing traditional gender stereotypes. Focus on universal qualities or professional skills.")
            bias_score += flag_score

    benevolent_sexism_phrases = ['needs protection', 'fragile', 'delicate', 'sensitive needs', 'best for her']
    if ('woman' in text_lower or 'female' in text_lower) and any(kw in text_lower for kw in benevolent_sexism_phrases):
        flag_message = "Implies women need special protection or are inherently vulnerable (benevolent sexism). **Requires Human Review.**"
        flag_score = 4
        if flag_message not in all_flags_messages:
            bias_categories["gender"].append({'message': flag_message, 'score': flag_score})
            all_flags_messages.append(flag_message)
            suggestions.append("Ensure safety messages are universal or focus on features, not gender-specific vulnerability. Confidence should stem from internal agency.")
            bias_score += flag_score

    problematic_contextual_phrases_and_cats = [
        (['primitive'], 'culture', "racial", 3), (['backward'], 'society', "racial", 3), (['exotic'], 'people', "racial", 3),
        (['ghetto'], 'neighborhood', "socioeconomic", 4), (['struggling'], 'community', "socioeconomic", 4), (['poverty', 'stricken'], 'community', "socioeconomic", 4),
        (['disabled'], 'unable', "disability", 4), (['handicap'], 'limitations', "disability", 4), (['wheelchair'], 'confined', "disability", 4),
        (['blind'], 'helpless', "disability", 4), (['deaf'], 'impaired', "disability", 4),
        (['fair', 'lovely'], 'skin', "racial", 5), (['fairness', 'cream'], 'skin', "racial", 5), (['skin', 'lightening'], 'cream', "physical", 5), (['brighten', 'skin'], 'cream', "physical", 5)
    ]
    for terms, context_keyword, category, score_add in problematic_contextual_phrases_and_cats:
        if any(term in text_lower for term in terms) and context_keyword in text_lower:
            flag_message = f"Use of problematic term(s) '{', '.join(terms)}' in '{context_keyword}' context. Requires urgent human review."
            if flag_message not in all_flags_messages:
                if category == "racial" or category == "physical":
                    if "racial" not in bias_categories: bias_categories["racial"] = []
                    if "physical" not in bias_categories: bias_categories["physical"] = []
                    bias_categories["racial"].append({'message': flag_message, 'score': score_add})
                    bias_categories["physical"].append({'message': flag_message, 'score': score_add}) 
                else:
                    if category not in bias_categories: bias_categories[category] = []
                    bias_categories[category].append({'message': flag_message, 'score': score_add})
                all_flags_messages.append(flag_message)
                suggestions.append("Review language for any unintended racial, cultural, religious, socio-economic, or disability-related stereotypes/insensitivities. Be cautious with terms implying superiority based on skin tone.")
                bias_score += score_add
                for term_to_highlight in terms:
                    pattern = r'\b' + re.escape(term_to_highlight) + r'\b'
                    if f"<mark>{term_to_highlight}</mark>" not in highlighted_text.lower():
                        highlighted_text = re.sub(pattern, lambda m: f"<mark>{m.group(0)}</mark>", highlighted_text, flags=re.IGNORECASE)

    filtered_bias_categories = {k: v for k, v in bias_categories.items() if v}

    return {
        "is_biased": len(all_flags_messages) > 0,
        "bias_categories": filtered_bias_categories,
        "suggestions": list(set(suggestions)),
        "bias_score": min(bias_score, 10),
        "highlighted_text": highlighted_text
    }

def check_for_pii(text):
    """Checks text for common patterns of Personal Identifiable Information (PII)"""
    pii_found = []

    emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if emails:
        pii_found.extend([f"Email: {e}" for e in emails])

    phone_numbers = re.findall(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', text)
    if phone_numbers:
        pii_found.extend([f"Phone: {p}" for p in phone_numbers])

    cc_numbers = re.findall(r'\b(?:\d[ -]*?){13,16}\b', text)
    cc_numbers = [cc for cc in cc_numbers if re.match(r'^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|6(?:011|5[0-9]{2})[0-9]{12}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|(?:2131|1800|35\d{3})\d{11})$', cc.replace(" ", "").replace("-", ""))]
    if cc_numbers:
        pii_found.extend([f"Credit Card (potential): {c}" for c in cc_numbers])
    
    ssn_numbers = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text)
    if ssn_numbers:
        pii_found.extend([f"SSN (potential): {s}" for s in ssn_numbers])

    return {
        "has_pii": len(pii_found) > 0,
        "detected_items": pii_found
    }

def check_for_harmful_content(text):
    """Checks text for overtly harmful, offensive, or manipulative content"""
    harmful_terms_direct = [
        'kill', 'murder', 'assault', 'bomb', 'terrorist', 'exploit', 'manipulate',
        'deceive', 'fraud', 'illegal scam', 'cheat', 'offensive', 'slur', 'discriminat',
        'hate crime', 'violence', 'racist', 'sexist', 'pedophile', 'child abuse', 'nazi',
        'supremacy', 'genocide', 'terrorize', 'threaten', 'extort', 'harass', 'bully',
        'illegal drugs', 'weapon sales', 'human trafficking', 'suicide', 'self-harm',
        'explicit sexual content', 'graphic violence', 'incite hatred', 'provoke violence'
    ]
    detected_harm = [term for term in harmful_terms_direct if term in text.lower()]

    return {
        "has_harmful_content": len(detected_harm) > 0,
        "detected_items": detected_harm
    }

def calculate_overall_bias_score(text_bias_score, visual_bias_score):
    """Calculates an overall bias score based on textual and visual analysis"""
    overall_score = text_bias_score + visual_bias_score
    normalized_score = min(overall_score, 10)
    return normalized_score

def create_gauge_chart(title, score, max_score=10):
    """Creates a gauge chart for bias scores (for Streamlit display and PDF embedding)"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, max_score]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_score*0.3], 'color': "green"},
                {'range': [max_score*0.3, max_score*0.7], 'color': "yellow"},
                {'range': [max_score*0.7, max_score], 'color': "red"}],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score}
        }
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20), height=250)
    return fig

def generate_image_analysis_pdf_report(
    image_file_name,
    extracted_text,
    text_bias_results,
    visual_analysis_results,
    pii_results,
    harmful_results,
    overall_ad_status,
    overall_bias_score,
    uploaded_image_bytes,
    ollama_insights="N/A"
):
    """Generates a PDF report for image advertisement analysis."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12) 

    try:
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt=clean_text_for_pdf("IMAGE ADVERTISEMENT ANALYSIS REPORT"), ln=1, align='C')
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, txt=clean_text_for_pdf(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=1, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Key Takeaways"), ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 7, txt=clean_text_for_pdf(f"Overall Ad Status: {overall_ad_status}"))
        pdf.multi_cell(0, 7, txt=clean_text_for_pdf(f"Combined Bias Score: {overall_bias_score}/10"))
        pdf.ln(5)

        if uploaded_image_bytes:
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt=clean_text_for_pdf("Uploaded Ad Image"), ln=1)
            pdf.ln(2)
            try:
                img_buffer = io.BytesIO(uploaded_image_bytes)
                with Image.open(io.BytesIO(uploaded_image_bytes)) as img:
                    img_width, img_height = img.size
                
                max_pdf_width = pdf.w - 20
                max_pdf_height = pdf.h - pdf.get_y() - 20

                aspect_ratio = img_width / img_height
                new_width = min(max_pdf_width, max_pdf_height * aspect_ratio)
                new_height = new_width / aspect_ratio

                if pdf.get_y() + new_height > pdf.h - 10:
                    pdf.add_page()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img_file:
                    Image.open(img_buffer).save(tmp_img_file.name, format='PNG')
                    tmp_img_file_path = tmp_img_file.name

                pdf.image(tmp_img_file_path, x=10, w=new_width, h=new_height)
                pdf.ln(new_height + 5)
                os.unlink(tmp_img_file_path)
            except Exception as e:
                pdf.multi_cell(0, 7, txt=clean_text_for_pdf(f"Could not embed uploaded image: {e}"))
            pdf.ln(5)

        pdf.ln(5)
        pdf.set_x(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Insights & Recommendations (from LLaVA)"), ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 7, txt=clean_text_for_pdf(ollama_insights))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Extracted Text (from OCR)"), ln=1)
        pdf.set_font("Arial", '', 12)
        if extracted_text:
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf(extracted_text))
        else:
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("No text could be extracted from the image."))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Textual Bias Analysis"), ln=1)
        pdf.set_font("Arial", '', 12)
        
        text_bias_severity_scores = {} 

        if text_bias_results and text_bias_results.get('is_biased', False):
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("Potential Textual Bias Detected!"))
            for category, flags_data in text_bias_results.get('bias_categories', {}).items():
                if flags_data:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt=clean_text_for_pdf(f"- {category.replace('_', ' ').title()} Bias:"), ln=1)
                    pdf.set_font("Arial", '', 12)
                    for flag_item in flags_data:
                        pdf.multi_cell(0, 7, txt=clean_text_for_pdf(f"  - {flag_item['message']}"))
            
            text_bias_severity_scores = {
                category.replace('_', ' ').title(): sum(item['score'] for item in flags_data)
                for category, flags_data in text_bias_results['bias_categories'].items()
            }
            if text_bias_severity_scores:
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, txt=clean_text_for_pdf("Textual Bias Breakdown by Severity:"), ln=1)
                pdf.ln(2)
                
                chart_width = 90
                chart_height = 60
                
                col_counter = 0
                for i, (bias_type, score) in enumerate(text_bias_severity_scores.items()):
                    fig_gauge = create_gauge_chart(bias_type, score)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        fig_gauge.write_image(tmp_file.name, format='png', width=400, height=250, scale=2)
                        tmp_file_path = tmp_file.name
                    
                    if col_counter == 0:
                        pdf.set_x(10)
                    
                    pdf.image(tmp_file_path, x=pdf.get_x(), w=chart_width, h=chart_height)
                    pdf.set_x(pdf.get_x() + chart_width + 5)
                    col_counter +=1
                    
                    if col_counter >= 2 or (i + 1) == len(text_bias_severity_scores):
                        pdf.ln(chart_height + 5)
                        col_counter = 0
                    
                    os.unlink(tmp_file_path)

        else:
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("No significant textual bias detected."))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Visual Context Analysis"), ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, txt=clean_text_for_pdf(f"Generated Image Caption: {visual_analysis_results.get('generated_caption', 'N/A') }"))
        
        visual_bias_severity_scores = {}

        if visual_analysis_results and visual_analysis_results.get('is_visually_biased', False):
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("Potential Visual Bias Detected!"))
            for category, flags_data in visual_analysis_results.get('bias_categories', {}).items():
                if flags_data:
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(0, 10, txt=clean_text_for_pdf(f"- {category.replace('_', ' ').title()} Bias:"), ln=1)
                    pdf.set_font("Arial", '', 12)
                    for flag_item in flags_data:
                        pdf.multi_cell(0, 7, txt=clean_text_for_pdf(f"  - {flag_item['message']}"))

            visual_bias_severity_scores = {
                category.replace('_', ' ').title(): sum(item['score'] for item in flags_data)
                for category, flags_data in visual_analysis_results['bias_categories'].items()
            }
            if visual_bias_severity_scores:
                pdf.ln(5)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, txt=clean_text_for_pdf("Visual Bias Breakdown by Severity:"), ln=1)
                pdf.ln(2)
                
                chart_width = 90
                chart_height = 60
                
                col_counter = 0
                for i, (bias_type, score) in enumerate(visual_bias_severity_scores.items()):
                    fig_gauge = create_gauge_chart(bias_type, score)
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                        fig_gauge.write_image(tmp_file.name, format='png', width=400, height=250, scale=2)
                        tmp_file_path = tmp_file.name
                    
                    if col_counter == 0:
                        pdf.set_x(10)
                    
                    pdf.image(tmp_file_path, x=pdf.get_x(), w=chart_width, h=chart_height)
                    pdf.set_x(pdf.get_x() + chart_width + 5)
                    col_counter +=1
                    
                    if col_counter >= 2 or (i + 1) == len(visual_bias_severity_scores):
                        pdf.ln(chart_height + 5)
                        col_counter = 0
                    
                    os.unlink(tmp_file_path)

        else:
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("No significant visual stereotypical context detected."))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Security Suggestions"), ln=1)
        pdf.set_font("Arial", '', 12)
        if pii_results and pii_results.get('has_pii', False):
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf(f"PII Detected! Found: {', '.join(pii_results.get('detected_items', []))}. This information should be redacted."))
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("Action: Remove or redact sensitive personal information before public display."))
        else:
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("No Personal Identifiable Information (PII) found."))
        pdf.ln(2)

        if harmful_results and harmful_results.get('has_harmful_content', False):
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf(f"Harmful Content Detected! Found: {', '.join(harmful_results.get('detected_items', []))}. Review for offensive or manipulative language."))
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("Action: Remove or rephrase offensive/manipulative language to maintain a positive brand image and ethical standards."))
        else:
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("No direct harmful content detected by keyword check."))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Heuristic-Based Recommendations"), ln=1)
        pdf.set_font("Arial", '', 12)
        visual_flags_messages = [flag['message'] for flag in visual_analysis_results.get('visual_flags', []) if isinstance(flag, dict)]
        all_suggestions = list(set(text_bias_results.get('suggestions', []) + visual_flags_messages))

        if all_suggestions or (pii_results and pii_results.get('has_pii', False)) or (harmful_results and harmful_results.get('has_harmful_content', False)):
            if all_suggestions:
                for suggestion in all_suggestions:
                    pdf.multi_cell(0, 7, txt=clean_text_for_pdf(f"- Bias Suggestion: {suggestion}"))
            if (pii_results and pii_results.get('has_pii', False)) or (harmful_results and harmful_results.get('has_harmful_content', False)):
                pdf.multi_cell(0, 7, txt=clean_text_for_pdf("- Review security findings above for specific actions."))
        else:
            pdf.multi_cell(0, 10, txt=clean_text_for_pdf("No specific recommendations needed for this ad based on heuristic analysis."))
        pdf.ln(5)

        if pdf.get_y() > 200: pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Important Considerations & Limitations"), ln=1)
        pdf.set_font("Arial", '', 10)
        limitations_text = """
        * AI Estimation: This tool uses AI models to estimate biases and risks based on learned patterns from vast datasets. It is not a substitute for human review and expert judgment.
        * Image Analysis (Caption-based Inference): The image analysis component works by first generating a descriptive caption of the image using a dedicated image captioning model. This caption is then fed into a natural language processing (NLP) model to infer potential biases. Its accuracy is dependent on the quality of the generated caption and the NLP model's ability to interpret visual context from text.
        * Keyword-based Detection: Some bias and harmful content detection relies on keyword matching, which may lead to false positives or miss nuanced issues. Human review is always recommended.
        * Context Matters: An ad's impact is highly dependent on its specific context (e.g., where it's placed, current events, cultural nuances). This tool provides a general guide.
        * Data Bias: AI models can inherit biases present in their training data. Results might reflect these biases.
        * Input Quality: The accuracy of the analysis heavily relies on the quality and clarity of the uploaded image and text.
        """
        for paragraph in limitations_text.strip().split('\n*'):
            if paragraph.strip():
                if pdf.get_y() + pdf.font_size * 3 > pdf.h - 10:
                    pdf.add_page()
                pdf.multi_cell(0, 7, txt=clean_text_for_pdf(paragraph.strip()))
                pdf.ln(2)
        pdf.ln(5)
        
        from io import BytesIO
        buffer = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        buffer.write(pdf_bytes)
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF report for Ad Analyzer: {e}")
        return None

# --- Functions for Ad Demographic Analyzer ---
DEMOGRAPHIC_GROUPS = {
    "Age Groups": [
        "kids (0-12 years)", "adolescents (13-18 years)", "young adults (19-35 years)",
        "adults (36-60 years)", "elderly (60+ years)"
    ],
    "Gender & Sexuality": [
        "male", "female", "non-binary", "straight people", "LGBTQ+ people"
    ],
    "Generations": [
        "Gen Z", "Millennials", "Gen X", "Baby Boomers", "Silent Generation"
    ],
    "Location & Lifestyle": [
        "urban population", "rural population", "suburban population",
        "students", "professionals", "stay-at-home parents", "retirees"
    ],
    "Socio-economic Status": [
        "high income", "middle income", "low income", "luxury consumers", "budget-conscious"
    ],
    "Interests & Hobbies": [
        "pet owners", "fitness enthusiasts", "gamers", "tech-savvy",
        "fashion-conscious", "foodies", "travelers", "book lovers", "outdoor adventurers"
    ]
}

ALL_DEMOGRAPHICS = [item for sublist in DEMOGRAPHIC_GROUPS.values() for item in sublist]

def preprocess_image_appeal(image):
    """Preprocesses an image for the appeal.py image model (ResNet)."""
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = image.convert("RGB")
    img = preprocess(img)
    img = img.unsqueeze(0)
    return img

def analyze_image_for_demographic_appeal_via_caption(image_bytes, captioner_pipeline, text_classifier_pipeline):
    """
    Analyzes image for demographic appeal by first generating a caption
    and then using zero-shot text classification on the caption.
    """
    demographic_scores = {dem: 0.0 for dem in ALL_DEMOGRAPHICS}

    if captioner_pipeline is None or text_classifier_pipeline is None:
        st.warning("Models not available for demographic appeal analysis.")
        return demographic_scores

    try:
        with Image.open(io.BytesIO(image_bytes)).convert("RGB") as image:
            caption_results = captioner_pipeline(image)
            generated_caption = ""
            if caption_results and caption_results[0] and 'generated_text' in caption_results[0]:
                generated_caption = caption_results[0]['generated_text']
                st.info(f"Generated Image Caption for Demographic Analysis: `{generated_caption}`")
            else:
                st.warning("Could not generate a descriptive caption for the image. Image-based demographic analysis will be limited.")
                return demographic_scores

        if generated_caption:
            classification_results = text_classifier_pipeline(generated_caption, ALL_DEMOGRAPHICS)
            
            for label, score in zip(classification_results['labels'], classification_results['scores']):
                demographic_scores[label] = score

            caption_lower = generated_caption.lower()

            if "child" in caption_lower or "kid" in caption_lower or "baby" in caption_lower:
                demographic_scores["kids (0-12 years)"] += 0.1
            if "teen" in caption_lower or "student" in caption_lower or "college" in caption_lower:
                demographic_scores["adolescents (13-18 years)"] += 0.05
                demographic_scores["young adults (19-35 years)"] += 0.05
                demographic_scores["students"] += 0.1
            if "elderly" in caption_lower or "senior" in caption_lower or "retired" in caption_lower:
                demographic_scores["elderly (60+ years)"] += 0.1
                demographic_scores["retirees"] += 0.1
            if "suit" in caption_lower or "office" in caption_lower or "business" in caption_lower:
                demographic_scores["professionals"] += 0.1
                demographic_scores["high income"] += 0.05
            if "gym" in caption_lower or "workout" in caption_lower or "sport" in caption_lower or "run" in caption_lower:
                demographic_scores["fitness enthusiasts"] += 0.1
                demographic_scores["outdoor adventurers"] += 0.05
            if "luxury" in caption_lower or "expensive" in caption_lower or "high-end" in caption_lower or "designer" in caption_lower:
                demographic_scores["luxury consumers"] += 0.15
                demographic_scores["high income"] += 0.1

            for dem in demographic_scores:
                demographic_scores[dem] = max(0.0, min(1.0, demographic_scores[dem]))

            total_score = sum(demographic_scores.values())
            if total_score > 0:
                normalized_scores = {k: v / total_score for k, v in demographic_scores.items()}
            else:
                normalized_scores = demographic_scores

            sorted_results = dict(sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True))
            return sorted_results

    except UnidentifiedImageError:
        st.error("Error: The uploaded file is not a recognized image format or is corrupted for image captioning.")
        return {dem: 0.0 for dem in ALL_DEMOGRAPHICS}
    except Exception as e:
        st.error(f"An error occurred during image-based demographic analysis: {e}")
        return {dem: 0.0 for dem in ALL_DEMOGRAPHICS}

def analyze_hashtags_appeal(hashtags, classifier):
    """Analyzes hashtags for demographic appeal."""
    try:
        hashtag_list = re.findall(r'#(\w+)', hashtags)

        if not hashtag_list:
            return {dem: 0.0 for dem in ALL_DEMOGRAPHICS}

        text_to_classify = " ".join(hashtag_list)

        result = classifier(text_to_classify, ALL_DEMOGRAPHICS)

        scores = {label: score for label, score in zip(result['labels'], result['scores'])}

        total_score = sum(scores.values())
        if total_score > 0:
            normalized_scores = {k: v / total_score for k, v in scores.items()}
        else:
            normalized_scores = scores

        sorted_scores = dict(sorted(normalized_scores.items(), key=lambda item: item[1], reverse=True))
        return sorted_scores

    except Exception as e:
        st.error(f"Error analyzing hashtags: {str(e)}")
        return {dem: 0.0 for dem in ALL_DEMOGRAPHICS}

def combine_results_appeal(image_results, text_results, image_weight=0.5, text_weight=0.5):
    """Combines image and text analysis results with weighting."""
    combined = {dem: 0.0 for dem in ALL_DEMOGRAPHICS}

    total_input_weight = image_weight + text_weight
    if total_input_weight == 0:
        return combined

    normalized_image_weight = image_weight / total_input_weight
    normalized_text_weight = text_weight / total_input_weight

    for dem in ALL_DEMOGRAPHICS:
        image_score = image_results.get(dem, 0.0) * normalized_image_weight
        text_score = text_results.get(dem, 0.0) * normalized_text_weight
        combined[dem] = image_score + text_score

    total_combined_score = sum(combined.values())
    if total_combined_score > 0:
        normalized_combined = {k: v / total_combined_score for k, v in combined.items()}
    else:
        normalized_combined = combined

    sorted_combined = dict(sorted(normalized_combined.items(), key=lambda item: item[1], reverse=True))
    return sorted_combined

def get_popular_hashtags_for_demographics(demographics_list):
    """
    Provides a curated list of popular hashtags for given demographic groups.
    This is a static mapping for quick lookup.
    """
    popular_hashtags_map = {
        "kids (0-12 years)": ["#KidsFun", "#Playtime", "#FamilyAdventures", "#ChildrensFashion", "#KidFriendly"],
        "adolescents (13-18 years)": ["#TeenLife", "#YouthCulture", "#GenZ", "#HighSchool", "#StudentLife"],
        "young adults (19-35 years)": ["#Millennials", "#Adulting", "#CareerGoals", "#TravelLife", "#FitnessJourney"],
        "adults (36-60 years)": ["#FamilyGoals", "#HomeLife", "#CareerGrowth", "#Wellness", "#ParentingTips"],
        "elderly (60+ years)": ["#Seniors", "#RetirementLife", "#GoldenYears", "#ActiveAging", "#Grandparents"],
        "male": ["#MensFashion", "#GamerLife", "#TechGadgets", "#SportsFan", "#AdventureTime"],
        "female": ["#WomensStyle", "#BeautyTips", "#Fashionista", "#SelfCare", "#EmpowerWomen"],
        "non-binary": ["#NonBinary", "#GenderNeutral", "#Inclusive", "#QueerJoy", "#BeyondBinary"],
        "straight people": ["#RelationshipGoals", "#LoveStory", "#CoupleGoals", "#FamilyLove", "#DatingLife"], 
        "LGBTQ+ people": ["#Pride", "#LGBTQCommunity", "#Queer", "#LoveIsLove", "#Equality"],
        "Gen Z": ["#GenZLife", "#TikTokTrends", "#YouthCulture", "#DigitalNative", "#FutureIsNow"],
        "Millennials": ["#MillennialLife", "#Adulting", "#CoffeeLover", "#Throwback", "#WorkLifeBalance"],
        "Gen X": ["#GenX", "#80sKid", "#RetroVibes", "#ClassicRock", "#ParentingHumor"],
        "Baby Boomers": ["#BabyBoomer", "#Nostalgia", "#VintageLife", "#Wisdom", "#TimelessStyle"],
        "Silent Generation": ["#HistoryLover", "#ClassicMovies", "#TimelessElegance", "#FamilyHeritage", "#Tradition"],
        "urban population": ["#CityLife", "#UrbanExploration", "#StreetPhotography", "#Cityscape", "#Downtown"],
        "rural population": ["#CountryLife", "#FarmLife", "#NatureLover", "#RuralLiving", "#SmallTown"],
        "suburban population": ["#SuburbanLife", "#FamilyFriendly", "#CommunityVibes", "#BackyardFun", "#Neighborhood"],
        "students": ["#StudentLife", "#StudyMotivation", "#CampusLife", "#CollegeLife", "#ExamSeason"],
        "professionals": ["#CareerGoals", "#WorkLife", "#BusinessTips", "#LinkedIn", "#ProfessionalDevelopment"],
        "stay-at-home parents": ["#SAHMLife", "#MomLife", "#DadLife", "#ParentingHacks", "#KidsActivities"],
        "retirees": ["#RetirementGoals", "#TravelMore", "#Hobbies", "#Relaxation", "#NewAdventures"],
        "high income": ["#LuxuryLifestyle", "#Investment", "#WealthManagement", "#HighEnd", "#Exclusive"],
        "middle income": ["#SmartSpending", "#FamilyBudget", "#ValueForMoney", "#HomeImprovement", "#EverydayLife"],
        "low income": ["#BudgetFriendly", "#SavingMoney", "#CommunitySupport", "#AffordableLiving", "#EssentialNeeds"],
        "luxury consumers": ["#LuxuryBrand", "#HauteCouture", "#FineLiving", "#ExclusiveAccess", "#PremiumQuality"],
        "budget-conscious": ["#BudgetFriendly", "#SavvyShopper", "#DealsAndSteals", "#AffordableFashion", "#SmartChoices"],
        "single": ["#SingleLife", "#DatingTips", "#SelfLove", "#Independent", "#SoloTravel"],
        "married": ["#MarriedLife", "#CoupleGoals", "#WeddingAnniversary", "#RelationshipAdvice", "#ForeverLove"],
        "parents": ["#Parenting", "#MomAndDad", "#FamilyFirst", "#KidsActivities", "#ParentingCommunity"],
        "single parents": ["#SingleParent", "#SuperMom", "#SuperDad", "#ParentingJourney", "#Strength"],
        "empty nesters": ["#EmptyNester", "#NewAdventures", "#TravelGoals", "#RediscoverYourself", "#Freedom"],
        "pet owners": ["#PetLover", "#DogsofInstagram", "#CatsofInstagram", "#PetCare", "#AnimalLover"],
        "fitness enthusiasts": ["#FitnessMotivation", "#Workout", "#GymLife", "#HealthyLifestyle", "#FitFam"],
        "gamers": ["#GamingLife", "#Gamer", "#Esports", "#VideoGames", "#TwitchStreamer"],
        "tech-savvy": ["#TechGadgets", "#Innovation", "#FutureTech", "#AI", "#CodingLife"],
        "fashion-conscious": ["#FashionStyle", "#OOTD", "#StyleInspo", "#TrendSetter", "#Fashionista"],
        "foodies": ["#Foodie", "#InstaFood", "#FoodPhotography", "#Delicious", "#EatingOut"],
        "travelers": ["#Wanderlust", "#TravelGram", "#Explore", "#AdventureTime", "#BucketList"],
        "book lovers": ["#Bookworm", "#ReadingCommunity", "#Bookstagram", "#Literary", "#GoodReads"],
        "outdoor adventurers": ["#OutdoorLife", "#Adventure", "#Hiking", "#NatureLover", "#ExploreMore"]
    }

    results = {}
    for dem in demographics_list:
        if dem in popular_hashtags_map:
            results[dem] = popular_hashtags_map[dem]
    return results


def generate_demographic_pdf_report(
    combined_results, 
    image_caption, 
    top_appealing_demographics, 
    uploaded_image_bytes=None,
    ollama_insights="N/A"
):
    """Generates a PDF report for Ad Demographic Analysis."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    try:
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, txt=clean_text_for_pdf("AD DEMOGRAPHIC ANALYSIS REPORT"), ln=1, align='C')
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, txt=clean_text_for_pdf(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), ln=1, align='C')
        pdf.ln(10)

        # Ensure clean slate before writing multi-line content
        pdf.ln(5)
        pdf.set_x(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Actionable Insights & Recommendations (from LLaVA)"), ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 7, txt=clean_text_for_pdf(ollama_insights))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Image Content Summary"), ln=1)
        pdf.set_font("Arial", '', 12)
        pdf.multi_cell(0, 10, txt=clean_text_for_pdf(f"Generated Caption: {image_caption if image_caption else 'N/A'}"))
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Overall Demographic Appeal"), ln=1)
        pdf.set_font("Arial", '', 12)
        
        if combined_results:
            sorted_demographics = sorted(combined_results.items(), key=lambda item: item[1], reverse=True)
            
            pdf.set_font("Arial", 'U', 12)
            pdf.cell(0, 10, txt=clean_text_for_pdf("Top 5 Appealing Demographics (Overall Ad Content):"), ln=1)
            pdf.set_font("Arial", '', 12)
            if sorted_demographics:
                for i, (dem, score) in enumerate(sorted_demographics[:5]):
                    pdf.cell(0, 7, txt=clean_text_for_pdf(f"{i+1}. {dem}: {score*100:.1f}%"), ln=1)
            else:
                pdf.cell(0, 7, txt=clean_text_for_pdf("No prominent demographic appeal data available."), ln=1)
            pdf.ln(5)

        else:
            pdf.cell(0, 10, txt=clean_text_for_pdf("No combined demographic analysis results available."), ln=1)
            pdf.ln(5)

        if uploaded_image_bytes:
            if pdf.get_y() > 150: pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, txt=clean_text_for_pdf("Uploaded Ad Image"), ln=1)
            pdf.ln(2)
            try:
                img_buffer = io.BytesIO(uploaded_image_bytes)
                with Image.open(io.BytesIO(uploaded_image_bytes)) as img:
                    img_width, img_height = img.size
                
                max_pdf_width = pdf.w - 20
                max_pdf_height = pdf.h - pdf.get_y() - 20

                aspect_ratio = img_width / img_height
                new_width = min(max_pdf_width, max_pdf_height * aspect_ratio, 180) # Max width
                new_height = new_width / aspect_ratio

                if pdf.get_y() + new_height > pdf.h - 10:
                    pdf.add_page()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img_file:
                    Image.open(img_buffer).save(tmp_img_file.name, format='PNG')
                    tmp_img_file_path = tmp_img_file.name

                pdf.image(tmp_img_file_path, x=10, w=new_width, h=new_height)
                pdf.ln(new_height + 5)
                os.unlink(tmp_img_file_path)
            except Exception as e:
                pdf.multi_cell(0, 7, txt=clean_text_for_pdf(f"Could not embed uploaded image: {e}"))
            pdf.ln(5)

        if pdf.get_y() > 180: pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Detailed Appeal by Demographic Group"), ln=1)
        pdf.ln(2)
        
        col_counter = 0
        for i, (group_name, demographics_list) in enumerate(DEMOGRAPHIC_GROUPS.items()):
            
            group_specific_scores = {}
            for dem in demographics_list:
                if combined_results.get(dem, 0) > 0.001: 
                    group_specific_scores[dem] = combined_results.get(dem, 0)

            total_group_score = sum(group_specific_scores.values())
            if total_group_score > 0:
                normalized_group_scores = {k: v / total_group_score for k, v in group_specific_scores.items()}
            else:
                normalized_group_scores = {}

            if normalized_group_scores:
                if pdf.get_y() > 200: 
                    pdf.add_page()
                    col_counter = 0

                pdf.set_font("Arial", 'B', 12)
                pdf.cell(0, 10, txt=clean_text_for_pdf(f"Appeal within {group_name}"), ln=1)
                pdf.set_font("Arial", '', 10)

                fig_pie, ax_pie = plt.subplots(figsize=(6, 6)) 
                labels = [clean_text_for_pdf(k) for k, v in normalized_group_scores.items()]
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

                ax_pie.set_title(clean_text_for_pdf(f'Distribution within {group_name}'), fontsize=12)
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    fig_pie.savefig(tmp_file.name, format='png', bbox_inches='tight', dpi=150)
                    tmp_file_path = tmp_file.name
                plt.close(fig_pie)

                chart_pdf_width = 90
                chart_pdf_height = chart_pdf_width

                if col_counter == 0:
                    pdf.set_x(10)
                
                pdf.image(tmp_file_path, x=pdf.get_x(), w=chart_pdf_width, h=chart_pdf_height)
                pdf.set_x(pdf.get_x() + chart_pdf_width + 5)
                col_counter += 1

                if col_counter >= 2 or (i + 1) == len(DEMOGRAPHIC_GROUPS):
                    pdf.ln(chart_pdf_height + 10)
                    col_counter = 0
                
                os.unlink(tmp_file_path)
        
        pdf.ln(10) 
        if pdf.get_y() > 180: pdf.add_page()
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, txt=clean_text_for_pdf("Overall Highest Appealing Individual Demographics"), ln=1)
        pdf.ln(2)

        top_individual_demographics = {k: v for k, v in combined_results.items() if v > 0.005}
        if top_individual_demographics:
            top_individual_demographics_sorted = dict(sorted(top_individual_demographics.items(), key=lambda item: item[1], reverse=True)[:10])

            fig_bar, ax_bar = plt.subplots(figsize=(10, 6)) 
            demographics_labels = [clean_text_for_pdf(k) for k in top_individual_demographics_sorted.keys()]
            scores = list(top_individual_demographics_sorted.values())

            bars = ax_bar.barh(demographics_labels, scores, color='lightcoral')
            ax_bar.set_xlabel('Appeal Score (Normalized)', fontsize=12)
            ax_bar.set_title(clean_text_for_pdf('Overall Highest Appealing Individual Demographics'), fontsize=16)
            ax_bar.invert_yaxis()

            for bar in bars:
                width = bar.get_width()
                ax_bar.text(width, bar.get_y() + bar.get_height()/2, f'{width*100:.1f}%',
                            ha='left', va='center', fontsize=9, color='black')
            
            ax_bar.xaxis.set_tick_params(labelbottom=True)
            plt.tight_layout()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                fig_bar.savefig(tmp_file.name, format='png', bbox_inches='tight', dpi=150)
                tmp_file_path = tmp_file.name
            plt.close(fig_bar)

            chart_pdf_width_full = pdf.w - 20
            chart_pdf_height_full = chart_pdf_width_full * (6/10)

            if pdf.get_y() + chart_pdf_height_full > pdf.h - 10:
                pdf.add_page()

            pdf.image(tmp_file_path, x=10, w=chart_pdf_width_full, h=chart_pdf_height_full)
            pdf.ln(chart_pdf_height_full + 5)
            os.unlink(tmp_file_path)
        else:
            pdf.multi_cell(0, 7, txt=clean_text_for_pdf("No significant individual demographic appeal detected across all categories."))
        pdf.ln(5)

        buffer = io.BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin1')
        buffer.write(pdf_bytes)
        buffer.seek(0)
        
        return buffer
    except Exception as e:
        st.error(f"Error generating PDF report for Demographic Analyzer: {e}")
        return None