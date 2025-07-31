# streamlit_frontend.py
# Enhanced Fashion Chatbot Streamlit Frontend

import streamlit as st
import requests
import json
import base64
from PIL import Image
import io
import os
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Fashion Stylist AI",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .upload-section {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #f5f5f5;
        align-self: flex-start;
    }
    .recommendation-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 10px;
        margin: 5px;
        background-color: #f9f9f9;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
UPLOAD_DIR = "temp_uploads"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# Sidebar
with st.sidebar:
    st.title("👗 Fashion Stylist AI")
    st.markdown("---")
    
    # User ID display
    st.info(f"**User ID:** {st.session_state.user_id}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.session_state.uploaded_images = []
        st.rerun()
    
    st.markdown("---")
    
    # Recent uploads
    st.subheader("📸 Recent Uploads")
    if st.session_state.uploaded_images:
        for i, img_info in enumerate(st.session_state.uploaded_images[-3:]):
            with st.expander(f"Image {i+1}"):
                st.image(img_info["path"], width=200)
                st.caption(f"Uploaded: {img_info['timestamp']}")
    else:
        st.info("No images uploaded yet")
    
    st.markdown("---")
    
    # Settings
    st.subheader("⚙️ Settings")
    show_recommendations = st.checkbox("Show Recommendations", value=True)
    show_analysis_details = st.checkbox("Show Analysis Details", value=True)

# Main content area
st.title("👗 Fashion Stylist AI Assistant")
st.markdown("Upload images of outfits or ask fashion-related questions!")

# Function definitions
def save_uploaded_file(uploaded_file):
    """Save uploaded file and return path"""
    try:
        # Create unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{st.session_state.user_id}_{timestamp}_{uploaded_file.name}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None

def encode_image_to_base64(image_path):
    """Encode image to base64 for API transmission"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return f"data:image/jpeg;base64,{encoded_string}"
    except Exception as e:
        st.error(f"Error encoding image: {str(e)}")
        return None

def call_api(endpoint, data=None, files=None):
    """Make API call to backend"""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        
        if files:
            response = requests.post(url, data=data, files=files, timeout=30)
        else:
            response = requests.post(url, json=data, timeout=30)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to backend API. Please ensure the backend is running.")
        return None
    except requests.exceptions.Timeout:
        st.error("API request timed out. Please try again.")
        return None
    except Exception as e:
        st.error(f"Error calling API: {str(e)}")
        return None

def display_chat_message(message, is_user=True):
    """Display a chat message with proper styling"""
    with st.container():
        if is_user:
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {message}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>Fashion Stylist:</strong> {message}
            </div>
            """, unsafe_allow_html=True)

def display_recommendations(recommendations):
    """Display product recommendations"""
    if not recommendations:
        return
    
    st.subheader("💡 Personalized Recommendations")
    
    cols = st.columns(min(len(recommendations), 3))
    for i, rec in enumerate(recommendations[:3]):
        with cols[i]:
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>{rec.get('name', 'Fashion Item')}</h4>
                <p><strong>Brand:</strong> {rec.get('brand', 'Unknown')}</p>
                <p><strong>Price:</strong> ${rec.get('price', 0)}</p>
                <p><strong>Color:</strong> {rec.get('color', 'Various')}</p>
                <p>{rec.get('description', 'Stylish fashion item')}</p>
            </div>
            """, unsafe_allow_html=True)

def display_image_analysis(analysis):
    """Display image analysis results"""
    if not analysis or not show_analysis_details:
        return
    
    with st.expander("🔍 Detailed Image Analysis", expanded=False):
        if isinstance(analysis, dict):
            for key, value in analysis.items():
                if key != "error" and value:
                    st.subheader(key.replace('_', ' ').title())
                    if isinstance(value, (list, dict)):
                        st.json(value)
                    else:
                        st.write(value)

# Image upload section
st.subheader("📸 Upload Fashion Image")
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['png', 'jpg', 'jpeg'],
    help="Upload an image of an outfit or fashion item for analysis"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=250)
    
    with col2:
        st.write("**Image Details:**")
        st.write(f"- **Filename:** {uploaded_file.name}")
        st.write(f"- **Size:** {uploaded_file.size} bytes")
        st.write(f"- **Type:** {uploaded_file.type}")
        
        # Analyze button
        if st.button("🔍 Analyze Image", type="primary"):
            with st.spinner("Analyzing your fashion image..."):
                # Save uploaded file
                file_path = save_uploaded_file(uploaded_file)
                
                if file_path:
                    # Add to session state
                    st.session_state.uploaded_images.append({
                        "path": file_path,
                        "name": uploaded_file.name,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Encode image for API
                    encoded_image = encode_image_to_base64(file_path)
                    
                    if encoded_image:
                        # Call backend API for image analysis
                        response = call_api("analyze-image", {
                            "user_id": st.session_state.user_id,
                            "image_data": encoded_image,
                            "query": "Analyze this fashion image"
                        })
                        
                        if response and response.get("success"):
                            analysis = response.get("analysis", {})
                            
                            # Display analysis
                            st.success("✅ Image analyzed successfully!")
                            
                            # Add to chat history
                            st.session_state.messages.append({
                                "type": "image_upload",
                                "content": f"Uploaded and analyzed: {uploaded_file.name}",
                                "analysis": analysis,
                                "timestamp": datetime.now()
                            })
                            
                            # Display analysis results
                            display_image_analysis(analysis)
                            
                            # Show recommendations if available
                            if show_recommendations and response.get("recommendations"):
                                display_recommendations(response["recommendations"])

# Chat interface
st.subheader("💬 Chat with Fashion Stylist")

# Display chat history
for message in st.session_state.messages:
    if message["type"] == "user":
        display_chat_message(message["content"], is_user=True)
    elif message["type"] == "bot":
        display_chat_message(message["content"], is_user=False)
        
        # Display recommendations if available
        if show_recommendations and message.get("recommendations"):
            display_recommendations(message["recommendations"])
    elif message["type"] == "image_upload":
        st.info(f"📸 {message['content']}")
        if show_analysis_details and message.get("analysis"):
            display_image_analysis(message["analysis"])

# Chat input
user_input = st.chat_input("Ask me anything about fashion, styling, or your uploaded images...")

if user_input:
    # Add user message to chat
    st.session_state.messages.append({
        "type": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })
    
    # Display user message immediately
    display_chat_message(user_input, is_user=True)
    
    # Get bot response
    with st.spinner("Fashion Stylist is thinking..."):
        # Prepare context from recent image analysis
        image_context = None
        if st.session_state.uploaded_images:
            latest_image = st.session_state.uploaded_images[-1]
            # Get analysis from recent messages
            for msg in reversed(st.session_state.messages):
                if msg["type"] == "image_upload" and msg.get("analysis"):
                    image_context = msg["analysis"]
                    break
        
        # Call backend API for chat response
        response = call_api("chat", {
            "user_id": st.session_state.user_id,
            "message": user_input,
            "image_context": image_context
        })
        
        if response and response.get("success"):
            bot_response = response.get("answer", "I apologize, but I couldn't process your request.")
            recommendations = response.get("recommendations", [])
            
            # Add bot message to chat
            st.session_state.messages.append({
                "type": "bot",
                "content": bot_response,
                "recommendations": recommendations,
                "timestamp": datetime.now()
            })
            
            # Display bot response
            display_chat_message(bot_response, is_user=False)
            
            # Display recommendations
            if show_recommendations and recommendations:
                display_recommendations(recommendations)
        else:
            error_message = "I'm having trouble connecting to my fashion knowledge base. Please try again!"
            st.session_state.messages.append({
                "type": "bot",
                "content": error_message,
                "timestamp": datetime.now()
            })
            display_chat_message(error_message, is_user=False)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>👗 Fashion Stylist AI - Your Personal Style Assistant</p>
    <p>Upload images, ask questions, and get personalized fashion advice!</p>
</div>
""", unsafe_allow_html=True)

# Cleanup old temporary files (optional)
try:
    import glob
    import time
    
    # Remove files older than 1 hour
    cutoff_time = time.time() - 3600
    for file_path in glob.glob(os.path.join(UPLOAD_DIR, "*")):
        if os.path.getmtime(file_path) < cutoff_time:
            os.remove(file_path)
except:
    pass  # Ignore cleanup errors
