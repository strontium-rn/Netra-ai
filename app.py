import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os
from PIL import Image
import tensorflow as tf

def load_model_safely():
    try:
        model = load_model("/Users/mac/Desktop/Netra.ai/netra-ai-model.h5", 
                      compile=False,
                      custom_objects=None)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize image to match model's expected input size
    # Adjust size based on your model's requirements
    image = image.resize((224, 224))
    
    # Convert to array and preprocess
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Normalize pixel values
    img_array = img_array / 255.0
    
    return img_array

def main():
    # Page configuration
    st.set_page_config(
        page_title="Netra.ai Prediction System",
        page_icon="👁️",
        layout="wide"
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main { 
            padding: 2rem;
        }
        .stButton>button {
            width: 100%;
            margin-top: 20px;
        }
        .prediction-box {
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
            background-color: #f0f2f6;
        }
        .upload-section {
            border: 2px dashed #cccccc;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header section
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🔬 Netra.ai Analysis System")
        st.markdown("---")

    # Sidebar for additional information
    with st.sidebar:
        st.header("About")
        st.info("""
        Netra.ai analyzes fundus photographs to assess eye health and provide insights about potential conditions.
        """)
        st.markdown("---")
        st.subheader("Instructions")
        st.write("1. Upload your fundus photograph")
        st.write("2. Click 'Generate Prediction'")
        st.write("3. Review the analysis results")
        
        # Add information about accepted image formats
        st.markdown("---")
        st.subheader("Supported Formats")
        st.write("- JPEG/JPG")
        st.write("- PNG")
        st.write("- BMP")

    # Main content
    model = load_model_safely()
    
    if model:
        st.subheader("Upload Fundus Photograph")
        
        # Create upload section with preview
        uploaded_file = st.file_uploader(
            "Choose a fundus photograph...", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a clear, well-lit fundus photograph"
        )

        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Fundus Photograph", use_column_width=True)
            
            # Center the predict button
            col1, col2, col3 = st.columns([1,1,1])
            with col2:
                predict_button = st.button("Generate Prediction", use_container_width=True)

            if predict_button:
                with st.spinner("Analyzing fundus photograph..."):
                    try:
                        # Preprocess the image
                        processed_image = preprocess_image(image)
                        
                        # Make prediction
                        prediction = model.predict(processed_image)
                        
                        # Display prediction in a nice format
                        st.markdown("---")
                        st.subheader("Analysis Results")
                        
                        # Create three columns for displaying results
                        col1, col2, col3 = st.columns(3)
                        
                        # Assuming your model outputs class probabilities
                        # Modify these based on your model's actual output format
                        with col1:
                            predicted_class = np.argmax(prediction[0])
                            class_names = ["Healthy", "Mild DR", "Moderate DR", "Severe DR"]  # Update with your classes
                            st.metric(label="Diagnosis", value=class_names[predicted_class])
                        
                        with col2:
                            confidence = np.max(prediction[0]) * 100
                            st.metric(label="Confidence", value=f"{confidence:.2f}%")
                        
                        with col3:
                            st.metric(label="Analysis Time", value="< 1 sec")

                        # Additional details in an expander
                        with st.expander("See detailed analysis"):
                            st.write("Class probabilities:")
                            for i, prob in enumerate(prediction[0]):
                                st.write(f"{class_names[i]}: {prob*100:.2f}%")
                            
                            # Add any additional analysis details here
                            st.write("\nNote: This analysis is for screening purposes only. Please consult with an eye care professional for proper diagnosis.")

                    except Exception as e:
                        st.error(f"An error occurred during analysis: {str(e)}")
                        st.write("Please ensure you've uploaded a valid fundus photograph and try again.")

if __name__ == "__main__":
    main()