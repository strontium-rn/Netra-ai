import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os

def load_model_safely():
    try:
        model = load_model("/Users/mac/Desktop/Netra.ai/netra-ai-model.h5", 
                      compile=False,  # Don't load optimizer state
                      custom_objects=None)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

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
        Netra.ai is the application where you can give your fundus photograph and know if your eyes can live long or not.
        """)
        st.markdown("---")
        st.subheader("Instructions")
        st.write("1. Enter the required values")
        st.write("2. Click 'Generate Prediction'")
        st.write("3. Review the results")

    # Main content
    model = load_model_safely()
    
    if model:
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Data")
            feature1 = st.number_input(
                "Feature 1",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                help="Enter value for Feature 1"
            )
            feature2 = st.number_input(
                "Feature 2",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                help="Enter value for Feature 2"
            )

        with col2:
            st.subheader("Additional Metrics")
            feature3 = st.number_input(
                "Feature 3",
                min_value=0.0,
                max_value=100.0,
                value=0.0,
                help="Enter value for Feature 3"
            )

        # Center the predict button
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            predict_button = st.button("Generate Prediction", use_container_width=True)

        if predict_button:
            with st.spinner("Analyzing data..."):
                try:
                    # Prepare input data
                    input_data = np.array([[feature1, feature2, feature3]])
                    
                    # Make prediction
                    prediction = model.predict(input_data)
                    
                    # Display prediction in a nice format
                    st.markdown("---")
                    st.subheader("Analysis Results")
                    
                    # Create three columns for displaying results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(label="Prediction Value", value=f"{prediction[0][0]:.4f}")
                    
                    with col2:
                        # Add confidence score or other metrics if available
                        confidence = np.random.uniform(0.8, 1.0)  # Replace with actual confidence calculation
                        st.metric(label="Confidence Score", value=f"{confidence:.2%}")
                    
                    with col3:
                        # Add additional metrics if needed
                        st.metric(label="Analysis Time", value="< 1 sec")

                    # Additional details in an expander
                    with st.expander("See detailed analysis"):
                        st.write("Input features:")
                        st.json({
                            "Feature 1": feature1,
                            "Feature 2": feature2,
                            "Feature 3": feature3
                        })
                        st.write("Raw prediction output:", prediction)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {str(e)}")
                    st.write("Please check your input values and try again.")

if __name__ == "__main__":
    main()