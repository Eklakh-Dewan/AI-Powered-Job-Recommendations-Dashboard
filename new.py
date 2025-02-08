import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(page_title="Skin Cancer Detection", layout="wide")

class SkinCancerModel:
    def __init__(self):
        self.img_height = 224
        self.img_width = 224
        self.class_names = ['Actinic keratoses', 'Basal cell carcinoma', 
                           'Benign keratosis', 'Dermatofibroma', 
                           'Melanoma', 'Melanocytic nevi', 'Vascular lesions']

    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        img = image.resize((self.img_height, self.img_width))
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)

    def dummy_predict(self, img_array):
        """Dummy prediction function for demo purposes"""
        # Replace this with actual model prediction when you have the model
        return np.random.uniform(0, 1, len(self.class_names))

def main():
    st.title("Skin Cancer Detection System")
    st.markdown("""
    This application uses deep learning to detect different types of skin cancer from images.
    Currently running in demo mode with simulated predictions.
    """)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload & Predict", "About"])

    if page == "Upload & Predict":
        show_prediction_page()
    else:
        show_about_page()

def show_prediction_page():
    st.header("Upload Image for Analysis")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    model = SkinCancerModel()
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image"):
            with st.spinner("Processing image..."):
                # Preprocess image
                processed_img = model.preprocess_image(image)
                
                # Get predictions
                predictions = model.dummy_predict(processed_img)
                
                # Display results
                st.subheader("Analysis Results")
                
                # Create bar chart
                fig = px.bar(
                    x=model.class_names,
                    y=predictions,
                    title="Prediction Probabilities",
                    labels={'x': 'Condition', 'y': 'Probability'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig)
                
                # Show top 3 predictions
                top_3_idx = predictions.argsort()[-3:][::-1]
                st.subheader("Top 3 Predictions:")
                for idx in top_3_idx:
                    st.write(f"{model.class_names[idx]}: {predictions[idx]:.2%}")

def show_about_page():
    st.header("About This Project")
    st.markdown("""
    ### Overview
    This is a demonstration version of a skin cancer detection system. 
    
    ### How to Use
    1. Navigate to the "Upload & Predict" page
    2. Upload an image of a skin lesion
    3. Click "Analyze Image" to see the predictions
    
    ### Important Note
    This is a demo version running with simulated predictions. For actual medical diagnosis, 
    please consult with healthcare professionals.
    
    ### Dataset Information
    The full version of this system is trained on the HAM10000 dataset, which contains 
    dermatoscopic images of various skin lesions.
    
    ### Technologies Used
    - Streamlit
    - TensorFlow
    - Plotly
    - Python
    """)

if __name__ == "__main__":
    main()
