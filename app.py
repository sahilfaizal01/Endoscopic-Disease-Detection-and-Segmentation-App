import streamlit as st
from preprocess import classify_image, segment_image


# Streamlit app
st.title("Endoscopic Disease Detection and Segmentation App")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    display_width = st.slider("Choose image display width", 100, 1000, 500, step=50)

    # Display the uploaded image
    st.image(uploaded_file, caption="Original Image", use_column_width=False, width=display_width)

    # Make predictions
    segmentation_result = segment_image(uploaded_file)

    # Display the predicted class (replace this with your actual class prediction logic)
    predicted_class = classify_image(uploaded_file) # Replace with your actual class prediction logic
    st.write(f"Predicted Class: {predicted_class}")

    # Display the segmented image
    st.image('segmented.png', caption="Segmented Image.", use_column_width=False, width=display_width)