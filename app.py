import streamlit as st
import pickle
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine
from PIL import Image
import os

# Load saved face embeddings from file
model_path = 'model/poopy-not-poopy.pkl'
with open(model_path, 'rb') as f:
    saved_embeddings = pickle.load(f)

def preprocess_image(image_path):
    """
    Preprocess the image to ensure it has consistent dimensions and format.
    Args:
    - image_path (str): Path to the image file.

    Returns:
    - preprocessed_image (np.array): Resized and normalized image.
    """
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))  # Resize to 224x224 pixels
    return np.array(image)

def get_embedding(image_path):
    """
    Extract face embedding from an uploaded image using DeepFace.
    Args:
    - image_path (str): Path to the image file.

    Returns:
    - embedding (np.array): Face embedding of the image, or None if there's an error or no face is detected.
    """
    try:
        # Use DeepFace to extract face embeddings
        result = DeepFace.represent(img_path=image_path, model_name='VGG-Face', enforce_detection=False)
        if result:
            embeddings = np.array(result[0]['embedding'])
            return embeddings
        return None
    except Exception as e:
        print(f"Error getting embedding for image {image_path}: {e}")
        return None

def find_best_match(embedding):
    """
    Find the best match for the uploaded image embedding from the saved embeddings.
    Args:
    - embedding (np.array): Face embedding of the uploaded image.

    Returns:
    - best_match (str): Name of the person with the closest matching embedding.
    - min_distance (float): Distance between the uploaded image embedding and the closest match.
    """
    min_distance = float('inf')
    best_match = None
    
    # Ensure the embedding is a flat array
    embedding = np.ravel(embedding)
    
    for person, embeddings in saved_embeddings.items():
        for saved_embedding in embeddings:
            saved_embedding = np.ravel(saved_embedding)
            # Check if shapes match before computing distance
            if embedding.shape[0] == saved_embedding.shape[0]:
                distance = cosine(embedding, saved_embedding)
                if distance < min_distance:
                    min_distance = distance
                    best_match = person

    # Define a threshold for distance (adjust as needed)
    threshold = 0.6
    if min_distance > threshold:
        return "Unknown", min_distance
    
    # Special case handling
    if best_match == 'Prapty':
        best_match = 'Bubblegum Babu 🐣'
    
    return best_match, min_distance

# Streamlit app interface
st.title("Who are you madafakah!!! 🫵🏻")
st.title("Poopy or Not Poopy? 🤷🏻‍♀️")
st.write("Upload an image to recognize poopies")

# File uploader widget for image files
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
    

    # Save the uploaded image temporarily
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get embedding for the uploaded image
    uploaded_image_embedding = get_embedding(temp_path)
    
    if uploaded_image_embedding is not None:
        # Find the best match from saved embeddings
        best_match, distance = find_best_match(uploaded_image_embedding)
        confidence = max(0, 1 - distance) * 100
        if best_match == "Unknown":
            st.write(f"Image does not match any known faces. Confidence: {confidence:.2f}%")
        else:
            st.write(f"Best Match: {best_match}")
            st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.write("Could not recognize the face or no face detected.")

    # Clean up the temporary file
    os.remove(temp_path)
