import numpy as np
from scipy.spatial import distance

def manhattan(v1, v2):
   
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sum(np.abs(v1-v2))
    return dist

def euclidean(v1, v2):
   
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.sqrt(np.sum(v1-v2)**2)
    return dist
    
def chebyshev(v1, v2):
    
    v1 = np.array(v1).astype('float')
    v2 = np.array(v2).astype('float')
    dist = np.max(np.abs(v1-v2))
    return dist

def canberra(v1, v2):
   
    
    return distance.canberra(v1, v2)

def retrieve_similar_image(features_db, query_features, distance, num_results):
    distances = []
    for instance in features_db:
        features, label, img_path = instance[ : -2], instance[-2], instance[-1]
        if distance == 'manhattan':
            dist = manhattan(query_features, features)
        if distance == 'euclidean':
            dist = euclidean(query_features, features)
        if distance == 'chebyshev':
            dist = chebyshev(query_features, features)
        if distance == 'canberra':
            dist = canberra(query_features, features)
        distances.append((img_path, dist, label))
    distances.sort(key=lambda x: x[1])
    return distances[ : num_results]
            
            # CSS pour personnaliser le style de l'application
st.markdown("""
<style>
    body, .main, .sidebar, .stApp {
        background-color: #1e1e1e; /* Fond gris foncé */
        color: #ffffff; /* Texte blanc */
        font-family: 'Arial', sans-serif; /* Police */
    }
    .stButton>button {
        background-color: #007bff; /* Couleur bleue */
        color: #ffffff;
        border: none;
        padding: 12px 24px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Couleur bleue plus foncée au survol */
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #2e2e2e; /* Fond gris */
        color: #ffffff; /* Texte blanc */
        border: 1px solid #666666;
        border-radius: 5px;
        padding: 12px;
    }
    .stSelectbox select {
        background-color: #2e2e2e;
        color: #ffffff;
        border: 1px solid #666666;
        border-radius: 5px;
        padding: 10px;
    }
    .stHeader {
        color: #007bff;
        text-align: center;
        padding: 20px;
        border-bottom: 2px solid #666666;
    }
    .gold-title {
        font-size: 36px; /* Taille de la police augmentée */
        background: -webkit-linear-gradient(#007bff, #0056b3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stImage {
        border: 2px solid #666666;
        border-radius: 10px;
        padding: 10px;
        background-color: #3e3e3e;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
    }
    .stSidebar {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stFooter {
        text-align: center;
        padding: 10px;
        background-color: #2e2e2e;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Configuration de l'application Streamlit
st.markdown('<h1 class="gold-title">Content-Based Image Retrieval</h1>', unsafe_allow_html=True)
st.write('This App retrieves images based on their content using GLCM and BiT descriptors.')

# Sidebar parameters
st.sidebar.header('Parameters')
num_images = st.sidebar.number_input('Number of similar images to display', min_value=1, max_value=10, value=5)
distance_measure = st.sidebar.selectbox('Select distance measure', ['Euclidean', 'Manhattan', 'Chebyshev', 'Canberra'])
descriptor_selected = st.sidebar.selectbox('Select descriptor', ['GLCM', 'BiT'])

# Load signatures from the database
signatures = load_signatures(descriptor_selected)

# Image upload section
st.header('Upload an Image')
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Extract features from the uploaded image based on the selected descriptor
    if descriptor_selected == 'GLCM':
        features = glcm("temp_image.png")
    elif descriptor_selected == 'BiT':
        features = bitdesc("temp_image.png")
    else:
        features = None
    
    if features is None:
        st.error("Error extracting features. Please try uploading a different image.")
    elif len(signatures) == 0:
        st.error("No features found in the database. Please ensure the dataset has been processed.")
    else:
        # Calculate similarities and retrieve similar images
        similar_images = calculate_similarity(signatures, features, distance_measure, num_images)
        
        # Display similar images
        if similar_images:
            st.header('Similar Images')
            for img_path in similar_images:
                img_abs_path = os.path.abspath(os.path.join('./Projet1_Dataset/Projet1_Dataset', img_path))
                if os.path.isfile(img_abs_path):
                    st.image(img_abs_path, caption=os.path.basename(img_abs_path), use_column_width=True)
                else:
                    st.warning(f"Cannot open image {img_path}. File not found.")
        else:
            st.info("No similar images found.")


