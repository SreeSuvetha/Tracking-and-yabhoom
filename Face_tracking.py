import cv2
import numpy as np
import os
from sklearn.decomposition import PCA
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics.pairwise import cosine_similarity

# Constants
FACE_DETECTOR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
IMG_SIZE = (128, 128)
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
THRESHOLD = 0.8
NUM_TEMPLATE_FRAMES = 70

  # Number of frames to use as templates

# Load face detector
face_cascade = cv2.CascadeClassifier(FACE_DETECTOR_PATH)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, IMG_SIZE)
    face_normalized = face_resized / 255.0
    return face_normalized, (x, y, w, h)

def extract_hog_features(image):
    hog_features = hog(image, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_PIXELS_PER_CELL,
                       cells_per_block=HOG_CELLS_PER_BLOCK, block_norm='L2-Hys', visualize=False)
    return hog_features

def extract_lbp_features(image):
    lbp = local_binary_pattern(image, LBP_N_POINTS, LBP_RADIUS, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_POINTS + 3),
                               range=(0, LBP_N_POINTS + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist

def combine_features(hog_features, lbp_features):
    return np.hstack([hog_features, lbp_features])

def compute_mean_feature_vector(feature_vectors):
    return np.mean(feature_vectors, axis=0)

def compare_with_template(live_feature_vector, template_feature_vectors):
    match_found = False
    for template_vector in template_feature_vectors:
        similarity = cosine_similarity([live_feature_vector], [template_vector])[0][0]
        if similarity >= THRESHOLD:
            match_found = True
            break
    return match_found

def compute_fisherfaces(images):
    flattened_images = [img.flatten() for img in images]
    flattened_images = np.array(flattened_images)

    # Set n_components to the minimum of number of samples and number of features
    num_samples, num_features = flattened_images.shape
    n_components = min(num_samples, num_features, 50)  # Ensure n_components is within the permissible range

    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(flattened_images)
    return pca_features

# Initialize variables for template feature vectors and counter
template_feature_vectors = []
frame_counter = 0

# Initialize video capture for live video stream (e.g., webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    preprocessed_frame, face_coords = preprocess_image(frame)
    if preprocessed_frame is not None:
        hog_features = extract_hog_features(preprocessed_frame)
        lbp_features = extract_lbp_features(preprocessed_frame)
        live_feature_vector = combine_features(hog_features, lbp_features)
        
        # Increment frame counter
        frame_counter += 1

        # Store features of first few detected faces as templates
        if len(template_feature_vectors) < NUM_TEMPLATE_FRAMES:
            template_feature_vectors.append(live_feature_vector)

        # Compare with the template feature vectors
        else:
            match_found = compare_with_template(live_feature_vector, template_feature_vectors)
            if match_found:
                cv2.putText(frame, "Match found!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                x, y, w, h = face_coords
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow('Live Tracking', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Reset frame counter if all template frames have been captured
    if frame_counter == NUM_TEMPLATE_FRAMES:
        frame_counter = 0

cap.release()
cv2.destroyAllWindows()
