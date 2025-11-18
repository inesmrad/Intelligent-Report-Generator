import os
import pickle
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from flask import Flask, request, jsonify

# --------------------- CONFIG ---------------------
BASE_DIR = os.path.dirname(__file__)
EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'embeddings .pkl')
PRODUCTS_INDEX_PATH = os.path.join(BASE_DIR, 'products_index.csv')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
INPUT_SIZE = 300

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------- LOAD PRECOMPUTED EMBEDDINGS ---------------------
with open(EMBEDDINGS_PATH, 'rb') as f:
    embeddings = pickle.load(f)

products_df = pd.read_csv(PRODUCTS_INDEX_PATH)
print(f"Loaded {len(products_df)} products and embeddings.")

# --------------------- CNN EMBEDDING MODEL ---------------------
base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(INPUT_SIZE, INPUT_SIZE, 3))

def get_embedding(img_path):
    img = Image.open(img_path).convert('RGB').resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.expand_dims(preprocess_input(np.array(img, dtype=np.float32)), axis=0)
    emb = base_model.predict(arr, verbose=0)
    emb = np.mean(emb, axis=(1,2)).reshape(1, -1)  # Global Average Pooling
    return emb

# --------------------- PREDICTION FUNCTION ---------------------
def predict_product(img_path, top_k=1):
    img_emb = get_embedding(img_path)
    product_names = products_df['name'].tolist()
    product_embs = np.array([embeddings[name] for name in product_names])
    sims = cosine_similarity(img_emb, product_embs).flatten()
    top_idxs = sims.argsort()[::-1][:top_k]
    top_products = [(product_names[i], float(sims[i])) for i in top_idxs]
    top_brand = products_df.loc[products_df['name'] == top_products[0][0], 'brand'].values[0]
    return top_brand, top_products[0]

# --------------------- FLASK API ---------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    filename = file.filename or "uploaded_image.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    try:
        brand, top_product = predict_product(file_path)
        return jsonify({
            'brand': brand,
            'product_name': top_product[0],
            'similarity': top_product[1]
        })
    except Exception as e:
        return jsonify({'error': f"Backend crashed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
