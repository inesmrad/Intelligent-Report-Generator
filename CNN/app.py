from flask import Flask, request, jsonify
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# LOAD BRAND MODEL + LABEL ENCODER + EMBEDDINGS
# ------------------------------------------------------------
brand_model = load_model('brand_cnn_model.h5')

with open('brand_encoder.pkl', 'rb') as f:
    le_brand = pickle.load(f)

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# ------------------------------------------------------------
# GET LAST CONV LAYER AS EMBEDDING LAYER
# ------------------------------------------------------------
last_conv_layer = None
for layer in reversed(brand_model.layers):
    if 'conv' in layer.name:
        last_conv_layer = layer
        break

embedding_model = Model(
    inputs=brand_model.input,
    outputs=last_conv_layer.output
)

# ------------------------------------------------------------
# IMAGE UPLOAD FOLDER
# ------------------------------------------------------------
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------------------------------
# PREDICT FUNCTION
# ------------------------------------------------------------
def predict_product(img_path):
    # Load + preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    # Step 1 — Brand prediction
    brand_pred = brand_model.predict(x)
    brand_idx = np.argmax(brand_pred)
    brand_name = le_brand.inverse_transform([brand_idx])[0]

    # Step 2 — Embedding
    conv_output = embedding_model.predict(x)  # Should be (1, H, W, C)
    img_emb = np.mean(conv_output, axis=(1, 2)).reshape(1, -1)

    # Step 3 — Filter products by brand
    brand_products = [name for name in embeddings.keys() if name.startswith(brand_name)]

    if len(brand_products) == 0:
        return brand_name, "Unknown product"

    brand_embeddings = np.array([embeddings[name] for name in brand_products])

    # Step 4 — Similarity
    similarities = cosine_similarity(img_emb, brand_embeddings).flatten()
    best_idx = np.argmax(similarities)

    predicted_product = brand_products[best_idx]
    return brand_name, predicted_product


# ------------------------------------------------------------
# FLASK API
# ------------------------------------------------------------
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    # FIX 1 — filename fallback
    filename = file.filename or "uploaded_image.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        brand, product = predict_product(file_path)
        return jsonify({'brand': brand, 'product': product})
    except Exception as e:
        return jsonify({'error': f"Backend crashed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
