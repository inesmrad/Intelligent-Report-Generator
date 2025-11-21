import streamlit as st
from PIL import Image
import requests

st.set_page_config(page_title="Product Finder", page_icon="🛒", layout="wide")

# --- Initialize session state ---
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "report_generated" not in st.session_state:
    st.session_state.report_generated = False
if "report_data" not in st.session_state:
    st.session_state.report_data = None

# --- File Upload Section ---
if st.session_state.uploaded_file is None:
    uploaded_file = st.file_uploader("📤 Upload a product image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        st.rerun()

else:
    # --- Side-by-side layout ---
    left_col, right_col = st.columns([1, 1])

    # Left column: show uploaded image
    with left_col:
        image = Image.open(st.session_state.uploaded_file)
        st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

        if st.button("🔁 Upload Another Image"):
            st.session_state.uploaded_file = None
            st.session_state.report_generated = False
            st.session_state.report_data = None
            st.rerun()

    # Right column: report generation
    with right_col:
        st.markdown("### ⚙️ Product Identification")

        if not st.session_state.report_generated:
            if st.button("Identify Product"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Send file properly as a file-like object
                        response = requests.post(
                            "http://127.0.0.1:5000/predict",
                            files={
                                "file": (
                                    st.session_state.uploaded_file.name,
                                    st.session_state.uploaded_file.getvalue(),
                                    st.session_state.uploaded_file.type
                                )
                            }
                        )


                        if response.status_code == 200:
                            st.session_state.report_data = response.json()
                            st.session_state.report_generated = True
                        else:
                            st.error("❌ Server error: " + response.text)
                    except Exception as e:
                        st.error(f"❌ Could not connect to backend: {e}")

        # Show report if generated
        if st.session_state.report_generated and st.session_state.report_data:
            result = st.session_state.report_data
            st.markdown("### 🧠 Identified Product")
            st.markdown(f"**Brand:** {result['brand']}")
            st.markdown(f"**Product Name:** {result['product_name']}")
            st.markdown(f"**Similarity Score:** {result['similarity']:.3f}")
