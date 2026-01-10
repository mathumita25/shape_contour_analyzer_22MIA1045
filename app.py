import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(page_title="Shape & Contour Analyzer", layout="wide")
st.title(" Shape & Contour Analyzer 22MIA1045")

st.write("Upload an image to detect shapes, count objects, and calculate area & perimeter.")

# Upload image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    detected = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # ignore small noise
            continue

        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            shape = "Quadrilateral"
        elif len(approx) == 5:
            shape = "Pentagon"
        else:
            shape = "Circle"

        detected.append((shape, area, perimeter, cnt))

    return detected

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    results = detect_shapes(image_bgr)

    annotated = image_bgr.copy()
    for shape, area, perimeter, cnt in results:
        cv2.drawContours(annotated, [cnt], -1, (0, 255, 0), 2)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image_np, use_column_width=True)

    with col2:
        st.subheader("Detected Shapes")
        st.image(
            cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            use_column_width=True
        )

    st.subheader("Detection Results")
    st.write(f"**Total Objects Detected:** {len(results)}")

    for i, (shape, area, perimeter, _) in enumerate(results, start=1):
        st.write(
            f"{i}. **{shape}** | Area: {area:.2f} | Perimeter: {perimeter:.2f}"
        )
