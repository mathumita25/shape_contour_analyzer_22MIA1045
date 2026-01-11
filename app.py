import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Shape Analyzer", layout="wide")

# -------------------- CSS --------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    background-color: #f7f2ea;
    font-family: 'Georgia', serif;
}

h1 {
    text-align: center;
    font-size: 64px;
    margin-bottom: 20px;
}

div[data-testid="stFileUploader"] {
    background: #1f1f1f;
    padding: 30px;
    border-radius: 10px;
    border: 2px solid black;
}

div[data-testid="stFileUploader"] * {
    color: white;
}

.results-box {
    background: white;
    border: 2px solid black;
    padding: 20px;
    border-radius: 6px;
    margin-top: 30px;
}

.results-box * {
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------- UI --------------------
st.markdown("<h1>Shape Analyzer</h1>", unsafe_allow_html=True)
uploaded = st.file_uploader("", ["png","jpg","jpeg"])

# -------------------- Shape Classifier --------------------
def classify(approx, cnt):
    sides = len(approx)
    area = cv2.contourArea(cnt)

    if area < 2000:
        return None

    peri = cv2.arcLength(cnt, True)

    if sides == 3:
        return "Triangle"

    if sides == 4:
        x,y,w,h = cv2.boundingRect(approx)
        ar = w / float(h)

        hull = cv2.convexHull(cnt)
        solidity = area / cv2.contourArea(hull)

        if 0.95 < ar < 1.05:
            return "Square"
        if solidity < 0.92:
            return "Trapezium"
        if ar > 1.2 or ar < 0.8:
            return "Rectangle"
        return "Parallelogram"

    if sides == 5:
        return "Pentagon"

    if sides == 6:
        return "Hexagon"

    if sides > 6:
        return "Circle"

    return None

# -------------------- Processing --------------------
if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    blur = cv2.GaussianBlur(gray, (5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,11,2)

    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes = []

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1500:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        shape = classify(approx, c)
        if shape:
            shapes.append({
                "name": shape,
                "area": int(area),
                "perimeter": int(peri)
            })

            cv2.drawContours(img,[c],-1,(0,255,0),3)

    st.image(img, use_column_width=True)

    st.markdown("<h2 style='text-align:center;'>Detection Results</h2>", unsafe_allow_html=True)
    st.markdown("<div class='results-box'>", unsafe_allow_html=True)

    for i,s in enumerate(shapes):
        st.markdown(f"<p><strong>{i+1}. {s['name']}</strong> — Area: {s['area']} | Perimeter: {s['perimeter']}</p>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
