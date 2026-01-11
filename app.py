import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

st.set_page_config(layout="wide")

# ---------------- WIX STYLE THEME ----------------
st.markdown("""
<style>

body {
    background-color: #f5f1ec;
}

.block-container {
    background-color: #f5f1ec;
    padding-top: 2rem;
}

.title {
    font-family: Georgia, serif;
    font-size: 64px;
    text-align: center;
    font-weight: 600;
    margin-bottom: 5px;
    color: #111;
}

.subtitle {
    text-align: center;
    letter-spacing: 6px;
    font-size: 13px;
    margin-bottom: 60px;
}

.upload-box {
    background: white;
    padding: 30px;
    border: 1px solid black;
    max-width: 800px;
    margin: auto;
    margin-bottom: 50px;
}

.image-frame {
    border: 2px solid black;
    background: white;
    padding: 15px;
    margin-bottom: 40px;
}

.results-title {
    text-align: center;
    font-family: Georgia, serif;
    font-size: 32px;
    margin-top: 40px;
    margin-bottom: 30px;
}

.result-row {
    background: white;
    border: 1px solid black;
    padding: 18px;
    margin-bottom: 15px;
    font-size: 18px;
}

footer, header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# ---------------- PAGE HEADER ----------------
st.markdown('<div class="subtitle">EVERYTHING IS PERSONAL. INCLUDING THIS ANALYZER.</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Shape Analyzer</div>', unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a shape image", ["png","jpg","jpeg"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MATH HELPERS ----------------
def angle(p1,p2,p0):
    d1 = p1-p0
    d2 = p2-p0
    return abs(np.degrees(np.arctan2(d1[1],d1[0]) - np.arctan2(d2[1],d2[0])))

def is_parallel(a,b):
    return abs(a-b) < 0.15

# ---------------- SHAPE CLASSIFIER ----------------
def classify(cnt):
    peri = cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,0.02*peri,True)
    v = len(approx)
    area = cv2.contourArea(cnt)
    if area < 800: return None

    if v > 6: return "Circle"
    if v == 3: return "Triangle"
    if v == 5: return "Pentagon"
    if v == 6: return "Hexagon"

    if v == 4:
        pts = approx.reshape(4,2)
        sides = [np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)]
        sides.sort()

        angles = []
        for i in range(4):
            angles.append(angle(pts[(i-1)%4], pts[(i+1)%4], pts[i]))

        right = sum(80<a<100 for a in angles)

        if right == 4 and abs(sides[0]-sides[3]) < 15:
            return "Square"
        if right == 4:
            return "Rectangle"

        def slope(p1,p2):
            if p2[0]-p1[0]==0: return 999
            return (p2[1]-p1[1])/(p2[0]-p1[0])

        s1=slope(pts[0],pts[1])
        s2=slope(pts[1],pts[2])
        s3=slope(pts[2],pts[3])
        s4=slope(pts[3],pts[0])

        if is_parallel(s1,s3) and is_parallel(s2,s4):
            return "Parallelogram"
        if is_parallel(s1,s3) or is_parallel(s2,s4):
            return "Trapezium"

        return "Quadrilateral"

    return "Unknown"

# ---------------- PROCESS IMAGE ----------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    image = np.array(img)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur,60,160)

    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges,kernel,2)
    edges = cv2.erode(edges,kernel,1)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = image.copy()
    results = []

    for cnt in contours:
        name = classify(cnt)
        if name:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt,True)
            results.append((name,area,peri))
            cv2.drawContours(out,[cnt],-1,(0,150,0),3)

    st.markdown('<div class="image-frame">', unsafe_allow_html=True)
    st.image(out, use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="results-title">Detection Results</div>', unsafe_allow_html=True)

    for i,(n,a,p) in enumerate(results,1):
        st.markdown(
            f'<div class="result-row">{i}. <b>{n}</b> — Area: {int(a)} | Perimeter: {int(p)}</div>',
            unsafe_allow_html=True
        )
