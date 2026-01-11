import streamlit as st
import cv2
import numpy as np
from PIL import Image
import math

# ---------- Custom CSS ----------
st.markdown("""
<style>
body {
    background-color: #f8f9fa;
}
.header-title {
    font-size: 42px;
    font-weight: bold;
    color: #222;
}
.header-sub {
    font-size: 18px;
    color: #444;
}
.upload-card {
    background: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
.result-card {
    background: white;
    padding: 15px;
    border-radius: 10px;
    margin-bottom: 12px;
    border-left: 5px solid #4a90e2;
}
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ---------- Page Header ----------
st.markdown('<div class="header-title">Shape & Contour Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="header-sub">Upload your image and detect shapes with area and perimeter</div>', unsafe_allow_html=True)
st.markdown("---")

# ---------- Upload Section ----------
with st.container():
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a shape image", ["png", "jpg", "jpeg"])
    st.markdown("</div>", unsafe_allow_html=True)

# ---------- Geometry Helpers ----------
def angle(pt1, pt2, pt0):
    dx1 = pt1[0] - pt0[0]; dy1 = pt1[1] - pt0[1]
    dx2 = pt2[0] - pt0[0]; dy2 = pt2[1] - pt0[1]
    return abs(math.degrees(math.atan2(dy1, dx1) - math.atan2(dy2, dx2)))

def is_parallel(l1, l2):
    return abs(l1 - l2) < 10

def shape_name(cnt):
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    v = len(approx); area = cv2.contourArea(cnt)
    if area < 700: return None
    if v > 6: return "Circle"
    if v == 3: return "Triangle"
    if v == 5: return "Pentagon"
    if v == 6: return "Hexagon"
    if v == 4:
        pts = approx.reshape(4,2)
        sides = []
        for i in range(4):
            p1 = pts[i]; p2 = pts[(i+1)%4]
            sides.append(np.linalg.norm(p1-p2))
        sides = sorted(sides)
        angles = []
        for i in range(4):
            p0 = pts[i]; p1 = pts[(i-1)%4]; p2 = pts[(i+1)%4]
            angles.append(angle(p1,p2,p0))
        right_angles = sum(80 < a < 100 for a in angles)
        if right_angles == 4 and abs(sides[0]-sides[3]) < 15: return "Square"
        if right_angles == 4: return "Rectangle"
        def slope(p1,p2):
            if p2[0]-p1[0]==0: return 999
            return (p2[1]-p1[1]) / (p2[0]-p1[0])
        s1= slope(pts[0],pts[1]); s2=slope(pts[1],pts[2])
        s3= slope(pts[2],pts[3]); s4=slope(pts[3],pts[0])
        if is_parallel(s1,s3) and is_parallel(s2,s4): return "Parallelogram"
        if is_parallel(s1,s3) or is_parallel(s2,s4): return "Trapezium"
        return "Quadrilateral"
    return "Unknown"

# ---------- Shape Detection ----------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    image = np.array(img)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = image.copy()
    results = []

    for cnt in contours:
        name = shape_name(cnt)
        if name:
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            results.append((name, area, peri))
            cv2.drawContours(output, [cnt], -1, (0,128,0), 3)

    st.image(output, use_column_width=True)

    st.markdown("### Detection Results")
    for i, (name, area, peri) in enumerate(results,1):
        st.markdown(f'<div class="result-card">{i}. **{name}** | Area: {int(area)} | Perimeter: {int(peri)}</div>', unsafe_allow_html=True)
