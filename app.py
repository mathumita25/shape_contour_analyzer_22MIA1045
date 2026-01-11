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
h1 { text-align:center; font-size:64px; }
.results-box {
    background:white;
    border:2px solid black;
    padding:20px;
    border-radius:6px;
}
.results-box * { color:black !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Shape Analyzer</h1>", unsafe_allow_html=True)
uploaded = st.file_uploader("", ["png","jpg","jpeg"])

# -------------------- Shape Classifier --------------------
def classify(approx, cnt):
    sides = len(approx)
    area = cv2.contourArea(cnt)
    if area < 1500:
        return None

    if sides == 3:
        return "Triangle"

    if sides == 4:
        x,y,w,h = cv2.boundingRect(approx)
        ar = w / float(h)

        hull = cv2.convexHull(cnt)
        solidity = area / cv2.contourArea(hull)

        if 0.95 < ar < 1.05:
            return "Square"
        if solidity < 0.9:
            return "Trapezium"
        if ar > 1.2 or ar < 0.8:
            return "Rectangle"
        return "Parallelogram"

    if sides == 5:
        return "Pentagon"

    if sides > 6:
        return "Circle"

    return None

# -------------------- Processing --------------------
if uploaded:
    img = np.array(Image.open(uploaded).convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    thresh = cv2.adaptiveThreshold(blur,255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,11,2)

    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shapes=[]
    counts = {
        "Square":0,"Rectangle":0,"Circle":0,
        "Triangle":0,"Trapezium":0,"Parallelogram":0,"Pentagon":0
    }

    for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c,0.02*peri,True)
        shape = classify(approx,c)

        if shape:
            counts[shape]+=1
            area=cv2.contourArea(c)

            shapes.append({
                "name":shape,
                "area":int(area),
                "perimeter":int(peri)
            })

            # draw contour
            cv2.drawContours(img,[c],-1,(0,255,0),3)

            # place label above shape
            x,y,w,h=cv2.boundingRect(c)
            cv2.putText(img,shape,(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,0),3)
            cv2.putText(img,shape,(x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),1)

    st.image(img,use_column_width=True)

    # -------------------- Results --------------------
    st.markdown("<h2 style='text-align:center'>Detection Results</h2>", unsafe_allow_html=True)
    st.markdown("<div class='results-box'>", unsafe_allow_html=True)

    for k,v in counts.items():
        if v>0:
            st.markdown(f"<p><strong>{k}:</strong> {v}</p>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    for i,s in enumerate(shapes):
        st.markdown(
            f"<p><strong>{i+1}. {s['name']}</strong> — Area: {s['area']} | Perimeter: {s['perimeter']}</p>",
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)
