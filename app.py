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
        pts = approx.reshape(4,2)
    
        # Compute side lengths
        def dist(a,b): 
            return np.linalg.norm(a-b)
    
        d1 = dist(pts[0], pts[1])
        d2 = dist(pts[1], pts[2])
        d3 = dist(pts[2], pts[3])
        d4 = dist(pts[3], pts[0])
    
        sides_equal = abs(d1-d2)<10 and abs(d2-d3)<10 and abs(d3-d4)<10
    
        # Compute angles
        def angle(a,b,c):
            ba = a - b
            bc = c - b
            cosang = np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc))
            return np.degrees(np.arccos(cosang))
    
        a1 = angle(pts[0],pts[1],pts[2])
        a2 = angle(pts[1],pts[2],pts[3])
        a3 = angle(pts[2],pts[3],pts[0])
        a4 = angle(pts[3],pts[0],pts[1])
    
        right_angles = all(80 < a < 100 for a in [a1,a2,a3,a4])
    
        # Check parallel sides using slopes
        def slope(p1,p2):
            return (p2[1]-p1[1])/(p2[0]-p1[0]+1e-5)
    
        s1 = slope(pts[0],pts[1])
        s2 = slope(pts[2],pts[3])
        s3 = slope(pts[1],pts[2])
        s4 = slope(pts[3],pts[0])
    
        parallel1 = abs(s1-s2) < 0.2
        parallel2 = abs(s3-s4) < 0.2
    
        if sides_equal and right_angles:
            return "Square"
        if right_angles:
            return "Rectangle"
        if parallel1 and parallel2:
            return "Parallelogram"
        if parallel1 or parallel2:
            return "Trapezium"
    

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
