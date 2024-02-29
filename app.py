from transformers import pipeline
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import random

pipe = pipeline(task='image-segmentation', model='nvidia/segformer-b0-finetuned-ade-512-512')

def segment_objects(output, img):
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result = np.zeros_like(img)
    for obj in output:
        label = obj['label']
        score = obj['score']
        mask = np.array(obj['mask'])
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            color = [random.randint(0, 255) for _ in range(3)]
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.drawContours(result, [cnt], -1, color, -1)  # Fill contour with color
                cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

    result = cv2.addWeighted(img, 0.5, result, 0.5, 0)  # Blend original and result for visibility
    return result

def save_uploaded_file(uploadedfile):
    with open(uploadedfile.name, "wb") as f:
        f.write(uploadedfile.getbuffer())
    return uploadedfile.name

def main():
    st.title("Image Segmentation")

    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'png', 'jpeg'])
    process = st.button("Process")

    if uploaded_file is not None and process:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        
        save_uploaded_file(uploaded_file)

        image = Image.open(uploaded_file)
        result = pipe(image)
        detected_objects = [i['label'] for i in result] # list of detected objects
        st.write("Detected Objects:", detected_objects)

        segmented_img = segment_objects(result, image)
        st.image(segmented_img, caption='Segmented Image.', use_column_width=True)
        
        # delete the saved image.

if __name__ == "__main__":
    main()