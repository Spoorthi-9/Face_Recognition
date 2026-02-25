import cv2
import os
import numpy as np
import pyttsx3
import requests
import threading  # To keep camera smooth

# --- 1. SETTINGS ---


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trained_model.yml')

data_path = 'face_data'
names_list = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]

# Confidence threshold (tune this for your setup)
THRESHOLD = 70

# --- 2. BACKGROUND FUNCTIONS ---
def background_tasks(name, is_known):
    """Send Telegram message and voice greeting in a background thread"""
    # A. Telegram
    msg = f"✅ Access Granted: {name}" if is_known else "⚠️ ALERT: Unknown person detected!"
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    try:
        requests.get(url, params={"chat_id": CHAT_ID, "text": msg}, timeout=5)
    except:
        print("Telegram failed (check internet)")

    # B. Voice Greeting
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        text = f"Welcome {name}" if is_known else "Access denied"
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except:
        pass

# --- 3. MAIN CAMERA LOOP ---
cap = cv2.VideoCapture(0)
last_person = ""

print("📸 System Started. Camera should be smooth now.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        last_person = ""

    for (x, y, w, h) in faces:
        id_num, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        # print(f"DEBUG: ID={id_num}, Confidence={confidence}")  # Uncomment for debugging

        # Check if recognized or unknown
        if confidence < THRESHOLD and id_num < len(names_list):
            name = names_list[id_num]
            label = name
            color = (0, 255, 0)
            is_known = True
        else:
            name = "Unknown"
            label = "Unknown"
            color = (0, 0, 255)
            is_known = False

        # Trigger background tasks only if new person
        if last_person != name:
            print(f"Detecting: {name}...")
            threading.Thread(target=background_tasks, args=(name, is_known), daemon=True).start()
            last_person = name

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Smart Access Control', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
