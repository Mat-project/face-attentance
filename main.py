import cv2
import face_recognition
import os
import numpy as np
import smtplib
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

known_faces_dir = "known_faces"
attendance_file = "logs/attendance.csv"
known_encodings = []
known_names = []

# Load known faces
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
        encoding = face_recognition.face_encodings(img)[0]
        known_encodings.append(encoding)
        known_names.append(filename.split(".")[0])

# Mark attendance
def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(attendance_file, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([name, dt_string, "Entered"])
    print(f"[+] Attendance marked for {name}")

# Send email alert
def send_alert(name):
    subject = f"Bunk Alert: {name} left the class"
    body = f"{name} left the class at {datetime.now().strftime('%H:%M:%S')}"
    msg = f"Subject: {subject}\n\n{body}"
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.sendmail(EMAIL_USER, EMAIL_TO, msg)
        print(f"[!] Alert sent for {name}")
    except Exception as e:
        print("[-] Email failed:", e)

video = cv2.VideoCapture(0)

attendance_logged = {}
exit_logged = {}

while True:
    ret, frame = video.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb, face_locations)

    current_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_names[best_match_index]
            current_names.append(name)

            if name not in attendance_logged:
                mark_attendance(name)
                attendance_logged[name] = time.time()
            else:
                # Reset if student returned
                exit_logged.pop(name, None)

    # Check if someone left
    for name in list(attendance_logged.keys()):
        if name not in current_names:
            if name not in exit_logged:
                exit_logged[name] = time.time()
            elif time.time() - exit_logged[name] > 120:  # 2 min
                send_alert(name)
                del attendance_logged[name]
                del exit_logged[name]

    # Display frame
    for (top, right, bottom, left), name in zip(face_locations, current_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
