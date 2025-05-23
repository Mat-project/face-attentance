import cv2
import face_recognition
import os
import numpy as np
import smtplib
import csv
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
EMAIL_TO = os.getenv("EMAIL_TO")

known_faces_dir = "known_faces"
attendance_file = "logs/attendance.csv"
known_encodings = []
known_names = []

print("Loading known faces...")
# Load known faces
for filename in os.listdir(known_faces_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        try:
            img = face_recognition.load_image_file(f"{known_faces_dir}/{filename}")
            # Use multiple jitters for better quality encodings
            encodings = face_recognition.face_encodings(img, num_jitters=3)
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(filename.split(".")[0])
                print(f"Loaded: {filename} - {filename.split('.')[0]}")
            else:
                print(f"No face found in {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

if not known_encodings:
    print("WARNING: No faces were loaded. Please check your known_faces directory.")

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

# Ensure attendance file directory exists
os.makedirs(os.path.dirname(attendance_file), exist_ok=True)

# Initialize webcam
print("Starting webcam...")
video = cv2.VideoCapture(0)

# Adjust camera settings if possible (to get higher resolution)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

attendance_logged = {}
exit_logged = {}

# Minimum face size (pixels)
MIN_FACE_SIZE = 30

# Set a stricter tolerance threshold (lower is stricter)
FACE_RECOGNITION_TOLERANCE = 0.5

print("System running. Press 'q' to exit.")
while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame from camera")
        time.sleep(1)
        continue

    # Resize frame - adjusted to get better detection
    scale = 0.75  # Increased from 0.5 for better detail
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame - adjusted parameters
    face_locations = face_recognition.face_locations(rgb, model="hog")
    
    # Debug information about detected faces
    if face_locations:
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_width = right - left
            face_height = bottom - top
            print(f"Face {i+1}: Size {face_width}x{face_height} pixels")
    
    current_names = []
    processed_faces = 0
    
    # Process faces if any were found
    if face_locations and known_encodings:
        try:
            # Get all face encodings at once
            face_encodings = face_recognition.face_encodings(rgb, face_locations, num_jitters=1)
            
            # Process each face encoding
            for i, face_encoding in enumerate(face_encodings):
                if i >= len(face_locations):
                    continue  # Safety check
                    
                top, right, bottom, left = face_locations[i]
                
                # Skip faces that are too small - reduced threshold
                face_width = right - left
                face_height = bottom - top
                if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
                    print(f"Skipping face {i+1}: too small ({face_width}x{face_height})")
                    continue
                
                processed_faces += 1
                
                # Compare with known faces - use stricter tolerance
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=FACE_RECOGNITION_TOLERANCE)
                
                # Get face distances for all known faces
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                # Debug information
                print("\nFace comparison results:")
                for j, (name, distance) in enumerate(zip(known_names, face_distances)):
                    match_status = "MATCH" if matches[j] else "NO MATCH"
                    confidence = (1 - distance) * 100
                    print(f"  {name}: distance={distance:.4f}, confidence={confidence:.1f}%, {match_status}")
                
                name = "Unknown"
                # Only accept a match if we have a True in matches
                if True in matches:
                    best_match_index = matches.index(True)
                    name = known_names[best_match_index]
                    # Double-check the match quality
                    match_confidence = (1 - face_distances[best_match_index]) * 100
                    print(f"Match found: {name} with {match_confidence:.1f}% confidence")
                else:
                    # Don't use face distance as fallback - this was causing false positives
                    print("No matches found with current tolerance")
                    
                current_names.append(name)
                
                # Update attendance records
                if name != "Unknown":
                    if name not in attendance_logged:
                        mark_attendance(name)
                        attendance_logged[name] = time.time()
                    else:
                        exit_logged.pop(name, None)
                
        except Exception as e:
            print(f"Error recognizing faces: {e}")
    
    if processed_faces == 0 and face_locations:
        print(f"Found {len(face_locations)} faces, but none were large enough to process")

    # Check for exits
    for name in list(attendance_logged.keys()):
        if name not in current_names:
            if name not in exit_logged:
                exit_logged[name] = time.time()
            elif time.time() - exit_logged[name] > 120:
                send_alert(name)
                del attendance_logged[name]
                del exit_logged[name]

    # Display
    for i, (top, right, bottom, left) in enumerate(face_locations):
        # Scale back up face locations
        top = int(top / scale)
        right = int(right / scale)
        bottom = int(bottom / scale)
        left = int(left / scale)
        
        # Get the name if available
        name = "Unknown"
        if i < len(current_names):
            name = current_names[i]
        
        # Draw box and label with confidence if available
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Add confidence display if we have a match
        display_text = name
        if name != "Unknown" and i < len(face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encodings[i], tolerance=FACE_RECOGNITION_TOLERANCE)
            if True in matches:
                idx = matches.index(True)
                confidence = (1 - face_recognition.face_distance(known_encodings, face_encodings[i])[idx]) * 100
                display_text = f"{name} ({confidence:.1f}%)"
        
        cv2.putText(frame, display_text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # Show the resulting image
    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("Shutting down...")
video.release()
cv2.destroyAllWindows()
