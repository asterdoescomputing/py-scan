import argparse
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
DB_PATH = DATA_DIR / "faces.db"
CAMERA_INDEX = 0
SAMPLES_PER_PERSON = 20
LOG_INTERVAL_SECONDS = 2


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def open_database():
    DATA_DIR.mkdir(exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            last_seen TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER NOT NULL,
            seen_at TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (person_id) REFERENCES people(id)
        )
        """
    )
    connection.commit()
    return connection


def detector():
    filename = "haarcascade_frontalface_default.xml"
    candidates = [
        Path(cv2.data.haarcascades) / filename,
        Path(cv2.__file__).parent / "data" / filename,
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise RuntimeError(
            "OpenCV's face detector file is missing. Reinstall the pinned dependency with "
            "`python -m pip install --force-reinstall -r requirements.txt`."
        )
    model = cv2.CascadeClassifier(str(path))
    if model.empty():
        raise RuntimeError(f"Could not load OpenCV's face detector from {path}.")
    return model


def largest_face(faces):
    return max(faces, key=lambda face: face[2] * face[3], default=None)


def enroll(name):
    name = name.strip()
    if not name:
        raise ValueError("Name cannot be empty.")

    connection = open_database()
    try:
        person = connection.execute(
            "SELECT id FROM people WHERE name = ?", (name,)
        ).fetchone()
        if person is None:
            cursor = connection.execute("INSERT INTO people (name) VALUES (?)", (name,))
            person_id = cursor.lastrowid
            connection.commit()
        else:
            person_id = person[0]
    finally:
        connection.close()

    person_dir = FACES_DIR / str(person_id)
    person_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError("Could not open the webcam.")

    face_detector = detector()
    saved = 0
    last_capture = 0.0
    print("Look at the camera. Press q to stop enrollment.")
    try:
        while saved < SAMPLES_PER_PERSON:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
            face = largest_face(faces)
            if face is not None:
                x, y, width, height = face
                if time.monotonic() - last_capture >= 0.25:
                    crop = gray[y : y + height, x : x + width]
                    crop = cv2.resize(crop, (200, 200))
                    saved += 1
                    cv2.imwrite(str(person_dir / f"{saved:03d}.png"), crop)
                    last_capture = time.monotonic()

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)

            cv2.imshow("Face enrollment", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {saved} face samples for {name}.")


def load_model(connection):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images = []
    labels = []
    people = {}
    for person_id, name in connection.execute("SELECT id, name FROM people ORDER BY id"):
        files = sorted((FACES_DIR / str(person_id)).glob("*.png"))
        for file in files:
            image = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                images.append(image)
                labels.append(person_id)
                people[person_id] = name

    if not images:
        raise RuntimeError("No enrolled faces found. Run `python app.py enroll --name NAME` first.")
    recognizer.train(images, np.asarray(labels, dtype=np.int32))
    return recognizer, people


def run_scanner():
    connection = open_database()
    try:
        recognizer, people = load_model(connection)
        face_detector = detector()
        camera = cv2.VideoCapture(CAMERA_INDEX)
        if not camera.isOpened():
            raise RuntimeError("Could not open the webcam.")

        print("Scanning. Press q to stop.")
        last_logged = {}
        try:
            while True:
                success, frame = camera.read()
                if not success:
                    raise RuntimeError("Could not read a frame from the webcam.")

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(100, 100))
                for x, y, width, height in faces:
                    crop = cv2.resize(gray[y : y + height, x : x + width], (200, 200))
                    person_id, confidence = recognizer.predict(crop)
                    known = person_id in people and confidence < 75
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 220, 0), 2)

                    if known and time.monotonic() - last_logged.get(person_id, 0) >= LOG_INTERVAL_SECONDS:
                        seen_at = now_iso()
                        connection.execute(
                            "INSERT INTO sightings (person_id, seen_at, confidence) VALUES (?, ?, ?)",
                            (person_id, seen_at, confidence),
                        )
                        connection.execute(
                            "UPDATE people SET last_seen = ? WHERE id = ?",
                            (seen_at, person_id),
                        )
                        connection.commit()
                        last_logged[person_id] = time.monotonic()
                        print(f"{seen_at}: {people[person_id]}")

                cv2.imshow("Face scanner", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            camera.release()
            cv2.destroyAllWindows()
    finally:
        connection.close()


def list_people():
    connection = open_database()
    try:
        rows = connection.execute(
            "SELECT name, COALESCE(last_seen, 'Never') FROM people ORDER BY name"
        ).fetchall()
    finally:
        connection.close()
    if not rows:
        print("No enrolled people.")
        return
    for name, last_seen in rows:
        print(f"{name}: {last_seen}")


def main():
    parser = argparse.ArgumentParser(description="Local webcam face enrollment and recognition.")
    commands = parser.add_subparsers(dest="command", required=True)
    enroll_command = commands.add_parser("enroll", help="Enroll a person using the webcam.")
    enroll_command.add_argument("--name", required=True, help="Name associated with the face.")
    commands.add_parser("run", help="Start recognition and log sightings.")
    commands.add_parser("list", help="Show enrolled people and their last-seen time.")
    args = parser.parse_args()

    if args.command == "enroll":
        enroll(args.name)
    elif args.command == "run":
        run_scanner()
    else:
        list_people()


if __name__ == "__main__":
    main()
