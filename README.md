# Py-Scan

A small, local-only Python prototype for experimenting with webcam face detection, face enrollment, and recognition logs.

## Setup

Use this only with informed consent from the people being enrolled. Face images and recognition logs are biometric data.

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m pip install -r requirements.txt
```

## Usage

Enroll a person. The camera captures 20 samples; press `q` to stop early:

```powershell
python app.py enroll --name Alice
```

Start scanning:

```powershell
python app.py run
```

Press `q` in the scanner window to stop. The scanner displays the camera feed with a simple box around detected faces. Known people are logged at most once every 2 seconds while continuously visible.

View each enrolled person's latest recorded time:

```powershell
python app.py list
```

Face samples and SQLite logs are created in `data/` when you run the app. This is an educational prototype, not a security or access-control system.

## Contact
For more information, reporting errors or any other business, please feel free to open an issue!
