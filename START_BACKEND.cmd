@echo off
REM SignBridge Backend Quick Start Script for Windows PowerShell
REM Copy these commands and paste into PowerShell to start the backend

REM 1. Navigate to backend folder
cd "C:\3rd year projects\individual project\SignBridge\signbridge-backend"

REM 2. Activate virtual environment
.venv\Scripts\Activate.ps1

REM 3. Start the backend (choose one):

REM Option A: Using uvicorn directly (recommended)
uvicorn app.main:app --host 127.0.0.1 --port 8000

REM Option B: Using Python module
REM python -m app.main

REM When started successfully, you should see:
REM - "All models loaded successfully!"
REM - "Uvicorn running on http://127.0.0.1:8000"
REM
REM Open http://127.0.0.1:8000/ in your browser to test

