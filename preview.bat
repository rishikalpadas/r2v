@echo off
rem Start the raster2vector preview and open it in the default browser.
cd /d "%~dp0"
start "" http://127.0.0.1:8765
.venv\Scripts\python.exe -m app.server --port 8765
