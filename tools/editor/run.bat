@echo off
rem Standalone start for editor (or auto-started by portal)
cd /d "%~dp0"
"%~dp0..\..\python\python.exe" app.py
pause
