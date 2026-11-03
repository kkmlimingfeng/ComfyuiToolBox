@echo off
rem Standalone start for dataset_tagger (or auto-started by portal)
cd /d "%~dp0"
"%~dp0..\..\python\python.exe" app.py
pause
