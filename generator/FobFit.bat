@echo off
rem Double-click to open the FobFit window.
cd /d "%~dp0"
where pythonw >nul 2>nul
if %errorlevel%==0 (
  start "" pythonw "%~dp0fobfit_gui.py" %*
) else (
  python "%~dp0fobfit_gui.py" %*
  if errorlevel 1 pause
)
