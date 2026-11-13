@echo off
rem Makes FobFit.lnk next to this file: a shortcut with the FobFit icon that opens the FobFit window through
rem pythonw, so no console window shows. Drag it to the desktop or the Start menu.
rem Run this again if you move the FobFit folder or install another Python.
cd /d "%~dp0"
set "FOBFIT_DIR=%~dp0"
set "FOBFIT_PYW="
for /f "delims=" %%p in ('where pythonw 2^>nul') do if not defined FOBFIT_PYW set "FOBFIT_PYW=%%p"
powershell -NoProfile -Command "$d = $env:FOBFIT_DIR; $s = (New-Object -ComObject WScript.Shell).CreateShortcut($d + 'FobFit.lnk'); if ($env:FOBFIT_PYW) { $s.TargetPath = $env:FOBFIT_PYW; $s.Arguments = [char]34 + $d + 'fobfit_gui.py' + [char]34; $s.WindowStyle = 1 } else { $s.TargetPath = $d + 'FobFit.bat'; $s.WindowStyle = 7 }; $s.WorkingDirectory = $d; $s.IconLocation = $d + 'assets\fobfit.ico,0'; $s.Description = 'FobFit generator: a key jacket from a 3D scan'; $s.Save()"
if exist "%~dp0FobFit.lnk" (echo Made FobFit.lnk) else (echo Could not make FobFit.lnk & pause)
if not defined FOBFIT_PYW (echo pythonw was not found: the shortcut opens FobFit.bat instead.)
