@echo off
REM Drag-and-drop face pipeline launcher.
REM Usage: drop a portrait image onto this .bat file. The pipeline runs
REM end-to-end and outputs land in outputs\characters\<slug>\.
REM
REM Slug is derived from the dropped filename: stem of the file, lowercased,
REM with non-alphanumerics collapsed to underscores. E.g.
REM   "Carolyn Lilipaly.jpg" -> slug "carolyn_lilipaly"
REM   "joe-lynch.webp"        -> slug "joe_lynch"

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: drop an image file onto this .bat, or run from a shell:
    echo     face_pipeline.bat ^<path-to-image^>
    echo.
    pause
    exit /b 1
)

set "IMG=%~1"
set "RAW=%~n1"

REM Derive a clean slug: lowercase + replace non-alphanumerics with underscore.
REM PowerShell one-liner is the cleanest way to do this in cmd.exe.
for /f "delims=" %%S in ('powershell -NoProfile -Command "(([string]'%RAW%').ToLower() -replace '[^a-z0-9]+','_').Trim('_')"') do set "SLUG=%%S"

echo.
echo === DG face pipeline ===
echo image : %IMG%
echo slug  : %SLUG%
echo.

REM cd to the pipeline directory so relative paths resolve correctly.
pushd "%~dp0.."

python dg_face_pipeline\run_full_pipeline.py ^
    --subject "%SLUG%" ^
    --image "%IMG%" ^
    --canonical dg_face_pipeline\canonical_face_model_v002.obj ^
    --subdivisions 0
set "PIPELINE_RC=%errorlevel%"

popd

if not %PIPELINE_RC%==0 (
    echo.
    echo Pipeline failed with exit code %PIPELINE_RC%.
    pause
    exit /b %PIPELINE_RC%
)

echo.
echo Done. Outputs at:
echo   dg_face_pipeline\outputs\characters\%SLUG%\
echo.
pause
