@echo off
setlocal

REM Docker image name
set IMAGE_NAME=aicomp-pandoc-report

REM Check if Docker image exists
docker images -q %IMAGE_NAME% 2>nul | findstr . >nul
if %errorlevel% neq 0 (
    echo Docker image '%IMAGE_NAME%' not found. Building it now...
    docker build -t %IMAGE_NAME% .
    if %errorlevel% neq 0 (
        echo Error: Failed to build Docker image.
        exit /b 1
    )
    echo Docker image built successfully.
) else (
    echo Docker image '%IMAGE_NAME%' already exists.
)

REM Create _build directory if it doesn't exist
if not exist "_build" mkdir _build

REM Run pandoc in Docker container
echo Running pandoc to generate report.pdf...
docker run --rm -v "%cd%:/root" --entrypoint "pandoc" %IMAGE_NAME% src/report.md --defaults defaults.yaml -o _build/report.pdf

if %errorlevel% equ 0 (
    echo Success! Report generated at _build\report.pdf
) else (
    echo Error: Failed to generate report.
    exit /b 1
)

endlocal

