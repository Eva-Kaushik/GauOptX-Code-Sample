@ECHO OFF

REM Navigate to the script's directory
pushd %~dp0

REM Define Sphinx build command
if "%SPHINXBUILD%" == "" (
    set SPHINXBUILD=sphinx-build
)

REM Define source and build directories
set SOURCEDIR=.
set BUILDDIR=_build
set SPHINXPROJ=GauOptX

REM Check if a target is provided
if "%1" == "" goto help

REM Verify if Sphinx is installed
%SPHINXBUILD% --version >NUL 2>&1
if errorlevel 9009 (
    echo.
    echo ERROR: 'sphinx-build' command not found!
    echo Please ensure Sphinx is installed and available in your system PATH.
    echo To install Sphinx, run: pip install sphinx
    exit /b 1
)

REM Run the Sphinx build process
%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
REM Display help message for available make targets
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd
