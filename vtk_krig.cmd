::!cmd
@echo off
setlocal
if not exist vulcan.chk goto PYTHON_APPDATA
if defined VULCAN_EXE goto PYTHON_VULCAN
:: first try to use Vulcan 11
for /d %%i in ("%ProgramFiles%\Maptek\Vulcan 11*") do set VULCAN_EXE=%%~si\bin\exe
if defined VULCAN_EXE goto PYTHON_VULCAN
:: then resort to newest version
for /d %%i in ("%ProgramFiles%\Maptek\Vulcan 20*") do set VULCAN_EXE=%%~si\bin\exe
if not defined VULCAN_EXE goto :EOF
:PYTHON_VULCAN
echo VULCAN_EXE=%VULCAN_EXE%
if defined VULCAN_BIN goto PYTHON_OK
set VULCAN_BIN=%VULCAN_EXE:~0,-4%
set VULCAN=%VULCAN_EXE:~0,-8%
set PATH=%VULCAN_EXE%;%VULCAN_BIN%;%VULCAN_BIN%\cygnus\bin;%VULCAN_BIN%\other\x86;%PATH%
set PERLLIB=%VULCAN%\lib\perl;%VULCAN%\lib\perl\site\lib
set VULCAN_VERSION_MAJOR=10
goto PYTHON_OK
:PYTHON_APPDATA
:: Search for the latest python distribution
if defined WINPYDIRBASE goto PYTHON_OK
for /d %%i in (%appdata%\WPy64*) do set WINPYDIRBASE=%%i
if not defined WINPYDIRBASE (
    echo.
    echo Python runtime not found. 
    echo Download Winpython and extract to APPDATA system folder:
    echo %APPDATA%
    echo.
    pause
    goto :EOF
)
call "%WINPYDIRBASE%\scripts\env.bat"
:PYTHON_OK
python -V
set python_cmd="%~dpn0.py"
if not exist %python_cmd% (
    set python_cmd=-m _gui "%~f0"
)
python %python_cmd%
if errorlevel 1 pause
