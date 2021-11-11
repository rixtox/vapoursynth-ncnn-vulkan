@echo off

REM Don't export variables
setlocal

REM Script directory
SET SCRIPT_DIR=%~dp0

REM Project directory
PUSHD .
CD "%SCRIPT_DIR%\.."
SET PROJECT_DIR=%CD%
POPD

SET GENERATOR=
SET PLATFORM=x64

:argloop
IF NOT "%1"=="" (
    IF "%1"=="-G" (
        SET GENERATOR=-G "%2"
        SHIFT
    )
    IF "%1"=="-A" (
        SET PLATFORM=%2
        SHIFT
    )
    IF "%1"=="-h" (
        GOTO :showhelp
    )
    SHIFT
    GOTO :argloop
)
GOTO :build

:showhelp
ECHO build.bat [options]
ECHO     -G GENERATOR       See cmake -G
ECHO     -A PLATFORM        Win32/x64       Default x64
EXIT

:build

REM Push current directory
PUSHD .

REM Create build directories
CD "%PROJECT_DIR%"
IF NOT EXIST build MKDIR build
cd build
IF NOT EXIST "%PLATFORM%" MKDIR "%PLATFORM%"

REM Navigate to build directory
CD "%PLATFORM%"

REM Generate CMake build system
cmake ..\.. %GENERATOR% -A %PLATFORM%

REM Execute CMake build system
cmake --build . --config Release --verbose

REM Pop current directory
POPD
