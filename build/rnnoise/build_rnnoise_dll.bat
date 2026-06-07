@echo off
REM Build an AVX2, self-contained rnnoise.dll with MSVC (VS2019 BuildTools).
REM Prereq: clone https://github.com/xiph/rnnoise.git @ 70f1d256 into .deps\rnnoise
REM and copy rnnoise_buffer.c into its src\ (see README.md). vcvars path hard-coded
REM because vswhere is broken on this machine.
setlocal
cd /d "%~dp0"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 ( echo [build] vcvars64 failed & exit /b 1 )

set ROOT=%~dp0..\..
set SRC=%ROOT%\.deps\rnnoise

cl /nologo /O2 /MT /LD /arch:AVX2 ^
  /I "%SRC%\include" /I "%SRC%\src" ^
  /DWIN32 /DRNNOISE_BUILD /DDLL_EXPORT ^
  "%SRC%\src\denoise.c" ^
  "%SRC%\src\rnn.c" ^
  "%SRC%\src\pitch.c" ^
  "%SRC%\src\kiss_fft.c" ^
  "%SRC%\src\celt_lpc.c" ^
  "%SRC%\src\nnet.c" ^
  "%SRC%\src\nnet_default.c" ^
  "%SRC%\src\parse_lpcnet_weights.c" ^
  "%SRC%\src\rnnoise_data.c" ^
  "%SRC%\src\rnnoise_tables.c" ^
  "%SRC%\src\rnnoise_buffer.c" ^
  /Fe:rnnoise.dll
set RC=%errorlevel%
del *.obj *.exp *.lib 2>nul
echo [build] cl exit code: %RC%
endlocal & exit /b %RC%
