@REM 2022/09/29 
@REM save_fig() for all available UCPs to 'img' folder
@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

SET PARAM=
IF /I "%*" == "/F" (
  SET PARAM=--overwrite
)

ECHO draw and save figures
FOR /R %%f IN (*.npy) DO (
  ECHO process %%~nxf
  python show.py --ucp %%f --silent %PARAM%
)

ECHO.

ECHO update index.html
python mk_index.py
