@REM 2022/09/29 
@REM apply all available UCPs to attack dataset
@ECHO OFF

SETLOCAL ENABLEDELAYEDEXPANSION

SET ATK_DATASET=imagenet-1k

REM clean test
python test.py -D %ATK_DATASET%

REM attack test
FOR /R %%f IN (*.npy) DO (
  ECHO process %%~nxf
  python test.py -D %ATK_DATASET% --ucp %%f --ex --resizer tile
  python test.py -D %ATK_DATASET% --ucp %%f --ex --resizer interpolate
)
