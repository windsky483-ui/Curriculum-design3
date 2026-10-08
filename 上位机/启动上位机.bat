@echo off
setlocal
set "APP=%~dp0mpu6050_monitor.py"
set "PYW=C:\Users\lyl\AppData\Local\Programs\Python\Python314\pythonw.exe"
if not exist "%PYW%" set "PYW=pythonw"
start "" "%PYW%" "%APP%"
endlocal
