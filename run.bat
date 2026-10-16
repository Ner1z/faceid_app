@echo off
chcp 65001 >nul
title NEON FaceID
echo [1/2] Установка зависимостей...
pip install -r requirements.txt
echo [2/2] Запуск NEON FaceID...
python main.py
pause
