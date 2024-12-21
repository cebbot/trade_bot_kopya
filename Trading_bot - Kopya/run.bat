 @echo off
cd /d %~dp0
echo Binance Trading Bot Baslatiliyor...
echo ================================
echo.
echo Gerekli kutuphaneler yukleniyor...
pip install -r requirements.txt
echo.
echo Bot baslatiliyor...
echo Web arayuzu http://127.0.0.1:5000 adresinde baslatilacak
echo Log dosyasini kontrol etmeyi unutmayin: logs/bot_logs.log
echo.
python Trade_bot.py
pause