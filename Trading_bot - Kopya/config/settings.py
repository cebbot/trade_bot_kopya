# --- START OF FILE settings.py ---
import os
import logging

# Dosya yolları
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Proje ana dizini
LOG_DIR = os.path.join(BASE_DIR, 'logs')
API_CONFIG_FILE = os.path.join(BASE_DIR, 'config', 'api_keys.txt')

# API anahtarlarını çevre değişkenlerinden oku
API_KEY = os.environ.get('BINANCE_API_KEY')
API_SECRET = os.environ.get('BINANCE_API_SECRET')

# Loglama ayarları
LOG_CONFIG = {
    'filename': os.path.join(LOG_DIR, 'bot_logs.log'),
    'format': '%(asctime)s - %(levelname)s - %(message)s',
    'level': logging.DEBUG  # Hata ayıklama için DEBUG seviyesi
}

# Trading parametreleri
TRADING_PARAMS = {
    'symbol': 'BTCUSDT',
    'order_type': 'MARKET',
    'stop_loss_percent': 0.02,  # %2
    'take_profit_percent': 0.03,  # %3
    'trailing_stop_percent': 0.015,  # %1.5
    'max_position_size': 0.01  # Maximum 0.01 BTC
}

# Risk yönetimi parametreleri
RISK_PARAMS = {
    'max_daily_trades': 10,
    'max_daily_loss': 100,  # USDT
    'max_position_risk': 50,  # USDT
    'risk_per_trade': 0.02,  # %2
    'max_open_trades': 3  # Artık aynı anda 3 pozisyon açabilir
}

# AI model parametreleri
AI_PARAMS = {
    'window_size': 60,  # Son 60 mum verisi
    'lstm_units': [256, 128, 64],  # LSTM katmanlarındaki nöron sayıları
    'dense_units': [32, 16],       # Dense katmanlarındaki nöron sayıları
    'dropout_rate': 0.3,
    'activation': 'relu',          # Aktivasyon fonksiyonu
    'optimizer': {
        'name': 'Adam',
        'learning_rate': 0.001,
        'clipnorm': 1.0
    },
    'epochs': 50,
    'batch_size': 32,
    'validation_split': 0.2
}

# Web sunucu ayarları
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True  # Flask hata ayıklama modu
}

# Duygu analizi parametreleri
SENTIMENT_PARAMS = {
    'news_update_interval': 3600,  # 1 saat
    'min_confidence': 0.6, # Duygu skoru esik degeri
    'news_sources': [
        {'url': 'https://min-api.cryptocompare.com/data/v2/news/?lang=EN', 'type': 'api'},
        {'url': 'https://www.coindesk.com/feed', 'type': 'rss'},
        {'url': 'https://cointelegraph.com/rss', 'type': 'rss'}
    ]
}

# Teknik analiz parametreleri
TECHNICAL_PARAMS = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2,
    'trend_ema': 200
}