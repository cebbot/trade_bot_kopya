import os
import time
import asyncio
import traceback
from binance.client import Client
from binance.enums import *
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO
import threading
import warnings
import ta

# Özel modüller
from ai_models.lstm_model import LSTMModel
from ai_models.sentiment_model import SentimentAnalyzer
from modules.risk_manager import RiskManager
from config.settings import API_KEY, API_SECRET, LOG_DIR, LOG_CONFIG, TRADING_PARAMS

warnings.filterwarnings('ignore')

# Flask uygulaması
app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading', logger=True, engineio_logger=True)

# Loglama ayarları
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(**LOG_CONFIG)

class TechnicalAnalyzer:
    def __init__(self):
        self.last_state = {
            'trend': 'NEUTRAL',
            'volatility': 'MEDIUM',
            'momentum': 'NEUTRAL'
        }

    def calculate_indicators(self, df):
        """Teknik indikatörleri hesapla"""
        try:
            # RSI
            df['RSI'] = ta.momentum.RSIIndicator(df['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            
            # Bollinger Bantları
            bollinger = ta.volatility.BollingerBands(df['close'])
            df['BB_upper'] = bollinger.bollinger_hband()
            df['BB_middle'] = bollinger.bollinger_mavg()
            df['BB_lower'] = bollinger.bollinger_lband()
            
            # EMA'lar
            df['EMA_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
            df['EMA_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
            df['EMA_50'] = ta.trend.EMAIndicator(df['close'], window=50).ema_indicator()
            df['EMA_200'] = ta.trend.EMAIndicator(df['close'], window=200).ema_indicator()
            
            # ATR (Volatilite için)
            df['ATR'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
            
            # Stokastik RSI
            stoch_rsi = ta.momentum.StochRSIIndicator(df['close'])
            df['Stoch_RSI_K'] = stoch_rsi.stochrsi_k()
            df['Stoch_RSI_D'] = stoch_rsi.stochrsi_d()
            
            # Piyasa durumunu güncelle
            self.update_market_state(df)
            
            return df.dropna()
            
        except Exception as e:
            logging.error(f"Gösterge hesaplama hatası: {str(e)}")
            return None

    def generate_signals(self, df):
        """Teknik analiz sinyalleri üret"""
        try:
            signals = []
            
            # RSI sinyalleri
            if df['RSI'].iloc[-1] < 30:
                signals.append({
                    'type': 'BUY',
                    'reason': 'RSI Oversold',
                    'confidence': min(0.7 + (30 - df['RSI'].iloc[-1]) / 100, 1.0),
                    'price': float(df['close'].iloc[-1])
                })
            elif df['RSI'].iloc[-1] > 70:
                signals.append({
                    'type': 'SELL',
                    'reason': 'RSI Overbought',
                    'confidence': min(0.7 + (df['RSI'].iloc[-1] - 70) / 100, 1.0),
                    'price': float(df['close'].iloc[-1])
                })
            
            # MACD sinyalleri
            if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] and \
               df['MACD'].iloc[-2] <= df['MACD_Signal'].iloc[-2]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'MACD Crossover',
                    'confidence': 0.8,
                    'price': float(df['close'].iloc[-1])
                })
            elif df['MACD'].iloc[-1] < df['MACD_Signal'].iloc[-1] and \
                 df['MACD'].iloc[-2] >= df['MACD_Signal'].iloc[-2]:
                signals.append({
                    'type': 'SELL',
                    'reason': 'MACD Crossover',
                    'confidence': 0.8,
                    'price': float(df['close'].iloc[-1])
                })
            
            # Bollinger Band sinyalleri
            if df['close'].iloc[-1] < df['BB_lower'].iloc[-1]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'BB Oversold',
                    'confidence': 0.75,
                    'price': float(df['close'].iloc[-1])
                })
            elif df['close'].iloc[-1] > df['BB_upper'].iloc[-1]:
                signals.append({
                    'type': 'SELL',
                    'reason': 'BB Overbought',
                    'confidence': 0.75,
                    'price': float(df['close'].iloc[-1])
                })
            
            # EMA Crossover sinyalleri
            if df['EMA_9'].iloc[-1] > df['EMA_21'].iloc[-1] and \
               df['EMA_9'].iloc[-2] <= df['EMA_21'].iloc[-2]:
                signals.append({
                    'type': 'BUY',
                    'reason': 'EMA Crossover',
                    'confidence': 0.7,
                    'price': float(df['close'].iloc[-1])
                })
            elif df['EMA_9'].iloc[-1] < df['EMA_21'].iloc[-1] and \
                 df['EMA_9'].iloc[-2] >= df['EMA_21'].iloc[-2]:
                signals.append({
                    'type': 'SELL',
                    'reason': 'EMA Crossover',
                    'confidence': 0.7,
                    'price': float(df['close'].iloc[-1])
                })
            
            # Stokastik RSI sinyalleri
            if df['Stoch_RSI_K'].iloc[-1] < 20 and df['Stoch_RSI_D'].iloc[-1] < 20:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Stoch RSI Oversold',
                    'confidence': 0.65,
                    'price': float(df['close'].iloc[-1])
                })
            elif df['Stoch_RSI_K'].iloc[-1] > 80 and df['Stoch_RSI_D'].iloc[-1] > 80:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Stoch RSI Overbought',
                    'confidence': 0.65,
                    'price': float(df['close'].iloc[-1])
                })
            
            return signals
            
        except Exception as e:
            logging.error(f"Sinyal üretme hatası: {str(e)}")
            return []

    def get_market_state(self):
        """Piyasa durumunu analiz et"""
        try:
            # Trend analizi
            if self.last_state['trend'] == 'UPTREND':
                trend = 'Yükseliş'
            elif self.last_state['trend'] == 'DOWNTREND':
                trend = 'Düşüş'
            else:
                trend = 'Yatay'
            
            # Volatilite analizi
            if self.last_state['volatility'] == 'HIGH':
                volatility = 'Yüksek'
            elif self.last_state['volatility'] == 'LOW':
                volatility = 'Düşük'
            else:
                volatility = 'Orta'
            
            # Momentum analizi
            if self.last_state['momentum'] == 'STRONG':
                momentum = 'Güçlü'
            elif self.last_state['momentum'] == 'WEAK':
                momentum = 'Zayıf'
            else:
                momentum = 'Nötr'
            
            return {
                'trend': trend,
                'volatility': volatility,
                'momentum': momentum
            }
            
        except Exception as e:
            logging.error(f"Piyasa durumu analiz hatası: {str(e)}")
            return {
                'trend': 'Belirsiz',
                'volatility': 'Belirsiz',
                'momentum': 'Belirsiz'
            }

    def update_market_state(self, df):
        """Piyasa durumunu güncelle"""
        try:
            # Trend analizi
            if df['EMA_50'].iloc[-1] > df['EMA_200'].iloc[-1]:
                if df['close'].iloc[-1] > df['EMA_50'].iloc[-1]:
                    self.last_state['trend'] = 'UPTREND'
                else:
                    self.last_state['trend'] = 'NEUTRAL'
            else:
                if df['close'].iloc[-1] < df['EMA_50'].iloc[-1]:
                    self.last_state['trend'] = 'DOWNTREND'
                else:
                    self.last_state['trend'] = 'NEUTRAL'
            
            # Volatilite analizi
            atr = df['ATR'].iloc[-1]
            atr_mean = df['ATR'].mean()
            if atr > atr_mean * 1.5:
                self.last_state['volatility'] = 'HIGH'
            elif atr < atr_mean * 0.5:
                self.last_state['volatility'] = 'LOW'
            else:
                self.last_state['volatility'] = 'MEDIUM'
            
            # Momentum analizi
            rsi = df['RSI'].iloc[-1]
            if rsi > 60:
                self.last_state['momentum'] = 'STRONG'
            elif rsi < 40:
                self.last_state['momentum'] = 'WEAK'
            else:
                self.last_state['momentum'] = 'NEUTRAL'
            
        except Exception as e:
            logging.error(f"Piyasa durumu güncelleme hatası: {str(e)}")

class BinanceBot:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.lock = asyncio.Lock()  # Asyncio kilidi eklendi
        
        # API anahtarlarını config/api_keys.txt dosyasından oku
        try:
            api_config_path = os.path.join(os.path.dirname(__file__), 'config', 'api_keys.txt')
            with open(api_config_path, 'r') as file:
                lines = file.readlines()
                for line in lines:
                  if line.startswith('API Key'):
                      self.api_key = lines[lines.index(line) + 1].strip()
                  elif line.startswith('Secret Key'):
                      self.api_secret = lines[lines.index(line) + 1].strip()

            if not self.api_key or not self.api_secret:
                raise ValueError("API anahtarları config/api_keys.txt dosyasında bulunamadı veya hatalı formattalar.")

        except (FileNotFoundError, IndexError):
            error_msg = f"API anahtarları dosyası ({api_config_path}) bulunamadı veya formatı hatalı."
            logging.error(error_msg)
            raise ValueError(error_msg)

        except Exception as e:
            logging.error(f"API anahtarları okunurken bir hata oluştu: {str(e)}")
            raise
        
        # Binance client'ı başlat
        try:
            self.client = Client(self.api_key, self.api_secret)
            # Test bağlantısı
            self.client.ping()
            server_time = self.client.get_server_time()
            logging.info("Binance bağlantısı başarılı")
        except Exception as e:
            logging.error(f"Binance bağlantısı kurulamadı: {str(e)}")
            if hasattr(e, 'status_code'):
                logging.error(f"HTTP Durum Kodu: {e.status_code}")
            if hasattr(e, 'response'):
                logging.error(f"Sunucu Yanıtı: {e.response.text if hasattr(e.response, 'text') else e.response}")
            raise
        
        # Zaman senkronizasyonu
        self.server_time_offset = 0
        self.sync_time()
        
        # Trading parametreleri
        self.symbol = TRADING_PARAMS['symbol']
        self.position = None
        self.stop_loss_percent = TRADING_PARAMS['stop_loss_percent']
        self.take_profit_percent = TRADING_PARAMS['take_profit_percent']
        self.trailing_stop_percent = TRADING_PARAMS['trailing_stop_percent']
        self.max_position_size = TRADING_PARAMS['max_position_size']
        
        # Pozisyon takibi
        self.last_price = None
        self.entry_price = None
        self.position_size = 0
        self.highest_price = None
        self.lowest_price = None
        self.position_id = None
        
        # Sinyal geçmişi
        self.signals = []
        
        # Modülleri başlat
        self.lstm_model = LSTMModel()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_manager = RiskManager()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Model eğitimi için veri topla
        self.train_model()
        
        print("Bot başlatıldı!")

    def sync_time(self):
        """Binance sunucu zamanı ile senkronizasyon"""
        try:
            server_time = self.client.get_server_time()
            local_time = int(time.time() * 1000)
            self.server_time_offset = server_time['serverTime'] - local_time
            logging.info(f"Sunucu zamanı senkronize edildi. Offset: {self.server_time_offset}ms")
        except Exception as e:
            logging.error(f"Zaman senkronizasyon hatası: {str(e)}")

    def train_model(self):
        """LSTM modelini eğit"""
        try:
            print("Model eğitimi için veri toplanıyor...")
            # Son 1000 saatlik veri
            klines = self.client.get_historical_klines(
                self.symbol,
                Client.KLINE_INTERVAL_1HOUR,
                "1000 hours ago UTC"
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Teknik göstergeleri hesapla
            df = TechnicalAnalyzer().calculate_indicators(df)
            
            if df is not None:
                print("Model eğitimi başlıyor...")
                self.lstm_model.train(df, epochs=50)
            
        except Exception as e:
            print(f"Model eğitim hatası: {str(e)}")
            logging.error(f"Model eğitim hatası: {str(e)}")

    def get_historical_data(self):
        """Geçmiş fiyat verilerini al"""
        try:
            print("Geçmiş veriler alınıyor...")
            klines = self.client.get_historical_klines(
                self.symbol,
                Client.KLINE_INTERVAL_1HOUR,
                "500 hours ago UTC"
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"Veri alındı: {len(df)} kayıt")
            return df
            
        except Exception as e:
            print(f"Veri alma hatası: {str(e)}")
            logging.error(f"Geçmiş veri alma hatası: {str(e)}")
            return None

    def analyze_market(self, df):
        """Piyasa analizi yap"""
        try:
            # Önce teknik analiz ile market durumunu güncelle
            self.technical_analyzer.update_market_state(df)
            
            # Teknik analiz sinyallerini al
            technical_signals = self.technical_analyzer.generate_signals(df)
            
            # LSTM tahminlerini al
            lstm_predictions = self.lstm_model.predict(df)
            
            # Duygu analizi yap
            sentiment = self.sentiment_analyzer.analyze()
            
            # Risk değerlendirmesi yap
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            return {
                'technical': technical_signals,
                'market_state': self.technical_analyzer.get_market_state(),
                'lstm': lstm_predictions,
                'sentiment': sentiment,
                'risk': risk_metrics
            }
            
        except Exception as e:
            logging.error(f"Analiz hatası: {str(e)}")
            return None

    def calculate_ai_metrics(self, df):
        """AI metriklerini hesapla"""
        try:
            # AI tahminleri
            lstm_prediction = self.lstm_model.predict(df)
            
            # AI metriklerini hazırla
            ai_metrics = {
                'confidence': float(max(lstm_prediction[0])) if lstm_prediction is not None else 0,
                'success_rate': self.calculate_ai_success_rate()
            }
            
            return ai_metrics
            
        except Exception as e:
            logging.error(f"AI metrikleri hesaplama hatası: {str(e)}")
            return None

    def calculate_ai_success_rate(self):
        """AI tahminlerinin başarı oranını hesapla"""
        try:
            # Son 100 sinyali al
            last_signals = self.signals[-100:]
            if not last_signals:
                return 0
            
            successful = 0
            for signal in last_signals:
                if signal['type'] == 'BUY':
                    # Alış sinyali sonrası fiyat yükseldiyse başarılı
                    if signal.get('exit_price', 0) > signal['price']:
                        successful += 1
                elif signal['type'] == 'SELL':
                    # Satış sinyali sonrası fiyat düştüyse başarılı
                    if signal.get('exit_price', 0) < signal['price']:
                        successful += 1
            
            return (successful / len(last_signals)) * 100
            
        except Exception as e:
            logging.error(f"Başarı oranı hesaplama hatası: {str(e)}")
            return 0
    
    def calculate_pnl(self):
        """Kar/Zarar (PnL) hesapla"""
        if self.position == "LONG" and self.entry_price:
            return ((self.last_price - self.entry_price) / self.entry_price) * 100
        else:
            return 0

    def execute_trade(self, side, quantity):
        """İşlem gerçekleştir"""
        try:
            # Risk kontrolü (İşlem gerçekleştirmeden önce)
            if side == SIDE_BUY:
                stop_loss_price = self.last_price * (1 - self.stop_loss_percent)
                risk_amount = (self.last_price - stop_loss_price) * quantity

                if not self.risk_manager.can_open_trade('BUY', risk_amount):
                    logging.warning("Risk limiti aşıldı - İşlem iptal edildi")
                    return None

            # İşlemi gerçekleştir
            order = self.client.create_order(
                symbol=self.symbol,
                side=side,
                type=TRADING_PARAMS['order_type'],
                quantity=quantity
            )

            # İşlem bilgilerini kaydet
            trade_info = {
                'id': order['orderId'],
                'timestamp': datetime.now(),
                'side': side,
                'quantity': quantity,
                'price': float(order['fills'][0]['price']),
                'commission': float(order['fills'][0]['commission'])
            }

            # Risk yöneticisine ekle
            if side == SIDE_BUY:
                self.risk_manager.add_trade(trade_info)
            else:
                # Satış işleminde önceki sinyali güncelle
                if self.signals:
                    last_signal = self.signals[-1]
                    last_signal['exit_price'] = float(order['fills'][0]['price'])
                    last_signal['exit_time'] = datetime.now()

                self.risk_manager.close_trade(
                    self.position_id,
                    (self.last_price - self.entry_price) * self.position_size
                )

            return order

        except Exception as e:
            logging.error(f"İşlem hatası: {str(e)}")
            return None

    def calculate_position_size(self):
        """İşlem büyüklüğünü hesapla"""
        # Bakiye bilgisini al
        account = self.client.get_account()
        balance = float([asset for asset in account['balances'] if asset['asset'] == 'USDT'][0]['free'])

        # Risk yönetimi kurallarını uygula
        risk_adjusted_balance = self.risk_manager.get_risk_adjusted_balance(balance)

        # Pozisyon büyüklüğünü hesapla
        position_size = min(
            risk_adjusted_balance / self.last_price,
            self.max_position_size
        )

        # Miktar formatını düzelt
        info = self.client.get_symbol_info(self.symbol)
        step_size = float([f for f in info['filters'] if f['filterType'] == 'LOT_SIZE'][0]['stepSize'])
        precision = len(str(step_size).split('.')[-1].rstrip('0'))

        # Miktarı yuvarla
        position_size = round(position_size - (position_size % step_size), precision)

        if position_size <= 0:
            logging.warning("Hesaplanan pozisyon büyüklüğü çok küçük veya negatif.")
            return 0

        return position_size

    async def run(self):
        """Ana bot döngüsü"""
        print("Bot çalışıyor...")
        last_price_check_time = time.time()
        
        
        while True:
            try:
                current_time = time.time()
                print("Veri alınıyor...")
                # Güncel fiyat verilerini al
                df = self.get_historical_data()
                if df is None:
                    print("Veri alınamadı!")
                    await asyncio.sleep(10)
                    continue
                
                # Her dakika başı fiyat kontrolü yap
                if current_time - last_price_check_time >= 60:
                    self.last_price = float(df['close'].iloc[-1])
                    last_price_check_time = current_time
                    print(f"Güncel fiyat: {self.last_price}")

                # Piyasa analizi
                analysis = self.analyze_market(df)
                if analysis is None:
                    await asyncio.sleep(10)
                    continue
                
                async with self.lock:
                    # Web arayüzünü güncelle
                    print("Web arayüzü güncelleniyor...")
                    socketio.emit('update_data', {
                        'status': 'running',
                        'position': {
                            'type': self.position,
                            'size': self.position_size,
                            'entry_price': self.entry_price,
                            'current_price': self.last_price,
                            'pnl': ((self.last_price - self.entry_price) / self.entry_price * 100) if self.entry_price else 0
                        },
                        'market_state': analysis['market_state'],
                        'market_data': {
                            'price': self.last_price,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'trend': analysis['technical']['trend'] if analysis and 'technical' in analysis and 'trend' in analysis['technical'] else 'SIDEWAYS',
                            'volatility': float(analysis['technical']['volatility']) if analysis and 'technical' in analysis and 'volatility' in analysis['technical'] else 0.0,
                            'chart_data': df.tail(100).to_dict(orient='list'),
                            'signals': {
                                'technical': to_serializable(analysis['technical']) if analysis and 'technical' in analysis else {},
                                'sentiment': self.sentiment_analyzer.get_current_sentiment() if self.sentiment_analyzer else 0,
                                'lstm': to_serializable(analysis.get('lstm', {}))
                            }
                        },
                        'ai_metrics': to_serializable(self.calculate_ai_metrics(df)),
                        'risk_metrics': to_serializable(self.risk_manager.get_risk_metrics() if self.risk_manager else {})
                    }, namespace='/')
                    print("Web arayüzü güncellendi")

                await asyncio.sleep(10)
                
            except Exception as e:
                print(f"Hata oluştu: {str(e)}")
                logging.error(f"Ana döngü hatası: {str(e)}")
                await asyncio.sleep(10)

def to_serializable(obj):
    """Numpy array ve diğer özel tipleri JSON serileştirilebilir hale getir"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, datetime):
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(item) for item in obj]
    return obj

async def update_data():
    """Veri güncellemesi yap ve WebSocket üzerinden gönder"""
    last_update_time = 0
    update_interval = 60  # Güncelleme aralığı (saniye)

    while True:
        try:
            bot = app.config.get('bot')
            if bot:
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    # Fiyat verilerini al
                    df = bot.get_historical_data()
                    if df is None:
                        print("Fiyat verisi alınamadı")
                        await asyncio.sleep(5)
                        continue

                    # Teknik analiz yap
                    analysis = bot.analyze_market(df)
                    if analysis is None:
                        print("Analiz yapılamadı")
                        await asyncio.sleep(5)
                        continue

                    async with bot.lock:
                        # Son fiyatı güncelle
                        bot.last_price = float(df['close'].iloc[-1])
                        
                        # Market durumunu hazırla
                        market_state = {
                            'price': float(bot.last_price),
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'trend': analysis['technical']['trend'] if analysis and 'technical' in analysis and 'trend' in analysis['technical'] else 'SIDEWAYS',
                            'volatility': float(analysis['technical']['volatility']) if analysis and 'technical' in analysis and 'volatility' in analysis['technical'] else 0.0,
                            'signals': {
                                'technical': to_serializable(analysis['technical']) if analysis and 'technical' in analysis else {},
                                'sentiment': bot.sentiment_analyzer.get_current_sentiment() if bot.sentiment_analyzer else 0,
                                'lstm': to_serializable(analysis.get('lstm', {}))
                            }
                        }
                        
                        # Pozisyon bilgilerini hazırla
                        position_info = {
                            'type': bot.position,
                            'size': float(bot.position_size) if bot.position_size else 0,
                            'entry_price': float(bot.entry_price) if bot.entry_price else 0,
                            'current_price': float(bot.last_price),
                            'pnl': 0  # PnL hesaplaması eklenecek
                        }
                        
                        if bot.position and bot.entry_price and bot.last_price:
                            if bot.position == 'LONG':
                                position_info['pnl'] = ((bot.last_price - bot.entry_price) / bot.entry_price) * 100
                            elif bot.position == 'SHORT':
                                position_info['pnl'] = ((bot.entry_price - bot.last_price) / bot.entry_price) * 100

                        # Veri yapısını hazırla
                        data = {
                            'market_state': market_state,
                            'position': position_info,
                            'risk_metrics': to_serializable(bot.risk_manager.get_risk_metrics() if bot.risk_manager else {})
                        }

                    # WebSocket üzerinden gönder
                    socketio.emit('update_data', data, namespace='/')
                    print("Veri güncellendi")

                    last_update_time = current_time
                else:
                    print("Veri güncelleme zamanı gelmedi")

        except Exception as e:
            print(f"Veri güncelleme hatası: {str(e)}")
            logging.error(f"Veri güncelleme hatası: {str(e)}")

        await asyncio.sleep(5)

@socketio.on('connect', namespace='/')
def handle_connect():
    """Client bağlantısı kurulduğunda ilk verileri gönder"""
    try:
        bot = app.config.get('bot')
        if bot:
            # Fiyat verilerini al
            df = bot.get_historical_data()
            if df is not None:
                # Teknik analiz yap
                analysis = bot.analyze_market(df)
                if analysis:
                    # İlk verileri gönder
                    data = {
                        'status': 'running',
                        'position': {
                            'type': bot.position,
                            'size': bot.position_size,
                            'entry_price': bot.entry_price,
                            'current_price': bot.last_price,
                            'pnl': bot.calculate_pnl() if bot.position else 0
                        },
                        'market_state': analysis['market_state'],
                        'market_data': {
                            'price': bot.last_price,
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'trend': analysis['technical']['trend'] if analysis and 'technical' in analysis and 'trend' in analysis['technical'] else 'SIDEWAYS',
                            'volatility': float(analysis['technical']['volatility']) if analysis and 'technical' in analysis and 'volatility' in analysis['technical'] else 0.0,
                            'chart_data': df.tail(100).to_dict(orient='list'),
                            'signals': {
                                'technical': to_serializable(analysis['technical']) if analysis and 'technical' in analysis else {},
                                'sentiment': bot.sentiment_analyzer.get_current_sentiment() if bot.sentiment_analyzer else 0,
                                'lstm': to_serializable(analysis.get('lstm', {}))
                            }
                        },
                        'ai_metrics': to_serializable(bot.calculate_ai_metrics(df)),
                        'risk_metrics': to_serializable(bot.risk_manager.get_risk_metrics() if bot.risk_manager else {})
                    }
                    socketio.emit('initial_data', data, namespace='/')
                    print("İlk veriler gönderildi:", data)
    except Exception as e:
        print(f"Bağlantı hatası: {str(e)}")
        logging.error(f"Bağlantı hatası: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Bot durumunu döndür"""
    try:
        bot = app.config['bot']
        if bot:
            status_data = {
                'status': 'running',
                'position': bot.position,
                'last_price': bot.last_price,
                'risk_metrics': to_serializable(bot.risk_manager.get_risk_metrics() if bot.risk_manager else {})
            }
            return jsonify(status_data)
        else:
            return jsonify({'status': 'error', 'message': 'Bot başlatılmamış.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@socketio.on('manual_trade')
def handle_manual_trade(data):
    """Manuel işlem sinyallerini işle"""
    try:
        bot = app.config['bot']
        if data['type'] == 'BUY' and bot.position is None:
            quantity = bot.calculate_position_size()
            if quantity > 0:
                order = bot.execute_trade(SIDE_BUY, quantity)
                if order:
                    bot.position = "LONG"
                    bot.position_size = quantity
                    bot.entry_price = bot.last_price
                    bot.highest_price = bot.last_price
                    bot.position_id = order['orderId']
        elif data['type'] == 'SELL' and bot.position == "LONG":
            order = bot.execute_trade(SIDE_SELL, bot.position_size)
            if order:
                bot.position = None
                bot.position_size = 0
                bot.entry_price = None
                bot.highest_price = None
                bot.position_id = None
    except Exception as e:
        print(f"Manuel işlem hatası: {str(e)}")
        logging.error(f"Manuel işlem hatası: {str(e)}")

def start_bot():
    """Bot'u başlat"""
    try:
        # Bot'u başlat
        bot = BinanceBot()
        app.config['bot'] = bot

        # Bot'un çalışacağı event loop'u al
        loop = asyncio.get_event_loop()

        # Veri güncelleme görevini başlat
        update_task = loop.create_task(update_data())

        # Bot'u asenkron olarak çalıştır
        bot_task = loop.create_task(bot.run())

        # Flask uygulamasını başlat (Flask-SocketIO'nun asenkron modunu kullanarak)
        socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)

        # Görevleri çalıştır
        loop.run_until_complete(asyncio.gather(update_task, bot_task))

    except Exception as e:
        print(f"Bot başlatma hatası: {str(e)}")
        logging.error(f"Bot başlatma hatası: {str(e)}")

if __name__ == "__main__":
    try:
        print("Bot başlatılıyor...")
        start_bot()
    except Exception as e:
        print(f"Başlatma hatası: {str(e)}")
        logging.error(f"Başlatma hatası: {str(e)}")