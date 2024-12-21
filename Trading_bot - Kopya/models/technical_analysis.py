import pandas as pd
import numpy as np
import ta
from ta.trend import MACD, SMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
import logging

class TechnicalAnalyzer:
    def __init__(self):
        self.indicators = {}
        self.signals = []
    
    def calculate_indicators(self, df):
        """Tüm teknik göstergeleri hesapla"""
        try:
            # RSI
            rsi = RSIIndicator(df['close'], window=14)
            self.indicators['RSI'] = rsi.rsi()
            
            # MACD
            macd = MACD(df['close'])
            self.indicators['MACD'] = macd.macd()
            self.indicators['MACD_Signal'] = macd.macd_signal()
            self.indicators['MACD_Hist'] = macd.macd_diff()
            
            # Stochastic
            stoch = StochasticOscillator(df['high'], df['low'], df['close'])
            self.indicators['Stoch_K'] = stoch.stoch()
            self.indicators['Stoch_D'] = stoch.stoch_signal()
            
            # Bollinger Bands
            bb = BollingerBands(df['close'])
            self.indicators['BB_upper'] = bb.bollinger_hband()
            self.indicators['BB_middle'] = bb.bollinger_mavg()
            self.indicators['BB_lower'] = bb.bollinger_lband()
            
            # Moving Averages
            self.indicators['MA20'] = SMAIndicator(df['close'], window=20).sma_indicator()
            self.indicators['MA50'] = SMAIndicator(df['close'], window=50).sma_indicator()
            self.indicators['MA200'] = SMAIndicator(df['close'], window=200).sma_indicator()
            
            # ATR
            atr = AverageTrueRange(df['high'], df['low'], df['close'])
            self.indicators['ATR'] = atr.average_true_range()
            
            # Ichimoku Cloud
            ichimoku = IchimokuIndicator(df['high'], df['low'])
            self.indicators['Ichimoku_Conversion'] = ichimoku.ichimoku_conversion_line()
            self.indicators['Ichimoku_Base'] = ichimoku.ichimoku_base_line()
            
            # Trend Direction
            self.indicators['Trend'] = np.where(
                df['close'] > self.indicators['MA50'],
                1,  # Yükselen trend
                np.where(
                    df['close'] < self.indicators['MA50'],
                    -1,  # Düşen trend
                    0  # Yatay trend
                )
            )
            
            return self.indicators
            
        except Exception as e:
            logging.error(f"Teknik gösterge hesaplama hatası: {str(e)}")
            return None
    
    def generate_signals(self, df):
        """Teknik analiz bazlı işlem sinyalleri üret"""
        try:
            self.signals = []
            
            # RSI Sinyalleri
            if self.indicators['RSI'].iloc[-1] < 30:
                self.signals.append({
                    'type': 'BUY',
                    'reason': 'RSI Oversold',
                    'confidence': 0.7
                })
            elif self.indicators['RSI'].iloc[-1] > 70:
                self.signals.append({
                    'type': 'SELL',
                    'reason': 'RSI Overbought',
                    'confidence': 0.7
                })
            
            # MACD Sinyalleri
            if (self.indicators['MACD'].iloc[-2] < self.indicators['MACD_Signal'].iloc[-2] and
                self.indicators['MACD'].iloc[-1] > self.indicators['MACD_Signal'].iloc[-1]):
                self.signals.append({
                    'type': 'BUY',
                    'reason': 'MACD Crossover',
                    'confidence': 0.6
                })
            elif (self.indicators['MACD'].iloc[-2] > self.indicators['MACD_Signal'].iloc[-2] and
                  self.indicators['MACD'].iloc[-1] < self.indicators['MACD_Signal'].iloc[-1]):
                self.signals.append({
                    'type': 'SELL',
                    'reason': 'MACD Crossover',
                    'confidence': 0.6
                })
            
            # Bollinger Bands Sinyalleri
            if df['close'].iloc[-1] < self.indicators['BB_lower'].iloc[-1]:
                self.signals.append({
                    'type': 'BUY',
                    'reason': 'BB Oversold',
                    'confidence': 0.65
                })
            elif df['close'].iloc[-1] > self.indicators['BB_upper'].iloc[-1]:
                self.signals.append({
                    'type': 'SELL',
                    'reason': 'BB Overbought',
                    'confidence': 0.65
                })
            
            # Trend Takibi
            if (self.indicators['Trend'].iloc[-1] == 1 and
                df['close'].iloc[-1] > self.indicators['MA20'].iloc[-1]):
                self.signals.append({
                    'type': 'BUY',
                    'reason': 'Trend Following',
                    'confidence': 0.55
                })
            elif (self.indicators['Trend'].iloc[-1] == -1 and
                  df['close'].iloc[-1] < self.indicators['MA20'].iloc[-1]):
                self.signals.append({
                    'type': 'SELL',
                    'reason': 'Trend Following',
                    'confidence': 0.55
                })
            
            # Ichimoku Sinyalleri
            if (self.indicators['Ichimoku_Conversion'].iloc[-1] > self.indicators['Ichimoku_Base'].iloc[-1] and
                df['close'].iloc[-1] > self.indicators['MA50'].iloc[-1]):
                self.signals.append({
                    'type': 'BUY',
                    'reason': 'Ichimoku Bullish',
                    'confidence': 0.6
                })
            elif (self.indicators['Ichimoku_Conversion'].iloc[-1] < self.indicators['Ichimoku_Base'].iloc[-1] and
                  df['close'].iloc[-1] < self.indicators['MA50'].iloc[-1]):
                self.signals.append({
                    'type': 'SELL',
                    'reason': 'Ichimoku Bearish',
                    'confidence': 0.6
                })
            
            return self.signals
            
        except Exception as e:
            logging.error(f"Sinyal üretme hatası: {str(e)}")
            return []
    
    def get_market_state(self):
        """Piyasa durumunu analiz et"""
        try:
            state = {
                'trend': 'NEUTRAL',
                'volatility': 'MEDIUM',
                'momentum': 'NEUTRAL',
                'support': None,
                'resistance': None
            }
            
            # Trend analizi
            if self.indicators['Trend'].iloc[-1] == 1:
                state['trend'] = 'BULLISH'
            elif self.indicators['Trend'].iloc[-1] == -1:
                state['trend'] = 'BEARISH'
            
            # Volatilite analizi
            atr = self.indicators['ATR'].iloc[-1]
            atr_percentage = atr / self.indicators['MA20'].iloc[-1] * 100
            
            if atr_percentage < 1:
                state['volatility'] = 'LOW'
            elif atr_percentage > 3:
                state['volatility'] = 'HIGH'
            
            # Momentum analizi
            rsi = self.indicators['RSI'].iloc[-1]
            if rsi > 60:
                state['momentum'] = 'BULLISH'
            elif rsi < 40:
                state['momentum'] = 'BEARISH'
            
            # Destek ve direnç seviyeleri
            state['support'] = float(self.indicators['BB_lower'].iloc[-1])
            state['resistance'] = float(self.indicators['BB_upper'].iloc[-1])
            
            return state
            
        except Exception as e:
            logging.error(f"Piyasa durum analizi hatası: {str(e)}")
            return None
