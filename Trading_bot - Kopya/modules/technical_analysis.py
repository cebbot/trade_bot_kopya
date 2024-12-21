import pandas as pd
import numpy as np
import ta
import logging

class TechnicalAnalyzer:
    def __init__(self):
        self.last_state = {
            'trend': 'NEUTRAL',
            'volatility': 'MEDIUM',
            'momentum': 'NEUTRAL'
        }

    def calculate_indicators(self, df):
        """Teknik göstergeleri hesapla"""
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
                    'confidence': 0.7 + (30 - df['RSI'].iloc[-1]) / 100,
                    'price': float(df['close'].iloc[-1])
                })
            elif df['RSI'].iloc[-1] > 70:
                signals.append({
                    'type': 'SELL',
                    'reason': 'RSI Overbought',
                    'confidence': 0.7 + (df['RSI'].iloc[-1] - 70) / 100,
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
