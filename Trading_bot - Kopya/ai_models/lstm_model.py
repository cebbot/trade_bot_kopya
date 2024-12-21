# --- START OF FILE lstm_model.py ---
"""
LSTM (Long Short-Term Memory) Model for Cryptocurrency Price Prediction
Bu modül, kripto para fiyat tahminleri için LSTM tabanlı derin öğrenme modelini içerir.

Özellikler:
- Çoklu teknik gösterge desteği
- Otomatik veri normalizasyonu
- Adaptif öğrenme oranı
- Dropout ve BatchNormalization ile aşırı öğrenme kontrolü
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import logging
from config.settings import AI_PARAMS

# TensorFlow uyarılarını azalt
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # Uyarıları gidermek için bu satır değiştirildi

class LSTMModel:
    def __init__(self):
        """
        LSTM modelini başlat ve yapılandır
        """
        self.model = None
        self.scaler = MinMaxScaler()
        self.window_size = AI_PARAMS['window_size']
        # Teknik göstergeler ve fiyat verileri
        self.feature_columns = [
            'close', 'volume', 'RSI', 'MACD', 'MACD_Signal',
            'BB_upper', 'BB_middle', 'BB_lower', 'high', 'low', 'open'
        ]
        
        self._build_model()
        print("LSTM model başarıyla yüklendi")

    def _build_model(self):
        """
        LSTM modelinin mimarisini oluştur
        - 3 LSTM katmanı
        - Dropout ve BatchNormalization ile regularizasyon
        - Adam optimizer ve sparse_categorical_crossentropy loss
        """
        try:
            input_shape = (self.window_size, len(self.feature_columns) + 6)
            
            self.model = Sequential([
                LSTM(units=AI_PARAMS['lstm_units'][0],
                     return_sequences=True,
                     input_shape=input_shape,
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                BatchNormalization(),
                Dropout(AI_PARAMS['dropout_rate']),
                
                LSTM(units=AI_PARAMS['lstm_units'][1],
                     return_sequences=True,
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                BatchNormalization(),
                Dropout(AI_PARAMS['dropout_rate']),
                
                LSTM(units=AI_PARAMS['lstm_units'][2],
                     kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                BatchNormalization(),
                Dropout(AI_PARAMS['dropout_rate']),
                
                Dense(AI_PARAMS['dense_units'][0], activation=AI_PARAMS['activation']),
                BatchNormalization(),
                Dropout(AI_PARAMS['dropout_rate']),
                
                Dense(3, activation='softmax')  # Çıktı katmanı (Sell, Hold, Buy)
            ])
            
            # Modeli derle
            optimizer_params = AI_PARAMS['optimizer']
            self.model.compile(
                optimizer=getattr(tf.keras.optimizers, optimizer_params['name'])(
                    learning_rate=optimizer_params['learning_rate'],
                    clipnorm=optimizer_params['clipnorm']
                ),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(), # Düzeltme: tf.keras.losses kullanıldı
                metrics=['accuracy']
            )
            
        except Exception as e:
            logging.error(f"Model oluşturma hatası: {str(e)}")
            raise

    def _prepare_data(self, df):
        """
        Veriyi model için hazırla
        
        Args:
            df (DataFrame): İşlenecek veri seti
            
        Returns:
            tuple: (X, y) - özellikler ve etiketler
        """
        try:
            # Ana özellikleri normalize et
            features = df[self.feature_columns].values
            scaled_features = self.scaler.fit_transform(features)
            
            # Ek özellikler hesapla
            price_momentum = df['close'].pct_change(5).values  # 5 periyotluk momentum
            volume_momentum = df['volume'].pct_change(5).values
            rsi_momentum = df['RSI'].diff().values
            
            # Bollinger Band sinyalleri
            bb_signal = np.zeros(len(df))
            bb_signal[df['close'] > df['BB_upper']] = 1  # Aşırı alım
            bb_signal[df['close'] < df['BB_lower']] = -1  # Aşırı satım
            
            # MACD sinyalleri
            macd_signal = np.zeros(len(df))
            macd_signal[df['MACD'] > df['MACD_Signal']] = 1
            macd_signal[df['MACD'] < df['MACD_Signal']] = -1
            
            # Volatilite hesapla
            volatility = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            
            # Ek özellikleri birleştir
            additional_features = np.column_stack([
                price_momentum,
                volume_momentum,
                rsi_momentum,
                bb_signal,
                macd_signal,
                volatility
            ])
            
            # NaN değerleri temizle
            additional_features = np.nan_to_num(additional_features, nan=0.0)
            
            # Tüm özellikleri birleştir
            all_features = np.column_stack([scaled_features, additional_features])
            
            X, y = [], []
            for i in range(len(df) - self.window_size):
                feature_window = all_features[i:(i + self.window_size)]
                X.append(feature_window)
                
                # Gelecek fiyat değişimi için daha uzun vadeli tahmin
                future_window = 10  # 10 periyoda çıkarıldı
                if i + self.window_size + future_window >= len(df):
                    future_price_change = 0
                else:
                    future_price = df['close'].iloc[i + self.window_size + future_window]
                    current_price = df['close'].iloc[i + self.window_size]
                    future_price_change = (future_price - current_price) / current_price
                
                # Eşik değerleri ayarlandı
                if future_price_change < -0.008:  # %0.8 düşüş
                    y.append(0)  # Sell
                elif future_price_change > 0.008:  # %0.8 yükseliş
                    y.append(2)  # Buy
                else:
                    y.append(1)  # Hold
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logging.error(f"Veri hazırlama hatası: {str(e)}")
            return None, None

    def train(self, df, epochs=None):
        """
        Modeli eğit
        
        Args:
            df (DataFrame): Eğitim veri seti
            epochs (int): Eğitim epochs sayısı
        
        Returns:
            History: Eğitim geçmişi
        """
        try:
            print("Model eğitimi başlıyor...")
            X, y = self._prepare_data(df)
            if X is None or y is None:
                return
            
            # Early stopping ve model checkpoint ekle
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=0.0001
                )
            ]
            
            epochs = epochs or AI_PARAMS['epochs']
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=AI_PARAMS['batch_size'],
                validation_split=AI_PARAMS['validation_split'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Model performansını değerlendir
            val_loss, val_acc = self.model.evaluate(
                X[int(len(X)*0.8):], 
                y[int(len(y)*0.8):],
                verbose=0
            )
            print(f"\nModel değerlendirmesi:")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_acc:.4f}")
            
            return history
            
        except Exception as e:
            logging.error(f"Model eğitim hatası: {str(e)}")
            return None

    def predict(self, df):
        """
        Tahmin yap
        
        Args:
            df (DataFrame): Tahmin veri seti
        
        Returns:
            Prediction: Tahmin sonucu
        """
        try:
            # Veriyi hazırla
            features = df[self.feature_columns].values
            scaled_features = self.scaler.transform(features)
            
            # Ek özellikler hesapla
            price_momentum = df['close'].pct_change(5).values
            volume_momentum = df['volume'].pct_change(5).values
            rsi_momentum = df['RSI'].diff().values
            
            # Bollinger Band sinyalleri
            bb_signal = np.zeros(len(df))
            bb_signal[df['close'] > df['BB_upper']] = 1
            bb_signal[df['close'] < df['BB_lower']] = -1
            
            # MACD sinyalleri
            macd_signal = np.zeros(len(df))
            macd_signal[df['MACD'] > df['MACD_Signal']] = 1
            macd_signal[df['MACD'] < df['MACD_Signal']] = -1
            
            # Volatilite hesapla
            volatility = df['close'].rolling(window=20).std() / df['close'].rolling(window=20).mean()
            
            # Ek özellikleri birleştir
            additional_features = np.column_stack([
                price_momentum,
                volume_momentum,
                rsi_momentum,
                bb_signal,
                macd_signal,
                volatility
            ])
            
            # NaN değerleri temizle
            additional_features = np.nan_to_num(additional_features, nan=0.0)
            
            # Tüm özellikleri birleştir
            all_features = np.column_stack([scaled_features, additional_features])
            
            # Son window_size kadar veriyi al
            X = all_features[-self.window_size:].reshape(1, self.window_size, len(self.feature_columns) + 6)
            
            # Tahmin yap ve güven skorlarını hesapla
            prediction = self.model.predict(X, verbose=0)
            confidence = np.max(prediction[0])
            
            # Tahmin ve güven skorunu logla
            action = ['SELL', 'HOLD', 'BUY'][np.argmax(prediction[0])]
            logging.info(f"LSTM Tahmin: {action} (Güven: {confidence:.2%})")
            
            return prediction
            
        except Exception as e:
            logging.error(f"Tahmin hatası: {str(e)}")
            return None

    def get_model_summary(self):
        """
        Model özetini döndür
        
        Returns:
            dict: Model özet bilgileri
        """
        try:
            model_info = {
                'window_size': self.window_size,
                'features': self.feature_columns,
                'architecture': [
                    f'LSTM({AI_PARAMS["lstm_units"][0]}) + BatchNorm + Dropout({AI_PARAMS["dropout_rate"]})',
                    f'LSTM({AI_PARAMS["lstm_units"][1]}) + BatchNorm + Dropout({AI_PARAMS["dropout_rate"]})',
                    f'LSTM({AI_PARAMS["lstm_units"][2]}) + BatchNorm + Dropout({AI_PARAMS["dropout_rate"]})'
                ]
            }
            return model_info
        except Exception as e:
            logging.error(f"Model özeti hatası: {str(e)}")
            return None