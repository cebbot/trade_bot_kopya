# --- START OF FILE risk_manager.py ---
import logging
from datetime import datetime, timedelta
from config.settings import RISK_PARAMS

class RiskManager:
    def __init__(self):
        self.daily_trades = []
        self.open_trades = []
        self.daily_pnl = 0
        self.max_drawdown = 0
        
        # Risk limitleri
        self.max_daily_trades = RISK_PARAMS['max_daily_trades']
        self.max_daily_loss = RISK_PARAMS['max_daily_loss']
        self.max_position_risk = RISK_PARAMS['max_position_risk']
        self.risk_per_trade = RISK_PARAMS['risk_per_trade']
        self.max_open_trades = RISK_PARAMS['max_open_trades']
        
        # Günlük verileri sıfırla
        self._reset_daily_data()

    def _reset_daily_data(self):
        """Günlük verileri sıfırla"""
        now = datetime.now()
        self.daily_trades = [t for t in self.daily_trades if t['timestamp'].date() == now.date()]
        self.daily_pnl = sum(t.get('pnl', 0) for t in self.daily_trades)

    def can_open_trade(self, side, risk_amount):
        """Yeni işlem açılabilir mi kontrol et"""
        try:
            self._reset_daily_data()
            
            # Açık işlem sayısı kontrolü
            if len(self.open_trades) >= self.max_open_trades:
                logging.warning("Maksimum açık işlem sayısına ulaşıldı")
                return False
            
            # Günlük işlem sayısı kontrolü
            if len(self.daily_trades) >= self.max_daily_trades:
                logging.warning("Günlük maksimum işlem sayısına ulaşıldı")
                return False
            
            # Günlük zarar limiti kontrolü
            if self.daily_pnl < -self.max_daily_loss:
                logging.warning("Günlük maksimum zarar limitine ulaşıldı")
                return False
            
            # İşlem risk miktarı kontrolü
            if risk_amount > self.max_position_risk:
                logging.warning(f"İşlem riski çok yüksek: {risk_amount}")
                return False
            
            return True
            
        except Exception as e:
            logging.error(f"Risk kontrolü hatası: {str(e)}")
            return False

    def add_trade(self, trade_info):
        """Yeni işlem ekle"""
        try:
            self.open_trades.append(trade_info)
            self.daily_trades.append(trade_info)
            logging.info(f"Yeni işlem eklendi: {trade_info}")
            
        except Exception as e:
            logging.error(f"İşlem ekleme hatası: {str(e)}")

    def close_trade(self, trade_id, pnl):
        """İşlemi kapat"""
        try:
            # İşlemi açık işlemlerden kaldır
            self.open_trades = [t for t in self.open_trades if t['id'] != trade_id]
            
            # PnL'i güncelle
            self.daily_pnl += pnl
            
            # Maksimum drawdown'ı güncelle
            if pnl < 0 and abs(pnl) > self.max_drawdown:
                self.max_drawdown = abs(pnl)
            
            # İlgili işlemin PnL'ini kaydet
            for trade in self.daily_trades:
                if trade['id'] == trade_id:
                    trade['pnl'] = pnl
                    trade['close_time'] = datetime.now()
                    break
            
            logging.info(f"İşlem kapatıldı - ID: {trade_id}, PnL: {pnl}")
            
        except Exception as e:
            logging.error(f"İşlem kapatma hatası: {str(e)}")

    def calculate_position_size(self, balance, current_price, stop_loss_price):
        """İşlem büyüklüğünü hesapla"""
        try:
            risk_amount = balance * self.risk_per_trade
            price_difference = abs(current_price - stop_loss_price)
            
            if price_difference == 0:
                return 0
                
            position_size = risk_amount / price_difference
            return position_size
            
        except Exception as e:
            logging.error(f"Pozisyon büyüklüğü hesaplama hatası: {str(e)}")
            return 0
    
    def get_risk_adjusted_balance(self, balance):
        """
        Riski göz önünde bulundurarak kullanılabilir bakiyeyi hesaplar.
        
        Args:
            balance (float): Toplam bakiye.

        Returns:
            float: Risk ayarlı kullanılabilir bakiye.
        """
        
        # Günlük zarara göre bakiyeyi düşür
        adjusted_balance = balance - abs(self.daily_pnl) if self.daily_pnl < 0 else balance
        
        # Minimum değeri 0 olarak belirle
        adjusted_balance = max(0, adjusted_balance)
        
        return adjusted_balance

    def adjust_position_for_volatility(self, position_size, volatility):
        """Volatiliteye göre pozisyon büyüklüğünü ayarla"""
        try:
            if volatility == 'HIGH':
                return position_size * 0.5  # Yüksek volatilitede pozisyonu yarıya düşür
            elif volatility == 'LOW':
                return position_size * 1.2  # Düşük volatilitede pozisyonu %20 artır
            return position_size
            
        except Exception as e:
            logging.error(f"Volatilite ayarlama hatası: {str(e)}")
            return position_size

    def evaluate_risk(self, df):
        """Piyasa riskini değerlendir"""
        try:
            # Volatilite hesapla
            returns = df['close'].pct_change()
            volatility = returns.std()
            
            # Trend yönü ve gücü
            price_change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
            trend_strength = abs(price_change)
            
            # Risk skoru hesapla (0-1 arası)
            risk_score = min(1.0, (volatility * 100 + trend_strength) / 2)
            
            return {
                'risk_score': risk_score,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'recommendation': 'HIGH_RISK' if risk_score > 0.7 else 'MEDIUM_RISK' if risk_score > 0.4 else 'LOW_RISK'
            }
            
        except Exception as e:
            logging.error(f"Risk değerlendirme hatası: {str(e)}")
            return None

    def get_risk_metrics(self):
        """Risk metriklerini döndür"""
        try:
            self._reset_daily_data()
            return {
                'daily_pnl': self.daily_pnl,
                'max_drawdown': self.max_drawdown,
                'open_trades': len(self.open_trades),
                'daily_trades': len(self.daily_trades)
            }
        except Exception as e:
            logging.error(f"Risk metrikleri hatası: {str(e)}")
            return {
                'daily_pnl': 0,
                'max_drawdown': 0,
                'open_trades': 0,
                'daily_trades': 0
            }
