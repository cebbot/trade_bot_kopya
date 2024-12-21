# --- START OF FILE sentiment_model.py ---
import nltk
from textblob import TextBlob
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime, timedelta
from config.settings import SENTIMENT_PARAMS
import feedparser
import time

# NLTK kaynaklarını bir kere indir
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

class SentimentAnalyzer:
    def __init__(self):
        # Son güncelleme zamanı
        self.last_update = None
        self.cached_sentiment = None
        
        # NLTK VADER sentiment analyzer
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        self.analyzer = SentimentIntensityAnalyzer()

    def get_news(self):
        """Kripto para haberleri topla"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        all_news = []
        successful_sources = 0
        
        for source in SENTIMENT_PARAMS['news_sources']:
            try:
                if len(all_news) >= 20:  # Yeterli haber toplandıysa dur
                    break
                    
                response = requests.get(source['url'], headers=headers, timeout=10)
                if response.status_code == 429:  # Rate limit
                    logging.warning(f"Rate limit aşıldı ({source['url']})")
                    time.sleep(2)
                    continue
                    
                response.raise_for_status()
                
                if source['type'] == 'api':
                    news = self._parse_api_news(response.json(), source['url'])
                elif source['type'] == 'rss':
                    news = self._parse_rss_news(response.content)
                
                if news:
                    all_news.extend(news)
                    successful_sources += 1
                    logging.info(f"{source['url']} kaynağından {len(news)} haber alındı")
                
            except requests.exceptions.RequestException as e:
                logging.warning(f"Haber kaynağı hatası ({source['url']}): {str(e)}")
                continue
                
            except Exception as e:
                logging.error(f"Haber toplama hatası ({source['url']}): {str(e)}")
                continue
        
        if not all_news:
            if successful_sources == 0:
                logging.error("Hiçbir haber kaynağına erişilemedi")
            else:
                logging.warning("Hiç haber bulunamadı")
            return []
            
        return all_news[:20]  # En güncel 20 haber
    
    def _parse_api_news(self, data, source):
        """API yanıtlarını işle"""
        news = []
        try:
            if 'cryptocompare.com' in source:
                for item in data.get('Data', []):
                    news.append({
                        'title': item.get('title', ''),
                        'summary': item.get('body', ''),
                        'published': item.get('published_on', '')
                    })
        except Exception as e:
            logging.error(f"API yanıtı işleme hatası ({source}): {str(e)}")
        return news
    
    def _parse_rss_news(self, content):
        """RSS feed'lerini işle"""
        try:
            feed = feedparser.parse(content)
            return [{'title': entry.title, 'summary': entry.summary, 'published': entry.published}
                   for entry in feed.entries[:10]]
        except Exception as e:
            logging.error(f"RSS işleme hatası: {str(e)}")
            return []

    def get_news_sentiment(self):
        """Kripto haber kaynaklarından duygu analizi yap"""
        news = self.get_news()
        if not news:
            logging.warning("Hiç haber bulunamadı")
            return 0

        sentiments = []
        for article in news:
            try:
                # VADER sentiment analizi
                vader_scores = self.analyzer.polarity_scores(article['summary'])
                vader_compound = vader_scores['compound']

                # TextBlob sentiment analizi
                blob = TextBlob(article['summary'])
                textblob_polarity = blob.sentiment.polarity

                # İki skorun ortalamasını al
                combined_sentiment = (vader_compound + textblob_polarity) / 2
                sentiments.append(combined_sentiment)

            except Exception as e:
                logging.error(f"Metin analizi hatası: {str(e)}")

        if not sentiments:
            return 0

        # Ağırlıklı ortalama hesapla
        weights = np.linspace(0.1, 1, len(sentiments))
        weighted_average = np.average(sentiments, weights=weights)
        logging.info(f"Duygu analizi tamamlandı: {weighted_average:.2f} ({len(sentiments)} makale)")
        return weighted_average

    def get_sentiment_signals(self):
        """Duygu analizi sinyalleri üret"""
        try:
            # Belirli aralıklarla güncelle
            if (self.last_update is None or 
                datetime.now() - self.last_update > timedelta(seconds=SENTIMENT_PARAMS['news_update_interval'])):
                
                sentiment_score = self.get_news_sentiment()
                self.cached_sentiment = sentiment_score
                self.last_update = datetime.now()
                logging.info(f"Duygu analizi güncellendi: {sentiment_score:.2f}")
            else:
                sentiment_score = self.cached_sentiment
            
            signals = []
            min_confidence = SENTIMENT_PARAMS['min_confidence']
            
            # Güçlü pozitif duygu -> Al sinyali
            if sentiment_score > min_confidence:
                signals.append({
                    'type': 'BUY',
                    'reason': 'Sentiment Analysis',
                    'confidence': abs(sentiment_score),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'details': 'Pozitif piyasa duygusu'
                })
                logging.info(f"Pozitif duygu sinyali üretildi: {sentiment_score:.2f}")
            
            # Güçlü negatif duygu -> Sat sinyali
            elif sentiment_score < -min_confidence:
                signals.append({
                    'type': 'SELL',
                    'reason': 'Sentiment Analysis',
                    'confidence': abs(sentiment_score),
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'details': 'Negatif piyasa duygusu'
                })
                logging.info(f"Negatif duygu sinyali üretildi: {sentiment_score:.2f}")
            
            return signals
            
        except Exception as e:
            logging.error(f"Duygu analizi sinyal hatası: {str(e)}")
            return []

    def analyze_text(self, text):
        """Metin duygu analizi"""
        try:
            # VADER sentiment analizi
            vader_scores = self.analyzer.polarity_scores(text)
            vader_compound = vader_scores['compound']
            
            # TextBlob sentiment analizi
            blob = TextBlob(text)
            textblob_polarity = blob.sentiment.polarity
            
            # İki skorun ortalamasını al
            combined_sentiment = (vader_compound + textblob_polarity) / 2
            
            return combined_sentiment
            
        except Exception as e:
            logging.error(f"Metin analizi hatası: {str(e)}")
            return 0

    def get_current_sentiment(self):
        """Mevcut duygu durumunu döndür"""
        try:
            if (self.last_update is None or 
                datetime.now() - self.last_update > timedelta(seconds=SENTIMENT_PARAMS['news_update_interval'])):
                sentiment_score = self.get_news_sentiment()
                self.cached_sentiment = sentiment_score
                self.last_update = datetime.now()
                logging.info(f"Duygu analizi güncellendi: {sentiment_score:.2f}")
                return sentiment_score
            return self.cached_sentiment if self.cached_sentiment is not None else 0
        except Exception as e:
            logging.error(f"Duygu durumu alma hatası: {str(e)}")
            return 0

    def analyze(self):
        """Mevcut piyasa duygu durumunu analiz et ve sinyal üret"""
        try:
            signals = self.get_sentiment_signals()
            if signals:
                return signals[0]  # En son üretilen sinyali döndür
            else:
                return {
                    'type': 'HOLD',
                    'reason': 'Sentiment Analysis',
                    'confidence': 0.5,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'details': 'Nötr piyasa duygusu'
                }
        except Exception as e:
            logging.error(f"Duygu analizi hatası: {str(e)}")
            return {
                'type': 'HOLD',
                'reason': 'Error',
                'confidence': 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'details': f'Hata: {str(e)}'
            }
