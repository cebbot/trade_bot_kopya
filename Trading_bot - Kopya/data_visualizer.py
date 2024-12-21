import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Örnek veri
data = {
    'close': [YOUR_CLOSE_DATA],  # Buraya close verilerini ekleyin
    'volume': [YOUR_VOLUME_DATA],  # Buraya volume verilerini ekleyin
    'RSI': [YOUR_RSI_DATA],       # Buraya RSI verilerini ekleyin
    'MACD': [YOUR_MACD_DATA]      # Buraya MACD verilerini ekleyin
}

# DataFrame oluştur
df = pd.DataFrame(data)

# Grafik oluştur
fig = make_subplots(rows=4, cols=1, 
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Price', 'Volume', 'RSI', 'MACD'),
                    row_heights=[0.4, 0.2, 0.2, 0.2])

# Fiyat grafiği
fig.add_trace(go.Scatter(y=df['close'], name='Price'), row=1, col=1)

# Volume grafiği
fig.add_trace(go.Bar(y=df['volume'], name='Volume'), row=2, col=1)

# RSI grafiği
fig.add_trace(go.Scatter(y=df['RSI'], name='RSI'), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# MACD grafiği
fig.add_trace(go.Scatter(y=df['MACD'], name='MACD'), row=4, col=1)

# Grafik düzeni
fig.update_layout(height=1000, title='Market Analysis', showlegend=True)

# Grafiği göster
fig.show()
