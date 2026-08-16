from market_data.candles import CandleCollector


collector = CandleCollector()


df = collector.collect_candles()


print(df.head())

print("\nColunas:")
print(df.columns)

print("\nÚltimo candle:")
print(df.tail(1))