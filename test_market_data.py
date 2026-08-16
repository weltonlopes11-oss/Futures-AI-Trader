from market_data.collector import MarketDataCollector


def main():

    collector = MarketDataCollector()


    candles = collector.get_candles(
        symbol="BTCUSDT"
    )


    if candles is not None:

        print(candles.head())


if __name__ == "__main__":

    main()