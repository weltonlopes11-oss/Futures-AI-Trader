from market_data.binance_rest import BinanceREST


def main():

    binance = BinanceREST()

    price = binance.get_price()

    print("----------------------")
    print("BTC USDT")
    print(price)
    print("----------------------")


if __name__ == "__main__":
    main()