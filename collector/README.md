# Futures AI Trader — Binance USD-M Collector

This service is a narrow market-data gateway for the frozen ETHUSDT 1H forward test.

## Contract

- Upstream: Binance USD-M Futures only (`fapi*.binance.com`).
- Market: `ETHUSDT` only.
- Interval: `1h` only.
- Response: native Binance kline array after schema validation.
- Source attestation header: `X-Market-Source: binance-usdm-futures`.
- No Spot fallback.
- No order execution.
- No Binance API key is required because klines are public market data.

## Vercel deployment

Import repository `weltonlopes11-oss/Futures-AI-Trader` as a new Vercel project and set the project Root Directory to `collector`.

`vercel.json` pins the function to the Sao Paulo region (`gru1`).

Configure the environment variable below for Production, Preview, and Development:

- `COLLECTOR_API_KEY`: a long random secret used only to authenticate the forward engine to this collector.

After deployment, the endpoint is:

`/api/klines?symbol=ETHUSDT&interval=1h&limit=200`

Requests must include:

`X-Collector-Key: <COLLECTOR_API_KEY>`

## Forward engine configuration

Configure these GitHub Actions secrets after the Vercel endpoint is live:

- `MARKET_DATA_COLLECTOR_URL`: full deployed `/api/klines` URL.
- `MARKET_DATA_COLLECTOR_KEY`: same value as `COLLECTOR_API_KEY`.

The Python market-data adapter rejects the collector response unless the source attestation header equals `binance-usdm-futures`.

## Future automatic trading

This collector remains market-data-only. Real order execution must be implemented as a separate Binance Futures execution adapter with restricted API credentials, idempotent client order IDs, fill reconciliation, position reconciliation, risk limits, and a kill switch.
