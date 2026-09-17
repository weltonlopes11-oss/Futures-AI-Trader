const BINANCE_FUTURES_ENDPOINTS = [
  'https://fapi.binance.com/fapi/v1/klines',
  'https://fapi1.binance.com/fapi/v1/klines',
  'https://fapi2.binance.com/fapi/v1/klines',
];

const ALLOWED_SYMBOLS = new Set(['ETHUSDT']);
const ALLOWED_INTERVALS = new Set(['1h']);
const DEFAULT_LIMIT = 200;
const MAX_LIMIT = 500;
const UPSTREAM_TIMEOUT_MS = 8000;

function validateKlines(payload) {
  if (!Array.isArray(payload) || payload.length === 0) return false;
  return payload.every((row) =>
    Array.isArray(row) &&
    row.length >= 12 &&
    Number.isFinite(Number(row[0])) &&
    Number.isFinite(Number(row[1])) &&
    Number.isFinite(Number(row[2])) &&
    Number.isFinite(Number(row[3])) &&
    Number.isFinite(Number(row[4])) &&
    Number.isFinite(Number(row[6]))
  );
}

async function fetchFromBinance(endpoint, symbol, interval, limit) {
  const url = new URL(endpoint);
  url.searchParams.set('symbol', symbol);
  url.searchParams.set('interval', interval);
  url.searchParams.set('limit', String(limit));

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), UPSTREAM_TIMEOUT_MS);

  try {
    const response = await fetch(url, {
      headers: {
        Accept: 'application/json',
        'User-Agent': 'Futures-AI-Trader-Collector/1.0',
      },
      cache: 'no-store',
      signal: controller.signal,
    });

    const contentType = response.headers.get('content-type') || '';
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    if (!contentType.toLowerCase().includes('application/json')) {
      throw new Error(`invalid content-type ${contentType || 'missing'}`);
    }

    const payload = await response.json();
    if (!validateKlines(payload)) {
      throw new Error('invalid kline payload');
    }
    return payload;
  } finally {
    clearTimeout(timeout);
  }
}

export default async function handler(req, res) {
  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET');
    return res.status(405).json({ error: 'method_not_allowed' });
  }

  const expectedKey = process.env.COLLECTOR_API_KEY;
  if (expectedKey) {
    const suppliedKey = req.headers['x-collector-key'];
    if (suppliedKey !== expectedKey) {
      return res.status(401).json({ error: 'unauthorized' });
    }
  }

  const symbol = String(req.query.symbol || 'ETHUSDT').toUpperCase();
  const interval = String(req.query.interval || '1h');
  const parsedLimit = Number.parseInt(String(req.query.limit || DEFAULT_LIMIT), 10);
  const limit = Number.isFinite(parsedLimit)
    ? Math.min(Math.max(parsedLimit, 1), MAX_LIMIT)
    : DEFAULT_LIMIT;

  if (!ALLOWED_SYMBOLS.has(symbol) || !ALLOWED_INTERVALS.has(interval)) {
    return res.status(400).json({ error: 'unsupported_market_request' });
  }

  const failures = [];
  for (const endpoint of BINANCE_FUTURES_ENDPOINTS) {
    try {
      const klines = await fetchFromBinance(endpoint, symbol, interval, limit);
      res.setHeader('Cache-Control', 'no-store, max-age=0');
      res.setHeader('X-Market-Source', 'binance-usdm-futures');
      return res.status(200).json(klines);
    } catch (error) {
      failures.push(`${new URL(endpoint).host}:${error?.name || 'Error'}:${String(error?.message || error)}`);
    }
  }

  console.error('All Binance Futures upstreams failed', failures);
  return res.status(502).json({
    error: 'binance_futures_unavailable',
    source: 'binance-usdm-futures',
  });
}
