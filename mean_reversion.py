#!/usr/bin/env python3
"""
Mean Reversion Bot - Candle-synced + Live-exit monitoring

Key fixes applied:
- Correct candle-synchronization using kline close timestamps (ensures indicators update only after candle CLOSE)
- Live ticker polling for exit monitoring between candle closes
- Robust BinancePublic retry compatibility (handles older urllib3)
- Safer RSI handling (avoid divide-by-zero / NaNs)
- Clear ternary parentheses for slippage calculations
- Clean session close on shutdown

Usage: python mean_reversion_fixed.py
"""

import requests
import time
from datetime import datetime, UTC
import pandas as pd
import numpy as np
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --------------------- Logging ---------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BinancePublic:
    def __init__(self, base_url: str = "https://api.binance.us", timeout: int = 10):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "MeanReversionBot/1.0"})

        # Retry strategy compatible across urllib3 versions
        retry_kwargs = dict(
            total=3,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
        )
        try:
            retries = Retry(**retry_kwargs, allowed_methods=("GET",))
        except TypeError:
            retries = Retry(**retry_kwargs, method_whitelist=("GET",))

        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _get(self, path: str, params: dict = None):
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_ticker(self, symbol: str):
        return self._get("/api/v3/ticker/price", params={"symbol": symbol})

    def get_klines(self, symbol: str, interval: str = "1m", limit: int = 200):
        return self._get("/api/v3/klines", params={"symbol": symbol, "interval": interval, "limit": limit})

    def get_24hr_ticker(self, symbol: str):
        return self._get("/api/v3/ticker/24hr", params={"symbol": symbol})


class MeanReversionBot:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_capital: float = 10000.0,
        timeframe: str = "1m",
    ):
        # Client
        self.client = BinancePublic()

        # Trading parameters
        self.symbol = symbol
        self.capital = float(initial_capital)
        self.available_capital = float(initial_capital)
        self.position = None
        self.trades_history = []

        # Risk Management
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.rr_ratio = 2.0  # 1:2 Risk-Reward ratio

        # Timeframe
        self.timeframe = timeframe
        self.check_interval = self._get_check_interval(timeframe)

        # Strategy parameters
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bb_period = 20
        self.bb_std = 2

        # SL/TP placement basis
        self.sl_buffer = 0.005  # 0.5% buffer from BB bands

        # Market model
        self.fee_rate = 0.00075  # 0.075% per side (example)
        self.slippage_pct = 0.0005  # 0.05% slippage on fill

        # Safety caps
        self.min_price_risk_pct = 0.0005  # 0.05% minimum price risk guard
        self.max_account_risk_pct = 0.9  # don't allocate more than 90% of account to a single position

    def _get_check_interval(self, timeframe: str) -> int:
        intervals = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "4h": 14400,
        }
        return intervals.get(timeframe, 60)

    def sleep_until_ms(self, timestamp_ms: int):
        """Sleep until the given epoch timestamp in milliseconds (with small buffer)."""
        now_ms = int(time.time() * 1000)
        wait_ms = timestamp_ms - now_ms
        if wait_ms > 0:
            # add small safety buffer of 200ms to ensure candle is closed on server
            time.sleep((wait_ms + 200) / 1000.0)

    # ---------- Indicators ----------
    def calculate_rsi(self, prices, period: int = 14) -> float:
        s = pd.Series(prices).astype(float)
        if len(s) < period + 1:
            return 50.0
        delta = s.diff().dropna()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
        # avoid division by zero; where roll_down == 0, set RS to large number so RSI -> 100
        roll_down_safe = roll_down.replace(0, np.nan)
        rs = roll_up / roll_down_safe
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(100)
        last_rsi = float(rsi.iloc[-1])
        return last_rsi

    def calculate_bollinger_bands(self, prices, period: int = 20, std: float = 2.0):
        s = pd.Series(prices).astype(float)
        if len(s) < period:
            raise ValueError("Not enough data for Bollinger Bands")
        window = s.iloc[-period:]
        sma = float(window.mean())
        std_dev = float(window.std(ddof=0))
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band

    # ---------- Market Data (candle-aware) ----------
    def get_market_data(self):
        """Fetch klines and return closes for CLOSED candles and the timestamp of the last closed candle's close in ms.

        Returns: (closes:list[float], last_closed_close_ts_ms:int) or (None, None) on error
        """
        try:
            klines = self.client.get_klines(self.symbol, interval=self.timeframe, limit=200)
            if not klines or not isinstance(klines, list):
                return None, None

            now_ms = int(time.time() * 1000)
            # Binance kline format: [open_time, open, high, low, close, volume, close_time, ...]
            last_kline_close_ms = int(klines[-1][6])

            # If last kline's close_time is in the future, that kline is still open -> use previous as last closed
            if last_kline_close_ms > now_ms:
                if len(klines) < 2:
                    return None, None
                closes = [float(k[4]) for k in klines[:-1]]
                last_closed_close_ms = int(klines[-2][6])
            else:
                closes = [float(k[4]) for k in klines]
                last_closed_close_ms = last_kline_close_ms

            return closes, last_closed_close_ms

        except Exception as e:
            logger.warning(f"Error fetching market data: {e}")
            return None, None

    # ---------- Position Sizing ----------
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> dict:
        risk_amount = self.available_capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        min_price_risk = entry_price * self.min_price_risk_pct
        price_risk = max(price_risk, min_price_risk)

        if price_risk == 0:
            raise ValueError("Price risk evaluated to zero; cannot size position")

        position_size = risk_amount / price_risk
        max_size = (self.available_capital * self.max_account_risk_pct) / entry_price
        position_size = min(position_size, max_size)
        position_value = position_size * entry_price

        return {
            "size": round(position_size, 6),
            "value": round(position_value, 2),
            "risk_amount": round(risk_amount, 2),
        }

    # ---------- Signals & Execution ----------
    def check_entry_signals(self, prices):
        current_price = float(prices[-1])
        rsi = self.calculate_rsi(prices, self.rsi_period)

        try:
            upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(prices, self.bb_period, self.bb_std)
        except ValueError:
            return None

        # LONG
        if rsi < self.rsi_oversold and current_price <= lower_bb * 1.002:
            stop_loss = lower_bb * (1 - self.sl_buffer)
            risk = current_price - stop_loss
            if risk <= 0:
                return None
            take_profit = current_price + (risk * self.rr_ratio)

            position_info = self.calculate_position_size(current_price, stop_loss)
            if position_info["value"] > self.available_capital:
                logger.info("Not enough available capital to open LONG after sizing cap")
                return None

            return {
                "signal": "LONG",
                "entry": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "size": position_info["size"],
                "value": position_info["value"],
                "risk_amount": position_info["risk_amount"],
                "rsi": rsi,
                "lower_bb": lower_bb,
                "middle_bb": middle_bb,
                "upper_bb": upper_bb,
            }

        # SHORT
        elif rsi > self.rsi_overbought and current_price >= upper_bb * 0.998:
            stop_loss = upper_bb * (1 + self.sl_buffer)
            risk = stop_loss - current_price
            if risk <= 0:
                return None
            take_profit = current_price - (risk * self.rr_ratio)

            position_info = self.calculate_position_size(current_price, stop_loss)
            if position_info["value"] > self.available_capital:
                logger.info("Not enough available capital to open SHORT after sizing cap")
                return None

            return {
                "signal": "SHORT",
                "entry": current_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "size": position_info["size"],
                "value": position_info["value"],
                "risk_amount": position_info["risk_amount"],
                "rsi": rsi,
                "lower_bb": lower_bb,
                "middle_bb": middle_bb,
                "upper_bb": upper_bb,
            }

        return None

    def execute_paper_trade(self, signal: dict):
        value = float(signal["value"])
        size = float(signal["size"])
        entry_price = float(signal["entry"])

        if value > self.available_capital:
            logger.info("Not enough available capital to open position after rounding. Skipping.")
            return

        self.available_capital -= value
        signal["reserved_capital"] = value

        effective_entry = entry_price * ((1 + self.slippage_pct) if signal["signal"] == "LONG" else (1 - self.slippage_pct))
        entry_fee = effective_entry * size * self.fee_rate
        signal["effective_entry"] = effective_entry
        signal["entry_fee"] = entry_fee

        self.position = signal
        self.position["entry_time"] = datetime.now(UTC)

        logger.info("PAPER TRADE EXECUTED - %s | Entry: %.2f | Size: %.6f | Reserved: $%.2f", signal["signal"], entry_price, size, value)
        logger.info("  RSI: %.2f | BB Lower: %.2f | BB Middle: %.2f | BB Upper: %.2f", signal["rsi"], signal["lower_bb"], signal["middle_bb"], signal["upper_bb"])

    # ---------- Exit management (uses live ticker) ----------
    def check_exit(self, current_price: float):
        if not self.position:
            return

        signal_type = self.position["signal"]
        size = float(self.position["size"])
        entry_price = float(self.position["entry"])
        take_profit = float(self.position["take_profit"])
        stop_loss = float(self.position["stop_loss"])

        if signal_type == "LONG":
            if current_price >= take_profit:
                pnl = (take_profit - entry_price) * size
                self.close_position(take_profit, "TAKE_PROFIT", pnl)
            elif current_price <= stop_loss:
                pnl = (stop_loss - entry_price) * size
                self.close_position(stop_loss, "STOP_LOSS", pnl)

        elif signal_type == "SHORT":
            if current_price <= take_profit:
                pnl = (entry_price - take_profit) * size
                self.close_position(take_profit, "TAKE_PROFIT", pnl)
            elif current_price >= stop_loss:
                pnl = (entry_price - stop_loss) * size
                self.close_position(stop_loss, "STOP_LOSS", pnl)

    def close_position(self, exit_price: float, reason: str, raw_pnl: float):
        if not self.position:
            return

        size = float(self.position["size"])
        reserved = float(self.position.get("reserved_capital", 0.0))
        entry_price = float(self.position["entry"])
        effective_entry = float(self.position.get("effective_entry", entry_price))

        effective_exit = exit_price * ((1 - self.slippage_pct) if self.position["signal"] == "LONG" else (1 + self.slippage_pct))
        exit_fee = effective_exit * size * self.fee_rate
        entry_fee = float(self.position.get("entry_fee", effective_entry * size * self.fee_rate))

        pnl_after_fees = raw_pnl - (entry_fee + exit_fee)

        self.position["exit_price"] = exit_price
        self.position["exit_time"] = datetime.now(UTC)
        self.position["exit_reason"] = reason
        self.position["pnl"] = pnl_after_fees
        self.position["entry_fee"] = entry_fee
        self.position["exit_fee"] = exit_fee

        self.available_capital += reserved + pnl_after_fees

        self.trades_history.append(self.position)

        value = float(self.position.get("value", reserved))
        pnl_pct = (pnl_after_fees / value * 100) if value != 0 else 0.0
        logger.info("POSITION CLOSED - %s | Exit: %.2f | PnL: %.2f (%.2f%%) | New Capital: %.2f", reason, exit_price, pnl_after_fees, pnl_pct, self.available_capital)

        self.position = None

    # ---------- Main loop & reporting ----------
    def run(self):
        logger.info("MEAN REVERSION BOT STARTING - Symbol: %s | Timeframe: %s | Initial Capital: $%.2f", self.symbol, self.timeframe, self.capital)

        iteration = 0
        min_required = max(self.bb_period, self.rsi_period + 1)
        interval_ms = int(self.check_interval * 1000)

        try:
            while True:
                iteration += 1
                closes, last_closed_close_ms = self.get_market_data()

                if not closes or len(closes) < min_required:
                    count = len(closes) if closes else 0
                    logger.info("Waiting for data (%s/%s)", count, min_required)
                    # If we have last_closed_close_ms, sleep until next candle close; else sleep a default
                    if last_closed_close_ms:
                        next_close_ms = last_closed_close_ms + interval_ms
                        self.sleep_until_ms(next_close_ms)
                    else:
                        time.sleep(self.check_interval)
                    continue

                current_price = float(closes[-1])

                # Evaluate entry/exit on CLOSED candle (indicators computed from closes)
                if self.position is None:
                    signal = self.check_entry_signals(closes)
                    if signal:
                        self.execute_paper_trade(signal)
                    else:
                        rsi = self.calculate_rsi(closes, self.rsi_period)
                        logger.info("#%s | Monitoring | Price: %.2f | RSI: %.1f | Capital: $%.2f", iteration, current_price, rsi, self.available_capital)

                else:
                    # If in position, we still want to check exit immediately using latest closed price
                    self.check_exit(current_price)

                # Determine next candle close and then monitor exits with live ticker until then
                next_close_ms = last_closed_close_ms + interval_ms

                # While waiting for next candle close, if we have a position poll live ticker frequently for exit
                last_heartbeat = 0
                while True:
                    now_ms = int(time.time() * 1000)
                    if now_ms >= next_close_ms:
                        break

                    if self.position:
                        try:
                            live = self.client.get_ticker(self.symbol)
                            live_price = float(live.get("price"))
                            self.check_exit(live_price)
                            if not self.position:
                                # position closed by live price, break to re-fetch new candles
                                break
                        except Exception as e:
                            logger.debug("Live ticker error: %s", e)
                                                # ---- LIVE PnL STREAMING (heartbeat every ~10s) ----
                        entry = float(self.position["entry"])
                        size = float(self.position["size"])
                        direction = self.position["signal"]

                        if direction == "LONG":
                            pnl_usd = (live_price - entry) * size
                            pnl_pct = (live_price - entry) / entry * 100
                        else:
                            pnl_usd = (entry - live_price) * size
                            pnl_pct = (entry - live_price) / entry * 100

                        sl_dist_pct = abs(live_price - self.position["stop_loss"]) / live_price * 100
                        tp_dist_pct = abs(self.position["take_profit"] - live_price) / live_price * 100

                        if now_ms - last_heartbeat > 10_000:
                            logger.info(
                                "[LIVE PnL] Price: %.2f | PnL: %+0.2f USD (%+.2f%%) | SL dist: %.2f%% | TP dist: %.2f%%",
                                live_price,
                                pnl_usd,
                                pnl_pct,
                                sl_dist_pct,
                                tp_dist_pct,
                            )
                            last_heartbeat = now_ms

                        time.sleep(2)  # poll every 2 seconds for exit
                    else:
                        # No open position -> sleep until candle close
                        self.sleep_until_ms(next_close_ms)
                        break

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            self.print_summary()
        except Exception as e:
            logger.exception("Unhandled exception in bot loop: %s", e)
            self.print_summary()

    def print_summary(self):
        if not self.trades_history:
            logger.info("No trades executed.")
        else:
            total_pnl = sum(t.get("pnl", 0.0) for t in self.trades_history)
            winning_trades = [t for t in self.trades_history if t.get("pnl", 0.0) > 0]
            losing_trades = [t for t in self.trades_history if t.get("pnl", 0.0) < 0]

            logger.info("TRADING SUMMARY")
            logger.info("Total Trades: %d", len(self.trades_history))
            logger.info("Winning Trades: %d (%.1f%%)", len(winning_trades), len(winning_trades) / len(self.trades_history) * 100)
            logger.info("Losing Trades: %d (%.1f%%)", len(losing_trades), len(losing_trades) / len(self.trades_history) * 100)
            logger.info("Total PnL: $%.2f", total_pnl)
            logger.info("Final Capital: $%.2f", self.available_capital)
            logger.info("Return: %.2f%%", (self.available_capital - self.capital) / self.capital * 100)

        # Close HTTP session cleanly
        try:
            self.client.session.close()
        except Exception:
            pass


if __name__ == "__main__":
    bot = MeanReversionBot(symbol="BTCUSDT", initial_capital=10000, timeframe="1m")
    bot.run()
