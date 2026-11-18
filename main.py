#!/usr/bin/env python3
"""cbBTC minute-bar trading bot for agent-fun/bars-replay/v1 (make.fun btc-race).

Strategy: trend following on the one-hour scale, with impact-aware
sizing and layered drawdown control.

The market model charges roughly 6 bps of fee+slippage per side plus
constant-product price impact that grows with order size relative to pool
depth, so a full-size round trip costs ~26 bps or more. Short-horizon
reversals in this market are worth only a few bps and cannot pay that toll;
what carries is trend persistence at the 30-120 minute scale. The bot
therefore:

- stays in cash unless an uptrend is established (fast EMA above slow EMA
  with a confirming positive trailing return), then rides it at near-full
  size, sized down against the pool's latest close depth so its own price
  impact stays in single-digit bps;
- exits when the trend rolls over (EMA cross back), when price gives back
  too much from its peak (trailing stop), on a hard stop, or on a long
  time stop;
- flattens and pauses if equity drawdown from its peak reaches 12%, and
  stops for the window at 20%, far inside the race's 25% guardrail.

The bot is long only, holds at most one position, is fully deterministic
(it reads nothing but the harness's own lines: no clock, no files, no
environment, no randomness), and speaks the protocol directly: one JSON
line in, exactly one reply line per batch out.
"""

import json
import sys
from collections import deque

SYMBOL = "CBBTC"

# --- strategy parameters -------------------------------------------------
HISTORY = 400            # closes kept for the trailing-return lookback
EMA_FAST = 25            # fast EMA span, in bars
EMA_SLOW = 120           # slow EMA span, in bars
EMA_GAP_MIN_BPS = 14.0   # the fast EMA must clear the slow one by this much: a bare cross is noise
RET_LOOKBACK = 60        # trailing return window for trend confirmation
RET_IN_BPS = 10.0        # enter only when the trailing return exceeds this
LONG_LEN = 240           # slow mean: the regime filter's horizon
RET_LONG_IN_BPS = 15.0   # ...and the trailing return over that horizon must exceed this
PROGRESS_BARS = 75       # a trade that has not worked within this many bars is wrong: exit
PROGRESS_MIN_BPS = 0.0   # "worked" means the close is at least this far above the entry
MIN_BARS = 130           # bars of history required before any order
SIZE_FRAC = 0.95         # share of cash a single entry may spend
IMPACT_SHARE = 0.00055   # cap an order's notional at this share of pool depth (~5.5 bps impact)
STOP_BPS = 110           # hard stop below the entry fill price
TRAIL_GIVE_BPS = 35      # exit when price gives this much back from its peak
MAX_HOLD = 720           # time stop, in bars
COOLDOWN_EXIT = 15       # bars to sit out after an exit
COOLDOWN_STOP = 45       # bars to sit out after a stopped trade
# drawdown control (share of peak equity)
DD_PAUSE = 0.12          # flatten and pause entries at this drawdown
DD_HALT = 0.20           # flatten and stop trading for the window at this one
DD_PAUSE_BARS = 180

BUY_SLIP_BPS = 300       # a buy that would fill further above the close than this is skipped
SELL_SLIP_BPS = 3000     # exits are worth filling: allow a wide band


class Bot:
    def __init__(self, hello):
        self.hello = hello
        self.warmup = int(hello.get("warmup_bars", 0) or 0)
        self.closes = deque(maxlen=HISTORY)
        self.seq = -1
        self.ema_fast = None
        self.ema_slow = None
        # position bookkeeping, refreshed from every batch's fills/account
        self.entry_px = None       # raw fill price of the open position
        self.entry_seq = None
        self.peak_px = None        # highest close since entry
        self.cooldown_until = -1   # seq before which no new entry is allowed
        self.halted = False
        self.peak_equity = None

    # -- order helpers ---------------------------------------------------
    def _buy(self, notional):
        return {
            "side": "buy",
            "client_id": f"b{self.seq}",
            "symbol": SYMBOL,
            "notional_quote_units": str(int(notional)),
            "max_slippage_bps": BUY_SLIP_BPS,
        }

    def _sell(self, qty):
        return {
            "side": "sell",
            "client_id": f"s{self.seq}",
            "symbol": SYMBOL,
            "qty": str(int(qty)),
            "max_slippage_bps": SELL_SLIP_BPS,
        }

    # -- main decision ---------------------------------------------------
    def on_batch(self, batch):
        self.seq = batch["seq"]
        account = batch["account"]
        cash = int(account["cash_quote_units"])
        equity = int(account["equity_quote_units"])
        held = None
        for pos in account["positions"]:
            if pos["symbol"] == SYMBOL and int(pos["qty"]) > 0:
                held = pos

        # learn the entry price from the fills this batch reports
        for fill in batch["fills"]:
            if fill["symbol"] != SYMBOL or fill["status"] != "filled":
                continue
            if fill["side"] == "buy":
                self.entry_px = int(fill["price"])
                self.entry_seq = self.seq
                self.peak_px = self.entry_px
            elif fill["side"] == "sell" and held is None:
                self.entry_px = None
                self.entry_seq = None
                self.peak_px = None

        bar = None
        for b in batch["bars"]:
            if b["symbol"] == SYMBOL:
                bar = b
        if bar is None:
            return []  # no new bar for our asset: nothing to decide on
        close = int(bar["c"])
        self.closes.append(close)
        if self.ema_fast is None:
            self.ema_fast = float(close)
            self.ema_slow = float(close)
        else:
            kf = 2.0 / (EMA_FAST + 1)
            ks = 2.0 / (EMA_SLOW + 1)
            self.ema_fast += (close - self.ema_fast) * kf
            self.ema_slow += (close - self.ema_slow) * ks

        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = 0.0
        if self.peak_equity:
            drawdown = 1.0 - equity / self.peak_equity

        if self.seq < self.warmup or len(self.closes) < MIN_BARS:
            return []

        # ---- manage an open position ----
        if held is not None:
            qty = int(held["qty"])
            if close > (self.peak_px or close):
                self.peak_px = close
            if drawdown >= DD_PAUSE or self._exit_signal(close):
                if drawdown >= DD_HALT:
                    self.halted = True
                stopped = (
                    self.entry_px is not None
                    and close <= self.entry_px * (1 - STOP_BPS / 10000.0)
                )
                pause = DD_PAUSE_BARS if drawdown >= DD_PAUSE else (
                    COOLDOWN_STOP if stopped else COOLDOWN_EXIT
                )
                self.cooldown_until = max(self.cooldown_until, self.seq + pause)
                return [self._sell(qty)]
            return []

        # ---- flat: look for an entry ----
        if self.halted or drawdown >= DD_PAUSE or self.seq < self.cooldown_until:
            return []
        if cash <= 0:
            return []

        # trend module: ride an established uptrend the slow horizon
        # agrees with (see _trend_conditions)
        if self._trend_conditions(close):
            notional = self._entry_size(cash, bar, SIZE_FRAC)
            if notional >= 1_000_000:  # protocol minimum buy is 1 USDC
                return [self._buy(notional)]
        return []

    def _trend_conditions(self, close):
        """An established uptrend the slow horizon agrees with. In a
        market sliding over hours, hourly "trends" are whipsaws, so the
        fast EMA must clear the slow one by a real margin, price must sit
        above the slow mean, and the slow mean itself must be rising: a
        merely positive long lookback still points "up" for hours after
        a top."""
        closes = list(self.closes)
        if len(closes) <= LONG_LEN + 60:
            return False
        past = closes[-RET_LOOKBACK - 1]
        ret_bps = (close / past - 1.0) * 10000.0 if past > 0 else 0.0
        ma_long = sum(closes[-LONG_LEN:]) / LONG_LEN
        ma_long_ago = sum(closes[-LONG_LEN - 60:-60]) / LONG_LEN
        past_long = closes[-LONG_LEN - 1]
        ret_long_bps = (close / past_long - 1.0) * 10000.0 if past_long > 0 else 0.0
        trend_up = (
            self.ema_fast > self.ema_slow * (1 + EMA_GAP_MIN_BPS / 10000.0)
            and close > self.ema_slow
        )
        regime_up = close > ma_long and ma_long > ma_long_ago and ret_long_bps >= RET_LONG_IN_BPS
        return trend_up and regime_up and ret_bps >= RET_IN_BPS

    def _entry_size(self, cash, bar, frac):
        """Notional for an entry: a share of cash, capped so the order's
        own price impact against the pool's latest close depth stays in
        single-digit bps."""
        size_cap = cash * frac
        c_depth = bar.get("c_depth") or []
        if c_depth:
            size_cap = min(size_cap, IMPACT_SHARE * int(c_depth[0]))
        return int(size_cap)

    def _exit_signal(self, close):
        if self.entry_px is None or self.entry_seq is None:
            return False
        held_bars = self.seq - self.entry_seq
        peak = self.peak_px or close
        if close <= self.entry_px * (1 - STOP_BPS / 10000.0):
            return True  # hard stop
        if held_bars >= MAX_HOLD:
            return True  # time stop
        if held_bars >= PROGRESS_BARS:
            if close < self.entry_px * (1 + PROGRESS_MIN_BPS / 10000.0):
                return True  # no progress: the trend premise did not play out
        if self.ema_fast < self.ema_slow:
            return True  # trend rolled over
        if peak > self.entry_px and close <= peak * (1 - TRAIL_GIVE_BPS / 10000.0):
            return True  # trailing stop
        return False


def main():
    bot = None
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "hello":
            bot = Bot(msg)
        elif kind == "batch":
            orders = bot.on_batch(msg) if bot is not None else []
            reply = {"type": "reply", "seq": msg["seq"], "orders": orders}
            out.write(json.dumps(reply, separators=(",", ":")) + "\n")
            out.flush()
        elif kind == "done":
            break


if __name__ == "__main__":
    main()
