
import requests
import csv
import time
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime, timezone



# ==========================================
# 1. SETTINGS
# ==========================================

# Data Collection
COLLECT_NEW_DATA = False
CSV_FILE = "btcusdt_final_test.csv"
SNAPSHOT_COUNT = 2000

# Trading Settings
QUANTITY = 0.001
FEE_RATE = 0.001

# OBI + OFI Strategy Settings
UPPER_THRESHOLD = 0.95
LOWER_THRESHOLD = -0.95
OFI_THRESHOLD = 0.001


# ==========================================
# 2. BINANCE API
# ==========================================

url = "https://api.binance.com/api/v3/depth"

params = {
    "symbol": "BTCUSDT",
    "limit": 5
}


def get_orderbook():

    response = requests.get(
        url,
        params=params,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    return {
        "update_id": data["lastUpdateId"],
        "bid_price": float(data["bids"][0][0]),
        "ask_price": float(data["asks"][0][0]),
        "bid_volume": float(data["bids"][0][1]),
        "ask_volume": float(data["asks"][0][1])
    }


# ==========================================
# 3. FEATURE CALCULATION
# ==========================================

def calculate_features(book):

    bid = book["bid_price"]
    ask = book["ask_price"]

    bid_volume = book["bid_volume"]
    ask_volume = book["ask_volume"]

    # Spread
    spread = round(ask - bid, 2)

    # Midprice
    midprice = (bid + ask) / 2

    # Total volume
    total_volume = bid_volume + ask_volume

    if total_volume <= 0:
        raise ValueError("Toplam hacim sifir")

    # Order Book Imbalance
    obi = (
        (bid_volume - ask_volume)
        / total_volume
    )

    # Microprice
    microprice = (
        ask * bid_volume
        + bid * ask_volume
    ) / total_volume

    return {
        "spread": spread,
        "midprice": midprice,
        "obi": obi,
        "microprice": microprice
    }


# ==========================================
# 4. ORDER FLOW IMBALANCE
# ==========================================

def calculate_ofi(old_book, new_book):

    # BID SIDE

    if new_book["bid_price"] > old_book["bid_price"]:

        bid_ofi = new_book["bid_volume"]

    elif new_book["bid_price"] < old_book["bid_price"]:

        bid_ofi = -old_book["bid_volume"]

    else:

        bid_ofi = (
            new_book["bid_volume"]
            - old_book["bid_volume"]
        )

    # ASK SIDE

    if new_book["ask_price"] < old_book["ask_price"]:

        ask_ofi = -new_book["ask_volume"]

    elif new_book["ask_price"] > old_book["ask_price"]:

        ask_ofi = old_book["ask_volume"]

    else:

        ask_ofi = (
            old_book["ask_volume"]
            - new_book["ask_volume"]
        )

    return bid_ofi + ask_ofi


# ==========================================
# 5. DATA COLLECTION
# ==========================================

def collect_data():

    old_book = None

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "timestamp",
            "update_id",
            "bid_price",
            "ask_price",
            "bid_volume",
            "ask_volume",
            "spread",
            "midprice",
            "obi",
            "microprice",
            "ofi"
        ])

        for i in range(SNAPSHOT_COUNT):

            try:

                book = get_orderbook()

                # Tekrar eden snapshot'lari atla
                if (
                    old_book is not None
                    and book["update_id"] <= old_book["update_id"]
                ):
                    continue

                # Feature calculation
                features = calculate_features(book)

                # OFI calculation
                if old_book is None:

                    ofi = None

                else:

                    ofi = calculate_ofi(
                        old_book,
                        book
                    )

                # Timestamp
                timestamp = datetime.now(
                    timezone.utc
                ).isoformat()

                # CSV row
                writer.writerow([
                    timestamp,
                    book["update_id"],
                    book["bid_price"],
                    book["ask_price"],
                    book["bid_volume"],
                    book["ask_volume"],
                    features["spread"],
                    features["midprice"],
                    features["obi"],
                    features["microprice"],
                    ofi
                ])

                file.flush()

                old_book = book

                print("Snapshot:", i + 1)

            except (
                requests.RequestException,
                ValueError,
                KeyError,
                IndexError
            ) as error:

                print("Hata:", error)

            finally:
                time.sleep(0.5)


    print("Data collection tamamlandi!")


# ==========================================
# 6. PANDAS DATA ANALYSIS
# ==========================================

if COLLECT_NEW_DATA:

    collect_data()


df = pd.read_csv(CSV_FILE)

print("\nDATASET:")
print(df.head())


# ==========================================
# 7. DATA CLEANING
# ==========================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values(
    "timestamp"
).reset_index(drop=True)

# Spread hassasiyetini duzelt
df["spread"] = df["spread"].round(2)

print("\nSPREAD VALUES:")
print(df["spread"].value_counts())


# ==========================================
# 8. MIDPRICE CHANGE ANALYSIS
# ==========================================

df["next_midprice"] = (
    df["midprice"].shift(-1)
)

df["midprice_change"] = (
    df["next_midprice"] - df["midprice"]
)

print("\nMIDPRICE CHANGES:")

print(
    df[[
        "timestamp",
        "midprice",
        "next_midprice",
        "midprice_change"
    ]].head(10)
)


# ==========================================
# 9. MICROPRICE SIGNAL
# ==========================================

df["microprice_signal"] = (
    df["microprice"] - df["midprice"]
)


# ==========================================
# 10. FEATURE CORRELATION ANALYSIS
# ==========================================

features = [
    "spread",
    "obi",
    "ofi",
    "microprice_signal"
]

analysis_df = df.dropna(
    subset=features + ["midprice_change"]
).copy()

print("\nFEATURE CORRELATIONS:")

for feature in features:

    # Sabit feature icin korelasyon tanimsizdir
    if analysis_df[feature].nunique() <= 1:

        print(feature, "correlation: undefined")

    else:

        correlation = analysis_df[feature].corr(
            analysis_df["midprice_change"]
        )

        print(
            feature,
            "correlation:",
            round(correlation, 4)
        )


# ==========================================
# 11. DESCRIPTIVE STATISTICS
# ==========================================

print("\nTOTAL OBSERVATIONS:")
print(len(analysis_df))

print("\nDESCRIPTIVE STATISTICS:")

print(
    analysis_df[
        features + ["midprice_change"]
    ].describe()
)


# ==========================================
# 12. MIDPRICE MOVEMENT CHECK
# ==========================================

total = len(analysis_df)

unchanged = (
    analysis_df["midprice_change"] == 0
).sum()

print("\nMIDPRICE MOVEMENT CHECK:")

print("Total observations:", total)
print("Unchanged midprice:", unchanged)

if total > 0:

    unchanged_pct = (
        unchanged / total * 100
    )

    print(
        "Unchanged percentage:",
        round(unchanged_pct, 2),
        "%"
    )


# ==========================================
# 13. FEATURE VISUALIZATION
# ==========================================

for feature in features:

    plt.figure(figsize=(8, 5))

    plt.scatter(
        analysis_df[feature],
        analysis_df["midprice_change"],
        alpha=0.5
    )

    plt.xlabel(feature)

    plt.ylabel("Next Midprice Change (USDT)")

    plt.title(
        feature + " vs Next Midprice Change"
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.show()




# ==========================================
# 14. MULTI-FEATURE SIGNAL GENERATION
# ==========================================

analysis_df["signal"] = 0

# LONG
analysis_df.loc[
    (analysis_df["obi"] > UPPER_THRESHOLD)
    & (analysis_df["ofi"] > OFI_THRESHOLD),
    "signal"
] = 1

# SHORT
analysis_df.loc[
    (analysis_df["obi"] < LOWER_THRESHOLD)
    & (analysis_df["ofi"] < -OFI_THRESHOLD),
    "signal"
] = -1

print("\n===== TRADING SIGNALS =====")

print(analysis_df["signal"].value_counts())



# ==========================================
# 15. MULTIPLE TRADES BACKTEST
# ==========================================

# Islem sonuclarini saklayacagimiz liste
trades = []

# Sinyalleri df index'i ile hizala
signals = analysis_df["signal"].reindex(
    df.index,
    fill_value=0
)

i = 0

while i < len(df) - 2:

    signal = signals.iloc[i]

    # Sinyal yoksa sonraki snapshot'a gec
    if signal == 0:

        i = i + 1
        continue

    # Giris ve cikis snapshot'lari
    entry = df.iloc[i + 1]
    exit_book = df.iloc[i + 2]

    # LONG TRADE
    if signal == 1:

        entry_price = entry["ask_price"]
        exit_price = exit_book["bid_price"]

        gross_pnl = (
            exit_price - entry_price
        ) * QUANTITY

    # SHORT TRADE
    elif signal == -1:

        entry_price = entry["bid_price"]
        exit_price = exit_book["ask_price"]

        gross_pnl = (
            entry_price - exit_price
        ) * QUANTITY

    # TRANSACTION COSTS

    entry_fee = (
        entry_price * QUANTITY * FEE_RATE
    )

    exit_fee = (
        exit_price * QUANTITY * FEE_RATE
    )

    total_fee = entry_fee + exit_fee

    # NET PNL
    net_pnl = gross_pnl - total_fee

    # ISLEM KAYDI
    trades.append({
        "signal_time": df.iloc[i]["timestamp"],
        "entry_time": entry["timestamp"],
        "exit_time": exit_book["timestamp"],
        "direction": "LONG" if signal == 1 else "SHORT",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "quantity": QUANTITY,
        "gross_pnl": gross_pnl,
        "total_fee": total_fee,
        "net_pnl": net_pnl
    })

    # Pozisyon kapanana kadarki sinyalleri atla
    i = i + 2


# ==========================================
# 16. BACKTEST RESULTS
# ==========================================

trades_df = pd.DataFrame(trades)

if not trades_df.empty:

    total_trades = len(trades_df)

    winning_trades = (
        trades_df["net_pnl"] > 0
    ).sum()

    losing_trades = (
        trades_df["net_pnl"] < 0
    ).sum()

    total_gross_pnl = (
        trades_df["gross_pnl"].sum()
    )

    total_fees = (
        trades_df["total_fee"].sum()
    )

    total_net_pnl = (
        trades_df["net_pnl"].sum()
    )

    win_rate = (
        winning_trades / total_trades * 100
    )

    print("\n===== BACKTEST RESULTS =====")

    print("Total Trades:", total_trades)

    print("Winning Trades:", winning_trades)

    print("Losing Trades:", losing_trades)

    print("Win Rate:", round(win_rate, 2), "%")

    print(
        "Gross PnL:",
        round(total_gross_pnl, 6),
        "USDT"
    )

    print(
        "Total Fees:",
        round(total_fees, 6),
        "USDT"
    )

    print(
        "Net PnL:",
        round(total_net_pnl, 6),
        "USDT"
    )

    print("\nTRADE HISTORY:")

    print(trades_df.to_string(index=False))

else:

    print("Backtest icin uygun islem bulunamadi.")



# ==========================================
# 17. CUMULATIVE PNL ANALYSIS
# ==========================================

if not trades_df.empty:

    # Her isleme bir numara ver
    trades_df["trade_number"] = range(
        1,
        len(trades_df) + 1
    )

    # Birikimli brut PnL
    trades_df["cumulative_gross_pnl"] = (
        trades_df["gross_pnl"].cumsum()
    )

    # Birikimli net PnL
    trades_df["cumulative_net_pnl"] = (
        trades_df["net_pnl"].cumsum()
    )

    # Grafik olustur
    plt.figure(figsize=(10, 5))

    plt.plot(
        trades_df["trade_number"],
        trades_df["cumulative_gross_pnl"],
        label="Gross PnL"
    )

    plt.plot(
        trades_df["trade_number"],
        trades_df["cumulative_net_pnl"],
        label="Net PnL"
    )

    plt.axhline(
        y=0,
        color="black",
        linestyle="--"
    )

    plt.xlabel("Trade Number")
    plt.ylabel("Cumulative PnL (USDT)")

    plt.title("OBI Strategy - Cumulative PnL")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ==========================================
# 18. TRANSACTION COST SENSITIVITY
# ==========================================

if not trades_df.empty:

    fee_rates = [0, 0.0001, 0.001]

    print("\n===== FEE SENSITIVITY =====")

    for fee_rate in fee_rates:

        # Her islemin toplam komisyonu
        fees = (
            trades_df["entry_price"]
            + trades_df["exit_price"]
        ) * QUANTITY * fee_rate

        # Her islemin net PnL degeri
        net_pnls = trades_df["gross_pnl"] - fees

        # Butun islemlerin toplam sonucu
        total_net_pnl = net_pnls.sum()

        print(
            f"Fee: %{fee_rate * 100:.3f}"
            f" | Net PnL: {total_net_pnl:.6f} USDT"
        )



# ==========================================
# 19. BREAK-EVEN ANALYSIS
# ==========================================

if not trades_df.empty:

    # Ortalama brut PnL
    avg_gross_pnl = trades_df["gross_pnl"].mean()

    # Ortalama net PnL
    avg_net_pnl = trades_df["net_pnl"].mean()

    # Toplam islem hacmi (USDT)
    total_turnover = (
        (trades_df["entry_price"] + trades_df["exit_price"])
        * trades_df["quantity"]
    ).sum()

    # Basa bas komisyon orani
    breakeven_fee = (
        trades_df["gross_pnl"].sum()
        / total_turnover
    )

    print("\n===== BREAK-EVEN ANALYSIS =====")

    print("Average Gross PnL:", round(avg_gross_pnl, 6))

    print("Average Net PnL:", round(avg_net_pnl, 6))

    print("Break-Even Fee (%):", round(breakeven_fee * 100, 6))

    print("Break-Even Fee (bps):", round(breakeven_fee * 10000, 4))


# ==========================================
# 20. SIGNAL DECAY ANALYSIS
# ==========================================

# 0'dan 20'ye kadar lag degerleri

# Longer signal decay analysis

lags = [
    0, 1, 2, 3, 5,
    10, 20, 30, 40, 50,
    75, 100, 125, 150,
    175, 200
]

obi_correlations = []
ofi_correlations = []

for lag in lags:

    price_start = df["midprice"].shift(-lag)

    price_end = df["midprice"].shift(-(lag + 1))

    future_change = price_end - price_start

    obi_corr = df["obi"].corr(future_change)

    ofi_corr = df["ofi"].corr(future_change)

    obi_correlations.append(obi_corr)

    ofi_correlations.append(ofi_corr)

    print(
        f"Lag {lag} | "
        f"OBI: {obi_corr:.4f} | "
        f"OFI: {ofi_corr:.4f}"
    )


# ==========================================
# 21. SIGNAL DECAY VISUALIZATION
# ==========================================

plt.figure(figsize=(9, 5))

plt.plot(
    lags,
    obi_correlations,
    marker="o",
    label="OBI"
)

plt.plot(
    lags,
    ofi_correlations,
    marker="o",
    label="OFI"
)

plt.axhline(
    y=0,
    color="black",
    linestyle="--"
)

plt.xlabel("Lag (Snapshots)")

plt.ylabel("Correlation")

plt.title("Signal Decay: OBI vs OFI")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("signal_decay.png", dpi=150)

plt.show()


# ==========================================
# 22. HOLDING PERIOD ANALYSIS
# ==========================================

def holding_period_backtest(holding_period):

    results = []

    i = 0

    while i + 1 + holding_period < len(df):

        signal = signals.iloc[i]

        if signal == 0:
            i += 1
            continue

        # Giris ve cikis snapshot'lari
        entry = df.iloc[i + 1]

        exit_book = df.iloc[i + 1 + holding_period]

        # LONG PnL
        if signal == 1:

            entry_price = entry["ask_price"]
            exit_price = exit_book["bid_price"]

            gross_pnl = (
                exit_price - entry_price
            ) * QUANTITY

        # SHORT PnL
        else:

            entry_price = entry["bid_price"]
            exit_price = exit_book["ask_price"]

            gross_pnl = (
                entry_price - exit_price
            ) * QUANTITY

        # Komisyonlar
        fee = (
            entry_price + exit_price
        ) * QUANTITY * FEE_RATE

        net_pnl = gross_pnl - fee

        results.append({
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl
        })

        # Pozisyon kapanana kadar yeni islem acma
        i += holding_period + 1

    return pd.DataFrame(results)


# ==========================================
# 23. COMPARE HOLDING PERIODS
# ==========================================

holding_periods = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 50, 60,70, 80, 90, 100, 150, 200, 250]
print("\n===== HOLDING PERIOD ANALYSIS =====")

for period in holding_periods:

    result = holding_period_backtest(period)

    if result.empty:
        print("Period:", period, "| No trades")
        continue

    print(
        f"Period: {period} | "
        f"Trades: {len(result)} | "
        f"Gross PnL: {result['gross_pnl'].sum():.6f} | "
        f"Net PnL: {result['net_pnl'].sum():.6f} | "
        f"Avg Net PnL: {result['net_pnl'].mean():.6f}"
    )




# ==========================================
# 25. STRONG OBI PRICE MOVEMENT ANALYSIS
# ==========================================

OBI_ENTRY_THRESHOLD = 0.90

future_periods = [10, 30, 50, 100]

print("\n===== STRONG OBI PRICE MOVEMENT =====")

for period in future_periods:

    gross_results = []
    net_results = []
    price_movements = []
    fees = []

    for i in range(len(df) - period - 1):

        obi_now = df.iloc[i]["obi"]

        # Determine signal direction
        if obi_now > OBI_ENTRY_THRESHOLD:
            direction = 1

        elif obi_now < -OBI_ENTRY_THRESHOLD:
            direction = -1

        else:
            continue

        # Enter after the signal
        entry_row = df.iloc[i + 1]

        # Exit after the selected holding period
        exit_row = df.iloc[i + 1 + period]

        if direction == 1:
            entry_price = entry_row["ask_price"]
            exit_price = exit_row["bid_price"]

            price_move = exit_price - entry_price

        else:
            entry_price = entry_row["bid_price"]
            exit_price = exit_row["ask_price"]

            price_move = entry_price - exit_price

        # Gross profit
        gross_pnl = price_move * QUANTITY

        # Binance transaction costs
        total_fee = (
            entry_price + exit_price
        ) * QUANTITY * FEE_RATE

        # Net profit
        net_pnl = gross_pnl - total_fee

        price_movements.append(price_move)
        gross_results.append(gross_pnl)
        fees.append(total_fee)
        net_results.append(net_pnl)

    if len(net_results) == 0:
        print(f"Period {period}: No signals")
        continue

    total_signals = len(net_results)

    avg_price_move = sum(price_movements) / total_signals
    avg_gross = sum(gross_results) / total_signals
    avg_fee = sum(fees) / total_signals
    avg_net = sum(net_results) / total_signals

    profitable_signals = sum(
        1 for pnl in net_results if pnl > 0
    )

    profitable_rate = (
        profitable_signals / total_signals
    ) * 100

    print(f"\nPeriod: {period}")
    print(f"Signals: {total_signals}")
    print(f"Average Price Move: {avg_price_move:.4f} USDT")
    print(f"Average Gross PnL: {avg_gross:.6f} USDT")
    print(f"Average Fee: {avg_fee:.6f} USDT")
    print(f"Average Net PnL: {avg_net:.6f} USDT")
    print(f"Net Profitable Signals: {profitable_rate:.2f}%")


# ==========================================
# 26. OBI STRENGTH VS FUTURE PRICE MOVEMENT
# ==========================================

obi_bins = [0.50, 0.70, 0.80, 0.90, 0.95, 1.01]

obi_labels = [
    "0.50 - 0.70",
    "0.70 - 0.80",
    "0.80 - 0.90",
    "0.90 - 0.95",
    "0.95 - 1.00"
]

future_periods = [10, 30, 50, 100]

print("\n===== OBI STRENGTH ANALYSIS =====")

for period in future_periods:

    results = []

    for i in range(len(df) - period - 1):

        obi_now = df.iloc[i]["obi"]

        if pd.isna(obi_now) or abs(obi_now) < 0.50:
            continue

        # Signal at snapshot i
        # Entry at snapshot i + 1
        # Exit after selected holding period

        entry_row = df.iloc[i + 1]
        exit_row = df.iloc[i + 1 + period]

        if obi_now > 0:
            direction = 1

            entry_price = entry_row["ask_price"]
            exit_price = exit_row["bid_price"]

            price_move = exit_price - entry_price

        else:
            direction = -1

            entry_price = entry_row["bid_price"]
            exit_price = exit_row["ask_price"]

            price_move = entry_price - exit_price

        gross_pnl = price_move * QUANTITY

        total_fee = (
            entry_price + exit_price
        ) * QUANTITY * FEE_RATE

        net_pnl = gross_pnl - total_fee

        results.append({
            "obi_strength": abs(obi_now),
            "price_move": price_move,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print(f"\nPeriod {period}: No signals")
        continue

    # Assign each signal to an OBI strength group
    result_df["obi_group"] = pd.cut(
        result_df["obi_strength"],
        bins=obi_bins,
        labels=obi_labels,
        right=False
    )

    # Calculate performance for every group
    grouped = result_df.groupby(
        "obi_group",
        observed=True
    ).agg(
        signals=("net_pnl", "size"),
        avg_price_move=("price_move", "mean"),
        avg_gross=("gross_pnl", "mean"),
        avg_net=("net_pnl", "mean"),
        profitable_rate=(
            "net_pnl",
            lambda x: (x > 0).mean() * 100
        )
    )

    print(f"\n===== PERIOD: {period} SNAPSHOTS =====")

    print(grouped.round(6).to_string())



# ==========================================
# 27. OBI ENTRY EVENT ANALYSIS
# ==========================================

ENTRY_THRESHOLD = 0.95

holding_periods_event = [10, 30, 50]

print("\n===== OBI ENTRY EVENT ANALYSIS =====")

for period in holding_periods_event:

    event_trades = []

    i = 1

    while i + 1 + period < len(df):

        previous_obi = df.iloc[i - 1]["obi"]
        current_obi = df.iloc[i]["obi"]

        # LONG: OBI crosses above +0.95
        if (
            previous_obi <= ENTRY_THRESHOLD
            and current_obi > ENTRY_THRESHOLD
        ):
            direction = 1

        # SHORT: OBI crosses below -0.95
        elif (
            previous_obi >= -ENTRY_THRESHOLD
            and current_obi < -ENTRY_THRESHOLD
        ):
            direction = -1

        else:
            i += 1
            continue

        # Enter one snapshot after the signal
        entry_idx = i + 1

        # Exit after the holding period
        exit_idx = entry_idx + period

        entry_row = df.iloc[entry_idx]
        exit_row = df.iloc[exit_idx]

        if direction == 1:

            entry_price = entry_row["ask_price"]
            exit_price = exit_row["bid_price"]

            price_move = exit_price - entry_price

        else:

            entry_price = entry_row["bid_price"]
            exit_price = exit_row["ask_price"]

            price_move = entry_price - exit_price

        gross_pnl = price_move * QUANTITY

        total_fee = (
            entry_price + exit_price
        ) * QUANTITY * FEE_RATE

        net_pnl = gross_pnl - total_fee

        event_trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "price_move": price_move,
            "gross_pnl": gross_pnl,
            "total_fee": total_fee,
            "net_pnl": net_pnl
        })

        # Avoid overlapping trades
        i = exit_idx

    event_df = pd.DataFrame(event_trades)

    print(f"\n===== PERIOD: {period} =====")

    if event_df.empty:

        print("No entry events found.")
        continue

    print("Trades:", len(event_df))

    print(
        "Average Price Move:",
        round(event_df["price_move"].mean(), 4)
    )

    print(
        "Average Gross PnL:",
        round(event_df["gross_pnl"].mean(), 6)
    )

    print(
        "Average Net PnL:",
        round(event_df["net_pnl"].mean(), 6)
    )

    print(
        "Net Profitable Rate:",
        round(
            (event_df["net_pnl"] > 0).mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Total Net PnL:",
        round(event_df["net_pnl"].sum(), 6)
    )


# ==========================================
# 28. BEST-CASE EXIT ANALYSIS
# ==========================================

BEST_CASE_THRESHOLD = 0.95
BEST_CASE_HOLDING = 30

best_case_trades = []

i = 1

while i + 1 + BEST_CASE_HOLDING < len(df):

    previous_obi = df.iloc[i - 1]["obi"]
    current_obi = df.iloc[i]["obi"]

    # 1. Check OBI threshold crossing
    if (
        previous_obi <= BEST_CASE_THRESHOLD
        and current_obi > BEST_CASE_THRESHOLD
    ):
        direction = 1  # LONG

    elif (
        previous_obi >= -BEST_CASE_THRESHOLD
        and current_obi < -BEST_CASE_THRESHOLD
    ):
        direction = -1  # SHORT

    else:
        i += 1
        continue

    # 2. Enter on the next snapshot
    entry_idx = i + 1
    exit_idx = entry_idx + BEST_CASE_HOLDING

    entry_row = df.iloc[entry_idx]
    fixed_exit_row = df.iloc[exit_idx]

    # All possible exit snapshots within holding period
    future_quotes = df.iloc[
        entry_idx + 1 : exit_idx + 1
    ]

    # 3. Calculate fixed and best-case exit
    if direction == 1:

        entry_price = entry_row["ask_price"]

        fixed_exit_price = fixed_exit_row["bid_price"]

        # Highest selling price observed
        best_exit_price = future_quotes["bid_price"].max()

        fixed_move = fixed_exit_price - entry_price
        best_move = best_exit_price - entry_price

    else:

        entry_price = entry_row["bid_price"]

        fixed_exit_price = fixed_exit_row["ask_price"]

        # Lowest buying-back price observed
        best_exit_price = future_quotes["ask_price"].min()

        fixed_move = entry_price - fixed_exit_price
        best_move = entry_price - best_exit_price

    # 4. Gross PnL
    fixed_gross = fixed_move * QUANTITY
    best_gross = best_move * QUANTITY

    # 5. Trading fees
    fixed_fee = (
        entry_price + fixed_exit_price
    ) * QUANTITY * FEE_RATE

    best_fee = (
        entry_price + best_exit_price
    ) * QUANTITY * FEE_RATE

    # 6. Net PnL
    fixed_net = fixed_gross - fixed_fee
    best_net = best_gross - best_fee

    # 7. Save results
    best_case_trades.append({
        "direction": "LONG" if direction == 1 else "SHORT",
        "fixed_move": fixed_move,
        "best_move": best_move,
        "fixed_gross": fixed_gross,
        "best_gross": best_gross,
        "fixed_fee": fixed_fee,
        "best_fee": best_fee,
        "fixed_net": fixed_net,
        "best_net": best_net
    })

    # Avoid overlapping trades
    i = exit_idx

# 8. Convert to DataFrame
best_case_df = pd.DataFrame(best_case_trades)

print("\n===== BEST-CASE EXIT ANALYSIS =====")

if best_case_df.empty:

    print("No entry events found.")

else:

    print("Trades:", len(best_case_df))

    print(
        "Average Fixed Price Move:",
        round(best_case_df["fixed_move"].mean(), 4)
    )

    print(
        "Average Best Price Move:",
        round(best_case_df["best_move"].mean(), 4)
    )

    print(
        "Average Fixed Net PnL:",
        round(best_case_df["fixed_net"].mean(), 6)
    )

    print(
        "Average Best-Case Net PnL:",
        round(best_case_df["best_net"].mean(), 6)
    )

    print(
        "Best-Case Profitable Rate:",
        round(
            (best_case_df["best_net"] > 0).mean() * 100,
            2
        ),
        "%"
    )

    print(
        "Best-Case Unprofitable Trades:",
        (best_case_df["best_net"] <= 0).sum()
    )

    print(
        "Total Best-Case Net PnL:",
        round(best_case_df["best_net"].sum(), 6)
    )


# ==========================================
# 29. DATA QUALITY & BACKTEST VALIDATION
# ==========================================

print("\n===== DATA QUALITY CHECK =====")

# Convert timestamps to datetime
timestamps = pd.to_datetime(
    df["timestamp"],
    utc=True,
    errors="coerce"
)

# Calculate time gaps between snapshots
intervals = timestamps.diff().dt.total_seconds()

print("Total Snapshots:", len(df))

print("First Timestamp:", timestamps.min())
print("Last Timestamp:", timestamps.max())

print("\nSnapshot Intervals (seconds):")
print(intervals.describe())

# Check missing or duplicated data
print("\nMissing Timestamps:", timestamps.isna().sum())

print(
    "Duplicate Update IDs:",
    df["update_id"].duplicated().sum()
)

print(
    "Non-increasing Update IDs:",
    (df["update_id"].diff() <= 0).sum()
)

# ==========================================
# BACKTEST VALIDATION
# ==========================================

print("\n===== BACKTEST VALIDATION =====")

if best_case_df.empty:
    print("No trades to validate.")

else:
    print("Best-Case Trades:", len(best_case_df))

    # Best exit should never be worse than fixed exit
    invalid_exits = (
        best_case_df["best_net"]
        < best_case_df["fixed_net"] - 1e-9
    ).sum()

    print("Best-Case Worse Than Fixed:", invalid_exits)

    # Check PnL arithmetic
    pnl_errors = (
        best_case_df["best_gross"]
        - best_case_df["best_fee"]
        - best_case_df["best_net"]
    ).abs()

    print(
        "Maximum PnL Calculation Error:",
        pnl_errors.max()
    )






