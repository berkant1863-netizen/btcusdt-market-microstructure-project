import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm

from statsmodels.tsa.stattools import coint


# ============================================================
# 1. SETTINGS
# ============================================================

START_DATE = "2020-01-01"

END_DATE = None


# ------------------------------------------------------------
# BIST STOCKS
# ------------------------------------------------------------

TICKERS = [

    "AKBNK.IS",
    "ALARK.IS",
    "ASELS.IS",
    "ASTOR.IS",
    "BIMAS.IS",

    "EKGYO.IS",
    "ENKAI.IS",
    "EREGL.IS",
    "FROTO.IS",
    "GARAN.IS",

    "GUBRF.IS",
    "HEKTS.IS",
    "ISCTR.IS",
    "KCHOL.IS",
    "KOZAA.IS",

    "KOZAL.IS",
    "KRDMD.IS",
    "OYAKC.IS",
    "PETKM.IS",
    "PGSUS.IS",

    "SAHOL.IS",
    "SASA.IS",
    "SISE.IS",
    "TCELL.IS",
    "THYAO.IS",

    "TOASO.IS",
    "TTKOM.IS",
    "TUPRS.IS",
    "ULKER.IS",
    "YKBNK.IS"

]


# ============================================================
# WALK-FORWARD SETTINGS
# ============================================================

# Number of historical trading days used for pair selection.
FORMATION_DAYS = 252

# Number of out-of-sample (OOS) trading days for selected pairs.
TRADING_DAYS = 63


# ============================================================
# ROLLING MODEL SETTINGS
# ============================================================

# Hedge ratio / beta
BETA_WINDOW = 60

# Spread z-score
ZSCORE_WINDOW = 20


# ============================================================
# PAIR SELECTION SETTINGS
# ============================================================

MIN_CORRELATION = 0.60

MAX_COINTEGRATION_PVALUE = 0.05

MIN_HALF_LIFE = 2

MAX_HALF_LIFE = 60



MAX_PAIRS_PER_WINDOW = 5


# ============================================================
# TRADING RULES
# ============================================================

ENTRY_Z = 2.0

EXIT_Z = 0.5

STOP_Z = 3.5


# ============================================================
# TRANSACTION COST
# ============================================================

ROUND_TRIP_COST_BPS = 10.0


# ============================================================
# 2. DOWNLOAD DATA
# ============================================================

print()

print("=" * 70)

print("DATA DOWNLOAD")

print("=" * 70)


if END_DATE is None:

    downloaded = yf.download(

        TICKERS,

        start=START_DATE,

        auto_adjust=True,

        progress=False

    )

else:

    downloaded = yf.download(

        TICKERS,

        start=START_DATE,

        end=END_DATE,

        auto_adjust=True,

        progress=False

    )


# ============================================================
# CLOSING PRICES
# ============================================================

if isinstance(
    downloaded.columns,
    pd.MultiIndex
):

    if (
        "Close"
        in
        downloaded.columns.get_level_values(0)
    ):

        prices = (
            downloaded["Close"]
            .copy()
        )

    else:

        raise ValueError(
            "Close fiyatları bulunamadı."
        )

else:

    prices = (
        downloaded[["Close"]]
        .copy()
    )


# ============================================================
# SORT
# ============================================================

prices = (
    prices
    .sort_index()
)


# ============================================================
# REMOVE STOCKS WITH TOO MUCH MISSING DATA
#
# At least 95% of each stock's price observations must be available.
# ============================================================

minimum_valid_rows = int(
    len(prices) * 0.95
)


prices = prices.dropna(

    axis=1,

    thresh=minimum_valid_rows

)


# ============================================================
# FILL SHORT GAPS
# ============================================================

prices = prices.ffill(
    limit=2
)


# ============================================================
# REMOVE DATES WITH REMAINING NaN VALUES
# ============================================================

prices = prices.dropna()


print()

print(
    "Başlangıç:",
    prices.index.min()
)

print(
    "Bitiş:",
    prices.index.max()
)

print(
    "Trading days:",
    len(prices)
)

print(
    "Hisse sayısı:",
    len(prices.columns)
)


if (
    len(prices)
    <
    FORMATION_DAYS
    +
    TRADING_DAYS
):

    raise ValueError(
        "Yeterli veri yok. "
        "START_DATE'i daha eski yap."
    )


# ============================================================
# LISTS USED TO STORE ALL RESULTS
# ============================================================

all_selected_pairs = []

all_daily_results = []

total_trade_entries = 0


# ============================================================
# 3. START WALK-FORWARD BACKTEST
#
# Example:
#
# First 252 trading days
# -> pair selection
#
# Next 63 trading days
# -> OOS trading
#
# The window then advances by 63 trading days.
# ============================================================

window_number = 0

start_index = FORMATION_DAYS


while (

    start_index
    +
    TRADING_DAYS

    <= len(prices)

):

    window_number += 1


    # ========================================================
    # FORMATION PERIOD
    # ========================================================

    formation_start = (
        start_index
        -
        FORMATION_DAYS
    )

    formation_end = (
        start_index
    )


    formation_prices = prices.iloc[

        formation_start:
        formation_end

    ].copy()


    # ========================================================
    # OOS TRADING PERIOD
    # ========================================================

    trading_start = (
        start_index
    )

    trading_end = (

        start_index
        +
        TRADING_DAYS

    )


    trading_prices = prices.iloc[

        trading_start:
        trading_end

    ].copy()


    print()

    print("=" * 70)

    print(
        "WINDOW",
        window_number
    )

    print("=" * 70)


    print(

        "Formation:",

        formation_prices.index[0].date(),

        "->",

        formation_prices.index[-1].date()

    )


    print(

        "Trading:",

        trading_prices.index[0].date(),

        "->",

        trading_prices.index[-1].date()

    )


    # ========================================================
    # 4. PAIR SELECTION
    # ========================================================

    pair_candidates = []


    symbols = list(
        formation_prices.columns
    )


    # Evaluate every possible pair of stocks.
    for i in range(
        len(symbols)
    ):

        for j in range(

            i + 1,

            len(symbols)

        ):

            stock_a = symbols[i]

            stock_b = symbols[j]


            pair_data = formation_prices[
                [
                    stock_a,
                    stock_b
                ]
            ].dropna()


            # Skip pairs with insufficient formation-period data.
            if len(pair_data) < 200:

                continue


            # =================================================
            # LOG PRICE
            # =================================================

            log_a = np.log(
                pair_data[stock_a]
            )

            log_b = np.log(
                pair_data[stock_b]
            )


            # =================================================
            # CORRELATION
            #
            # Calculate correlation of daily returns, not prices.
            # =================================================

            returns = (
                pair_data
                .pct_change()
                .dropna()
            )


            correlation = (

                returns[stock_a]
                .corr(
                    returns[stock_b]
                )

            )


            if (

                pd.isna(
                    correlation
                )

                or

                correlation
                <
                MIN_CORRELATION

            ):

                continue


            # =================================================
            # COINTEGRATION TEST
            # =================================================

            try:

                _, p_value, _ = coint(

                    log_a,

                    log_b

                )

            except Exception:

                continue


            if (

                pd.isna(
                    p_value
                )

                or

                p_value
                >
                MAX_COINTEGRATION_PVALUE

            ):

                continue


            # =================================================
            # FORMATION OLS
            #
            # log(A)
            # =
            # alpha
            # +
            # beta * log(B)
            # =================================================

            X = sm.add_constant(
                log_b
            )


            ols_model = sm.OLS(

                log_a,

                X

            ).fit()


            alpha = (
                ols_model.params.iloc[0]
            )


            beta = (
                ols_model.params.iloc[1]
            )


            # Discard the pair if beta is invalid or too small.
            if (

                beta <= 0.05

                or

                not np.isfinite(beta)

            ):

                continue


            # =================================================
            # SPREAD
            # =================================================

            spread = (

                log_a

                -

                (
                    alpha
                    +
                    beta * log_b
                )

            )


            # =================================================
            # HALF-LIFE
            #
            # How quickly does the spread revert toward its mean?
            # =================================================

            spread_lag = (
                spread.shift(1)
            )


            spread_change = (

                spread
                -
                spread_lag

            )


            half_life_data = pd.DataFrame({

                "spread_change":
                    spread_change,

                "spread_lag":
                    spread_lag

            }).dropna()


            if len(
                half_life_data
            ) < 50:

                continue


            X_half = sm.add_constant(

                half_life_data[
                    "spread_lag"
                ]

            )


            half_model = sm.OLS(

                half_life_data[
                    "spread_change"
                ],

                X_half

            ).fit()


            lambda_value = (
                half_model.params.iloc[1]
            )


            # A non-negative lambda does not indicate mean reversion.
            if lambda_value >= 0:

                continue


            half_life = (

                -np.log(2)

                /

                lambda_value

            )


            if (

                half_life
                <
                MIN_HALF_LIFE

                or

                half_life
                >
                MAX_HALF_LIFE

                or

                not np.isfinite(
                    half_life
                )

            ):

                continue


            # =================================================
            # PAIR PASSES THE FILTERS
            # =================================================

            pair_candidates.append({

                "stock_a":
                    stock_a,

                "stock_b":
                    stock_b,

                "correlation":
                    correlation,

                "coint_pvalue":
                    p_value,

                "formation_beta":
                    beta,

                "half_life":
                    half_life

            })


    # ========================================================
    # CANDIDATE DATAFRAME
    # ========================================================

    candidates_df = pd.DataFrame(
        pair_candidates
    )


    if len(
        candidates_df
    ) == 0:

        print(
            "Bu window'da uygun pair bulunamadı."
        )

        start_index += (
            TRADING_DAYS
        )

        continue


    # ========================================================
    # PAIR RANKING
    #
    # Ranking priority:
    #
    # 1) lower cointegration p-value
    # 2) higher correlation
    # 3) shorter half-life
    # ========================================================

    candidates_df = (

        candidates_df

        .sort_values(

            [
                "coint_pvalue",
                "correlation",
                "half_life"
            ],

            ascending=[
                True,
                False,
                True
            ]

        )

        .reset_index(
            drop=True
        )

    )


    # ========================================================
    # 5. DIVERSIFICATION
    #
    # Do not include the same stock in more than one pair
    # within the same trading window.
    # ========================================================

    selected_pairs = []

    used_symbols = set()


    for _, pair in (
        candidates_df.iterrows()
    ):

        stock_a = pair[
            "stock_a"
        ]

        stock_b = pair[
            "stock_b"
        ]


        if (

            stock_a
            in
            used_symbols

            or

            stock_b
            in
            used_symbols

        ):

            continue


        selected_pairs.append(
            pair
        )


        used_symbols.add(
            stock_a
        )

        used_symbols.add(
            stock_b
        )


        if (

            len(selected_pairs)
            >=
            MAX_PAIRS_PER_WINDOW

        ):

            break


    if (
        len(selected_pairs)
        ==
        0
    ):

        print(
            "Pair kalmadı."
        )

        start_index += (
            TRADING_DAYS
        )

        continue


    selected_pairs_df = pd.DataFrame(
        selected_pairs
    )


    print()

    print(
        "Selected pairs:"
    )


    print(

        selected_pairs_df[
            [
                "stock_a",
                "stock_b",
                "correlation",
                "coint_pvalue",
                "half_life"
            ]
        ]

        .to_string(
            index=False
        )

    )


    # ========================================================
    # SAVE SELECTED PAIRS
    # ========================================================

    for _, pair in (
        selected_pairs_df.iterrows()
    ):

        all_selected_pairs.append({

            "window":
                window_number,

            "formation_start":
                formation_prices.index[0],

            "formation_end":
                formation_prices.index[-1],

            "trading_start":
                trading_prices.index[0],

            "trading_end":
                trading_prices.index[-1],

            "stock_a":
                pair["stock_a"],

            "stock_b":
                pair["stock_b"],

            "correlation":
                pair["correlation"],

            "coint_pvalue":
                pair["coint_pvalue"],

            "formation_beta":
                pair["formation_beta"],

            "half_life":
                pair["half_life"]

        })


    # ========================================================
    # 6. OOS TRADING
    # ========================================================

    pair_daily_returns = []


    for _, pair in (
        selected_pairs_df.iterrows()
    ):

        stock_a = pair[
            "stock_a"
        ]

        stock_b = pair[
            "stock_b"
        ]


        # ====================================================
        # LOAD HISTORICAL FORMATION DATA FOR ROLLING
        # CALCULATIONS AT THE START OF THE TRADING WINDOW
        # ====================================================

        lookback_days = max(

            BETA_WINDOW,

            ZSCORE_WINDOW

        ) + 10


        history = (

            formation_prices[
                [
                    stock_a,
                    stock_b
                ]
            ]

            .tail(
                lookback_days
            )

        )


        pair_trade_prices = trading_prices[
            [
                stock_a,
                stock_b
            ]
        ]


        combined = pd.concat(

            [
                history,
                pair_trade_prices
            ]

        )


        combined = (

            combined
            .dropna()
            .copy()

        )


        # ====================================================
        # LOG PRICES
        # ====================================================

        log_a = np.log(
            combined[stock_a]
        )

        log_b = np.log(
            combined[stock_b]
        )


        # ====================================================
        # 7. ROLLING BETA
        #
        # beta
        # =
        # covariance(A,B)
        # /
        # variance(B)
        # ====================================================

        rolling_cov = (

            log_a

            .rolling(
                BETA_WINDOW
            )

            .cov(
                log_b
            )

        )


        rolling_var = (

            log_b

            .rolling(
                BETA_WINDOW
            )

            .var()

        )


        beta = (

            rolling_cov

            /

            rolling_var

        )


        # ====================================================
        # ROLLING ALPHA
        # ====================================================

        alpha = (

            log_a
            .rolling(
                BETA_WINDOW
            )
            .mean()

            -

            beta

            *

            log_b
            .rolling(
                BETA_WINDOW
            )
            .mean()

        )


        # ====================================================
        # SPREAD
        # ====================================================

        spread = (

            log_a

            -

            (
                alpha
                +
                beta * log_b
            )

        )


        # ====================================================
        # Z-SCORE
        # ====================================================

        spread_mean = (

            spread

            .rolling(
                ZSCORE_WINDOW
            )

            .mean()

        )


        spread_std = (

            spread

            .rolling(
                ZSCORE_WINDOW
            )

            .std()

        )


        zscore = (

            spread
            -
            spread_mean

        ) / spread_std


        # ====================================================
        # IMPORTANT:
        #
        # Do not use a day's closing-price information
        # to make a trading decision for that same day.
        #
        # Shift the signals by one trading day.
        #
        # Therefore:
        #
        # yesterday's information
        # ->
        # today's position
        #
        # This reduces the risk of look-ahead bias.
        # ====================================================

        beta_signal = (
            beta.shift(1)
        )

        zscore_signal = (
            zscore.shift(1)
        )


        # ====================================================
        # PAIR DATAFRAME
        # ====================================================

        pair_df = pd.DataFrame({

            "price_a":
                combined[stock_a],

            "price_b":
                combined[stock_b],

            "beta":
                beta_signal,

            "zscore":
                zscore_signal

        })


        # Keep only out-of-sample trading dates.
        pair_df = pair_df.reindex(
            trading_prices.index
        )


        pair_df = (
            pair_df
            .dropna()
            .copy()
        )


        if len(
            pair_df
        ) < 10:

            continue


        # ====================================================
        # 8. TRADE LOOP
        # ====================================================

        position = 0

        # position:
        #
        #  1 = LONG SPREAD
        # -1 = SHORT SPREAD
        #  0 = FLAT


        pair_result = []

        pair_entry_count = 0


        dates = list(
            pair_df.index
        )


        # The loop stops before the final date because returns
        # are calculated from each date to the next date.
        for k in range(
            len(dates) - 1
        ):

            date = dates[k]

            next_date = dates[
                k + 1
            ]


            z = pair_df.loc[
                date,
                "zscore"
            ]


            current_beta = pair_df.loc[
                date,
                "beta"
            ]


            old_position = (
                position
            )


            # =================================================
            # ENTRY
            # =================================================

            if position == 0:

                # ---------------------------------------------
                # The spread is unusually low.
                #
                # LONG SPREAD:
                #
                # LONG A
                # SHORT B
                # ---------------------------------------------

                if z <= -ENTRY_Z:

                    position = 1

                    pair_entry_count += 1


                # ---------------------------------------------
                # The spread is unusually high.
                #
                # SHORT SPREAD:
                #
                # SHORT A
                # LONG B
                # ---------------------------------------------

                elif z >= ENTRY_Z:

                    position = -1

                    pair_entry_count += 1


            # =================================================
            # LONG SPREAD EXIT
            # =================================================

            elif position == 1:

                # The spread has moved close to its mean.
                if abs(z) <= EXIT_Z:

                    position = 0


                # The spread has moved too far against the position.
                elif z <= -STOP_Z:

                    position = 0


            # =================================================
            # SHORT SPREAD EXIT
            # =================================================

            elif position == -1:

                if abs(z) <= EXIT_Z:

                    position = 0


                elif z >= STOP_Z:

                    position = 0


            # =================================================
            # TRANSACTION COST
            # =================================================

            cost_bps = 0.0


            # A position was opened.
            if (

                old_position == 0

                and

                position != 0

            ):

                cost_bps = (

                    ROUND_TRIP_COST_BPS

                    /

                    2

                )


            # A position was closed.
            elif (

                old_position != 0

                and

                position == 0

            ):

                cost_bps = (

                    ROUND_TRIP_COST_BPS

                    /

                    2

                )


            # If the position reverses directly.
            elif (

                old_position != 0

                and

                position != 0

                and

                old_position
                !=
                position

            ):

                cost_bps = (
                    ROUND_TRIP_COST_BPS
                )


            # =================================================
            # 9. DAILY RETURNS
            # =================================================

            return_a = (

                pair_df.loc[
                    next_date,
                    "price_a"
                ]

                /

                pair_df.loc[
                    date,
                    "price_a"
                ]

                -

                1

            )


            return_b = (

                pair_df.loc[
                    next_date,
                    "price_b"
                ]

                /

                pair_df.loc[
                    date,
                    "price_b"
                ]

                -

                1

            )


            # =================================================
            # PAIR RETURN
            #
            # LONG SPREAD:
            #
            # +A
            # -beta B
            #
            # SHORT SPREAD:
            #
            # -A
            # +beta B
            #
            # Normalize by gross exposure.
            # =================================================

            if position == 0:

                gross_return = 0.0

            else:

                gross_return = (

                    position

                    *

                    (
                        return_a

                        -

                        current_beta
                        *
                        return_b
                    )

                    /

                    (
                        1

                        +

                        abs(
                            current_beta
                        )
                    )

                )


            # =================================================
            # NET RETURN
            # =================================================

            net_return = (

                gross_return

                -

                (
                    cost_bps
                    /
                    10000
                )

            )


            pair_result.append({

                "date":
                    next_date,

                "window":
                    window_number,

                "stock_a":
                    stock_a,

                "stock_b":
                    stock_b,

                "position":
                    position,

                "zscore":
                    z,

                "beta":
                    current_beta,

                "gross_return":
                    gross_return,

                "cost_bps":
                    cost_bps,

                "net_return":
                    net_return

            })


        total_trade_entries += (
            pair_entry_count
        )


        if len(
            pair_result
        ) > 0:

            pair_result_df = pd.DataFrame(
                pair_result
            )


            pair_daily_returns.append(
                pair_result_df
            )


    # ========================================================
    # 10. COMBINE PAIRS INTO A WINDOW PORTFOLIO
    #
    # Equal weight.
    # ========================================================

    if len(
        pair_daily_returns
    ) > 0:

        window_pairs_df = pd.concat(

            pair_daily_returns,

            ignore_index=True

        )


        window_portfolio = (

            window_pairs_df

            .groupby(
                "date"
            )

            .agg(

                gross_return=(
                    "gross_return",
                    "mean"
                ),

                net_return=(
                    "net_return",
                    "mean"
                ),

                pair_count=(
                    "stock_a",
                    "size"
                )

            )

            .reset_index()

        )


        window_portfolio[
            "window"
        ] = window_number


        all_daily_results.append(
            window_portfolio
        )


    # ========================================================
    # ADVANCE TO THE NEXT WALK-FORWARD WINDOW
    # ========================================================

    start_index += (
        TRADING_DAYS
    )


# ============================================================
# 11. FINAL OOS RESULTS
# ============================================================

print()

print("=" * 70)

print("FINAL OOS RESULTS")

print("=" * 70)


if len(
    all_daily_results
) == 0:

    print(
        "OOS sonuç üretilemedi."
    )

    raise SystemExit


# ============================================================
# COMBINE ALL OOS WINDOWS
# ============================================================

oos = pd.concat(

    all_daily_results,

    ignore_index=True

)


oos = (

    oos

    .sort_values(
        "date"
    )

    .set_index(
        "date"
    )

)


# ============================================================
# EQUITY CURVE
# ============================================================

oos["equity"] = (

    1

    +

    oos[
        "net_return"
    ]

).cumprod()


oos[
    "gross_equity"
] = (

    1

    +

    oos[
        "gross_return"
    ]

).cumprod()


# ============================================================
# TOTAL RETURN
# ============================================================

total_net_return = (

    oos[
        "equity"
    ].iloc[-1]

    -

    1

)


total_gross_return = (

    oos[
        "gross_equity"
    ].iloc[-1]

    -

    1

)


# ============================================================
# SHARPE RATIO
# ============================================================

daily_mean = (

    oos[
        "net_return"
    ].mean()

)


daily_std = (

    oos[
        "net_return"
    ].std()

)


if (

    daily_std > 0

    and

    np.isfinite(
        daily_std
    )

):

    sharpe = (

        daily_mean

        /

        daily_std

        *

        np.sqrt(252)

    )

else:

    sharpe = np.nan


# ============================================================
# MAX DRAWDOWN
# ============================================================

running_max = (

    oos[
        "equity"
    ]

    .cummax()

)


drawdown = (

    oos[
        "equity"
    ]

    /

    running_max

    -

    1

)


max_drawdown = (
    drawdown.min()
)


# ============================================================
# POSITIVE DAY RATE
# ============================================================

positive_day_rate = (

    (
        oos[
            "net_return"
        ]
        >
        0
    )

    .mean()

)


# ============================================================
# AVERAGE DAILY RETURN BPS
# ============================================================

average_daily_bps = (

    oos[
        "net_return"
    ].mean()

    *

    10000

)


# ============================================================
# VOLATILITY
# ============================================================

annualized_volatility = (

    oos[
        "net_return"
    ].std()

    *

    np.sqrt(252)

)


# ============================================================
# PRINT RESULTS
# ============================================================

print()

print(
    "OOS days:",
    len(oos)
)


print(
    "Walk-forward windows:",
    window_number
)


print(
    "Trade entries:",
    total_trade_entries
)


print()

print(
    "Gross return:",
    round(
        total_gross_return
        *
        100,
        2
    ),
    "%"
)


print(
    "Net return:",
    round(
        total_net_return
        *
        100,
        2
    ),
    "%"
)


print()

print(
    "Annualized Sharpe:",
    round(
        sharpe,
        3
    )
)


print(
    "Annualized volatility:",
    round(
        annualized_volatility
        *
        100,
        2
    ),
    "%"
)


print(
    "Max drawdown:",
    round(
        max_drawdown
        *
        100,
        2
    ),
    "%"
)


print()

print(
    "Average daily NET:",
    round(
        average_daily_bps,
        3
    ),
    "bps"
)


print(
    "Positive day rate:",
    round(
        positive_day_rate
        *
        100,
        2
    ),
    "%"
)


# ============================================================
# 12. SAVE OUTPUT FILES
# ============================================================

selected_pairs_output = pd.DataFrame(
    all_selected_pairs
)


selected_pairs_output.to_csv(

    "pairs_selected_walkforward.csv",

    index=False

)


oos.to_csv(
    "pairs_oos_daily_returns.csv"
)


print()

print("=" * 70)

print("FILES SAVED")

print("=" * 70)


print(
    "pairs_selected_walkforward.csv"
)


print(
    "pairs_oos_daily_returns.csv"
)