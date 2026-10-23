BTCUSDT Market Microstructure Research

This project investigates the ability of Order Book Imbalance (OBI) to predict short-term price movements using Binance BTCUSDT order book data, and whether this predictive information can be translated into a profitable trading strategy after transaction costs.

Order book data was collected using Python and the Binance REST API. Key market microstructure features, including OBI, OFI, bid-ask spread, midprice, and microprice, were calculated. Their relationships with future price movements were examined through correlation analysis, signal decay analysis, and backtesting.

The research developed entry rules based on strong OBI signals and evaluated the effects of different holding periods and transaction costs on strategy performance. The selected strategy was then tested on an independent dataset of 2,000 order book snapshots.

In the final test, 50 trades were simulated, with only one remaining profitable after transaction costs. To investigate the impact of exit timing, a theoretical Best-Case Exit Analysis was conducted by selecting the most favorable observed exit price within a 30-snapshot window for each trade. Even under this assumption, 49 out of 50 trades failed to overcome transaction costs.

Key Finding: Although order book imbalance may contain predictive information about short-term price movements, this statistical advantage does not necessarily translate into economic profitability when the resulting price movements are insufficient to cover trading costs.

This Quantitative Research project focuses on market microstructure, financial data analysis, statistical signal research, and transaction-cost-aware backtesting. The findings are limited to the observed market period and simulation assumptions and do not represent live trading performance.

