---_marker_description: Project-specific guidelines for the ML Crypto Momentum Trader.
globs: ["src/trader/**/*.py", "scripts/**/*.py", "config/**/*.yaml"]
---_marker_

# ML Crypto Momentum Trader: Project-Specific Guidelines

## 1. Data Handling and Preprocessing (`data/`, `scripts/data_processing/`)
- **Raw Data Structure**: Maintain the existing `data/raw/YYYY/MM/` structure for incoming minute aggregate data.
- **Processed Data**: Store processed features and datasets in `data/processed/` with clear train/validation/test splits, preferably in efficient formats like Parquet or Feather.
- **Feature Naming**: 
    - Use a consistent prefix for engineered features, e.g., `feat_`.
    - Suffixes should indicate the nature of the feature, e.g., `_sma_20` (20-period Simple Moving Average), `_rsi_14` (14-period RSI), `_lag_5` (5-period lag).
    - For momentum features, be explicit, e.g., `feat_mom_10` (10-period momentum), `feat_roc_5` (5-period Rate of Change).
- **Data Integrity**: Implement checks for missing data, outliers, and inconsistencies in raw data before processing.
- **Time Zones**: Ensure all timestamp data is consistently handled, preferably in UTC. Clearly document any timezone conversions.
- **Windowing/Sequencing**: For sequence-based models (LSTMs, Transformers), clearly define and document sequence length, stride, and how target variables are constructed.

## 2. Model Development (`src/trader/models/`, `scripts/training/`)
- **Model Abstraction**: Define a base model class or interface if multiple model architectures are explored, ensuring consistent methods for training, prediction, saving, and loading.
- **Hyperparameters**: Manage hyperparameters through the configuration system. Store best hyperparameter sets with corresponding experiment results.
- **Experiment Tracking** (e.g., `wandb`, `mlflow`):
    - Log all relevant experiment details: dataset version, feature set, model architecture, hyperparameters, training metrics, validation metrics, random seeds.
    - Store or link to saved model artifacts and evaluation results.
    - Use meaningful names/tags for experiments (e.g., `lstm_v1_btc_usdt_1m_mom_features`).
- **Target Variable Definition**: Clearly define the target variable for momentum trading (e.g., future price change, binary up/down movement, specific profit target).

## 3. Training and Evaluation (`scripts/training/`, `scripts/evaluation/`)
- **Time Series Splits**: 
    - For splitting time-series data, use methods that preserve temporal order, such as walk-forward validation or a fixed split point for train/validation and a later one for test.
    - Ensure no data leakage between sets.
- **Momentum-Specific Metrics**: 
    - Beyond standard ML metrics (accuracy, F1, MSE), prioritize trading-specific metrics: Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Profit Factor, Win/Loss Ratio, Average Win/Loss size.
    - Evaluate performance across different market regimes (e.g., trending, ranging, volatile) if possible.
- **Backtesting Rigor**: 
    - Account for simulated transaction costs (slippage, commissions).
    - Ensure backtests are realistic and avoid look-ahead bias.
    - Consider the impact of order book depth if simulating actual trade execution.
- **Baseline Models**: Always compare ML model performance against sensible baselines (e.g., buy-and-hold, simple moving average crossover strategy).

## 4. Configuration (`config/`)
- **Centralized Configuration**: Use YAML or similar for all project configurations (data paths, feature parameters, model hyperparameters, training settings).
- **Schema/Validation**: Consider using Pydantic or similar to define and validate configuration structures.
- **Version Control Configs**: Track changes to configuration files in Git.

## 5. Code Structure and Utilities (`src/trader/utils/`)
- **Utility Functions**: Common helper functions (e.g., for loading data, saving artifacts, specific calculations) should be placed in `src/trader/utils/`.
- **Environment Consistency**: Ensure `requirements.txt` or `pyproject.toml` is always up-to-date to maintain a consistent environment across development and deployment.

## 6. Logging and Profiling (`logs/`)
- **Structured Logging**: Implement structured logging, especially for training runs, to capture key information in a parseable format.
- **Profiling**: For performance-critical sections (e.g., data loading, feature computation, model inference), use profiling tools (`cProfile`, `line_profiler`, `torch.profiler`) to identify bottlenecks. 