# Do Headlines Improve Momentum Signals?

An interactive research project investigating whether Apple news sentiment adds useful information to a simple stock-momentum signal.

The project compares Apple (`AAPL`) with the S&P 500 ETF (`SPY`). It begins with an encouraging headline-tone result, then challenges that result through chronological, walk-forward, non-overlapping, ablation, and failure-case analyses. The final conclusion is deliberately cautious: headline tone showed some period-dependent ranking value, but no tested model was consistently reliable.

> **Current interface status:** the Streamlit app is designed and tested for desktop use. Mobile optimization is planned after the desktop content and design are finalized.

## Research question

Can information available at market close—recent price momentum and the tone of recent Apple headlines—help predict whether AAPL will outperform SPY over the next five trading days?

For each prediction date:

- **Momentum:** AAPL's return over the previous five trading days
- **Headline tone:** average FinBERT score for eligible headlines over the previous five trading days
- **News volume:** number of eligible headlines in that same window
- **Target:** `1` when AAPL's next five-trading-day return exceeds SPY's, otherwise `0`

Only information available by the prediction date is used as a model input. Future returns are reserved for evaluating the prediction.

## Interactive app

The desktop Streamlit app turns the analysis into four connected pages:

1. **Research Overview** explains the question, timeline, and evaluation design in accessible language.
2. **Stock & News Explorer** lets users inspect a held-out date, the information known at market close, model forecasts, representative headlines, and the outcome revealed five trading days later.
3. **Model Comparison** compares accuracy and ROC AUC, removes individual sentiment features through ablation, and tests robustness across time and non-overlapping schedules.
4. **Failures & Limitations** explores representative mistakes, explains the models' modest confidence, and separates supported findings from claims the evidence cannot justify.

## Data and experiment design

| Item | Design choice |
|---|---|
| Primary stock | Apple (`AAPL`) |
| Benchmark | S&P 500 ETF (`SPY`) |
| Price-data range | January 1, 2023–January 1, 2025 |
| Final prediction cutoff | November 22, 2024 |
| Lookback window | Five trading days |
| Prediction horizon | Five trading days |
| Price source | Yahoo Finance through `yfinance` |
| Sentiment model | `ProsusAI/finbert` |
| Predictive models | Interpretable logistic regressions |

The news cutoff reflects the end of reliable coverage in the available dataset. Later price observations are used only to calculate the outcomes of late-November predictions, never as prediction inputs.

News timestamps are converted from UTC to New York time. Articles published before 4:00 p.m. are assigned to that trading day; articles published after the close, on weekends, or on market holidays are assigned to the next trading day.

## Main findings

- The majority-class baseline achieved **50.0% accuracy** in the initial held-out window.
- Momentum alone achieved **52.5% accuracy** and **50.3% ROC AUC**.
- Tone alone achieved **62.7% accuracy** and **67.2% ROC AUC** in that particular window.
- Tone plus headline count achieved **63.6% accuracy** but a lower **61.8% ROC AUC**.
- Headline count alone fell below the baseline, suggesting the initial improvement was not simply caused by having more news.
- Walk-forward and non-overlapping tests showed substantial variation across periods and schedules. No model was consistently reliable.
- Most tone-model probabilities were close to 50%, indicating modest confidence even when the final classification was correct.

These are research findings, not evidence of a profitable trading strategy.

## Sentiment quality control

The source dataset included sentiment labels, but those labels agreed with a manually reviewed 30-headline sample only **40%** of the time. FinBERT agreed with the same sample **76.7%** of the time, so the project uses FinBERT probabilities rather than the original labels.

The continuous headline score is:

```text
positive probability - negative probability
```

This produces a value from `-1` to `+1` while retaining more information than a positive/neutral/negative label.

## Evaluation approach

- **Chronological split:** training observations occur before testing observations.
- **Safety gap:** five trading days separate training and test targets to reduce leakage.
- **Walk-forward testing:** models repeatedly train on the past and evaluate later periods.
- **Non-overlapping testing:** every fifth prediction date is evaluated across five starting offsets so five-day outcomes do not share trading days.
- **Ablation:** tone and headline count are removed separately to identify which feature contributed to the result.
- **Failure analysis:** representative false positives, false negatives, reversals, and conflicting-news periods are inspected after aggregate evaluation.

## Limitations

- The experiment covers one company and a relatively short historical period.
- Daily five-day outcomes overlap; non-overlapping tests reduce this issue but create small samples.
- A positive headline is not the same as positive market impact.
- The 30-headline human review is useful quality control but remains small and subjective.
- The study identifies associations, not causal effects of news on prices.
- The models are intentionally simple and may miss nonlinear or changing relationships.
- Accuracy and ROC AUC do not account for transaction costs, taxes, slippage, position sizing, or investment risk.

## Future work

- Predefine and test **sentiment dispersion**, such as the standard deviation of headline scores, on new unseen data.
- Replicate the experiment with companies that have sufficiently complete timestamped news histories.
- Collect a longer unseen period and pre-specify earnings versus non-earnings comparisons.
- Evaluate probability calibration and choose classification thresholds using validation data rather than test results.
- Add trading costs and risk controls only if the research is extended into a strategy simulation.
- Optimize the application for mobile devices after the desktop version is finalized.

## Project structure

```text
app.py             Streamlit research application
notebooks/         Numbered research pipeline, from prices through failure analysis
data/raw/          Local source datasets (not committed)
data/processed/    Generated features, predictions, and results (not committed)
src/               Reusable Python package scaffold
archive/           Earlier exploratory datasets
requirements.txt   Python dependencies
```

Raw and generated CSV files are excluded from version control because of dataset size and redistribution considerations. To reproduce the complete app, place the required source files in `data/raw/` and run the numbered notebooks in order. The notebooks generate the files consumed by `app.py` in `data/processed/`.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
jupyter lab
```

Run the numbered notebooks beginning with `notebooks/01_price_data.ipynb`. After the processed files have been generated, launch the desktop app:

```bash
streamlit run app.py
```

Then open the local URL printed by Streamlit, normally `http://localhost:8501`.

## Disclaimer

This project is an educational research exercise. It does not provide investment advice or represent a deployable trading strategy. Yahoo Finance is convenient for prototyping but is not an institutional market-data feed.
