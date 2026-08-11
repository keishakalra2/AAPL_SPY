from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parent
APP_DATA_DIR = PROJECT_ROOT / "data" / "app"

st.set_page_config(
    page_title="Do Headlines Improve Momentum Signals?",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f6f4ef; color: #17211d; }
      [data-testid="stSidebar"] { background: #10261f; }
      [data-testid="stSidebar"] * { color: #f3f0e8 !important; }
      [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.16); }
      .block-container { max-width: 1180px; padding-top: 2.6rem; padding-bottom: 4rem; }
      h1, h2, h3 { letter-spacing: -0.035em; color: #14231d; }
      h1 { font-size: clamp(2.7rem, 5vw, 4.8rem) !important; line-height: .98 !important; }
      h2 { margin-top: 2.2rem !important; }
      p, li { line-height: 1.65; }
      .eyebrow { color: #b45e3b; font-size: .76rem; font-weight: 800; letter-spacing: .16em; text-transform: uppercase; }
      .hero-copy { color: #52615a; font-size: 1.18rem; line-height: 1.65; max-width: 760px; margin: 1.1rem 0 1.8rem; }
      .hero-copy.compact-copy { margin-bottom: .25rem; }
      .compact-section-title { font-size: 2rem; margin: .7rem 0 .2rem; letter-spacing: -.035em; color: #14231d; }
      .question-card { background: #14392f; border-radius: 18px; color: #f8f3e8; padding: 1.6rem 1.8rem; margin: 1rem 0 2rem; }
      .question-card small { color: #b9d1c6; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
      .question-card p { font-family: Georgia, serif; font-size: 1.45rem; line-height: 1.4; margin: .65rem 0 0; }
      .metric-card { background: #fffdf8; border: 1px solid #e5e0d5; border-radius: 16px; padding: 1.15rem 1.25rem; min-height: 128px; }
      .metric-value { color: #173f34; font-size: 2rem; font-weight: 750; letter-spacing: -.04em; }
      .metric-label { color: #68736d; font-size: .84rem; margin-top: .25rem; }
      .tooltip-card { position: relative; cursor: help; }
      .info-dot { display: inline-flex; align-items: center; justify-content: center; width: 1.15rem; height: 1.15rem; margin-left: .3rem; border-radius: 50%; background: #dbe8e2; color: #245444; font-size: .72rem; font-weight: 800; vertical-align: middle; }
      .tooltip-content { visibility: hidden; opacity: 0; position: absolute; z-index: 20; left: 1rem; top: calc(100% + .55rem); width: min(330px, 86vw); padding: .9rem 1rem; border-radius: 12px; background: #10261f; color: #f7f3e9; font-size: .8rem; line-height: 1.5; box-shadow: 0 12px 28px rgba(15, 35, 29, .22); transform: translateY(-4px); transition: opacity .15s ease, transform .15s ease; }
      .tooltip-content strong { color: #cce3d8; }
      .tooltip-card:hover .tooltip-content, .tooltip-card:focus .tooltip-content, .tooltip-card:focus-within .tooltip-content { visibility: visible; opacity: 1; transform: translateY(0); }
      .comparison-card { border: 1px solid #e1ddd3; border-radius: 16px; padding: 1.2rem 1.3rem; margin-bottom: .8rem; min-height: 150px; }
      .prediction-card.correct { background: #e5f3e8; border-color: #b8d9c0; }
      .prediction-card.incorrect { background: #f8e5e1; border-color: #e8bbb1; }
      .actual-card { background: #fffdf8; }
      .prediction-card.momentum { border-left: 6px solid #bf6842; }
      .prediction-card.tone { border-left: 6px solid #5472a4; }
      .prediction-card.momentum .card-kicker, .prediction-card.momentum .card-outcome { color: #934725; }
      .prediction-card.tone .card-kicker, .prediction-card.tone .card-outcome { color: #405e91; }
      .card-kicker { color: #5e6a64; font-size: .78rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
      .card-outcome { color: #173f34; font-size: 1.55rem; font-weight: 750; letter-spacing: -.03em; margin: .45rem 0 .35rem; }
      .card-detail { color: #59665f; font-size: .9rem; line-height: 1.5; }
      .status-pill { display: inline-block; border-radius: 999px; padding: .28rem .62rem; margin-top: .8rem; font-size: .78rem; font-weight: 800; }
      .status-pill.correct { color: #22683e; background: #cde7d3; }
      .status-pill.incorrect { color: #963f32; background: #f1c9c1; }
      .actual-grid { display: grid; grid-template-columns: 1fr 1fr; gap: .8rem; margin-top: 1rem; }
      .actual-stat { border-top: 1px solid #ded9cf; padding-top: .8rem; }
      .actual-stat strong { color: #173f34; display: block; font-size: 1.45rem; letter-spacing: -.03em; }
      .actual-stat span { color: #68736d; font-size: .78rem; }
      .failure-card { background: #fffdf8; border: 1px solid #e3ddd2; border-radius: 16px; padding: 1.2rem 1.35rem; margin: .8rem 0; }
      .failure-card .case-date { color: #b45e3b; font-size: .76rem; font-weight: 800; letter-spacing: .1em; text-transform: uppercase; }
      .failure-card h4 { color: #17211d; margin: .4rem 0 .7rem; }
      .failure-card p { color: #59655f; font-size: .9rem; margin: .25rem 0; }
      .limitation-card { background: #fffdf8; border-top: 4px solid #b45e3b; border-radius: 14px; padding: 1.1rem 1.2rem; min-height: 185px; }
      .limitation-card h4 { margin: 0 0 .45rem; color: #17211d; }
      .limitation-card p { margin: 0; color: #5e6a64; font-size: .9rem; }
      .method-card { background: #fffdf8; border-left: 4px solid #d78255; border-radius: 0 14px 14px 0; padding: 1.1rem 1.25rem; height: 100%; }
      .method-number { color: #b45e3b; font-size: .75rem; font-weight: 800; letter-spacing: .12em; }
      .method-card h4 { color: #17211d; margin: .4rem 0; }
      .method-card p { color: #5e6a64; font-size: .92rem; margin: 0; }
      .finding { background: #e7efe9; border-radius: 16px; padding: 1.35rem 1.5rem; margin-top: 1rem; }
      .finding strong { color: #14392f; }
      .caveat { background: #f5e8df; border-radius: 16px; padding: 1.25rem 1.5rem; color: #633b2c; }
      .takeaway { background: #e9edf4; border-left: 5px solid #5472a4; border-radius: 0 14px 14px 0; padding: 1.1rem 1.3rem; color: #34445d; margin: 1rem 0; }
      .takeaway strong { color: #263b5b; }
      div[data-testid="stMetric"] { background: #fffdf8; border: 1px solid #e5e0d5; border-radius: 14px; padding: 1rem; }
      .footer-note { color: #7b827e; font-size: .8rem; border-top: 1px solid #ddd8ce; padding-top: 1rem; margin-top: 3rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_project_summary() -> dict:
    combined = pd.read_csv(APP_DATA_DIR / "aapl_momentum_sentiment.csv")
    headlines = pd.read_csv(APP_DATA_DIR / "headlines.csv")
    return {
        "observations": int((pd.to_datetime(combined["date"]) <= "2024-11-22").sum()),
        "headlines": len(headlines),
        "reviewed": 30,
        "start": pd.to_datetime(combined["date"]).min().strftime("%b %Y"),
        "end": "Nov 2024",
    }


@st.cache_data
def load_explorer_data():
    predictions = pd.read_csv(
        APP_DATA_DIR / "explorer_predictions.csv", parse_dates=["date"], index_col="date"
    ).sort_index()
    headlines = pd.read_csv(
        APP_DATA_DIR / "headlines.csv",
        parse_dates=["published_at", "trading_date"],
    )
    prices = pd.read_csv(
        APP_DATA_DIR / "prices.csv",
        index_col="date",
        parse_dates=True,
    )[["AAPL", "SPY"]]
    return predictions, headlines, prices


@st.cache_data
def load_model_page_data():
    explorer = pd.read_csv(APP_DATA_DIR / "explorer_predictions.csv", parse_dates=["date"])
    comparison = pd.read_csv(APP_DATA_DIR / "model_comparison_predictions.csv", parse_dates=["date"])
    walk_forward = pd.read_csv(APP_DATA_DIR / "walk_forward_results.csv")
    non_overlapping = pd.read_csv(APP_DATA_DIR / "non_overlapping_results.csv")

    actual = explorer["outperformed"].astype(int)
    fixed_specs = {
        "Majority baseline": (
            comparison["Majority baseline_prediction"],
            comparison["Majority baseline_probability"],
        ),
        "Momentum only": (explorer["momentum_prediction"], explorer["momentum_probability"]),
        "Tone only": (explorer["tone_prediction"], explorer["tone_probability"]),
        "Momentum + tone + count": (
            comparison["Combined_prediction"],
            comparison["Combined_probability"],
        ),
    }
    fixed_rows = []
    for name, (prediction, probability) in fixed_specs.items():
        fixed_rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(actual, prediction.astype(int)),
                "ROC AUC": roc_auc_score(actual, probability),
            }
        )
    fixed = pd.DataFrame(fixed_rows)
    ablation = pd.DataFrame(
        {
            "Model": ["Majority baseline", "Tone only", "Headline count only", "Tone + count"],
            "Accuracy": [0.5000, 0.6271, 0.4407, 0.6356],
            "ROC AUC": [0.5000, 0.6716, 0.4723, 0.6179],
        }
    )
    return fixed, ablation, walk_forward, non_overlapping


@st.cache_data
def load_limitations_data():
    failures = pd.read_csv(APP_DATA_DIR / "failure_cases.csv", parse_dates=["date"])
    predictions = pd.read_csv(APP_DATA_DIR / "explorer_predictions.csv", parse_dates=["date"])
    predictions["relative_return"] = (
        predictions["aapl_future_return_5d"] - predictions["spy_future_return_5d"]
    )
    predictions["tone_confidence"] = (predictions["tone_probability"] - 0.5).abs()
    return failures, predictions


summary = load_project_summary()

with st.sidebar:
    st.markdown("### SIGNAL / STUDY")
    st.caption("AAPL × SPY · 2023–2024")
    st.divider()
    page = st.radio(
        "Explore",
        [
            "Research Overview",
            "Stock & News Explorer",
            "Model Comparison",
            "Failures & Limitations",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Educational research—not investment advice.")


if page == "Research Overview":
    st.markdown('<div class="eyebrow">An interpretable market-signals study</div>', unsafe_allow_html=True)
    st.title("Do headlines improve momentum signals?")
    st.markdown(
        '<div class="hero-copy">A focused experiment testing whether the tone of recent Apple news adds useful information to a simple price-momentum baseline.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="question-card">
          <small>Research question</small>
          <p>Does recent financial-news sentiment improve predictions of whether Apple will outperform the broader market over the next five trading days?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(4)
    metric_values = [
        (f"{summary['observations']:,}", "Eligible prediction dates"),
        (f"{summary['headlines']:,}", "FinBERT-scored headlines"),
        ("5 days", "Signal and outcome horizon"),
        (f"{summary['start']}–{summary['end']}", "Research window"),
    ]
    for column, (value, label) in zip(metric_columns, metric_values):
        with column:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )

    st.header("The experiment")
    method_columns = st.columns(4)
    methods = [
        ("01", "Measure momentum", "Calculate Apple’s adjusted-price return over the previous five trading days."),
        ("02", "Score headlines", "Use FinBERT probabilities to measure the tone of recent Apple coverage."),
        ("03", "Define the outcome", "Record whether AAPL beats SPY over the following five trading days."),
        ("04", "Test forward in time", "Train on earlier dates and evaluate on later dates that the models never saw."),
    ]
    for column, (number, title, copy) in zip(method_columns, methods):
        with column:
            st.markdown(
                f'<div class="method-card"><div class="method-number">STEP {number}</div><h4>{title}</h4><p>{copy}</p></div>',
                unsafe_allow_html=True,
            )

    st.header("What was compared")
    comparison = pd.DataFrame(
        {
            "Approach": ["Majority baseline", "Momentum only", "Headline tone", "Momentum + tone"],
            "Question": [
                "What if we always predict the most common training outcome?",
                "Do recent winners continue to outperform?",
                "Does recent news tone rank future relative performance?",
                "Does tone add useful information to momentum?",
            ],
            "Inputs": ["None", "5-day AAPL return", "5-day average FinBERT score", "Momentum and headline tone"],
        }
    )
    st.dataframe(comparison, hide_index=True, width="stretch")

    st.header("Current finding")
    st.markdown(
        """
        <div class="finding">
          <strong>Headline tone showed more ranking value than momentum, but no model produced consistently reliable classifications.</strong><br><br>
          Tone performed well in some held-out periods and poorly in others. Its relationship was mildly contrarian: more negative coverage sometimes preceded relative recovery. Walk-forward and non-overlapping tests showed that the result was sensitive to the evaluation period and weekly starting date.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Designed to avoid easy mistakes")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            - Features use only information available by the prediction date.
            - After-close and weekend headlines move to the next trading day.
            - Training always occurs before testing; dates are never shuffled.
            - A safety gap prevents training outcomes from crossing into testing.
            """
        )
    with right:
        st.markdown(
            """
            - Dataset sentiment was checked against 30 human-reviewed articles.
            - Ablation separated headline tone from headline volume.
            - Walk-forward tests checked stability across time.
            - Non-overlapping schedules reduced repeated five-day outcomes.
            """
        )

    st.header("How to read this study")
    st.caption("No finance or machine-learning background required—open any topic for a short explanation.")

    with st.expander("What does “outperform the market” mean?"):
        st.markdown(
            """
            We compare Apple with **SPY**, a fund used here as a stand-in for the broad U.S. stock market.

            - If AAPL rises 3% and SPY rises 1%, AAPL outperformed.
            - If AAPL falls 1% and SPY falls 3%, AAPL still outperformed—it lost less.
            - If AAPL rises 2% and SPY rises 4%, AAPL underperformed even though its price increased.

            This study predicts **relative performance**, not simply whether Apple’s price goes up.
            """
        )

    with st.expander("What is a walk-forward test, and why do we use it?"):
        st.markdown(
            """
            A walk-forward test repeatedly asks the model to learn from the past and take a test on the future:

            ```text
            Round 1: [ train Jan–Jun 2023 ] [ test Jul–Sep 2023 ]
            Round 2: [ train Jan–Sep 2023       ] [ test Oct–Dec 2023 ]
            Round 3: [ train Jan–Dec 2023                ] [ test Jan–Mar 2024 ]
            ```

            The training window grows after each round, but the test period always comes later. This is closer to real life than randomly mixing old and new dates. It also shows whether a result persists or works only during one unusually favorable period.
            """
        )

    with st.expander("What is a non-overlapping test, and why is it important?"):
        st.markdown(
            """
            Every answer in this project covers the **next five trading days**. If we predict every day, neighboring answers share most of the same future:

            ```text
            Monday prediction:    Tue · Wed · Thu · Fri · Mon
            Tuesday prediction:         Wed · Thu · Fri · Mon · Tue
            ```

            Those two tests overlap, so they are not fully independent. One market event can influence several rows and make the dataset appear larger than it really is.

            A non-overlapping test keeps only every fifth prediction date:

            ```text
            Prediction 1: days 1–5
            Prediction 2: days 6–10
            Prediction 3: days 11–15
            ```

            This produces fewer examples, but each result contains more distinct information. We tested all five possible starting offsets so the conclusion did not depend on an arbitrarily chosen starting day.
            """
        )

    with st.expander("What is an ablation comparison?"):
        st.markdown(
            """
            Ablation means removing one ingredient at a time to find out what contributed to a result. Our initial sentiment model used both headline tone and headline count, so we compared:

            - Tone only
            - Headline count only
            - Tone and count together

            Tone retained most of the performance, while headline count alone did not. This prevented us from incorrectly crediting “sentiment” when the useful input might have been news volume.
            """
        )

    with st.expander("How should I read accuracy and ROC AUC?"):
        metric_left, metric_right = st.columns(2)
        with metric_left:
            st.markdown(
                """
                **Accuracy**

                The percentage of yes/no predictions that were correct. An accuracy of 60% means 60 correct classifications out of every 100.

                Accuracy depends on the probability cutoff and can be misleading when one outcome is much more common.
                """
            )
        with metric_right:
            st.markdown(
                """
                **ROC AUC**

                Measures whether the model generally ranks actual outperforming dates above non-outperforming dates.

                - 50%: no useful ranking
                - Above 50%: some ranking information
                - 100%: perfect ranking
                """
            )

    st.markdown(
        """
        <div class="caveat"><strong>Scope:</strong> This is a single-company case study using one historical period. It evaluates a research hypothesis—not a deployable trading strategy. Transaction costs, taxes, and execution constraints are outside the current analysis.</div>
        """,
        unsafe_allow_html=True,
    )
elif page == "Stock & News Explorer":
    predictions, headlines, prices = load_explorer_data()
    st.markdown('<div class="eyebrow">One date, fully unpacked</div>', unsafe_allow_html=True)
    st.title("Stock & News Explorer")
    st.markdown(
        '<div class="hero-copy">Choose an unseen test date to see what the models knew, what they predicted, and what happened next.</div>',
        unsafe_allow_html=True,
    )

    selected_date = st.select_slider(
        "Prediction date",
        options=list(predictions.index),
        value=predictions.index[-1],
        format_func=lambda value: value.strftime("%B %d, %Y"),
        help="Every available date belongs to the held-out test period and was not used for model training.",
    )
    row = predictions.loc[selected_date]
    trading_dates = pd.DatetimeIndex(prices.index)
    position = trading_dates.get_loc(selected_date)
    past_date = trading_dates[position - 5]
    future_date = trading_dates[position + 5]
    feature_dates = trading_dates[position - 4 : position + 1]

    st.markdown("#### Closing Bell Snapshot")
    st.caption("The information available when the market closed on the selected prediction date.")
    known_columns = st.columns(3)
    known_values = [
        (f"{row['aapl_momentum_5d']:+.2%}", "AAPL return over the previous 5 trading days", None),
        (
            f"{row['average_sentiment_5d']:+.2f}",
            "Average headline tone, from −1 to +1",
            "For each headline, we subtract FinBERT’s negative probability from its positive probability. "
            "We then average those scores across headlines from the selected date and previous four trading days. "
            "Near +1 means consistently positive coverage, near −1 means consistently negative coverage, and near 0 means neutral or mixed coverage. "
            "We use probabilities instead of only labels so a slightly positive headline is not treated the same as a strongly positive one.",
        ),
        (f"{int(row['headline_count_5d'])}", "Headlines in the 5-trading-day window", None),
    ]
    for column, (value, label, tooltip) in zip(known_columns, known_values):
        with column:
            if tooltip:
                st.markdown(
                    f"""
                    <div class="metric-card tooltip-card" tabindex="0" aria-label="{label}. Focus or hover for explanation.">
                      <div class="metric-value">{value}</div>
                      <div class="metric-label">{label}<span class="info-dot" aria-hidden="true">i</span></div>
                      <div class="tooltip-content"><strong>How this is calculated</strong><br>{tooltip}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-value">{value}</div><div class="metric-label">{label}</div></div>',
                    unsafe_allow_html=True,
                )

    chart_start = max(0, position - 20)
    chart_end = min(len(prices), position + 11)
    chart_prices = prices.iloc[chart_start:chart_end]
    normalized = chart_prices.div(chart_prices.iloc[0]).mul(100)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(x=normalized.index, y=normalized["AAPL"], name="AAPL", line=dict(color="#b45e3b", width=3))
    )
    figure.add_trace(
        go.Scatter(x=normalized.index, y=normalized["SPY"], name="SPY", line=dict(color="#2f6d5c", width=2))
    )
    figure.add_vline(x=selected_date.timestamp() * 1000, line_dash="dash", line_color="#17211d")
    figure.add_annotation(
        x=selected_date,
        y=1.04,
        yref="paper",
        text="Prediction date",
        showarrow=False,
        font=dict(color="#17211d", size=12),
    )
    figure.update_layout(
        title="Price movement around the prediction date",
        yaxis_title="Normalized value (chart starts at 100)",
        xaxis_title=None,
        paper_bgcolor="#f6f4ef",
        plot_bgcolor="#fffdf8",
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=20, r=20, t=80, b=20),
        hovermode="x unified",
    )
    figure.update_xaxes(showgrid=False)
    figure.update_yaxes(gridcolor="#e5e0d5")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Momentum compares {past_date.strftime('%b %d')} with {selected_date.strftime('%b %d')}. "
        f"The outcome compares {selected_date.strftime('%b %d')} with {future_date.strftime('%b %d')}."
    )

    comparison_left, comparison_right = st.columns([1.08, 0.92], gap="large")
    prediction_specs = [
        ("Momentum model", "momentum", row["momentum_prediction"], row["momentum_probability"], row["momentum_correct"]),
        ("Headline-tone model", "tone", row["tone_prediction"], row["tone_probability"], row["tone_correct"]),
    ]
    with comparison_left:
        st.header("Model Forecasts")
        st.caption("Two different signals making the same yes-or-no decision.")
        for name, model_class, prediction, probability, correct in prediction_specs:
            label = "AAPL outperforms" if int(prediction) == 1 else "AAPL does not outperform"
            is_correct = str(correct).lower() == "true"
            status_class = "correct" if is_correct else "incorrect"
            status_label = "✓ Correct prediction" if is_correct else "✕ Incorrect prediction"
            st.markdown(
                f"""
                <div class="comparison-card prediction-card {model_class} {status_class}">
                  <div class="card-kicker">{name}</div>
                  <div class="card-outcome">{label}</div>
                  <div class="card-detail">{probability:.1%} estimated probability of AAPL outperforming SPY</div>
                  <span class="status-pill {status_class}">{status_label}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.caption("Probabilities near 50% indicate uncertainty. At 50% or above, the model predicts outperformance. Baseline and combined results appear on the Model Comparison page.")

    actual_label = "AAPL outperformed SPY" if int(row["outperformed"]) == 1 else "AAPL underperformed SPY"
    with comparison_right:
        st.header("Five-Day Scorecard")
        st.caption("The result revealed only after the prediction was made.")
        st.markdown(
            f"""
            <div class="comparison-card actual-card">
              <div class="card-kicker">Five trading days later</div>
              <div class="card-outcome">{actual_label}</div>
              <div class="card-detail">Outcome measured from {selected_date.strftime('%b %d')} through {future_date.strftime('%b %d, %Y')}.</div>
              <div class="actual-grid">
                <div class="actual-stat"><strong>{row['aapl_future_return_5d']:+.2%}</strong><span>AAPL return</span></div>
                <div class="actual-stat"><strong>{row['spy_future_return_5d']:+.2%}</strong><span>SPY return</span></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    direction = "positive" if row["aapl_momentum_5d"] >= 0 else "negative"
    tone_description = "positive" if row["average_sentiment_5d"] > 0.05 else "negative" if row["average_sentiment_5d"] < -0.05 else "mixed or neutral"
    actual_description = "outperformed" if int(row["outperformed"]) == 1 else "underperformed"
    st.markdown(
        f"""
        <div class="finding">
          Apple entered this date with <strong>{direction} five-day momentum</strong> and <strong>{tone_description} recent coverage</strong>.
          Over the following five trading days, AAPL returned <strong>{row['aapl_future_return_5d']:+.2%}</strong>
          while SPY returned <strong>{row['spy_future_return_5d']:+.2%}</strong>, so Apple <strong>{actual_description}</strong> the market.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Headlines behind the sentiment score")
    st.caption(
        "These articles were assigned to the selected date or previous four trading dates after applying market-close, weekend, and holiday rules."
    )
    recent = headlines.loc[headlines["trading_date"].isin(feature_dates)].copy()
    recent = recent.sort_values("published_at", ascending=False)

    positive = recent.nlargest(2, "sentiment_score")
    negative = recent.nsmallest(2, "sentiment_score")
    neutral_pool = recent.assign(distance_from_neutral=recent["sentiment_score"].abs())
    neutral = neutral_pool.loc[~neutral_pool.index.isin(positive.index.union(negative.index))].nsmallest(
        2, "distance_from_neutral"
    )

    headline_columns = st.columns(3)
    groups = [
        ("Most positive", positive, "#2f7d65"),
        ("Closest to neutral", neutral, "#68736d"),
        ("Most negative", negative, "#b45e3b"),
    ]
    for column, (heading, group, color) in zip(headline_columns, groups):
        with column:
            st.markdown(f"**{heading}**")
            for article in group.itertuples():
                st.markdown(
                    f"[{article.title}]({article.link})  \n"
                    f"<span style='color:{color};font-weight:700'>{article.sentiment_score:+.2f}</span> · "
                    f"{article.published_at.strftime('%b %d, %I:%M %p UTC')}",
                    unsafe_allow_html=True,
                )
                st.divider()

    with st.expander(f"Show all {len(recent)} headlines in this window"):
        full_headlines = recent[["published_at", "title", "finbert_label", "sentiment_score", "link"]].copy()
        full_headlines["published_at"] = full_headlines["published_at"].dt.strftime("%Y-%m-%d %H:%M UTC")
        full_headlines["sentiment_score"] = full_headlines["sentiment_score"].round(3)
        st.dataframe(
            full_headlines,
            hide_index=True,
            width="stretch",
            column_config={"link": st.column_config.LinkColumn("Source")},
        )

    with st.expander("Known then vs. revealed later"):
        known, revealed = st.columns(2)
        with known:
            st.markdown(
                """
                **Available at prediction time**

                - Historical AAPL and SPY prices
                - Previous five-day momentum
                - Headlines available by market close
                - FinBERT sentiment probabilities
                """
            )
        with revealed:
            st.markdown(
                """
                **Revealed only afterward**

                - AAPL’s following five-day return
                - SPY’s following five-day return
                - The correct outperformance answer
                - Whether each prediction was correct
                """
            )

elif page == "Model Comparison":
    fixed, ablation, walk_forward, non_overlapping = load_model_page_data()
    model_colors = {
        "Majority baseline": "#8a918d",
        "Momentum only": "#bf6842",
        "Tone only": "#5472a4",
        "Headline count only": "#b79b50",
        "Tone + count": "#6f8d75",
        "Momentum + tone": "#6f8d75",
        "Momentum + tone + count": "#6f8d75",
    }

    st.markdown('<div class="eyebrow">From one promising result to a tougher conclusion</div>', unsafe_allow_html=True)
    st.title("Model Comparison")
    st.markdown(
        '<div class="hero-copy compact-copy">The first test made headline tone look strongest. Repeating the experiment across time and with non-overlapping outcomes showed why one good score is not enough.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="compact-section-title">First Look: June 7–November 22, 2024</h2>', unsafe_allow_html=True)
    st.caption("One unseen test window containing 118 later dates; every model was trained only on earlier observations.")
    fixed_columns = st.columns(4)
    for column, result in zip(fixed_columns, fixed.itertuples()):
        with column:
            color = model_colors[result.Model]
            st.markdown(
                f"""
                <div class="metric-card" style="border-top:5px solid {color}">
                  <div class="metric-value">{result.Accuracy:.1%}</div>
                  <div class="metric-label">{result.Model}<br>accuracy in this window</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    fixed_figure = go.Figure()
    for metric in ["Accuracy", "ROC AUC"]:
        fixed_figure.add_trace(
            go.Bar(
                name=metric,
                x=fixed["Model"],
                y=fixed[metric] * 100,
                text=(fixed[metric] * 100).map(lambda value: f"{value:.1f}%"),
                textposition="outside",
            )
        )
    fixed_figure.update_layout(
        barmode="group",
        yaxis_title="Score (%)",
        yaxis_range=[0, 80],
        paper_bgcolor="#f6f4ef",
        plot_bgcolor="#fffdf8",
        legend=dict(orientation="h", y=1.12),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    fixed_figure.update_yaxes(gridcolor="#e5e0d5")
    st.plotly_chart(fixed_figure, width="stretch")
    st.markdown(
        """
        <div class="takeaway"><strong>Initial takeaway:</strong> Tone correctly classified 62.7% of dates and ranked outcomes better than momentum. That made it the strongest model <em>in this particular window</em>—not necessarily everywhere.</div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Accuracy vs. ROC AUC"):
        st.markdown(
            """
            **Accuracy** asks: “After converting probabilities into yes/no predictions, how many answers were correct?”

            **ROC AUC** asks: “Did the model generally give higher probabilities to dates when AAPL actually outperformed?” A score of 50% means no useful ranking. ROC AUC can be informative even when a fixed 50% cutoff produces unstable classifications.
            """
        )

    st.header("Ingredient Check: Tone or News Volume?")
    st.caption("The sentiment model originally used both the average tone and the number of headlines. Ablation removes one ingredient at a time.")

    def make_ablation_ladder(metric):
        values = ablation.set_index("Model")[metric]
        steps = [
            ("1  Full model: tone + count", "Tone + count", "Both ingredients included"),
            ("2A  Remove headline count", "Tone only", "Tone remains; count is removed"),
            ("2B  Remove headline tone", "Headline count only", "Count remains; tone is removed"),
            ("3  No signal features", "Majority baseline", "Neither ingredient is used"),
        ]
        labels = [step[0] for step in steps]
        scores = [values.loc[step[1]] * 100 for step in steps]
        colors = [model_colors[step[1]] for step in steps]
        descriptions = [step[2] for step in steps]
        figure = go.Figure(
            go.Bar(
                x=scores,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{score:.1f}%" for score in scores],
                textposition="outside",
                customdata=descriptions,
                hovertemplate="%{y}<br>%{customdata}<br>" + metric + ": %{x:.1f}%<extra></extra>",
            )
        )
        figure.add_vline(x=50, line_dash="dash", line_color="#c34f43", line_width=2)
        figure.add_annotation(
            x=50,
            y=1.08,
            yref="paper",
            text="50% baseline reference",
            showarrow=False,
            font=dict(color="#a43d34", size=12),
        )
        figure.update_layout(
            xaxis_title=f"{metric} (%)",
            xaxis_range=[0, 75],
            yaxis_title=None,
            paper_bgcolor="#f6f4ef",
            plot_bgcolor="#fffdf8",
            showlegend=False,
            margin=dict(l=20, r=35, t=70, b=20),
            height=430,
        )
        figure.update_xaxes(gridcolor="#e5e0d5")
        figure.update_yaxes(autorange="reversed")
        return figure

    accuracy_tab, auc_tab = st.tabs(["Accuracy", "ROC AUC"])
    with accuracy_tab:
        st.plotly_chart(make_ablation_ladder("Accuracy"), width="stretch")
        st.caption("Removing headline count reduced accuracy by less than one percentage point. Removing tone pushed accuracy below the no-feature baseline.")
    with auc_tab:
        st.plotly_chart(make_ablation_ladder("ROC AUC"), width="stretch")
        st.caption("Tone alone produced the strongest probability ranking. Headline count alone ranked outcomes no better than the baseline.")
    st.markdown(
        """
        <div class="takeaway"><strong>What contributed?</strong> Tone alone retained nearly all the result. Headline count alone fell below the baseline, so the initial improvement was not simply caused by having more news.</div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Reality Check: Did It Work Across Time?")
    st.caption("Walk-forward testing repeats the experiment across six later periods, always training on dates that came first.")
    walk_figure = go.Figure()
    period_order = walk_forward["Period"].drop_duplicates().tolist()
    for model_name in ["Majority baseline", "Momentum only", "Tone only", "Momentum + tone"]:
        model_data = walk_forward.loc[walk_forward["Model"].eq(model_name)].set_index("Period").loc[period_order]
        walk_figure.add_trace(
            go.Scatter(
                x=period_order,
                y=model_data["Accuracy"] * 100,
                name=model_name,
                mode="lines+markers",
                line=dict(color=model_colors[model_name], width=3),
                marker=dict(size=8),
                hovertemplate="%{x}<br>Accuracy: %{y:.1f}%<extra>%{fullData.name}</extra>",
            )
        )
    walk_figure.add_hline(y=50, line_dash="dash", line_color="#6f7773")
    walk_figure.update_layout(
        yaxis_title="Accuracy (%)",
        yaxis_range=[15, 80],
        paper_bgcolor="#f6f4ef",
        plot_bgcolor="#fffdf8",
        legend=dict(orientation="h", y=1.16),
        margin=dict(l=20, r=20, t=90, b=20),
        hovermode="x unified",
    )
    walk_figure.update_yaxes(gridcolor="#e5e0d5")
    st.plotly_chart(walk_figure, width="stretch")

    weighted_rows = []
    for model_name, group in walk_forward.groupby("Model"):
        weighted_rows.append(
            {
                "Model": model_name,
                "Weighted accuracy": (group["Accuracy"] * group["Test rows"]).sum() / group["Test rows"].sum(),
                "Weighted ROC AUC": (group["ROC AUC"] * group["Test rows"]).sum() / group["Test rows"].sum(),
            }
        )
    weighted = pd.DataFrame(weighted_rows).sort_values("Weighted ROC AUC", ascending=False)
    st.dataframe(
        weighted,
        hide_index=True,
        width="stretch",
        column_config={
            "Weighted accuracy": st.column_config.NumberColumn(format="percent"),
            "Weighted ROC AUC": st.column_config.NumberColumn(format="percent"),
        },
    )
    st.markdown(
        """
        <div class="caveat"><strong>Important change in the story:</strong> Tone’s accuracy did not consistently beat the baseline across periods. Its probability ranking remained more promising, but the yes/no results depended heavily on the test dates.</div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Stricter Check: Non-Overlapping Weeks")
    st.caption("Keeping only every fifth prediction reduces repeated outcomes. We tested all five possible starting offsets.")
    schedule_table = non_overlapping.pivot(index="Schedule", columns="Model", values="Accuracy") * 100
    schedule_figure = go.Figure()
    for model_name in ["Majority baseline", "Momentum only", "Tone only", "Momentum + tone"]:
        schedule_figure.add_trace(
            go.Scatter(
                x=schedule_table.index,
                y=schedule_table[model_name],
                name=model_name,
                mode="lines+markers",
                line=dict(color=model_colors[model_name], width=3),
                marker=dict(size=9),
            )
        )
    schedule_figure.add_hline(y=50, line_dash="dash", line_color="#6f7773")
    schedule_figure.update_layout(
        xaxis_title="Every-fifth-day starting schedule",
        yaxis_title="Accuracy (%)",
        yaxis_range=[25, 75],
        paper_bgcolor="#f6f4ef",
        plot_bgcolor="#fffdf8",
        legend=dict(orientation="h", y=1.16),
        margin=dict(l=20, r=20, t=90, b=20),
    )
    schedule_figure.update_xaxes(dtick=1)
    schedule_figure.update_yaxes(gridcolor="#e5e0d5")
    st.plotly_chart(schedule_figure, width="stretch")

    nonoverlap_summary = non_overlapping.groupby("Model").agg(
        mean_accuracy=("Accuracy", "mean"),
        lowest_accuracy=("Accuracy", "min"),
        highest_accuracy=("Accuracy", "max"),
        mean_roc_auc=("ROC AUC", "mean"),
    ).reset_index()
    st.dataframe(
        nonoverlap_summary,
        hide_index=True,
        width="stretch",
        column_config={
            "mean_accuracy": st.column_config.NumberColumn("Average accuracy", format="percent"),
            "lowest_accuracy": st.column_config.NumberColumn("Lowest accuracy", format="percent"),
            "highest_accuracy": st.column_config.NumberColumn("Highest accuracy", format="percent"),
            "mean_roc_auc": st.column_config.NumberColumn("Average ROC AUC", format="percent"),
        },
    )

    st.header("Bottom Line")
    st.markdown(
        """
        <div class="finding">
          <strong>No model was consistently reliable across every test.</strong><br><br>
          Momentum added little predictive value. Headline tone showed the most interesting ranking signal, often in a contrarian direction, but its classification accuracy changed with the period and weekly starting date. The evidence supports further research—not a trading claim.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Why we stopped adding more models"):
        st.markdown(
            """
            Testing many algorithms until one produces a favorable score would risk **model-shopping**—selecting a result that fits this historical sample by chance. We kept the comparison simple and interpretable, then challenged the initial result with stricter tests.

            Future work includes sentiment dispersion, additional companies, longer histories, and new unseen periods. Those ideas should be defined before evaluating new data.
            """
        )

elif page == "Failures & Limitations":
    failures, predictions = load_limitations_data()

    st.markdown('<div class="eyebrow">Where the story gets more credible</div>', unsafe_allow_html=True)
    st.title("Failures & Limitations")
    st.markdown(
        '<p class="page-subtitle">A useful model story includes the mistakes. This page shows where the signals broke, how uncertain the forecasts were, and what this experiment cannot prove.</p>',
        unsafe_allow_html=True,
    )

    st.header("Where the Signals Broke")
    st.write(
        "Choose a type of failure, then inspect a real date from the held-out test period. "
        "These cases help explain *how* the models failed—not just how often."
    )

    category = st.selectbox(
        "Failure pattern",
        failures["failure_category"].drop_duplicates().tolist(),
    )
    category_cases = failures.loc[failures["failure_category"] == category].sort_values("date")
    selected_date = st.selectbox(
        "Example date",
        category_cases["date"].dt.strftime("%B %d, %Y").tolist(),
    )
    case = category_cases.loc[
        category_cases["date"].dt.strftime("%B %d, %Y") == selected_date
    ].iloc[0]
    actual_label = "AAPL outperformed SPY" if case["outperformed"] else "AAPL underperformed SPY"
    momentum_label = "outperform" if case["momentum_prediction"] else "underperform"
    tone_label = "outperform" if case["tone_prediction"] else "underperform"
    relative_return = case["aapl_future_return_5d"] - case["spy_future_return_5d"]

    st.markdown(
        f"""
        <div class="failure-card">
          <div class="case-date">{case['date']:%B %d, %Y} · {case['failure_category']}</div>
          <h4>{actual_label} over the next five trading days</h4>
          <p>AAPL returned <strong>{case['aapl_future_return_5d']:+.2%}</strong> versus
          <strong>{case['spy_future_return_5d']:+.2%}</strong> for SPY—a relative result of
          <strong>{relative_return:+.2%}</strong>.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        st.metric("Momentum forecast", momentum_label.title(), f"{case['momentum_probability']:.1%} probability")
        st.caption(f"Known input: {case['aapl_momentum_5d']:+.2%} AAPL momentum")
    with right:
        st.metric("Headline-tone forecast", tone_label.title(), f"{case['tone_probability']:.1%} probability")
        st.caption(f"Known input: {case['average_sentiment_5d']:+.2f} average tone")

    with st.expander("Headlines available around this example", expanded=True):
        for headline in str(case["headline_context"]).split(" | "):
            st.markdown(f"- {headline}")
    st.caption(
        "Important: these examples were selected after the outcomes were known. They illustrate failure modes; "
        "the performance tests on the Model Comparison page are the unbiased evaluation."
    )

    st.header("Most Forecasts Were Close Calls")
    median_distance = predictions["tone_confidence"].median()
    within_five = (predictions["tone_confidence"] <= 0.05).mean()
    close_count = int((predictions["tone_confidence"] <= 0.05).sum())
    total_predictions = len(predictions)
    lowest_probability = predictions["tone_probability"].min()
    highest_probability = predictions["tone_probability"].max()

    st.markdown(
        f"""
        <div class="finding">
          <strong>{close_count} of {total_predictions} forecasts were between 45% and 55%.</strong><br><br>
          That range is close to a 50–50 guess. In other words, the model usually leaned slightly toward one outcome—it rarely expressed strong confidence.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(within_five, text=f"{within_five:.1%} of forecasts were near 50–50")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Most cautious forecast", f"{lowest_probability:.1%}")
        st.caption(f"Only {(0.5 - lowest_probability):.1%} below 50% → a slight lean toward underperformance")
    with c2:
        st.metric("The decision point", "50.0%")
        st.caption("Below 50% means underperform; 50% or above means outperform")
    with c3:
        st.metric("Most confident forecast", f"{highest_probability:.1%}")
        st.caption(f"Only {(highest_probability - 0.5):.1%} above 50% → a modest lean toward outperformance")

    with st.expander("Why probabilities near 50% matter"):
        st.write(
            f"A probability near 50% means the model sees little separation between the two possible outcomes. "
            f"Across this test, the typical forecast was just {median_distance:.1%} away from 50%. "
            "The model still had to choose a label, but that label could come from a very small lean. "
            "So a correct classification does not necessarily mean the signal was strong."
        )

    st.header("The Boundaries of This Study")
    limitations = [
        ("One company", "The analysis covers AAPL only. Its news cycle and investor attention may not represent other stocks."),
        ("A short era", "The usable experiment spans 2023 through November 22, 2024, so it includes only a limited set of market conditions."),
        ("Overlapping outcomes", "Daily five-day targets reuse many of the same trading days. The every-fifth-day test reduces this problem but leaves only about 23 cases per schedule."),
        ("Tone is not impact", "A positive headline can already be expected, irrelevant to valuation, or outweighed by other events."),
        ("Small human review", "Only 30 headlines were manually labeled. FinBERT agreed more often than the dataset labels, but that check is still small and subjective."),
        ("Association, not causation", "The results cannot show that headlines caused later returns; news and prices may respond to the same underlying events."),
        ("Simple models", "Logistic regression keeps the experiment interpretable, but it may miss nonlinear relationships and changing market regimes."),
        ("No trading simulation", "Accuracy and ROC AUC do not include transaction costs, taxes, slippage, position sizing, or investment risk."),
        ("Researcher choices", "Window length, cutoff time, features, and thresholds were design decisions. Trying many alternatives could create a lucky result."),
    ]
    for start in range(0, len(limitations), 3):
        cols = st.columns(3)
        for col, (title, body) in zip(cols, limitations[start:start + 3]):
            with col:
                st.markdown(
                    f'<div class="limitation-card"><h4>{title}</h4><p>{body}</p></div>',
                    unsafe_allow_html=True,
                )

    st.header("What the Evidence Supports")
    can_col, cannot_col = st.columns(2)
    with can_col:
        st.subheader("What we can say")
        st.markdown(
            """
            - Headline tone contained period-dependent ranking information.
            - Momentum was weak in these tests.
            - The original dataset sentiment labels were unreliable in the manual review.
            - Stricter robustness tests changed the initial conclusion.
            """
        )
    with cannot_col:
        st.subheader("What we cannot say")
        st.markdown(
            """
            - Headlines caused AAPL's later returns.
            - Positive news guarantees outperformance.
            - Any model here is a profitable trading strategy.
            - The result generalizes to other companies or future periods.
            """
        )

    st.header("Future Work—Defined Before the Next Test")
    with st.expander("Sentiment dispersion: the next feature to test", expanded=True):
        st.markdown(
            """
            **Average tone can hide disagreement.** Five very positive and five very negative headlines can average to neutral, just like ten genuinely neutral headlines.

            Sentiment dispersion would measure how spread out the headline scores are—typically with their standard deviation. High dispersion could flag a conflicted news day. It is listed as future work because adding it after seeing these results and reporting only a favorable score would weaken the evaluation. The feature should be defined first, then tested on new unseen data.
            """
        )
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**Broaden the evidence**")
        st.markdown("- Add companies with sufficient timestamped news\n- Collect a longer, fully unseen period\n- Pre-specify earnings and non-earnings comparisons")
    with f2:
        st.markdown("**Move closer to a real strategy test**")
        st.markdown("- Choose classification thresholds on validation data\n- Test probability calibration\n- Include costs, slippage, sizing, and risk")

    st.markdown(
        """
        <div class="caveat"><strong>Responsible conclusion:</strong> This project is useful because it found a nuanced, unstable signal and documented where it failed—not because it produced a trading system.</div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.error("Page not found.")

st.markdown(
    '<div class="footer-note">Do Headlines Improve Momentum Signals? · Educational research project · Not investment advice</div>',
    unsafe_allow_html=True,
)
