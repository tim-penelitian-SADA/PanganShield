"""
pages/output.py
Halaman hasil forecasting, evaluasi model, stacking ensemble,
dan analisis risiko komoditas.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import (
    evaluate_prediction,
    format_rupiah,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_trained_model,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Output Model — KomoditasAI",
    page_icon="📈",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / output",
    title="Hasil Forecasting",
    caption=(
        "Evaluasi performa base learner, proses stacking ensemble, "
        "prediksi harga, dan analisis risiko komoditas."
    ),
)


# ============================================================
# HELPER
# ============================================================

def fit_sarima_array(values, order, seasonal_order):
    """Fit SARIMA pada array observasi."""

    values = np.asarray(values, dtype=float)

    return SARIMAX(
        values,
        order=order,
        seasonal_order=seasonal_order,
        trend="c" if order[1] == 0 else "n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(
        disp=False,
        maxiter=150,
        method="lbfgs",
    )


def append_observation(result, actual):
    """Tambahkan observasi baru tanpa refit penuh."""

    try:
        return result.append(
            np.asarray([actual], dtype=float),
            refit=False,
        )
    except Exception:
        return None


def sarima_one_step_predictions(
    history_values,
    actual_future_values,
    order,
    seasonal_order,
    refit_every=0,
):
    """
    Menghasilkan prediksi SARIMA one-step-ahead
    menggunakan observasi aktual secara sequential.
    """

    history = list(
        np.asarray(
            history_values,
            dtype=float,
        )
    )

    future = np.asarray(
        actual_future_values,
        dtype=float,
    )

    predictions = []

    result = fit_sarima_array(
        history,
        order,
        seasonal_order,
    )

    for step, actual in enumerate(
        future,
        start=1,
    ):

        pred = float(
            result.forecast(
                steps=1
            )[0]
        )

        predictions.append(pred)

        history.append(
            float(actual)
        )

        appended = append_observation(
            result,
            actual,
        )

        need_refit = (
            appended is None
            or (
                refit_every
                and step % refit_every == 0
            )
        )

        if need_refit:
            result = fit_sarima_array(
                history,
                order,
                seasonal_order,
            )
        else:
            result = appended

    return np.asarray(
        predictions
    )


# ============================================================
# LOAD TRAINING RESULT
# ============================================================

trained, data, params = require_trained_model()

best_svr = trained["best_svr"]
best_rf = trained["best_rf"]

best_sarima_order = trained[
    "best_sarima_order"
]

best_sarima_seasonal_order = trained[
    "best_sarima_seasonal_order"
]

X_train = data["X_train"]
X_test = data["X_test"]

y_train = data["y_train"]
y_test = data["y_test"]

train_series = data["train_series"]
test_series = data["test_series"]

harga = data["harga"]

cv_splits = params["cv_splits"]
random_state = params["random_state"]

commodity_name = params.get(
    "commodity_column",
    "Komoditas",
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

st.markdown(
    "### 📄 Dataset & Model Overview"
)

with st.container(border=True):

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Komoditas",
            commodity_name,
            "Dataset aktif",
        )

    with c2:
        st.metric(
            "Data Train",
            f"{len(train_series):,}".replace(
                ",",
                ".",
            ),
            "Observasi",
        )

    with c3:
        st.metric(
            "Data Test",
            f"{len(test_series):,}".replace(
                ",",
                ".",
            ),
            "Observasi",
        )

    with c4:
        st.metric(
            "Model Ensemble",
            "3 Base Learners",
            "SARIMA + SVR + Random Forest",
        )


st.write("")


# ============================================================
# 1. BASE MODEL
# ============================================================

st.markdown(
    "### 1. 🔄 Membangun Base Learners"
)

with st.spinner(
    "Membangun base learners..."
):

    # SVR
    best_svr.fit(
        X_train,
        y_train,
    )

    # Random Forest
    best_rf.fit(
        X_train,
        y_train,
    )

st.success(
    "✓ Base learners berhasil dibangun."
)


# ============================================================
# 2. BASE LEARNER PREDICTION
# ============================================================

st.markdown(
    "### 2. 📈 Prediksi Base Learners"
)

with st.spinner(
    "Menghasilkan prediksi..."
):

    # --------------------------------------------------------
    # SARIMA
    # --------------------------------------------------------

    sarima_test_pred = pd.Series(
        sarima_one_step_predictions(
            history_values=train_series.values,
            actual_future_values=test_series.values,
            order=best_sarima_order,
            seasonal_order=best_sarima_seasonal_order,
        ),
        index=test_series.index,
    )

    # --------------------------------------------------------
    # SVR
    # --------------------------------------------------------

    svr_train_pred = pd.Series(
        best_svr.predict(
            X_train
        ),
        index=X_train.index,
    )

    svr_test_pred = pd.Series(
        best_svr.predict(
            X_test
        ),
        index=X_test.index,
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_train_pred = pd.Series(
        best_rf.predict(
            X_train
        ),
        index=X_train.index,
    )

    rf_test_pred = pd.Series(
        best_rf.predict(
            X_test
        ),
        index=X_test.index,
    )

st.success(
    "✓ Prediksi seluruh base learners selesai."
)


# ============================================================
# 3. BASE PREDICTION CHART
# ============================================================

st.markdown(
    "### 3. 📊 Perbandingan Base Learners"
)

with st.container(border=True):

    col1, col2, col3 = st.columns([1, 8, 1])

    with col2:

        fig, ax = plt.subplots(
            figsize=(10, 5),
            dpi=120,
        )

        ax.plot(
            test_series.index,
            test_series,
            linewidth=2.8,
            label="Actual",
        )

        ax.plot(
            sarima_test_pred.index,
            sarima_test_pred,
            linewidth=1.8,
            label="SARIMA",
        )

        ax.plot(
            svr_test_pred.index,
            svr_test_pred,
            linewidth=1.8,
            label="SVR",
        )

        ax.plot(
            rf_test_pred.index,
            rf_test_pred,
            linewidth=1.8,
            label="Random Forest",
        )

        ax.set_xlabel(
            "Tanggal"
        )

        ax.set_ylabel(
            f"Harga {commodity_name}"
        )

        ax.set_title(
            "Perbandingan Aktual dan Prediksi Base Learners"
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend(
            frameon=False
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# BASE PREDICTION TABLE
# ============================================================

with st.expander(
    "📋 Lihat Detail Prediksi Base Learners"
):

    prediction_df = pd.DataFrame(
        {
            "Actual": test_series,
            "SARIMA": sarima_test_pred,
            "SVR": svr_test_pred,
            "Random Forest": rf_test_pred,
        }
    )

    st.dataframe(
        prediction_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 4. EVALUATION
# ============================================================

st.markdown(
    "### 4. 📊 Evaluasi Performa Model"
)

sarima_metric = evaluate_prediction(
    test_series,
    sarima_test_pred,
    train_series,
    "SARIMA",
)

svr_metric = evaluate_prediction(
    y_test,
    svr_test_pred,
    y_train,
    "SVR",
)

rf_metric = evaluate_prediction(
    y_test,
    rf_test_pred,
    y_train,
    "Random Forest",
)

metric_table = pd.DataFrame(
    [
        sarima_metric,
        svr_metric,
        rf_metric,
    ]
)


METRIC_FORMAT = {
    "RMSE": "{:.2f}",
    "MAE": "{:.2f}",
    "MAPE (%)": "{:.2f}",
    "sMAPE (%)": "{:.2f}",
    "MASE": "{:.3f}",
    "R2": "{:.4f}",
    "Bias": "{:.2f}",
}


with st.container(border=True):

    st.dataframe(
        metric_table.style.format(
            METRIC_FORMAT
        ),
        use_container_width=True,
        hide_index=True,
    )

    best_model = (
        metric_table
        .sort_values(
            "RMSE"
        )
        .iloc[0]
    )

    st.success(
        f"Model dengan performa terbaik berdasarkan RMSE "
        f"adalah **{best_model['Model']}** "
        f"dengan RMSE **{best_model['RMSE']:.2f}**."
    )


# ============================================================
# 5. OOF META FEATURES
# ============================================================

st.markdown(
    "### 5. 🧩 Pembentukan Meta Features"
)


tscv = TimeSeriesSplit(
    n_splits=cv_splits
)

oof_svr = pd.Series(
    index=X_train.index,
    dtype=float,
)

oof_rf = pd.Series(
    index=X_train.index,
    dtype=float,
)

oof_sarima = pd.Series(
    index=train_series.index,
    dtype=float,
)


progress = st.progress(
    0,
    text="Membangun OOF prediction...",
)


for fold, (
    tr_idx,
    val_idx,
) in enumerate(
    tscv.split(X_train),
    start=1,
):

    X_tr = X_train.iloc[
        tr_idx
    ]

    X_val = X_train.iloc[
        val_idx
    ]

    y_tr = y_train.iloc[
        tr_idx
    ]

    y_val = y_train.iloc[
        val_idx
    ]

    # --------------------------------------------------------
    # SVR
    # --------------------------------------------------------

    svr_fold = clone(
        best_svr
    )

    svr_fold.fit(
        X_tr,
        y_tr,
    )

    oof_svr.iloc[
        val_idx
    ] = svr_fold.predict(
        X_val
    )

    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    rf_fold = clone(
        best_rf
    )

    rf_fold.fit(
        X_tr,
        y_tr,
    )

    oof_rf.iloc[
        val_idx
    ] = rf_fold.predict(
        X_val
    )

    # --------------------------------------------------------
    # SARIMA
    # --------------------------------------------------------

    validation_start = (
        y_val.index.min()
    )

    first_val_position = (
        train_series.index.get_loc(
            validation_start
        )
    )

    history_values = (
        train_series
        .iloc[
            :first_val_position
        ]
        .values
    )

    future_values = (
        train_series
        .loc[
            y_val.index
        ]
        .values
    )

    oof_sarima.loc[
        y_val.index
    ] = sarima_one_step_predictions(
        history_values=history_values,
        actual_future_values=future_values,
        order=best_sarima_order,
        seasonal_order=best_sarima_seasonal_order,
    )

    progress.progress(
        int(
            fold
            / cv_splits
            * 100
        ),
        text=(
            f"Membangun OOF "
            f"fold {fold}/{cv_splits}"
        ),
    )


progress.empty()


oof_table = pd.DataFrame(
    {
        "SARIMA": oof_sarima,
        "SVR": oof_svr,
        "RF": oof_rf,
        "Target": y_train,
    }
).dropna()


st.success(
    f"✓ OOF berhasil dibuat dengan "
    f"{len(oof_table):,} observasi."
)


with st.expander(
    "📋 Lihat OOF Prediction"
):

    st.dataframe(
        oof_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 6. META LEARNER
# ============================================================

st.markdown(
    "### 6. 🧠 Meta Learner"
)

meta_learner = LinearRegression()

meta_learner.fit(
    oof_table[
        [
            "SARIMA",
            "SVR",
            "RF",
        ]
    ],
    oof_table[
        "Target"
    ],
)


meta_test = pd.DataFrame(
    {
        "SARIMA": sarima_test_pred,
        "SVR": svr_test_pred,
        "RF": rf_test_pred,
    }
).loc[
    y_test.index
]


ensemble_pred = pd.Series(
    meta_learner.predict(
        meta_test
    ),
    index=y_test.index,
)


coef_table = pd.DataFrame(
    {
        "Base Learner": [
            "SARIMA",
            "SVR",
            "Random Forest",
        ],
        "Koefisien": [
            meta_learner.coef_[0],
            meta_learner.coef_[1],
            meta_learner.coef_[2],
        ],
    }
)


coef_table[
    "Koefisien"
] = coef_table[
    "Koefisien"
].round(4)


with st.container(border=True):

    st.markdown(
        "#### Bobot Base Learners"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "SARIMA",
            f"{meta_learner.coef_[0]:.4f}",
        )

    with c2:
        st.metric(
            "SVR",
            f"{meta_learner.coef_[1]:.4f}",
        )

    with c3:
        st.metric(
            "Random Forest",
            f"{meta_learner.coef_[2]:.4f}",
        )

    st.dataframe(
        coef_table,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        f"Intercept Meta Learner: "
        f"{meta_learner.intercept_:.4f}"
    )


# ============================================================
# 7. STACKING ENSEMBLE
# ============================================================

st.markdown(
    "### 7. 🚀 Stacking Ensemble"
)

stacking_metric = evaluate_prediction(
    y_test,
    ensemble_pred,
    y_train,
    "Stacking Ensemble",
)


with st.container(border=True):

    st.markdown(
        "#### Model Utama"
    )

    st.subheader(
        "Stacking Ensemble"
    )

    st.caption(
        "SARIMA + SVR + Random Forest"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "RMSE",
            f"{stacking_metric['RMSE']:.2f}",
        )

    with c2:
        st.metric(
            "MAE",
            f"{stacking_metric['MAE']:.2f}",
        )

    with c3:
        st.metric(
            "MAPE",
            f"{stacking_metric['MAPE (%)']:.2f}%",
        )

    with c4:
        st.metric(
            "R²",
            f"{stacking_metric['R2']:.4f}",
        )


# ============================================================
# STACKING CHART
# ============================================================

st.markdown(
    "#### 📈 Aktual vs Prediksi Ensemble"
)

prediction_result = pd.DataFrame(
    {
        "Actual": y_test,
        "SARIMA": sarima_test_pred,
        "SVR": svr_test_pred,
        "Random Forest": rf_test_pred,
        "Stacking Ensemble": ensemble_pred,
    }
)

with st.container(border=True):

    col1, col2, col3 = st.columns([1, 8, 1])

    with col2:

        fig, ax = plt.subplots(
            figsize=(10, 5),
            dpi=120,
        )

        ax.plot(
            prediction_result.index,
            prediction_result["Actual"],
            linewidth=2.8,
            label="Actual",
        )

        ax.plot(
            prediction_result.index,
            prediction_result["Stacking Ensemble"],
            linewidth=2.8,
            label="Stacking Ensemble",
        )

        ax.set_xlabel(
            "Tanggal"
        )

        ax.set_ylabel(
            f"Harga {commodity_name}"
        )

        ax.set_title(
            "Aktual vs Prediksi Stacking Ensemble"
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend(
            frameon=False
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

# ============================================================
# PREDICTION RESULT TABLE
# ============================================================

with st.expander(
    "📋 Lihat Detail Hasil Prediksi"
):

    st.dataframe(
        prediction_result,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 8. VALUE AT RISK
# ============================================================

st.markdown(
    "### 8. ⚠️ Analisis Risiko"
)

meta_oof_pred = pd.Series(
    meta_learner.predict(
        oof_table[
            [
                "SARIMA",
                "SVR",
                "RF",
            ]
        ]
    ),
    index=oof_table.index,
)


abs_error = (
    oof_table["Target"]
    - meta_oof_pred
).abs()


var90 = abs_error.quantile(
    0.90
)

var95 = abs_error.quantile(
    0.95
)

var99 = abs_error.quantile(
    0.99
)


var_table = pd.DataFrame(
    {
        "Hari": range(
            1,
            6,
        )
    }
)


var_table["VaR 90%"] = (
    var90
    * np.sqrt(
        var_table["Hari"]
    )
)

var_table["VaR 95%"] = (
    var95
    * np.sqrt(
        var_table["Hari"]
    )
)

var_table["VaR 99%"] = (
    var99
    * np.sqrt(
        var_table["Hari"]
    )
)


# ============================================================
# VAR SUMMARY
# ============================================================

with st.container(border=True):

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "VaR 90% · 1 Hari",
            format_rupiah(
                var90
            ),
        )

    with c2:
        st.metric(
            "VaR 95% · 1 Hari",
            format_rupiah(
                var95
            ),
        )

    with c3:
        st.metric(
            "VaR 99% · 1 Hari",
            format_rupiah(
                var99
            ),
        )


st.write("")


# ============================================================
# VAR TABLE
# ============================================================

with st.container(border=True):

    st.markdown(
        "#### VaR Berdasarkan Horizon"
    )

    st.dataframe(
        var_table.style.format(
            {
                "VaR 90%": "Rp{:,.2f}",
                "VaR 95%": "Rp{:,.2f}",
                "VaR 99%": "Rp{:,.2f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# VAR CHART
# ============================================================

st.markdown(
    "#### 📈 Perkembangan Risiko 1–5 Hari"
)

with st.container(border=True):

    col1, col2, col3 = st.columns([1, 8, 1])

    with col2:

        fig, ax = plt.subplots(
            figsize=(9, 5),
            dpi=120,
        )

        for col in [
            "VaR 90%",
            "VaR 95%",
            "VaR 99%",
        ]:

            ax.plot(
                var_table["Hari"],
                var_table[col],
                marker="o",
                linewidth=2.5,
                label=col,
            )

        ax.set_xlabel(
            "Horizon Risiko (Hari)"
        )

        ax.set_ylabel(
            "Nilai VaR (Rp)"
        )

        ax.set_xticks(
            var_table["Hari"]
        )

        ax.grid(
            alpha=0.25
        )

        ax.legend(
            frameon=False
        )

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)


# ============================================================
# RISK INTERPRETATION
# ============================================================

with st.container(border=True):

    st.markdown(
        "#### 💡 Interpretasi Risiko"
    )

    st.metric(
        "Estimasi VaR 95% · 1 Hari",
        format_rupiah(
            var95
        ),
    )

    st.metric(
        "Estimasi VaR 95% · 5 Hari",
        format_rupiah(
            var_table.iloc[-1][
                "VaR 95%"
            ]
        ),
    )

    st.write(
        "Estimasi VaR dihitung berdasarkan distribusi "
        "absolute error pada prediksi Stacking Ensemble "
        "dan diperluas menggunakan pendekatan "
        "square-root-of-time."
    )