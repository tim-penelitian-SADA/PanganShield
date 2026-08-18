import time
from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import loguniform, randint
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX

from utils import (
    clean_commodity_series,
    init_session_state,
    inject_custom_css,
    page_header,
    render_sidebar,
    require_dataset,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Input Parameter Model — KomoditasAI",
    page_icon="⚙️",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / input parameter",
    title=" Input Parameter Model",
    caption=(
        "Konfigurasi pembagian data, feature engineering, tuning model, "
        "dan parameter forecasting sebelum model dijalankan."
    ),
)


# ============================================================
# DATASET
# ============================================================

df, date_column, commodity_column = require_dataset()

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce",
    dayfirst=True,
)

df[commodity_column] = clean_commodity_series(
    df,
    commodity_column,
)

df = (
    df.dropna(subset=[date_column, commodity_column])
    .sort_values(date_column)
)

harga = (
    df.set_index(date_column)[commodity_column]
    .astype(float)
    .sort_index()
)

harga = harga[
    ~harga.index.duplicated(keep="last")
]


# ============================================================
# GLOBAL CONFIG
# ============================================================

RANDOM_STATE = 42
END_DATE = harga.index.max()

MIN_LAG = 1
MAX_LAG = 30

DEFAULT_TRAIN_END_DATE = pd.Timestamp("2025-12-31")

SVR_ITERATIONS_BY_PROFILE = {
    "Fast": 15,
    "Balanced": 28,
    "Thorough": 50,
}

RF_ITERATIONS_BY_PROFILE = {
    "Fast": 15,
    "Balanced": 24,
    "Thorough": 40,
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_time_series_features(
    series: pd.Series,
    max_lag: int,
) -> pd.DataFrame:
    """
    Membuat fitur time series tanpa data leakage.

    Seluruh rolling statistics menggunakan shift(1)
    sehingga hanya menggunakan informasi sebelum waktu t.
    """

    s = series.astype(float).copy()

    frame = pd.DataFrame(index=s.index)
    frame["target"] = s

    # -------------------------
    # Lag features
    # -------------------------

    for lag in range(1, max_lag + 1):
        frame[f"lag_{lag}"] = s.shift(lag)

    # -------------------------
    # Rolling features
    # -------------------------

    shifted = s.shift(1)

    for window in [3, 5, 7, 10, 14, 21, 30]:
        frame[f"roll_mean_{window}"] = (
            shifted.rolling(window).mean()
        )

        frame[f"roll_std_{window}"] = (
            shifted.rolling(window).std()
        )

        frame[f"roll_min_{window}"] = (
            shifted.rolling(window).min()
        )

        frame[f"roll_max_{window}"] = (
            shifted.rolling(window).max()
        )

    # -------------------------
    # Exponential moving average
    # -------------------------

    frame["ewm_mean_5"] = (
        shifted.ewm(span=5, adjust=False).mean()
    )

    frame["ewm_mean_14"] = (
        shifted.ewm(span=14, adjust=False).mean()
    )

    # -------------------------
    # Difference
    # -------------------------

    frame["diff_1"] = (
        s.shift(1) - s.shift(2)
    )

    frame["diff_5"] = (
        s.shift(1) - s.shift(6)
    )

    # -------------------------
    # Calendar features
    # -------------------------

    idx = pd.DatetimeIndex(frame.index)

    frame["day_of_week"] = idx.dayofweek
    frame["day_of_month"] = idx.day
    frame["month"] = idx.month
    frame["quarter"] = idx.quarter

    frame["day_of_year_sin"] = np.sin(
        2 * np.pi * idx.dayofyear / 365.25
    )

    frame["day_of_year_cos"] = np.cos(
        2 * np.pi * idx.dayofyear / 365.25
    )

    return (
        frame
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def make_svr_estimator(
    C=100.0,
    gamma="scale",
    epsilon=0.05,
):
    """Membuat estimator SVR dengan StandardScaler."""

    x_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=C,
                    gamma=gamma,
                    epsilon=epsilon,
                    cache_size=1000,
                ),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=x_pipeline,
        transformer=StandardScaler(),
    )


def valid_tscv(
    n_samples,
    requested_splits,
):
    """Menentukan jumlah fold TimeSeriesSplit yang valid."""

    n_splits = min(
        requested_splits,
        max(2, n_samples // 60),
    )

    n_splits = min(
        n_splits,
        n_samples - 1,
    )

    return TimeSeriesSplit(
        n_splits=n_splits
    )


def infer_seasonal_period(index):
    """Mengestimasi seasonal period berdasarkan pola weekday/weekend."""

    index = pd.DatetimeIndex(index)

    weekday_ratio = np.mean(
        index.dayofweek < 5
    )

    weekend_ratio = np.mean(
        index.dayofweek >= 5
    )

    if (
        weekday_ratio > 0.90
        and weekend_ratio < 0.10
    ):
        return 5

    return 7


def adf_test(series):
    """Augmented Dickey-Fuller test."""

    result = adfuller(
        pd.Series(series).dropna(),
        autolag="AIC",
    )

    return {
        "Statistik ADF": result[0],
        "ADF p-value": result[1],
        "ADF lag": result[2],
        "Kesimpulan ADF": (
            "Stasioner"
            if result[1] < 0.05
            else "Belum stasioner"
        ),
    }


def kpss_test(series):
    """KPSS test."""

    result = kpss(
        pd.Series(series).dropna(),
        regression="c",
        nlags="auto",
    )

    return {
        "Statistik KPSS": result[0],
        "KPSS p-value": result[1],
        "KPSS lag": result[2],
        "Kesimpulan KPSS": (
            "Stasioner"
            if result[1] > 0.05
            else "Belum stasioner"
        ),
    }


def sarima_grid_search(
    series,
    seasonal_period,
    d_value,
    profile="balanced",
):
    """
    Pemilihan parameter SARIMA berdasarkan AIC
    pada data train.
    """

    y = np.asarray(
        series,
        dtype=float,
    )

    if profile == "thorough":
        p_values = q_values = [0, 1, 2, 3]
        seasonal_values = [0, 1, 2]
        max_complexity = 7

    else:
        p_values = q_values = [0, 1, 2]
        seasonal_values = [0, 1]
        max_complexity = 5

    d_values = list(
        dict.fromkeys(
            [d_value, 0, 1]
        )
    )

    D_values = [0, 1]

    candidates = []

    for (
        p,
        d,
        q,
        P,
        D,
        Q,
    ) in product(
        p_values,
        d_values,
        q_values,
        seasonal_values,
        D_values,
        seasonal_values,
    ):

        if p + q + P + Q > max_complexity:
            continue

        if (
            p == q == P == Q == 0
            and d == D == 0
        ):
            continue

        candidates.append(
            (
                (p, d, q),
                (
                    P,
                    D,
                    Q,
                    seasonal_period,
                ),
            )
        )

    records = []

    best_result = None
    best_aic = np.inf
    best_spec = None

    for order, seasonal_order in candidates:

        try:

            model = SARIMAX(
                y,
                order=order,
                seasonal_order=seasonal_order,
                trend=(
                    "c"
                    if order[1] == 0
                    else "n"
                ),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            result = model.fit(
                disp=False,
                maxiter=150,
                method="lbfgs",
            )

            records.append(
                {
                    "order": order,
                    "seasonal_order": seasonal_order,
                    "AIC": result.aic,
                    "BIC": result.bic,
                    "Converged": bool(
                        result.mle_retvals.get(
                            "converged",
                            True,
                        )
                    ),
                }
            )

            if (
                np.isfinite(result.aic)
                and result.aic < best_aic
            ):
                best_aic = result.aic
                best_result = result
                best_spec = (
                    order,
                    seasonal_order,
                )

        except Exception:
            continue

    if len(records) == 0:
        raise RuntimeError(
            "Tidak ada kandidat SARIMA yang berhasil difit."
        )

    result_table = (
        pd.DataFrame(records)
        .sort_values("AIC")
    )

    if best_result is None:
        raise RuntimeError(
            "Seluruh kandidat SARIMA gagal. "
            "Coba tetapkan seasonal period = 5."
        )

    return (
        best_spec,
        best_result,
        result_table,
    )


# ============================================================
# SECTION 01
# DATASET AKTIF
# ============================================================

st.markdown("### 01 · Dataset Aktif")

st.caption(
    "Ringkasan dataset yang digunakan sebagai dasar "
    "proses pemodelan."
)

m1, m2, m3, m4 = st.columns(4)


with m1:
    st.metric(
        "Komoditas",
        commodity_column,
    )


with m2:
    st.metric(
        "Observasi",
        f"{len(harga):,}".replace(
            ",",
            ".",
        ),
    )


with m3:
    st.metric(
        "Tanggal Mulai",
        harga.index.min().strftime(
            "%d %b %Y"
        ),
    )


with m4:
    st.metric(
        "Tanggal Akhir",
        harga.index.max().strftime(
            "%d %b %Y"
        ),
    )


# ============================================================
# SECTION 02
# TRAIN TEST SPLIT
# ============================================================

st.markdown("### 02 · Pembagian Data Train & Test")

with st.container(border=True):

    split_col, summary_col = st.columns(
        [1.25, 1],
        gap="large",
    )

    with split_col:

        split_method = st.radio(
            "Metode Pembagian Data",
            [
                "Persentase",
                "Tanggal (Advanced)",
            ],
            horizontal=True,
        )

        if split_method == "Persentase":

            train_ratio = st.slider(
                "Proporsi Data Train (%)",
                min_value=50,
                max_value=95,
                value=80,
                step=5,
            )

            test_ratio = (
                100 - train_ratio
            )

            split_index = int(
                len(harga)
                * train_ratio
                / 100
            )

            train_series = (
                harga.iloc[:split_index]
                .copy()
            )

            test_series = (
                harga.iloc[split_index:]
                .copy()
            )

            split_note = (
                f"Data dibagi secara kronologis "
                f"{train_ratio}% train dan "
                f"{test_ratio}% test."
            )

        else:

            train_end_date = st.date_input(
                "Tanggal Akhir Data Train",
                value=min(
                    DEFAULT_TRAIN_END_DATE.date(),
                    harga.index.max().date(),
                ),
                min_value=harga.index.min().date(),
                max_value=harga.index.max().date(),
            )

            train_series = (
                harga.loc[
                    harga.index
                    <= pd.Timestamp(
                        train_end_date
                    )
                ]
                .copy()
            )

            test_series = (
                harga.loc[
                    harga.index
                    > pd.Timestamp(
                        train_end_date
                    )
                ]
                .copy()
            )

            total = len(harga)

            train_ratio = round(
                len(train_series)
                / total
                * 100,
                1,
            )

            test_ratio = round(
                len(test_series)
                / total
                * 100,
                1,
            )

            split_note = (
                "Data dibagi berdasarkan "
                "tanggal yang dipilih."
            )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if (
        len(train_series) == 0
        or len(test_series) == 0
    ):
        st.error(
            "Pembagian data menghasilkan train "
            "atau test kosong. Silakan ubah "
            "rasio atau tanggal."
        )
        st.stop()

    effective_train_end = (
        train_series.index.max()
    )

    effective_test_start = (
        test_series.index.min()
    )

    with summary_col:

        st.caption("RINGKASAN PEMBAGIAN")

        m1, m2 = st.columns(2)

        with m1:
            st.metric(
                "Train",
                f"{len(train_series):,}".replace(
                    ",",
                    ".",
                ),
            )

        with m2:
            st.metric(
                "Test",
                f"{len(test_series):,}".replace(
                    ",",
                    ".",
                ),
            )

        st.caption(
            f"Train: {effective_train_end.strftime('%d %b %Y')}"
        )

        st.caption(
            f"Test mulai: {effective_test_start.strftime('%d %b %Y')}"
        )


st.info(split_note)


# ------------------------------------------------------------
# SPLIT VISUALIZATION
# ------------------------------------------------------------

with st.expander(
    "📈 Lihat Visualisasi Pembagian Data",
    expanded=False,
):

    fig, ax = plt.subplots(
        figsize=(13, 4)
    )

    ax.plot(
        train_series.index,
        train_series,
        label="Train",
        linewidth=1.3,
    )

    ax.plot(
        test_series.index,
        test_series,
        label="Test",
        linewidth=1.3,
    )

    ax.axvline(
        effective_test_start,
        color="red",
        linestyle="--",
        linewidth=1.3,
    )

    ax.set_title(
        "Pembagian Data Train dan Test"
    )

    ax.set_xlabel("Tanggal")
    ax.set_ylabel(
        f"Harga {commodity_column}"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    st.pyplot(
        fig,
        use_container_width=True,
    )

    plt.close(fig)


st.write("")


# ============================================================
# SECTION 03
# FEATURE ENGINEERING
# ============================================================

st.markdown("### 03 · Feature Engineering")

with st.container(border=True):

    feature_col, cv_col = st.columns(
        2,
        gap="large",
    )

    with feature_col:

        st.markdown("#### Automatic Lag Selection")

        st.markdown(
            """
            Lag optimum akan dipilih otomatis
            menggunakan **SVR Baseline** dengan
            evaluasi **TimeSeriesSplit Cross Validation**.
            """
        )

        st.info(
            "Metode: SVR Baseline + CV RMSE"
        )

        st.caption(
            f"Rentang kandidat lag: "
            f"{MIN_LAG} sampai {MAX_LAG}"
        )

    with cv_col:

        st.markdown("#### Cross Validation")

        cv_splits = st.slider(
            "TimeSeriesSplit (Fold)",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
            key="cv_split",
        )

        effective_cv = min(
            cv_splits,
            max(
                2,
                len(train_series) // 60,
            ),
        )

        effective_cv = min(
            effective_cv,
            len(train_series) - 1,
        )

        st.metric(
            "Effective Fold",
            effective_cv,
        )

        st.caption(
            "Jumlah fold akan disesuaikan "
            "apabila jumlah data tidak mencukupi."
        )


st.write("")


# ============================================================
# SECTION 04
# SARIMA
# ============================================================

st.markdown("### 04 · Konfigurasi SARIMA")

recommended_seasonal = (
    infer_seasonal_period(
        train_series.index
    )
)

adf_result = adfuller(
    train_series.dropna(),
    autolag="AIC",
)

recommended_d = (
    0
    if adf_result[1] < 0.05
    else 1
)


sarima_col, diff_col = st.columns(
    2,
    gap="large",
)


# ------------------------------------------------------------
# SEASONAL PERIOD
# ------------------------------------------------------------

with sarima_col:

    with st.container(border=True):

        st.markdown(
            "#### 📅 Seasonal Period"
        )

        st.metric(
            "Suggested Seasonal Period",
            recommended_seasonal,
        )

        auto_season = st.toggle(
            "Gunakan nilai otomatis",
            value=True,
            key="auto_seasonal_period",
        )

        if auto_season:

            seasonal_period = (
                recommended_seasonal
            )

            st.number_input(
                "Seasonal Period",
                value=seasonal_period,
                disabled=True,
            )

        else:

            seasonal_period = (
                st.number_input(
                    "Seasonal Period",
                    min_value=2,
                    max_value=365,
                    value=recommended_seasonal,
                    step=1,
                )
            )

        st.caption(
            "Periodisitas musiman yang digunakan "
            "dalam pencarian parameter SARIMA."
        )


# ------------------------------------------------------------
# DIFFERENCING
# ------------------------------------------------------------

with diff_col:

    with st.container(border=True):

        st.markdown(
            "#### 🔄 Differencing"
        )

        st.metric(
            "ADF p-value",
            f"{adf_result[1]:.4f}",
        )

        if recommended_d == 0:

            st.caption(
                "Interpretasi: data sudah stasioner."
            )

        else:

            st.caption(
                "Interpretasi: data belum stasioner."
            )

        auto_d = st.toggle(
            "Gunakan nilai otomatis",
            value=True,
            key="auto_differencing",
        )

        if auto_d:

            d_value = recommended_d

            st.number_input(
                "Differencing (d)",
                value=d_value,
                disabled=True,
            )

        else:

            d_value = st.number_input(
                "Differencing (d)",
                min_value=0,
                max_value=2,
                value=recommended_d,
                step=1,
            )

        st.caption(
            "Nilai differencing yang digunakan "
            "dalam grid search SARIMA."
        )


st.write("")


# ============================================================
# SECTION 05
# MACHINE LEARNING
# ============================================================

st.markdown("### 05 · Machine Learning")

svr_col, rf_col = st.columns(
    2,
    gap="large",
)


# ------------------------------------------------------------
# SVR
# ------------------------------------------------------------

with svr_col:

    with st.container(border=True):

        st.markdown(
            "#### Support Vector Regression"
        )

        svr_profile = st.radio(
            "Search Profile",
            [
                "Fast",
                "Balanced",
                "Thorough",
            ],
            index=1,
            horizontal=True,
            key="svr_profile",
        )

        svr_iterations = (
            SVR_ITERATIONS_BY_PROFILE[
                svr_profile
            ]
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Search Iteration",
                svr_iterations,
            )

        with c2:

            st.metric(
                "CV Fold",
                effective_cv,
            )

        st.caption(
            "RandomizedSearchCV digunakan "
            "untuk mencari kombinasi parameter "
            "SVR terbaik."
        )


# ------------------------------------------------------------
# RANDOM FOREST
# ------------------------------------------------------------

with rf_col:

    with st.container(border=True):

        st.markdown(
            "#### Random Forest"
        )

        rf_profile = st.radio(
            "Search Profile",
            [
                "Fast",
                "Balanced",
                "Thorough",
            ],
            index=1,
            horizontal=True,
            key="rf_profile",
        )

        rf_iterations = (
            RF_ITERATIONS_BY_PROFILE[
                rf_profile
            ]
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(
                "Search Iteration",
                rf_iterations,
            )

        with c2:

            st.metric(
                "CV Fold",
                effective_cv,
            )

        st.caption(
            "RandomizedSearchCV digunakan "
            "untuk mencari kombinasi hyperparameter "
            "Random Forest terbaik."
        )


st.write("")
st.write("")


# ============================================================
# RUN MODEL
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin:10px 0 14px 0;
    ">
        <div style="
            font-size:13px;
            color:#747784;
            margin-bottom:8px;
        ">
            Semua konfigurasi sudah siap
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

jalankan = st.button(
    "▶  Jalankan Model",
    type="primary",
    use_container_width=True,
)


# ============================================================
# MODEL EXECUTION
# ============================================================

if jalankan:

    params = {
        "commodity_column": commodity_column,
        "date_column": date_column,
        "train_end": effective_train_end,
        "test_start": effective_test_start,
        "train_size": len(train_series),
        "test_size": len(test_series),
        "max_lag": MAX_LAG,
        "svr_iterations": svr_iterations,
        "rf_iterations": rf_iterations,
        "seasonal_period": seasonal_period,
        "d": d_value,
        "svr_profile": svr_profile,
        "rf_profile": rf_profile,
        "cv_splits": effective_cv,
        "random_state": RANDOM_STATE,
    }

    st.session_state.model_params = params

    # ========================================================
    # PREPARATION
    # ========================================================

    with st.spinner(
        "Mempersiapkan data dan menjalankan proses tuning model..."
    ):

        # ----------------------------------------------------
        # SVR LAG OPTIMIZATION
        # ----------------------------------------------------

        lag_results = []

        max_candidate = min(
            MAX_LAG,
            max(
                MIN_LAG,
                len(train_series) // 10,
            ),
        )

        for lag in range(
            MIN_LAG,
            max_candidate + 1,
        ):

            feature_data = (
                make_time_series_features(
                    train_series,
                    max_lag=lag,
                )
            )

            if len(feature_data) < 80:
                continue

            X_lag = (
                feature_data
                .drop(columns="target")
            )

            y_lag = (
                feature_data["target"]
            )

            cv = valid_tscv(
                len(X_lag),
                cv_splits,
            )

            scores = cross_val_score(
                make_svr_estimator(),
                X_lag,
                y_lag,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )

            lag_results.append(
                {
                    "Lag": lag,
                    "CV RMSE": -scores.mean(),
                    "CV RMSE Std": scores.std(),
                    "Jumlah fitur": X_lag.shape[1],
                    "Jumlah observasi": len(X_lag),
                }
            )

        lag_table = (
            pd.DataFrame(lag_results)
            .sort_values("CV RMSE")
        )

        if lag_table.empty:

            st.error(
                "Penentuan lag gagal karena "
                "data terlalu sedikit."
            )

            st.stop()

        optimal_lag = int(
            lag_table.iloc[0]["Lag"]
        )

        best_cv_rmse = float(
            lag_table.iloc[0]["CV RMSE"]
        )

        # ----------------------------------------------------
        # LAG PLOT
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(9, 4)
        )

        lag_plot = (
            lag_table
            .sort_values("Lag")
        )

        ax.plot(
            lag_plot["Lag"],
            lag_plot["CV RMSE"],
            marker="o",
        )

        ax.axvline(
            optimal_lag,
            linestyle="--",
            color="red",
            label=(
                f"Optimal Lag = "
                f"{optimal_lag}"
            ),
        )

        ax.set_xlabel("Lag")
        ax.set_ylabel("CV RMSE")
        ax.set_title(
            "Lag Optimization using TimeSeriesSplit"
        )

        ax.grid(alpha=0.25)
        ax.legend()

        # ----------------------------------------------------
        # STATIONARITY
        # ----------------------------------------------------

        stationarity_results = pd.DataFrame(
            [
                {
                    "Transformasi": "Level",
                    **adf_test(train_series),
                    **kpss_test(train_series),
                },
                {
                    "Transformasi": "First difference",
                    **adf_test(
                        train_series.diff()
                    ),
                    **kpss_test(
                        train_series.diff()
                    ),
                },
            ]
        )

        # ----------------------------------------------------
        # SUPERVISED DATA
        # ----------------------------------------------------

        supervised = (
            make_time_series_features(
                harga,
                max_lag=optimal_lag,
            )
        )

        X_all = (
            supervised
            .drop(columns="target")
        )

        y_all = (
            supervised["target"]
        )

        train_mask = (
            X_all.index
            <= effective_train_end
        )

        test_mask = (
            (X_all.index >= effective_test_start)
            & (X_all.index <= END_DATE)
        )

        X_train = (
            X_all.loc[train_mask]
            .copy()
        )

        y_train = (
            y_all.loc[train_mask]
            .copy()
        )

        X_test = (
            X_all.loc[test_mask]
            .copy()
        )

        y_test = (
            y_all.loc[test_mask]
            .copy()
        )

        if (
            len(X_train) < 100
            or len(X_test) < 10
        ):

            st.error(
                f"Data supervised tidak cukup. "
                f"Train={len(X_train)}, "
                f"Test={len(X_test)}"
            )

            st.stop()


    # ========================================================
    # SARIMA
    # ========================================================

    st.markdown(
        "### 🔎 Mencari Parameter SARIMA Terbaik"
    )

    with st.spinner(
        "Melakukan grid search SARIMA..."
    ):

        start = time.time()

        (
            best_sarima_spec,
            fitted_sarima_train,
            sarima_search_table,
        ) = sarima_grid_search(
            train_series,
            seasonal_period=seasonal_period,
            d_value=d_value,
            profile="balanced",
        )

        (
            best_sarima_order,
            best_sarima_seasonal_order,
        ) = best_sarima_spec

        elapsed = (
            time.time() - start
        ) / 60

    st.success(
        f"SARIMA selesai dalam "
        f"{elapsed:.2f} menit."
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Best Order",
        str(best_sarima_order),
    )

    c2.metric(
        "Seasonal Order",
        str(
            best_sarima_seasonal_order
        ),
    )

    with st.expander(
        "Lihat seluruh kandidat SARIMA"
    ):

        st.dataframe(
            sarima_search_table,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # SVR RANDOM SEARCH
    # ========================================================

    st.markdown(
        "### 🤖 Tuning Support Vector Regression"
    )

    svr_search_space = {
        "regressor__svr__C": loguniform(
            1e-1,
            2e3,
        ),
        "regressor__svr__gamma": loguniform(
            1e-5,
            1.0,
        ),
        "regressor__svr__epsilon": loguniform(
            1e-3,
            0.5,
        ),
    }

    svr_search = RandomizedSearchCV(
        estimator=make_svr_estimator(),
        param_distributions=svr_search_space,
        n_iter=svr_iterations,
        scoring="neg_root_mean_squared_error",
        cv=valid_tscv(
            len(X_train),
            effective_cv,
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    with st.spinner(
        "Melakukan tuning SVR..."
    ):

        start = time.time()

        svr_search.fit(
            X_train,
            y_train,
        )

        elapsed = (
            time.time() - start
        ) / 60

    best_svr = (
        svr_search.best_estimator_
    )

    st.success(
        f"Tuning SVR selesai "
        f"({elapsed:.2f} menit)"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Best CV RMSE",
        f"{-svr_search.best_score_:.3f}",
    )

    c2.metric(
        "Jumlah Iterasi",
        svr_iterations,
    )

    with st.expander(
        "Lihat Best Parameter SVR",
        expanded=False,
    ):

        best_svr_params = svr_search.best_params_

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "C",
                f"{best_svr_params['regressor__svr__C']:.4f}",
            )

        with c2:
            gamma_value = best_svr_params[
                "regressor__svr__gamma"
            ]

            if isinstance(gamma_value, str):
                gamma_display = gamma_value
            else:
                gamma_display = f"{gamma_value:.6f}"

            st.metric(
                "Gamma",
                gamma_display,
            )

        with c3:
            st.metric(
                "Epsilon",
                f"{best_svr_params['regressor__svr__epsilon']:.4f}",
            )


    # ========================================================
    # RANDOM FOREST
    # ========================================================

    st.markdown(
        "### 🌲 Tuning Random Forest"
    )

    rf_model = RandomForestRegressor(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_search_space = {
        "n_estimators": randint(
            300,
            1001,
        ),
        "max_depth": [
            None,
            5,
            8,
            12,
            16,
            24,
            32,
        ],
        "min_samples_split": randint(
            2,
            16,
        ),
        "min_samples_leaf": randint(
            1,
            10,
        ),
        "max_features": [
            "sqrt",
            "log2",
            0.5,
            0.75,
            1.0,
        ],
        "bootstrap": [True],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_search_space,
        n_iter=rf_iterations,
        scoring="neg_root_mean_squared_error",
        cv=valid_tscv(
            len(X_train),
            effective_cv,
        ),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    with st.spinner(
        "Melakukan tuning Random Forest..."
    ):

        start = time.time()

        rf_search.fit(
            X_train,
            y_train,
        )

        elapsed = (
            time.time() - start
        ) / 60

    best_rf = (
        rf_search.best_estimator_
    )

    st.success(
        f"Tuning Random Forest selesai "
        f"({elapsed:.2f} menit)"
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "Best CV RMSE",
        f"{-rf_search.best_score_:.3f}",
    )

    c2.metric(
        "Jumlah Iterasi",
        rf_iterations,
    )

    with st.expander(
        "Lihat Best Parameter Random Forest",
        expanded=False,
    ):


        best_rf_params = rf_search.best_params_

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "n_estimators",
                f"{best_rf_params['n_estimators']:,}".replace(
                    ",",
                    ".",
                ),
            )

        with c2:
            max_depth = best_rf_params["max_depth"]

            st.metric(
                "Max Depth",
                "Unlimited"
                if max_depth is None
                else str(max_depth),
            )

        with c3:
            st.metric(
                "Max Features",
                str(
                    best_rf_params["max_features"]
                ),
            )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Min Samples Split",
                str(
                    best_rf_params[
                        "min_samples_split"
                    ]
                ),
            )

        with c2:
            st.metric(
                "Min Samples Leaf",
                str(
                    best_rf_params[
                        "min_samples_leaf"
                    ]
                ),
            )

        with c3:
            st.metric(
                "CV RMSE",
                f"{-rf_search.best_score_:.3f}",
            )


    # ========================================================
    # SAVE MODEL RESULT
    # ========================================================

    st.session_state.model_result = {

        "best_svr": best_svr,

        "best_rf": best_rf,

        "best_sarima_order":
            best_sarima_order,

        "best_sarima_seasonal_order":
            best_sarima_seasonal_order,

        "fitted_sarima_train":
            fitted_sarima_train,

        "sarima_search_table":
            sarima_search_table,

        "svr_best_params":
            svr_search.best_params_,

        "svr_best_score":
            -svr_search.best_score_,

        "rf_best_params":
            rf_search.best_params_,

        "rf_best_score":
            -rf_search.best_score_,
    }


    # ========================================================
    # RESULT SUMMARY
    # ========================================================

    st.markdown(
        "### 🔎 Hasil Konfigurasi Otomatis"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Optimal Lag SVR",
        optimal_lag,
    )

    c2.metric(
        "SARIMA Seasonal Period",
        seasonal_period,
    )

    c3.metric(
        "Train",
        f"{len(X_train):,}".replace(
            ",",
            ".",
        ),
    )

    c4.metric(
        "Test",
        f"{len(X_test):,}".replace(
            ",",
            ".",
        ),
    )


    # --------------------------------------------------------
    # LAG OPTIMIZATION CHART
    # --------------------------------------------------------

    st.markdown(
        "#### 📈 Hasil Optimasi Lag"
    )

    chart_col1, chart_col2, chart_col3 = st.columns(
        [1, 2.5, 1]
    )

    with chart_col2:

        st.pyplot(
            fig,
            use_container_width=True,
        )

    plt.close(fig)


    # --------------------------------------------------------
    # STATIONARITY
    # --------------------------------------------------------

    with st.expander(
        "Lihat hasil uji stasioneritas"
    ):

        st.dataframe(
            stationarity_results,
            use_container_width=True,
            hide_index=True,
        )


    # --------------------------------------------------------
    # LAG TABLE
    # --------------------------------------------------------

    with st.expander(
        "Lihat kandidat lag SVR"
    ):

        st.dataframe(
            lag_table,
            use_container_width=True,
            hide_index=True,
        )


    # ========================================================
    # SAVE PREPROCESSING
    # ========================================================

    st.session_state.model_data = {

        "harga": harga,

        "train_series":
            train_series,

        "test_series":
            test_series,

        "X_train":
            X_train,

        "y_train":
            y_train,

        "X_test":
            X_test,

        "y_test":
            y_test,

        "optimal_lag":
            optimal_lag,

        "stationarity_results":
            stationarity_results,

        "seasonal_period":
            seasonal_period,

        "effective_train_end":
            effective_train_end,

        "effective_test_start":
            effective_test_start,

        "best_cv_rmse":
            best_cv_rmse,

        "best_svr":
            best_svr,

        "best_rf":
            best_rf,

        "best_sarima_order":
            best_sarima_order,

        "best_sarima_seasonal_order":
            best_sarima_seasonal_order,
    }