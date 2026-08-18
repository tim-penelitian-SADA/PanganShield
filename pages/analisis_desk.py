import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm
import streamlit as st

from scipy import stats
from statsmodels.stats.diagnostic import linear_rainbow, linear_reset

from utils import (
    clean_commodity_series,
    format_id,
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
    page_title="Analisis Deskriptif",
    page_icon="📊",
    layout="wide",
)

init_session_state()
inject_custom_css()
render_sidebar()

page_header(
    breadcrumb="app / analisis deskriptif",
    title="Analisis Deskriptif",
    caption=(
        "Ringkasan statistik dan visualisasi pola data harga "
        "sebelum pemodelan dilakukan."
    ),
)


# ============================================================
# CONSTANT
# ============================================================

RANDOM_STATE = 42


# ============================================================
# VALIDASI & PEMBERSIHAN DATA
# ============================================================

df, date_column, commodity_column = require_dataset()

working_df = df[[date_column, commodity_column]].copy()

working_df[date_column] = pd.to_datetime(
    working_df[date_column],
    dayfirst=True,
    errors="coerce",
)

working_df = (
    working_df
    .dropna(subset=[date_column])
    .sort_values(date_column)
)

st.session_state.df = working_df

df[commodity_column] = clean_commodity_series(
    df,
    commodity_column,
)

df = df.dropna(subset=[commodity_column])

harga = df[commodity_column]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normality_tests(
    series: pd.Series,
    label: str,
):
    """
    Menjalankan Shapiro-Wilk, Jarque-Bera,
    dan D'Agostino K².
    """

    x = (
        pd.Series(series)
        .dropna()
        .astype(float)
    )

    if len(x) > 5000:
        sample = x.sample(
            5000,
            random_state=RANDOM_STATE,
        )
        note = (
            "Shapiro menggunakan sampel acak "
            "5.000 observasi."
        )
    else:
        sample = x
        note = (
            "Shapiro menggunakan seluruh observasi."
        )

    shapiro_stat, shapiro_p = stats.shapiro(sample)

    jb = stats.jarque_bera(x)

    if len(x) >= 8:

        dagostino = stats.normaltest(x)

        dagostino_stat = dagostino.statistic
        dagostino_p = dagostino.pvalue

    else:

        dagostino_stat = np.nan
        dagostino_p = np.nan

    return {
        "Variabel": label,
        "N": len(x),
        "Shapiro": shapiro_stat,
        "Shapiro p-value": shapiro_p,
        "Jarque-Bera": jb.statistic,
        "JB p-value": jb.pvalue,
        "D'Agostino": dagostino_stat,
        "D'Agostino p-value": dagostino_p,
        "Kesimpulan": (
            "Normal"
            if shapiro_p > 0.05
            else "Tidak Normal"
        ),
        "Catatan": note,
    }


def build_lag_regression_data(
    series,
    n_lags=5,
):
    """
    Membentuk dataset regresi menggunakan
    lag harga.
    """

    frame = pd.DataFrame({
        "y": series
    })

    for lag in range(1, n_lags + 1):
        frame[f"lag_{lag}"] = series.shift(lag)

    return frame.dropna()


def format_rupiah(value):
    """
    Format angka menjadi format Rupiah.
    """

    return f"Rp {format_id(value, 0)}"


def conclusion_badge(
    title,
    value,
    description,
    status="neutral",
):
    """
    Small custom status card.
    """

    if status == "good":
        icon = "●"
        background = "#F2F8F4"
        border = "#D8EBDD"
    elif status == "warning":
        icon = "●"
        background = "#FFF8ED"
        border = "#F2DFC1"
    else:
        icon = "●"
        background = "#F6F7FA"
        border = "#E4E6EB"

    st.html(
        f"""
        <div style="
            background:{background};
            border:1px solid {border};
            border-radius:12px;
            padding:14px 16px;
            margin-top:8px;
        ">

            <div style="
                font-size:11px;
                color:#777B88;
                margin-bottom:5px;
            ">
                {title}
            </div>

            <div style="
                font-size:16px;
                font-weight:700;
                color:#31333F;
                margin-bottom:3px;
            ">
                <span style="font-size:10px;">
                    {icon}
                </span>
                {value}
            </div>

            <div style="
                font-size:11px;
                line-height:1.5;
                color:#747784;
            ">
                {description}
            </div>

        </div>
        """
    )


# ============================================================
# METRIK RINGKAS
# ============================================================

mean_val = harga.mean()
std_val = harga.std()
skew_val = stats.skew(harga)
kurt_val = stats.kurtosis(
    harga,
    fisher=False,
)


# ============================================================
# RATA-RATA PERUBAHAN BULANAN
# ============================================================

plot_df = (
    df
    .sort_values(date_column)
    .copy()
)

monthly = (
    plot_df
    .set_index(date_column)[commodity_column]
    .resample("MS")
    .mean()
)

monthly_change = (
    monthly
    .pct_change()
    .mean()
    * 100
)

if pd.isna(monthly_change):

    mean_delta = "-"

elif monthly_change > 0:

    mean_delta = (
        f"▲ {monthly_change:.2f}% / bulan"
    )

elif monthly_change < 0:

    mean_delta = (
        f"▼ {abs(monthly_change):.2f}% / bulan"
    )

else:

    mean_delta = "Tidak berubah"


# ============================================================
# VOLATILITAS
# ============================================================

cv = std_val / mean_val

if cv < 0.10:

    volatility_delta = "▼ Rendah"
    volatility_color = "inverse"

elif cv < 0.20:

    volatility_delta = "■ Sedang"
    volatility_color = "off"

else:

    volatility_delta = "▲ Tinggi"
    volatility_color = "normal"


# ============================================================
# SKEWNESS
# ============================================================

if abs(skew_val) < 0.50:

    skew_delta = "● Simetris"
    skew_color = "normal"

elif skew_val > 0:

    skew_delta = "▶ Miring ke kanan"
    skew_color = "inverse"

else:

    skew_delta = "◀ Miring ke kiri"
    skew_color = "off"


# ============================================================
# KURTOSIS
# ============================================================

if kurt_val < 3:

    kurt_delta = "▼ Platykurtic"
    kurt_color = "inverse"

elif kurt_val <= 3.5:

    kurt_delta = "● Mesokurtic"
    kurt_color = "off"

else:

    kurt_delta = "▲ Leptokurtic"
    kurt_color = "normal"


# ============================================================
# SECTION: RINGKASAN
# ============================================================

st.markdown("### Ringkasan Data")

st.caption(
    "Indikator utama yang menggambarkan karakteristik "
    "distribusi dan perubahan harga komoditas."
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Rata-rata Harga",
        format_rupiah(mean_val),
        mean_delta,
        delta_color="off",
    )

with m2:
    st.metric(
        "Volatilitas",
        format_id(std_val, 1),
        volatility_delta,
        delta_color=volatility_color,
    )

with m3:
    st.metric(
        "Skewness",
        f"{skew_val:.2f}",
        skew_delta,
        delta_color=skew_color,
    )

with m4:
    st.metric(
        "Kurtosis",
        f"{kurt_val:.2f}",
        kurt_delta,
        delta_color=kurt_color,
    )


# ============================================================
# SECTION: TREN HARGA
# ============================================================

st.markdown("### 📈 Tren Harga Historis")

st.caption(
    "Pergerakan harga berdasarkan seluruh periode "
    "data yang tersedia."
)

with st.container(border=True):

    plot_df = (
        df
        .sort_values(date_column)
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=plot_df[date_column],
            y=plot_df[commodity_column],
            mode="lines",
            name=commodity_column,
            line=dict(
                color="#FF4B4B",
                width=1.8,
            ),
            fill="tozeroy",
            fillcolor="rgba(255,75,75,0.07)",
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Harga: Rp %{y:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        margin=dict(
            l=10,
            r=10,
            t=10,
            b=10,
        ),
        height=340,
        yaxis_title="Rp/kg",
        xaxis_title=None,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#E5E6EA",
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#F0F1F4",
        zeroline=False,
        linecolor="#E5E6EA",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


# ============================================================
# SECTION: DISTRIBUSI HARGA
# ============================================================

st.markdown("### 📊 Distribusi Harga")

st.caption(
    "Statistik deskriptif dan bentuk distribusi "
    "harga komoditas."
)

col1, col2 = st.columns(
    2,
    gap="medium",
)


# ============================================================
# STATISTIK DESKRIPTIF
# ============================================================

with col1:

    with st.container(border=True):

        st.markdown(
            "#### Tabel Statistik Deskriptif"
        )

        stat_table = pd.DataFrame(
            {
                "Statistik": [
                    "Mean",
                    "Median",
                    "Std. Deviasi",
                    "Minimum",
                    "Maksimum",
                    "Skewness",
                    "Kurtosis",
                ],
                "Nilai": [
                    f"{mean_val:,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{harga.median():,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{std_val:,.1f}".replace(
                        ",",
                        ".",
                    ),
                    f"{harga.min():,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{harga.max():,.0f}".replace(
                        ",",
                        ".",
                    ),
                    f"{skew_val:.2f}",
                    f"{kurt_val:.2f}",
                ],
            }
        )

        st.dataframe(
            stat_table,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# HISTOGRAM
# ============================================================

with col2:

    with st.container(border=True):

        st.markdown(
            "#### Histogram Harga"
        )

        fig_hist = go.Figure()

        fig_hist.add_trace(
            go.Histogram(
                x=df[commodity_column],
                nbinsx=8,
                marker_color="#FF7A6E",
                marker_line_color="#FFFFFF",
                marker_line_width=1,
            )
        )

        fig_hist.update_layout(
            margin=dict(
                l=10,
                r=10,
                t=10,
                b=10,
            ),
            height=280,
            xaxis_title="Rp/kg",
            yaxis_title="Frekuensi",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
        )

        fig_hist.update_xaxes(
            showgrid=False,
            linecolor="#E5E6EA",
        )

        fig_hist.update_yaxes(
            showgrid=True,
            gridcolor="#F0F1F4",
            zeroline=False,
            linecolor="#E5E6EA",
        )

        st.plotly_chart(
            fig_hist,
            use_container_width=True,
        )


# ============================================================
# SECTION: UJI NORMALITAS
# ============================================================

st.markdown("### 📋 Uji Normalitas")

st.caption(
    "Pengujian dilakukan pada level harga dan "
    "first difference untuk melihat karakteristik distribusi data."
)

hasil_normalitas = pd.DataFrame(
    [
        normality_tests(
            harga,
            "Level Harga",
        ),
        normality_tests(
            harga.diff(),
            "First Difference",
        ),
    ]
)


with st.container(border=True):

    st.markdown(
        "#### Hasil Pengujian"
    )

    st.dataframe(
        hasil_normalitas,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    level_result = hasil_normalitas.iloc[0]
    diff_result = hasil_normalitas.iloc[1]

    level_status = (
        "good"
        if level_result["Kesimpulan"] == "Normal"
        else "warning"
    )

    diff_status = (
        "good"
        if diff_result["Kesimpulan"] == "Normal"
        else "warning"
    )

    c1, c2 = st.columns(2)

    with c1:

        conclusion_badge(
            "Level Harga",
            level_result["Kesimpulan"],
            (
                "Berdasarkan Shapiro-Wilk dengan "
                f"p-value {level_result['Shapiro p-value']:.4f}."
            ),
            level_status,
        )

    with c2:

        conclusion_badge(
            "First Difference",
            diff_result["Kesimpulan"],
            (
                "Berdasarkan Shapiro-Wilk dengan "
                f"p-value {diff_result['Shapiro p-value']:.4f}."
            ),
            diff_status,
        )

    # ========================================================
    # Q-Q PLOTS
    # ========================================================

    st.markdown(
        "#### Q-Q Plot"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.caption(
            "Level Harga"
        )

        fig_qq_1, ax1 = plt.subplots(
            figsize=(5, 4),
        )

        sm.qqplot(
            harga.dropna(),
            line="45",
            fit=True,
            ax=ax1,
        )

        ax1.set_title(
            "Q-Q Plot Level Harga",
            fontsize=11,
        )

        ax1.grid(
            alpha=0.20,
        )

        plt.tight_layout()

        st.pyplot(
            fig_qq_1,
            use_container_width=True,
        )

        plt.close(fig_qq_1)

    with col2:

        st.caption(
            "First Difference"
        )

        fig_qq_2, ax2 = plt.subplots(
            figsize=(5, 4),
        )

        sm.qqplot(
            harga.diff().dropna(),
            line="45",
            fit=True,
            ax=ax2,
        )

        ax2.set_title(
            "Q-Q Plot First Difference",
            fontsize=11,
        )

        ax2.grid(
            alpha=0.20,
        )

        plt.tight_layout()

        st.pyplot(
            fig_qq_2,
            use_container_width=True,
        )

        plt.close(fig_qq_2)


# ============================================================
# SECTION: UJI LINEARITAS
# ============================================================

st.markdown("### 📈 Uji Linearitas")

st.caption(
    "Pengujian hubungan linear dilakukan menggunakan "
    "Rainbow Test dan Ramsey RESET."
)


# ============================================================
# LAG REGRESSION DATA
# ============================================================

n_lags = min(
    5,
    max(
        1,
        len(harga) // 50,
    ),
)

lin_data = build_lag_regression_data(
    harga,
    n_lags=n_lags,
)

X = sm.add_constant(
    lin_data.drop(
        columns="y"
    )
)

y = lin_data["y"]

model = sm.OLS(
    y,
    X,
).fit()


# ============================================================
# LINEARITY TEST
# ============================================================

rainbow_stat, rainbow_p = linear_rainbow(
    model
)

reset = linear_reset(
    model,
    power=2,
    use_f=True,
)

hasil_linearitas = pd.DataFrame(
    {
        "Uji": [
            "Rainbow Test",
            "Ramsey RESET",
        ],
        "Statistik": [
            rainbow_stat,
            float(reset.fvalue),
        ],
        "p-value": [
            rainbow_p,
            float(reset.pvalue),
        ],
        "Keputusan": [
            (
                "Linear"
                if rainbow_p > 0.05
                else "Tidak Linear"
            ),
            (
                "Linear"
                if float(reset.pvalue) > 0.05
                else "Tidak Linear"
            ),
        ],
    }
)


with st.container(border=True):

    # ========================================================
    # TEST RESULT
    # ========================================================

    st.markdown(
        "#### Hasil Pengujian"
    )

    st.dataframe(
        hasil_linearitas,
        use_container_width=True,
        hide_index=True,
    )

    # ========================================================
    # TEST STATUS
    # ========================================================

    rainbow_decision = (
        "Linear"
        if rainbow_p > 0.05
        else "Tidak Linear"
    )

    reset_decision = (
        "Linear"
        if float(reset.pvalue) > 0.05
        else "Tidak Linear"
    )

    c1, c2 = st.columns(2)

    with c1:

        conclusion_badge(
            "Rainbow Test",
            rainbow_decision,
            (
                f"p-value = {rainbow_p:.4f}. "
                "H0 tidak ditolak jika p-value > 0,05."
            ),
            (
                "good"
                if rainbow_p > 0.05
                else "warning"
            ),
        )

    with c2:

        conclusion_badge(
            "Ramsey RESET",
            reset_decision,
            (
                f"p-value = {float(reset.pvalue):.4f}. "
                "H0 tidak ditolak jika p-value > 0,05."
            ),
            (
                "good"
                if float(reset.pvalue) > 0.05
                else "warning"
            ),
        )

# ========================================================
# SCATTER PLOT
# ========================================================

st.markdown(
    "#### Scatter Plot Lag-1"
)

st.caption(
    "Hubungan antara harga pada periode t-1 "
    "dan harga pada periode t."
)

col1, col2, col3 = st.columns([1, 8, 1])

with col2:
    fig_scatter, ax = plt.subplots(
        figsize=(9, 5),
        dpi=120,
    )

    x = lin_data["lag_1"]
    y = lin_data["y"]

    ax.scatter(
        x,
        y,
        alpha=0.45,
    )

    coef = np.polyfit(x, y, 1)

    xx = np.linspace(
        x.min(),
        x.max(),
        200,
    )

    yy = np.polyval(
        coef,
        xx,
    )

    ax.plot(
        xx,
        yy,
        linewidth=2,
    )

    ax.set_xlabel("Harga t-1")
    ax.set_ylabel("Harga t")
    ax.grid(alpha=0.20)

    fig_scatter.tight_layout()

    st.pyplot(fig_scatter)

    plt.close(fig_scatter)


# ============================================================
# OLS MODEL SUMMARY
# ============================================================

with st.expander(
    "📋 Ringkasan Model OLS",
    expanded=False,
):

    st.caption(
        "Ringkasan model regresi OLS yang digunakan "
        "sebagai dasar pengujian linearitas hubungan "
        "antara harga saat ini dan nilai lag."
    )

    # ========================================================
    # MODEL FIT
    # ========================================================

    st.markdown("#### Model Fit")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "R-squared",
            f"{model.rsquared:.3f}",
        )

    with c2:
        st.metric(
            "Adjusted R-squared",
            f"{model.rsquared_adj:.3f}",
        )

    with c3:
        st.metric(
            "F-statistic",
            f"{model.fvalue:,.2f}".replace(
                ",",
                ".",
            ),
        )

    with c4:
        st.metric(
            "Prob (F-statistic)",
            (
                "< 0.001"
                if model.f_pvalue < 0.001
                else f"{model.f_pvalue:.4f}"
            ),
        )


    st.write("")


    # ========================================================
    # KOEFISIEN MODEL
    # ========================================================

    st.markdown("#### Koefisien Model")

    coef_table = pd.DataFrame(
        {
            "Variabel": model.params.index,
            "Koefisien": model.params.values,
            "Std. Error": model.bse.values,
            "t-statistic": model.tvalues.values,
            "p-value": model.pvalues.values,
            "CI Lower": model.conf_int()[0].values,
            "CI Upper": model.conf_int()[1].values,
        }
    )

    # Format tampilan
    coef_display = coef_table.copy()

    coef_display["Koefisien"] = (
        coef_display["Koefisien"]
        .map(lambda x: f"{x:,.4f}")
    )

    coef_display["Std. Error"] = (
        coef_display["Std. Error"]
        .map(lambda x: f"{x:,.4f}")
    )

    coef_display["t-statistic"] = (
        coef_display["t-statistic"]
        .map(lambda x: f"{x:,.3f}")
    )

    coef_display["p-value"] = (
        coef_display["p-value"]
        .map(
            lambda x:
            "< 0.001"
            if x < 0.001
            else f"{x:.4f}"
        )
    )

    coef_display["CI Lower"] = (
        coef_display["CI Lower"]
        .map(lambda x: f"{x:,.4f}")
    )

    coef_display["CI Upper"] = (
        coef_display["CI Upper"]
        .map(lambda x: f"{x:,.4f}")
    )

    st.dataframe(
        coef_display,
        use_container_width=True,
        hide_index=True,
    )


    st.write("")


    # ========================================================
    # DIAGNOSTIC
    # ========================================================

    st.markdown("#### Diagnostic Model")

    # Ambil diagnostic dari statsmodels
    omnibus = sm.stats.omni_normtest(
        model.resid
    )

    jb = stats.jarque_bera(
        model.resid
    )

    durbin_watson = sm.stats.stattools.durbin_watson(
        model.resid
    )

    condition_number = model.condition_number

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Durbin-Watson",
            f"{durbin_watson:.3f}",
        )

    with c2:
        st.metric(
            "Jarque-Bera",
            f"{jb.statistic:,.2f}".replace(
                ",",
                ".",
            ),
        )

    with c3:
        st.metric(
            "JB p-value",
            (
                "< 0.001"
                if jb.pvalue < 0.001
                else f"{jb.pvalue:.4f}"
            ),
        )

    with c4:
        st.metric(
            "Condition Number",
            f"{condition_number:,.0f}".replace(
                ",",
                ".",
            ),
        )


    st.write("")


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.markdown("#### Interpretasi Singkat")

    if model.rsquared >= 0.80:
        fit_status = "good"
        fit_text = (
            f"Model memiliki R² sebesar "
            f"{model.rsquared:.3f}, sehingga sebagian besar "
            "variasi harga dapat dijelaskan oleh variabel "
            "lag dalam model."
        )
    elif model.rsquared >= 0.50:
        fit_status = "neutral"
        fit_text = (
            f"Model memiliki R² sebesar "
            f"{model.rsquared:.3f}, menunjukkan kemampuan "
            "penjelasan model berada pada tingkat sedang."
        )
    else:
        fit_status = "warning"
        fit_text = (
            f"Model memiliki R² sebesar "
            f"{model.rsquared:.3f}, sehingga kemampuan "
            "model dalam menjelaskan variasi harga relatif terbatas."
        )

    conclusion_badge(
        "Model Fit",
        f"R² = {model.rsquared:.3f}",
        fit_text,
        fit_status,
    )