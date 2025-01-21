import pandas as pd
from orbit.models import DLT, LGT, ETS, KTR
from prophet import Prophet
import torch
from chronos import ChronosPipeline
from keras.models import Sequential
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import acf
from sklearn.model_selection import TimeSeriesSplit
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import os
import plotly.graph_objects as go
import streamlit as st
import base64
import io
from datetime import datetime, timedelta
import optuna
from pytorch_forecasting import NBeats, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
from lightning.pytorch import Trainer

st.set_page_config(page_title="Прогноз параметров скважины", layout="wide")

# Заголовок страницы
st.title("Прогноз параметров скважины")

# Боковая панель с элементами управления
st.sidebar.header("Настройки")

# Для Chronos
device_map_chronos = "cpu"  # Установите "cuda" для GPU или "cpu" для CPU
chronos_model_name = "amazon/chronos-t5-tiny"

# Для LSTM
lstm_epochs = 10
lstm_batch_size = 16
lstm_n_steps = 2  # Количество подпоследовательностей
lstm_n_length = 7  # Длина каждой подпоследовательности

# Для TFT
tft_learning_rate = 0.03
tft_hidden_size = 16
tft_attention_head_size = 2
tft_dropout = 0.1
tft_hidden_continuous_size = 8
tft_max_epochs = 10

# Для N-BEATS
nbeats_learning_rate = 1e-3
nbeats_log_interval = 10
nbeats_log_val_interval = 1
nbeats_weight_decay = 1e-2
nbeats_widths = [32, 512]
nbeats_backcast_loss_ratio = 1.0
nbeats_max_epochs = 10

# Глобальные переменные для Orbit
estimator_dlt = 'stan-mcmc'
global_trend_option_dlt = 'linear'
n_bootstrap_draws_dlt = 500
regression_penalty_dlt = 'fixed_ridge'
estimator_lgt = 'stan-mcmc'
n_bootstrap_draws_lgt = 500
regression_penalty_lgt = 'fixed_ridge'
estimator_ets = 'stan-mcmc'
n_bootstrap_draws_ets = 500
estimator_ktr = 'pyro-svi'
n_bootstrap_draws_ktr = 500
num_steps_ktr = 200

# Для Prophet
seasonality_mode = 'additive'  # additive или multiplicative
yearly_seasonality = True
weekly_seasonality = True

# Хранилище истории прогнозов
if 'history' not in st.session_state: 
    st.session_state.history = []

def synchronize_forecast_and_test(forecast_values, test_values):
    """Привести прогноз и тестовую выборку к одинаковым размерам"""
    if len(forecast_values) < len(test_values):
        test_values = test_values.iloc[:len(forecast_values)]
    elif len(forecast_values) > len(test_values):
        forecast_values = forecast_values[:len(test_values)]
    return forecast_values, test_values

# Добавляем данные в историю
def add_to_history(well_name, horizon, forecast_combined,
                   forecast_test, forecast,
                   model_name, rmse, mae, mape, fig_original, 
                   target_parameter, estimator_dlt, estimator_lgt, estimator_ets, estimator_ktr, global_trend_option_dlt, 
                   n_bootstrap_draws_dlt, n_bootstrap_draws_lgt, n_bootstrap_draws_ets, n_bootstrap_draws_ktr, 
                   regression_penalty_dlt, regression_penalty_lgt, num_steps_ktr, fill_method, seasonality_mode, 
                   chronos_model_name, device_map_chronos, lstm_epochs, lstm_batch_size, lstm_n_steps, lstm_n_length,
                   tft_learning_rate, tft_hidden_size, tft_attention_head_size, tft_dropout, tft_hidden_continuous_size, 
                   tft_max_epochs, nbeats_learning_rate, nbeats_log_interval, nbeats_log_val_interval, nbeats_weight_decay, 
                   nbeats_widths, nbeats_backcast_loss_ratio, nbeats_max_epochs):
    st.session_state.history.append({
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "well_name": well_name,
        "horizon": horizon,
        "forecast_combined": forecast_combined,
        "forecast_test": forecast_test,
        "forecast": forecast,
        "model_name": model_name,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "fig_original": fig_original,
        "target_parameter": target_parameter,
        "estimator_dlt": estimator_dlt,
        "estimator_lgt": estimator_lgt,
        "estimator_ets": estimator_ets,
        "estimator_ktr": estimator_ktr,
        "global_trend_option_dlt": global_trend_option_dlt,
        "n_bootstrap_draws_dlt": n_bootstrap_draws_dlt,
        "n_bootstrap_draws_lgt": n_bootstrap_draws_lgt,
        "n_bootstrap_draws_ets": n_bootstrap_draws_ets,
        "n_bootstrap_draws_ktr": n_bootstrap_draws_ktr,
        "regression_penalty_dlt": regression_penalty_dlt,
        "regression_penalty_lgt": regression_penalty_lgt,
        "num_steps_ktr": num_steps_ktr,
        "fill_method": fill_method,
        "seasonality_mode": seasonality_mode,
        "chronos_model_name": chronos_model_name,
        "device_map_chronos": device_map_chronos,
        "tft_learning_rate": tft_learning_rate,
        "tft_hidden_size": tft_hidden_size,
        "tft_attention_head_size": tft_attention_head_size,
        "tft_dropout": tft_dropout,
        "tft_hidden_continuous_size": tft_hidden_continuous_size,
        "tft_max_epochs": tft_max_epochs,
        "nbeats_learning_rate": nbeats_learning_rate,
        "nbeats_log_interval": nbeats_log_interval,
        "nbeats_log_val_interval": nbeats_log_val_interval,
        "nbeats_weight_decay": nbeats_weight_decay,
        "nbeats_widths": nbeats_widths,
        "nbeats_backcast_loss_ratio": nbeats_backcast_loss_ratio,
        "nbeats_max_epochs": nbeats_max_epochs
    })

     
# Загрузка файла
uploaded_file = st.sidebar.file_uploader("Анализируемый файл", type=["xlsx"], help="Выберите файл с данными в формате xlsx.")

# Инструкции для пользователя
st.markdown("""
    ## Инструкции
    1. Загрузите файл с данными в формате xlsx.
    2. Выберите скважину(ы) из списка.
    3. Выберите параметр для прогнозирования.
    4. Введите количество дней прогноза.
    5. Выберите модель для прогнозирования.
    6. Нажмите кнопку "Запустить прогноз".
    7. Посмотрите результаты прогноза и график.
    8. Нажмите кнопку "Перезапустить" для запуска нового прогноза.
    9. Нажмите кнопку "Просмотреть историю прогнозов" для просмотра прошлых прогнозов.
    10. Нажмите кнопку "Анализ пропущенных данных" для анализа пропущенных значений в выбранном параметре.

    **Рекомендуется использовать модель DLT и интерполяцию.**\n
    **Спасибо)**
    """)

# Добавление состояния приложения
if "data_loaded" not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.well_names = []
    st.session_state.available_parameters = []

if uploaded_file is not None:

    # Сохранение файла
    file_path = os.path.join("uploaded_files", uploaded_file.name)
    os.makedirs("uploaded_files", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Определяем названия листов
    xls = pd.ExcelFile(file_path)
    data_sheets = xls.sheet_names    
   
    # Считываем названия скважин (со второго листа, т.к. первый - информационный)
    df_wells = pd.read_excel(file_path, sheet_name=data_sheets[0], header=None)
    st.session_state.well_names = df_wells.iloc[0, 1:].tolist()

    # Считываем доступные параметры (названия листов)
    st.session_state.available_parameters = data_sheets
    st.session_state.data_loaded = True

# Вывод предупреждения при отсутствии данных
if not st.session_state.data_loaded:
    st.warning("Загрузите файл с данными в формате xlsx.")
    st.stop()

# Интерфейс
st.sidebar.header("Выбор скважин")
selected_wells = st.sidebar.multiselect("Выберите скважину(ы)", st.session_state.well_names, help="Выберите скважину(ы) для прогнозирования.")

st.sidebar.header("Выбор параметра")
target_parameter = st.sidebar.selectbox("Параметр для прогнозирования:", st.session_state.available_parameters, help="Выберите параметр скважины для прогнозирования.")

parameter_change = st.sidebar.selectbox("Параметр, который вы хотите изменить:", st.session_state.available_parameters, help="Выберите параметр скважины для изменения.")
value_change = st.sidebar.number_input("На какую величину?", value=25, help="Укажите в абсолютном значении разницу между текущей и желаемой величиной на конец прогноза")

horizon = st.sidebar.number_input("Количество дней прогноза:", min_value=1, value=30, help="Укажите количество дней для прогноза.")

st.sidebar.header("Выбор модели")
model_options = ["DLT", "LGT", "ETS", "KTR", "Prophet", "Chronos", "LSTM", "TFT", "N-BEATS"]
selected_model = st.sidebar.selectbox("Модель:", model_options, help="Выберите модель для прогнозирования.")

if st.sidebar.button("Описание моделей"):
    st.sidebar.markdown("**DLT (Dynamic Linear Trend):** Эта модель подходит для прогнозирования временных рядов с трендом, сезонностью и автокорреляцией. Учитывает влияние регрессоров.")
    st.sidebar.markdown("**LGT (Local Linear Trend):** Модель для прогнозирования временных рядов с локальным трендом.")
    st.sidebar.markdown("**ETS (Exponential Smoothing):** Модель экспоненциального сглаживания для временных рядов.")
    st.sidebar.markdown("**KTR (Kalman Trend Regression):** Регрессионная модель с калмановским фильтром для учёта трендов и сезонности.")
    st.sidebar.markdown("**Prophet:** Модель от Meta (Facebook) для прогнозирования временных рядов с трендами, сезонностью и праздниками.")
    st.sidebar.markdown("**Chronos:** Модель от Amazon, использующая предварительно обученные трансформеры для прогнозирования временных рядов.")
    st.sidebar.markdown("**LSTM (Long Short-Term Memory):** Нейронная сеть с долговременной и краткосрочной памятью для прогнозирования временных рядов.")
    st.sidebar.markdown("**TFT (Temporal Fusion Transformer):** Подходит для временных рядов с множественными источниками данных. Поддерживает регрессоры и сезонность.")
    st.sidebar.markdown("**N-BEATS:** Нейронная сеть для прогнозирования временных рядов. Подходит для широкого спектра временных рядов.")


# Функция для создания фрейма данных по выбранной скважине
def create_well_dataframe(well_name, data_sheets, file):
    dfs = {}
    for sheet_name in data_sheets:
        df = pd.read_excel(file, sheet_name=sheet_name, header=None)
        df.columns = df.iloc[0]
        df = df[1:]
        df.index = pd.to_datetime(df.iloc[:, 0])
        dfs[sheet_name] = df[well_name]
    df_combined = pd.DataFrame(dfs)
    df_combined['ds'] = df_combined.index
    return df_combined

# Элементы боковой панели для выбора гиперпараметров
# Настройка модели DLT
if selected_model == "DLT":
    if 'estimator_dlt' not in st.session_state:
        st.session_state.estimator_dlt = 'stan-mcmc'
    estimator_dlt = st.sidebar.selectbox('Estimator DLT', ['stan-map', 'stan-mcmc'], index=list(['stan-map', 'stan-mcmc']).index(st.session_state.estimator_dlt))
    if 'global_trend_option_dlt' not in st.session_state:
        st.session_state.global_trend_option_dlt = 'linear'
    global_trend_option_dlt = st.sidebar.selectbox('Global Trend Option DLT', ['flat', 'linear', 'loglinear', 'logistic'], index=list(['flat', 'linear', 'loglinear', 'logistic']).index(st.session_state.global_trend_option_dlt))
    if 'n_bootstrap_draws_dlt' not in st.session_state:
        st.session_state.n_bootstrap_draws_dlt = 500
    n_bootstrap_draws_dlt = st.sidebar.slider('N Bootstrap Draws DLT', 100, 1000, st.session_state.n_bootstrap_draws_dlt)
    if 'regression_penalty_dlt' not in st.session_state:
        st.session_state.regression_penalty_dlt = 'fixed_ridge'
    regression_penalty_dlt = st.sidebar.selectbox('Regression Penalty DLT', ['fixed_ridge', 'lasso', 'auto_ridge'], index=list(['fixed_ridge', 'lasso', 'auto_ridge']).index(st.session_state.regression_penalty_dlt))

# Настройка модели LGT
if selected_model == "LGT":
    if 'estimator_lgt' not in st.session_state:
        st.session_state.estimator_lgt = 'stan-mcmc'
    estimator_lgt = st.sidebar.selectbox('Estimator LGT', ['stan-mcmc', 'pyro-svi'], index=list(['stan-mcmc', 'pyro-svi']).index(st.session_state.estimator_lgt))
    if 'n_bootstrap_draws_lgt' not in st.session_state:
        st.session_state.n_bootstrap_draws_lgt = 500
    n_bootstrap_draws_lgt = st.sidebar.slider('N Bootstrap Draws LGT', 100, 1000, st.session_state.n_bootstrap_draws_lgt)
    if 'regression_penalty_lgt' not in st.session_state:
        st.session_state.regression_penalty_lgt = 'fixed_ridge'
    regression_penalty_lgt = st.sidebar.selectbox('Regression Penalty LGT', ['fixed_ridge', 'lasso', 'auto_ridge'], index=list(['fixed_ridge', 'lasso', 'auto_ridge']).index(st.session_state.regression_penalty_lgt))

# Настройка модели ETS
if selected_model == "ETS":
    if 'estimator_ets' not in st.session_state:
        st.session_state.estimator_ets = 'stan-mcmc'
    estimator_ets = st.sidebar.selectbox('Estimator ETS', ['stan-map', 'stan-mcmc'], index=list(['stan-map', 'stan-mcmc']).index(st.session_state.estimator_ets))
    if 'n_bootstrap_draws_ets' not in st.session_state:
        st.session_state.n_bootstrap_draws_ets = 500
    n_bootstrap_draws_ets = st.sidebar.slider('N Bootstrap Draws ETS', 100, 1000, st.session_state.n_bootstrap_draws_ets)

# Настройка модели KTR
if selected_model == "KTR":
    if 'estimator_ktr' not in st.session_state:
        st.session_state.estimator_ktr = 'pyro-svi'
    estimator_ktr = st.sidebar.selectbox('Estimator KTR', ['pyro-svi'], index=list(['pyro-svi']).index(st.session_state.estimator_ktr))
    if 'n_bootstrap_draws_ktr' not in st.session_state:
        st.session_state.n_bootstrap_draws_ktr = 500
    n_bootstrap_draws_ktr = st.sidebar.slider('N Bootstrap Draws KTR', 100, 1000, st.session_state.n_bootstrap_draws_ktr)
    if 'num_steps_ktr' not in st.session_state:
        st.session_state.num_steps_ktr = 200
    num_steps_ktr = st.sidebar.slider('Num Steps KTR', 100, 500, st.session_state.num_steps_ktr)

# Настройка модели Prophet
if selected_model == "Prophet":
    if 'seasonality_mode' not in st.session_state:
        st.session_state.seasonality_mode = 'additive'
    seasonality_mode = st.sidebar.selectbox("Seasonality Mode", ["additive", "multiplicative"], index=["additive", "multiplicative"].index(st.session_state.seasonality_mode))
    yearly_seasonality = st.sidebar.checkbox("Yearly Seasonality", value=True)
    weekly_seasonality = st.sidebar.checkbox("Weekly Seasonality", value=True)

# Настройка модели Chronos
if selected_model == "Chronos":
    if 'device_map_chronos' not in st.session_state:
        st.session_state.device_map_chronos = 'cpu'
    device_map_chronos = st.sidebar.selectbox('Устройство для Chronos', ['cuda','cpu'], index=['cuda', 'cpu'].index(st.session_state.device_map_chronos))
    if 'chronos_model_name' not in st.session_state:
        st.session_state.chronos_model_name = 'amazon/chronos-t5-tiny'
    amazon_options = ['amazon/chronos-t5-tiny', 'amazon/chronos-t5-mini', 'amazon/chronos-t5-small', 'amazon/chronos-t5-base', 'amazon/chronos-t5-large']
    chronos_model_name = st.sidebar.selectbox("Модель:", amazon_options, help="Выберите модель Amazon для прогнозирования.")

# Настройка модели LSTM
if selected_model == "LSTM":
    if 'lstm_epochs' not in st.session_state:
        st.session_state.lstm_epochs = 20
    lstm_epochs = st.sidebar.slider('Количество эпох LSTM', 10, 100, st.session_state.lstm_epochs)
    
    if 'lstm_batch_size' not in st.session_state:
        st.session_state.lstm_batch_size = 16
    lstm_batch_size = st.sidebar.slider('Размер батча LSTM', 8, 64, st.session_state.lstm_batch_size)
    
    if 'lstm_n_steps' not in st.session_state:
        st.session_state.lstm_n_steps = 2
    lstm_n_steps = st.sidebar.slider('Количество шагов LSTM', 1, 5, st.session_state.lstm_n_steps)
    
    if 'lstm_n_length' not in st.session_state:
        st.session_state.lstm_n_length = 7
    lstm_n_length = st.sidebar.slider('Длина подпоследовательности LSTM', 1, 30, st.session_state.lstm_n_length)

# Настройка модели TFT
if selected_model == "TFT":
    tft_learning_rate = st.sidebar.slider('Learning Rate TFT', 0.001, 0.1, 0.03)
    tft_hidden_size = st.sidebar.slider('Hidden Size TFT', 8, 128, 16)
    tft_attention_head_size = st.sidebar.slider('Attention Head Size TFT', 1, 4, 2)
    tft_dropout = st.sidebar.slider('Dropout TFT', 0.1, 0.3, 0.1)
    tft_hidden_continuous_size = st.sidebar.slider('Hidden Continuous Size TFT', 8, 128, 8)
    tft_max_epochs = st.sidebar.slider('Max Epochs TFT', 10, 100, tft_max_epochs)

# Настройка модели N-BEATS
if selected_model == "N-BEATS":
    nbeats_learning_rate = st.sidebar.slider('Learning Rate N-BEATS', 0.001, 0.1, 0.01)
    nbeats_widths = st.sidebar.multiselect('Widths N-BEATS', [32, 64, 128, 256, 512], default=[32, 512])
    nbeats_max_epochs = st.sidebar.slider('Max Epochs N-BEATS', 10, 100, 15)
    nbeats_backcast_loss_ratio = st.sidebar.slider('Backcast Loss Ratio', 0.0, 1.0, 1.0)
    nbeats_log_interval = st.sidebar.slider('Log Interval', 10, 100, nbeats_max_epochs)
    nbeats_log_val_interval = st.sidebar.slider('Log Val Interval', 1, 10, 1)
    nbeats_weight_decay = st.sidebar.slider('Weight decay', 1e-5, 1e-1, 1e-2)

# Для N-BEATS
nbeats_learning_rate = 1e-3
nbeats_log_interval = 10
nbeats_log_val_interval = 1
nbeats_weight_decay = 1e-2
nbeats_widths = [32, 512]
nbeats_backcast_loss_ratio = 1.0
nbeats_max_epochs = 15
    
# Кнопка "Дополнительная информация"
if st.sidebar.button("Дополнительная информация"):
    if selected_model == "DLT":
        st.sidebar.markdown("**Estimator DLT:**  Метод, используемый для обучения модели.  Stan-map - это более быстрый, но менее точный метод.  Stan-mcmc - это более точный, но более медленный метод.")
        st.sidebar.markdown("**Global trend option DLT:**  Тип глобального тренда в данных.  'flat' - отсутствие тренда, 'linear' - линейный тренд, 'loglinear' - логарифмический тренд, 'logistic' - логистический тренд.")
        st.sidebar.markdown("**Bootstrap draws DLT:**  Количество сэмплов, которые используются для оценки неопределенности прогноза.  Увеличьте это значение, чтобы получить более точную оценку неопределенности.")
        st.sidebar.markdown("**Regression penalty DLT:**  Тип регуляризации для регрессии.  'fixed_ridge' - фиксированная регуляризация типа L2, 'lasso' - регуляризация типа L1, 'auto_ridge' - автоматический выбор типа регуляризации. ")
    elif selected_model == "LGT":
        st.sidebar.markdown("**Estimator LGT:**  Метод, используемый для обучения модели.  Stan-mcmc - это более точный, но более медленный метод.  Pyro-svi - это более быстрый, но менее точный метод.")
        st.sidebar.markdown("**Bootstrap draws LGT:**  Количество сэмплов, которые используются для оценки неопределенности прогноза.  Увеличьте это значение, чтобы получить более точную оценку неопределенности.")
        st.sidebar.markdown("**Regression penalty LGT:**  Тип регуляризации для регрессии.  'fixed_ridge' - фиксированная регуляризация типа L2, 'lasso' - регуляризация типа L1, 'auto_ridge' - автоматический выбор типа регуляризации. ")
    elif selected_model == "ETS":
        st.sidebar.markdown("**Estimator ETS:**  Метод, используемый для обучения модели.  Stan-map - это более быстрый, но менее точный метод.  Stan-mcmc - это более точный, но более медленный метод.")
        st.sidebar.markdown("**Bootstrap draws ETS:**  Количество сэмплов, которые используются для оценки неопределенности прогноза.  Увеличьте это значение, чтобы получить более точную оценку неопределенности.")
    elif selected_model == "KTR":
        st.sidebar.markdown("**Estimator KTR:**  Метод, используемый для обучения модели.  Pyro-svi - это единственный метод у модели.")
        st.sidebar.markdown("**Bootstrap draws KTR:**  Количество сэмплов, которые используются для оценки неопределенности прогноза.  Увеличьте это значение, чтобы получить более точную оценку неопределенности.")
        st.sidebar.markdown("**Num steps KTR:**  Количество шагов, которые используются для обучения модели.  Увеличение этого значения может улучшить точность модели, но обучение будет дольше.")
    elif selected_model == "Prophet":
        st.sidebar.markdown("**Prophet:** Модель от Meta (Facebook) для прогнозирования временных рядов с учётом сезонности и трендов. Подходит для данных с пропусками.")
    elif selected_model == "Chronos":
        st.sidebar.markdown("**Chronos:** Модель от Amazon для прогнозирования временных рядов. Поддерживает использование трансформеров и гибкую настройку параметров.")
        st.sidebar.markdown("**Модель:** Выберите предобученную модель (например, amazon/chronos-t5-large).")
        st.sidebar.markdown("**Устройство:** Определите, будет ли использоваться CUDA или CPU для вычислений.")    
    elif selected_model == "LSTM":
        st.sidebar.markdown("**LSTM (Long Short-Term Memory):** Нейронная сеть, которая учитывает долгосрочную и краткосрочную память для прогнозирования временных рядов.")
        st.sidebar.markdown("**Эпохи:** Количество эпох обучения сети.")
        st.sidebar.markdown("**Батч:** Размер батча для обучения.")
        st.sidebar.markdown("**Количество шагов:** Количество временных шагов, которые учитываются в каждой подпоследовательности.")
        st.sidebar.markdown("**Длина подпоследовательности:** Длина каждой подпоследовательности для прогнозирования.")
    elif selected_model == "TFT":
        st.sidebar.markdown("**TFT:** Temporal Fusion Transformer для прогнозирования временных рядов.")
        st.sidebar.markdown("- Learning Rate: скорость обучения.")
        st.sidebar.markdown("- Hidden Size: размер скрытых слоев.")
        st.sidebar.markdown("- Attention Head Size: количество голов в multi-head attention.")
    elif selected_model == "N-BEATS":
        st.sidebar.markdown("**N-BEATS:** Нейросеть для прогнозирования временных рядов.")
        st.sidebar.markdown("- Widths: ширина блоков сети.")
        st.sidebar.markdown("- Learning Rate: скорость обучения.")
   
# Кнопка для запуска автоматической настройки
if st.sidebar.button("Автонастройка"):
    for well_name in selected_wells:
        # Создание DataFrame
        df_well = create_well_dataframe(well_name, data_sheets, file_path)
        df_well_original = df_well.copy()  # Копия исходного DataFrame

        # Обработка пропущенных значений
        df_well.replace('', np.nan, inplace=True)
        df_well.replace(0, np.nan, inplace=True)
        df_well = df_well.interpolate(method='linear', limit_direction='both')
        df_well = df_well.dropna()

        # Разделение данных на обучающие и тестовые выборки
        train_size_train = int(len(df_well) * 0.7)
        train_size_test = int(len(df_well) * 0.95)
        df_train = df_well[:train_size_train]
        df_test = df_well[train_size_test:]

        # Удаляем строки, где целевая переменная равна 0
        df_train = df_well[df_well[target_parameter] != 0].copy()
        df_train.fillna(method='ffill', inplace=True)

        # Вычисляем автокорреляцию для целевой переменной
        acf_values_target = acf(df_train[target_parameter], fft=True, nlags=20)
        st.write(f"Автокорреляция для {target_parameter}: {acf_values_target}")

        # Проверяем автокорреляцию регрессоров
        relevant_regressors = []
        for regressor in df_train.columns:
            if regressor != target_parameter and regressor != 'ds':
                acf_values = acf(df_train[regressor], fft=True, nlags=20)
                if max(acf_values[1:]) > 0.2:
                    relevant_regressors.append(regressor)
                else:
                    st.write(f"Регрессор {regressor} имеет низкую автокорреляцию и исключен.")
        
        st.write(f'Регрессоры с высокой автокорреляцией: {relevant_regressors}')

        # Проверка корреляции:
        for regressor in relevant_regressors:
            # Проверяем корреляцию с целевой переменной
            correlation = df_train[regressor].corr(df_train[target_parameter])
            if abs(correlation) < 0.1:  
                st.write(f"Низкая корреляция: {regressor} (r = {correlation:.2f}). Не учитываем его в модели.")
                relevant_regressors.remove(regressor)

        # Информативный график корреляции
        correlation_data = {
            regressor: df_train[regressor].corr(df_train[target_parameter])
            for regressor in df_train.columns if regressor != target_parameter and regressor != 'ds'
        }
        correlation_df = pd.DataFrame.from_dict(correlation_data, orient='index', columns=['Correlation'])
        correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)

        # Убираем 'ds' из регрессоров
        relevant_regressors = [x for x in relevant_regressors if x != 'ds']

        # Оцениваем p-value для каждого регрессора
        for regressor in relevant_regressors:
            try:
                # Используем тест Спирмена для проверки корреляции
                correlation, p_value = stats.spearmanr(df_train[regressor], df_train[target_parameter])
            except ValueError:
                st.write(f"Ошибка: Невозможно рассчитать тест Спирмена для {regressor}. Все значения x одинаковы.")

        # Фильтруем регрессоры с p-value < 0.05:
        significant_regressors = [
            regressor for regressor in relevant_regressors
            if stats.spearmanr(df_train[regressor], df_train[target_parameter])[1] < 0.05
        ]      

    def objective(trial):
        # Локальная инициализация
        df_well = create_well_dataframe(well_name, data_sheets, file_path)
        df_well['well_id'] = str(well_name)
        # Определение гиперпараметров для каждой модели
        if selected_model == "DLT":  # Если выбрана модель DLT
            estimator_dlt = trial.suggest_categorical('estimator_dlt', ['stan-map', 'stan-mcmc'])  # Выбор метода оптимизации для DLT
            global_trend_option_dlt = trial.suggest_categorical('global_trend_option_dlt', ['flat', 'linear', 'loglinear', 'logistic'])  # Выбор типа тренда для DLT
            n_bootstrap_draws_dlt = trial.suggest_int('n_bootstrap_draws_dlt', 100, 1000)  # Количество бутстреп-выборок для DLT
            regression_penalty_dlt = trial.suggest_categorical('regression_penalty_dlt', ['fixed_ridge', 'lasso', 'auto_ridge'])  # Регуляризация для DLT
            model = DLT(response_col=target_parameter, regressor_col=significant_regressors,  # Создание объекта модели DLT
                        estimator=estimator_dlt, seasonality=12,  # Задание параметров модели
                        global_trend_option=global_trend_option_dlt, n_bootstrap_draws=n_bootstrap_draws_dlt,
                        regression_penalty=regression_penalty_dlt)
            model.fit(df=df_train)
            forecast_test = model.predict(df_test)
            rmse = mean_squared_error(df_test[target_parameter].astype(float), forecast_test["prediction"], squared=False)
            
        elif selected_model == "LGT":  # Аналогично для модели LGT
            estimator_lgt = trial.suggest_categorical('estimator_lgt', ['stan-mcmc', 'pyro-svi'])
            n_bootstrap_draws_lgt = trial.suggest_int('n_bootstrap_draws_lgt', 100, 1000)
            regression_penalty_lgt = trial.suggest_categorical('regression_penalty_lgt', ['fixed_ridge', 'lasso', 'auto_ridge'])
            model = LGT(response_col=target_parameter, regressor_col=significant_regressors,
                        estimator=estimator_lgt, seasonality=12,  # Задание параметров модели
                        regression_penalty=regression_penalty_lgt)
            model.fit(df=df_train)
            forecast_test = model.predict(df_test)
            rmse = mean_squared_error(df_test[target_parameter].astype(float), forecast_test["prediction"], squared=False)
        elif selected_model == "ETS":  # Аналогично для модели ETS
            estimator_ets = trial.suggest_categorical('estimator_ets', ['stan-mcmc', 'stan-map'])
            n_bootstrap_draws_ets = trial.suggest_int('n_bootstrap_draws_ets', 100, 1000)
            model = ETS(response_col=target_parameter,
                        estimator=estimator_ets, seasonality=12,  # Задание параметров модели
                        n_bootstrap_draws=n_bootstrap_draws_ets)
            model.fit(df=df_train)
            forecast_test = model.predict(df_test)
            rmse = mean_squared_error(df_test[target_parameter].astype(float), forecast_test["prediction"], squared=False)  
        elif selected_model == "KTR":  # Аналогично для модели KTR
            estimator_ktr = trial.suggest_categorical('estimator_ktr', ['pyro-svi'])
            n_bootstrap_draws_ktr = trial.suggest_int('n_bootstrap_draws_ktr', 100, 1000)
            num_steps_ktr = trial.suggest_int('num_steps_ktr', 100, 500)
            model = KTR(response_col=target_parameter, regressor_col=significant_regressors,
                        estimator=estimator_ktr, seasonality=12,  # Задание параметров модели
                        n_bootstrap_draws=n_bootstrap_draws_ktr, num_steps=num_steps_ktr)
            model.fit(df=df_train)
            forecast_test = model.predict(df_test)
            rmse = mean_squared_error(df_test[target_parameter].astype(float), forecast_test["prediction"], squared=False)
        elif selected_model == "Prophet":  # Настройка модели Prophet
            seasonality_mode = trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"])
            yearly_seasonality = trial.suggest_categorical("yearly_seasonality", [True, False])
            weekly_seasonality = trial.suggest_categorical("weekly_seasonality", [True, False])
            model = Prophet(
                seasonality_mode=seasonality_mode,
                yearly_seasonality=yearly_seasonality,
                weekly_seasonality=weekly_seasonality
            )
            df_train_prophet = df_train.rename(columns={"ds": "ds", target_parameter: "y"})
            model.fit(df_train_prophet)

            forecast_test = model.predict(df_test.rename(columns={"ds": "ds"})[["ds"]])
            rmse = mean_squared_error(df_test[target_parameter].astype(float), forecast_test["yhat"], squared=False)
        elif selected_model == "Chronos":
            chronos_model_name = trial.suggest_categorical("chronos_model_name", "amazon/chronos-t5-tiny")
            device_map = trial.suggest_categorical("device_map", "cpu")
            model = ChronosPipeline.from_pretrained(
                chronos_model_name,
                device_map=device_map,
                torch_dtype=torch.bfloat16
            )

            # Прогноз на реальных данных
            context_full = torch.tensor(df_well[target_parameter].values, dtype=torch.float32)  # Весь контекст доступных данных
            prediction_length = len(df_test)  # Длина прогноза по всем тестовым данным
            forecast_test = pipeline.predict(context_full, prediction_length)
            
            low, median, high = np.quantile(forecast_test[0].numpy(), [0.1, 0.5, 0.9], axis=0)
            forecast_test = pd.DataFrame({
                'ds': df_test['ds'], 
                'yhat': median,
                'yhat_lower': low,
                'yhat_upper': high
            })

            forecast_values = forecast_test['yhat'].values
            real_values, forecast_values = synchronize_forecast_and_test(df_test[target_parameter], forecast_values)

            # Метрики
            rmse = mean_squared_error(real_values.astype(float), forecast_values, squared=False)
        elif selected_model == "LSTM":
            lstm_epochs = trial.suggest_int("epochs", 10, 100)
            lstm_batch_size = trial.suggest_int("batch_size", 8, 64)
            n_steps = trial.suggest_int("n_input", 1, 5)
            n_length = trial.suggest_int("n_output", 7, 30)

            # Подготовка данных для LSTM
            def to_supervised(train, n_input, n_out=7):
                data = train.values
                X, y = [], []
                for i in range(len(data)):
                    end_ix = i + n_input
                    if end_ix + n_out <= len(data):
                        X.append(data[i:end_ix])
                        y.append(data[end_ix:end_ix + n_out])
                return np.array(X), np.array(y)
            
            def build_lstm_model(n_input, n_output):
                model = Sequential()
                model.add(LSTM(trial.suggest_int("lstm_units", 50, 200), activation='relu', input_shape=(n_input, 1)))
                model.add(RepeatVector(n_output))
                model.add(LSTM(trial.suggest_int("lstm_units", 50, 200), activation='relu', return_sequences=True))
                model.add(TimeDistributed(Dense(1)))
                model.compile(optimizer='adam', loss='mse')
                return model

            n_input = lstm_n_steps * lstm_n_length
            X, y = to_supervised(df_train[[target_parameter]], n_input)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            y = y.reshape((y.shape[0], y.shape[1], 1))

            # Построение и обучение модели
            model = build_lstm_model(n_input, y.shape[1])
            model.fit(X, y, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=0)

            # Прогноз на реальных данных
            X_test, y_test = to_supervised(df_test[[target_parameter]], n_input)
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
            yhat = model.predict(X_test, verbose=0)
            
            forecast_test = pd.DataFrame({
                'ds': df_test['ds'].iloc[:len(yhat)],  # Привязка оси x к длине прогнозируемых значений
                'yhat': yhat[:, -1, 0][:len(df_test)]  # Последний шаг прогнозируемой последовательности
            })
            
            forecast_values = forecast_test['yhat'].values
            real_values, forecast_values = synchronize_forecast_and_test(df_test[target_parameter], forecast_values)

            # Метрики
            rmse = mean_squared_error(real_values.astype(float), forecast_values, squared=False)
            
        elif selected_model == "TFT":
            
            # Подготовка данных
            df_well['well_id'] = str(well_name)
            df_well['time_idx'] = range(len(df_well))  # Создание индекса времени

            # Приведение типов к float32
            df_well = df_well.astype({col: 'float32' for col in df_well.select_dtypes(include=['float64']).columns})
            
            # Создание TimeSeriesDataSet для тренировки
            training = TimeSeriesDataSet(
                df_well,
                time_idx="time_idx",
                target=target_parameter,
                group_ids=["well_id"],
                time_varying_known_reals=["time_idx"],
                time_varying_unknown_reals=[target_parameter],
            )

            # Validation на всем диапазоне теста
            validation = TimeSeriesDataSet.from_dataset(
                training, df_well, predict=False
            )

            train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
            val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

            tft = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=trial.suggest_loguniform("tft_learning_rate", 1e-4, 1e-1),
                hidden_size=trial.suggest_int("tft_hidden_size", 8, 128),
                attention_head_size=trial.suggest_int("tft_attention_head_size", 1, 4),
                dropout=trial.suggest_uniform("tft_dropout", 0.1, 0.3),
                hidden_continuous_size=tft_hidden_continuous_size,
                loss=QuantileLoss(),
            ).float() 

            model = Trainer(max_epochs=tft_max_epochs, accelerator="auto")
            model.fit(tft, train_dataloader, val_dataloader)

            # Прогноз на тестовых данных
            predictions = tft.predict(validation, mode="prediction")

            # Учитываем минимальную длину между df_test['ds'] и предсказаниями
            min_length = min(len(df_test['ds']), len(predictions.flatten()))
            forecast_test = pd.DataFrame({
                'ds': df_test['ds'].iloc[:min_length],
                'yhat': predictions.flatten().numpy()[:min_length]
            })
            
            forecast_values = forecast_test['yhat'].values
            real_values, forecast_values = synchronize_forecast_and_test(df_test[target_parameter], forecast_values)

            # Метрики
            rmse = mean_squared_error(real_values.astype(float), forecast_values, squared=False)
            
        elif selected_model == "N-BEATS":

            # Подготовка данных
            df_well['well_id'] = str(well_name)
            df_well['time_idx'] = range(len(df_well))

            if target_parameter not in df_well.columns:
                st.error(f"Целевой параметр '{target_parameter}' не найден в данных.")
                st.stop()

            # Приведение типов к float32
            df_well = df_well.astype({col: 'float32' for col in df_well.select_dtypes(include=['float64']).columns})

            training = TimeSeriesDataSet(
                df_well,
                time_idx="time_idx",
                target=target_parameter,
                group_ids=["well_id"],
                time_varying_unknown_reals=[target_parameter],
            )

            validation = TimeSeriesDataSet.from_dataset(training, df_well, predict=False)

            train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
            val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

            # Настройка модели
            nbeats = NBeats.from_dataset(
                training,
                learning_rate = trial.suggest_loguniform("nbeats_learning_rate", 1e-4, 1e-1),
                widths = trial.suggest_categorical("nbeats_widths", [[32, 512], [64, 256], [128, 256]]),
                backcast_loss_ratio = trial.suggest_uniform("nbeats_backcast_loss_ratio", 0.5, 1.5),
                log_interval = trial.suggest_uniform("nbeats_log_interval", 1, 50),
                log_val_interval = trial.suggest_uniform("nbeats_log_val_interval", 0.1, 10),
                weight_decay = trial.suggest_loguniform("nbeats_weight_decay", 1e-5, 1e-2),
            ).float()

            trainer = Trainer(max_epochs=nbeats_max_epochs, accelerator="auto", gradient_clip_val=0.01)
            trainer.fit(nbeats, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

            # Прогноз на тестовых данных
            predictions = nbeats.predict(validation, mode="prediction")

            # Учитываем минимальную длину между df_test['ds'] и предсказаниями
            min_length = min(len(df_test['ds']), len(predictions.flatten()))
            forecast_test = pd.DataFrame({
                'ds': df_test['ds'].iloc[:min_length],
                'yhat': predictions.flatten().numpy()[:min_length]
            })

            forecast_values = forecast_test['yhat'].values
            real_values, forecast_values = synchronize_forecast_and_test(df_test[target_parameter], forecast_values)

            # Метрики
            rmse = mean_squared_error(real_values, forecast_values, squared=False)

        return rmse

    # Запуск оптимизации Optuna
    study = optuna.create_study(direction="minimize")  # Создание объекта Optuna для поиска минимального значения RMSE
    study.optimize(objective, n_trials=50)  # Запуск оптимизации, n_trials - количество испытаний 

    # Вывод результатов
    best_params = study.best_params  # Получение лучших найденных гиперпараметров
    st.markdown(f"\n**Наиболее оптимальные гиперпараметры: {best_params}.**") # Вывод оптимальных гиперпараметров на экран
    st.write(f"\n**Пожалуйста, укажите их в панели настройки.**")
    
    if selected_model == "DLT":
        st.session_state.estimator_dlt = best_params.get('estimator_dlt', 'stan-mcmc')
        st.session_state.global_trend_option_dlt = best_params.get('global_trend_option_dlt', 'linear')
        st.session_state.n_bootstrap_draws_dlt = best_params.get('n_bootstrap_draws_dlt', 500)
        st.session_state.regression_penalty_dlt = best_params.get('regression_penalty_dlt', 'fixed_ridge')
    elif selected_model == "LGT":
        st.session_state.estimator_lgt = best_params.get('estimator_lgt', 'stan-mcmc')
        st.session_state.n_bootstrap_draws_lgt = best_params.get('n_bootstrap_draws_lgt', 500)
        st.session_state.regression_penalty_lgt = best_params.get('regression_penalty_lgt', 'fixed_ridge')
    elif selected_model == "ETS":
        st.session_state.estimator_ets = best_params.get('estimator_ets', 'stan-mcmc')
        st.session_state.n_bootstrap_draws_ets = best_params.get('n_bootstrap_draws_ets', 500)
    elif selected_model == "KTR":
        st.session_state.estimator_ktr = best_params.get('estimator_ktr', 'pyro-svi')
        st.session_state.n_bootstrap_draws_ktr = best_params.get('n_bootstrap_draws_ktr', 500)
        st.session_state.num_steps_ktr = best_params.get('num_steps_ktr', 200)
    elif selected_model == "Prophet":
        st.session_state.update({
            "seasonality_mode": best_params.get("seasonality_mode", "additive"),
            "yearly_seasonality": best_params.get("yearly_seasonality", True),
            "weekly_seasonality": best_params.get("weekly_seasonality", True)
        })
    elif selected_model == "Chronos":
        st.session_state.chronos_model_name = best_params.get("chronos_model_name", "amazon/chronos-t5-tiny")
        st.session_state.device_map_chronos = best_params.get("device_map_chronos", "cpu")
    elif selected_model == "LSTM":
        st.session_state.lstm_epochs = best_params.get("epochs", 20)
        st.session_state.lstm_batch_size = best_params.get("batch_size", 16)
        st.session_state.lstm_n_steps = best_params.get("n_input", 2)
        st.session_state.lstm_n_length = best_params.get("n_output", 7)
    elif selected_model == "TFT":
        st.session_state.tft_learning_rate = best_params.get("tft_learning_rate", 0.03)
        st.session_state.tft_hidden_size = best_params.get("tft_hidden_size", 16)
        st.session_state.tft_attention_head_size = best_params.get("tft_attention_head_size", 2)
        st.session_state.tft_dropout = best_params.get("tft_dropout", 0.1)
        st.session_state.tft_hidden_continuous_size = best_params.get("tft_hidden_continuous_size", 8)
    elif selected_model == "N-BEATS":
        st.session_state.nbeats_learning_rate = best_params.get("nbeats_learning_rate", 1e-3)
        st.session_state.nbeats_widths = best_params.get("nbeats_widths", [32, 512])
        st.session_state.nbeats_backcast_loss_ratio = best_params.get("nbeats_backcast_loss_ratio", 1.0)
        st.session_state.nbeats_backcast_loss_ratio = best_params.get("nbeats_log_interval", 10)
        st.session_state.nbeats_backcast_loss_ratio = best_params.get("nbeats_log_val_interval", 1.0)
        st.session_state.nbeats_backcast_loss_ratio = best_params.get("nbeats_weight_decay", 1e-2)

st.sidebar.header("Настройки графика")
graph_color = st.sidebar.color_picker("Прогноз", value="#FF0000", help="Выберите цвет линии прогноза.")
graph_color_test = st.sidebar.color_picker("Тестовый прогноз", value="#00FF00", help="Выберите цвет линии прогноза на тестовых данных.")
graph_color_actual = st.sidebar.color_picker("Фактические данные", value="#000000", help="Выберите цвет линии фактических данных.")
graph_line_width = st.sidebar.slider("Толщина линии", min_value=1, max_value=5, value=2, help="Установите толщину линий на графике.")
graph_font_size = st.sidebar.slider("Размер шрифта", min_value=10, max_value=20, value=12, help="Установите размер шрифта для текста на графике.")
show_regressors = st.sidebar.checkbox("Отобразить релевантные регрессоры на графике прогноза", value=True)
background_color = st.sidebar.color_picker("Цвет фона", value="#FFFFFF", help="Выберите цвет фона для графика.")

st.sidebar.header("Выбор способа заполнения")

# Заполнение пропущенных значений
fill_method = st.sidebar.radio(
    "Способ:",
    ("Интерполяция", "Медиана ненулевых значений"),
    help="Выберете способ заполнения пропусков и нулей у регрессоров."
)

# Кнопка "Информация о методах заполнения"
if st.sidebar.button("Информация о методах заполнения"):
    st.sidebar.markdown("**Интерполяция:**  Этот метод позволяет заполнить пропущенные значения плавно,  сохраняя общий тренд данных.  Он лучше всего подходит для данных, которые имеют плавное изменение.")
    st.sidebar.markdown("**Медиана ненулевых значений:**  Этот метод  заполняет пропущенные значения медианой ненулевых значений.  Он подходит для данных, которые могут иметь выбросы.")

st.markdown("## Результаты прогнозирования")

# Функция для загрузки всех данных с листов
def load_all_data(file_path, target_parameter, well_names):
    
    # Создаем список для хранения всех DataFrames
    dfs_list = []

    for sheet_name in data_sheets:
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        df.columns = df.iloc[0]
        df = df[1:]
        df.index = pd.to_datetime(df.iloc[:, 0])
        dfs_list.append(df[well_names])

    # Объединяем DataFrames по столбцам
    df_combined = pd.concat(dfs_list, axis=1)
    df_combined['ds'] = df_combined.index
    return df_combined.reset_index()

# Функция для анализа пропущенных данных
def analyze_missing_data(df_well, target_parameter):
    st.markdown("## Анализ пропущенных данных")

    for well_name in selected_wells:
        st.markdown(f"### Скважина {well_name}")
        df = df_well[df_well.columns.drop('ds')]
        for col in df.columns:
            total_values = len(df[col])
            null_values = df[col].isnull().sum()
            zero_values = (df[col] == 0).sum()
            missing_percentage = (null_values / total_values) * 100
            zero_percentage = (zero_values / total_values) * 100
            г
            st.markdown(f"**Параметр: {col}**")
            st.markdown(f"   - Общее количество значений: {total_values}")
            st.markdown(f"   - Количество пропущенных значений: {null_values} ({missing_percentage:.2f}%)")
            st.markdown(f"   - Количество нулевых значений: {zero_values} ({zero_percentage:.2f}%)")

# Кнопка запуска прогноза
if st.button("Запустить прогноз"):
    # Запуск прогнозирования для нескольких скважин
    for well_name in selected_wells:
        # Создание DataFrame
        df_well = create_well_dataframe(well_name, data_sheets, file_path)
        df_well_original = df_well.copy()

        # Обработка пропущенных значений
        df_well.replace('', np.nan, inplace=True)
        df_well.replace(0, np.nan, inplace=True)

        if fill_method == "Интерполяция":
            df_well = df_well.interpolate(method='linear', limit_direction='both')
        elif fill_method == "Медиана ненулевых значений":
            for col in df_well.columns:
                df_well[col] = df_well[col].fillna(df_well[col].dropna().median())
        df_well = df_well.dropna()

        # Удаляем строки, где целевая переменная равна 0
        df_train = df_well[df_well[target_parameter] != 0].copy()
        df_train.fillna(method='ffill', inplace=True)

        # Вычисляем автокорреляцию для целевой переменной
        acf_values_target = acf(df_train[target_parameter], fft=True, nlags=20)
        st.write(f"Автокорреляция для {target_parameter}: {acf_values_target}")

        # График автокорреляции для целевой переменной
        plt.figure(figsize=(10, 5))
        plt.bar(range(len(acf_values_target)), acf_values_target)
        plt.title(f"Автокорреляция для {target_parameter}")
        plt.xlabel("Лаг")
        plt.ylabel("Корреляция")
        st.pyplot(plt)

        # Проверяем автокорреляцию регрессоров
        relevant_regressors = []
        for regressor in df_train.columns:
            if regressor != target_parameter and regressor != 'ds':
                acf_values = acf(df_train[regressor], fft=True, nlags=20)
                if max(acf_values[1:]) > 0.2:
                    relevant_regressors.append(regressor)
                else:
                    st.write(f"Регрессор {regressor} имеет низкую автокорреляцию и исключен.")
        
        st.write(f'Регрессоры с высокой автокорреляцией: {relevant_regressors}')

        # Проверка корреляции:
        for regressor in relevant_regressors:
            # Проверяем корреляцию с целевой переменной
            correlation = df_train[regressor].corr(df_train[target_parameter])
            if abs(correlation) < 0.1:  
                st.write(f"Низкая корреляция: {regressor} (r = {correlation:.2f}). Не учитываем его в модели.")
                relevant_regressors.remove(regressor)

        # Информативный график корреляции
        correlation_data = {
            regressor: df_train[regressor].corr(df_train[target_parameter])
            for regressor in df_train.columns if regressor != target_parameter and regressor != 'ds'
        }
        correlation_df = pd.DataFrame.from_dict(correlation_data, orient='index', columns=['Correlation'])
        correlation_df = correlation_df.sort_values(by='Correlation', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=correlation_df['Correlation'], y=correlation_df.index, palette="coolwarm")
        plt.axvline(x=0.1, color='green', linestyle='--', label='Threshold: 0.1')
        plt.axvline(x=-0.1, color='green', linestyle='--')
        plt.title("Корреляция регрессоров с целевой переменной")
        plt.xlabel("Correlation")
        plt.ylabel("Regressors")
        plt.legend()
        st.pyplot(plt)

        # Убираем 'ds' из регрессоров
        relevant_regressors = [x for x in relevant_regressors if x != 'ds']

        # Оцениваем p-value для каждого регрессора
        for regressor in relevant_regressors:
            try:
                # Используем тест Спирмена для проверки корреляции
                correlation, p_value = stats.spearmanr(df_train[regressor], df_train[target_parameter])
            except ValueError:
                st.write(f"Ошибка: Невозможно рассчитать тест Спирмена для {regressor}. Все значения x одинаковы.")

        # Фильтруем регрессоры с p-value < 0.05:
        significant_regressors = [
            regressor for regressor in relevant_regressors
            if stats.spearmanr(df_train[regressor], df_train[target_parameter])[1] < 0.05
        ]      

        # Реализуем кросс-валидацию
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(df_train):
            train, test = df_train.iloc[train_index], df_train.iloc[test_index]
            st.write(f"Размеры Train: {len(train)}, Test: {len(test)}")
           
            # Выбираем модель прогнозирования
            if selected_model == "DLT":
                model = DLT(
                    response_col=target_parameter,
                    regressor_col=relevant_regressors,
                    estimator=estimator_dlt,
                    regression_penalty=regression_penalty_dlt,
                    seasonality=12,
                    global_trend_option=global_trend_option_dlt,
                    n_bootstrap_draws=n_bootstrap_draws_dlt
                )
                model.fit(train)
                forecast_test = model.predict(test)

                forecast_values = forecast_test["prediction"].values
                real_values = test[target_parameter].values

                # Синхронизация длин
                min_length = min(len(forecast_values), len(real_values))
                forecast_values = forecast_values[:min_length]
                real_values = real_values[:min_length]

                # Метрики
                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100

                # Оценка точности модели
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                # Прогноз на будущее
                last_date = df_well.index[-1]  # Последняя дата реальных данных
                future_dates = pd.date_range(last_date, periods=horizon, freq='D')  # Начинаем прогноз с последней точки

                # Добавляем регрессоры в df_future
                df_future = pd.DataFrame({'ds': future_dates})
                for col in relevant_regressors:
                    df_future[col] = df_well[col].iloc[-1]

                if parameter_change:
                    last_date = df_well.index[-1]
                    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
                    changed_values = np.linspace(df_well[parameter_change].iloc[-1],
                                                 df_well[parameter_change].iloc[-1] + value_change, horizon)
                    df_future[parameter_change] = changed_values

                forecast = model.predict(df_future)

                # SHAP-анализ после обучения
                #st.write("SHAP-анализ после обучения модели")
                #explainer = shap.Explainer(model, train[relevant_regressors])
                #shap_values = explainer(test[relevant_regressors])
                
                #shap.summary_plot(shap_values, test[relevant_regressors], show=False)
                #st.pyplot(plt)

                #shap.summary_plot(shap_values, test[relevant_regressors], plot_type="bar", show=False)
                #st.pyplot(plt)

                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index()
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                # Добавляем показ регрессоров на графике
                if show_regressors:

                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')
            
            elif selected_model == "LGT":
                model = LGT(response_col=target_parameter, regressor_col=significant_regressors, estimator=estimator_lgt, regression_penalty = regression_penalty_lgt, seasonality=12, n_bootstrap_draws=n_bootstrap_draws_lgt)
                model.fit(train)
                forecast_test = model.predict(test)
                forecast_values = forecast_test["prediction"].values
                real_values = test[target_parameter].values

                # Синхронизация длин
                min_length = min(len(forecast_values), len(real_values))
                forecast_values = forecast_values[:min_length]
                real_values = real_values[:min_length]

                # Метрики
                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100

                # Оценка точности модели
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                # Прогноз на будущее
                last_date = df_well.index[-1]  # Последняя дата реальных данных
                future_dates = pd.date_range(last_date, periods=horizon, freq='D')  # Начинаем прогноз с последней точки

                # Добавляем регрессоры в df_future
                df_future = pd.DataFrame({'ds': future_dates})

                # Добавляем регрессоры из прошлого
                for col in significant_regressors:
                    df_future[col] = df_well[col].iloc[-1]

                # Добавляем измененный параметр как регрессор
                if parameter_change:
                    last_date = df_well.index[-1]
                    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
                    changed_values = np.linspace(df_well[parameter_change].iloc[-1],
                                                 df_well[parameter_change].iloc[-1] + value_change, horizon)
                    df_future[parameter_change] = changed_values

                # Генерируем прогноз
                forecast = model.predict(df_future)
                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index()
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                # Добавляем показ регрессоров на графике
                if show_regressors:

                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')
                
            elif selected_model == "ETS":
                model = ETS(response_col=target_parameter, estimator=estimator_ets, seasonality=12, n_bootstrap_draws=n_bootstrap_draws_ets)
                model.fit(train)
                forecast_test = model.predict(test)
                forecast_values = forecast_test["prediction"].values
                real_values = test[target_parameter].values

                # Синхронизация длин
                min_length = min(len(forecast_values), len(real_values))
                forecast_values = forecast_values[:min_length]
                real_values = real_values[:min_length]

                # Метрики
                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100

                # Оценка точности модели
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                # Прогноз на будущее
                last_date = df_well.index[-1]  # Последняя дата реальных данных
                future_dates = pd.date_range(last_date, periods=horizon, freq='D')  # Начинаем прогноз с последней точки

                # Добавляем регрессоры в df_future
                df_future = pd.DataFrame({'ds': future_dates})

                # Добавляем регрессоры из прошлого
                for col in significant_regressors:
                    df_future[col] = df_well[col].iloc[-1]

                # Добавляем измененный параметр как регрессор
                if parameter_change:
                    last_date = df_well.index[-1]
                    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
                    changed_values = np.linspace(df_well[parameter_change].iloc[-1],
                                                 df_well[parameter_change].iloc[-1] + value_change, horizon)
                    df_future[parameter_change] = changed_values

                # Генерируем прогноз
                forecast = model.predict(df_future)
                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index()
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                # Добавляем показ регрессоров на графике
                if show_regressors:

                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')
                
            elif selected_model == "KTR":
                model = KTR(response_col=target_parameter, regressor_col=significant_regressors, estimator=estimator_ktr, seasonality=12, n_bootstrap_draws=n_bootstrap_draws_ktr, num_steps=num_steps_ktr)
                model.fit(train)
                forecast_test = model.predict(test)
                forecast_values = forecast_test["prediction"].values
                real_values = test[target_parameter].values

                # Синхронизация длин
                min_length = min(len(forecast_values), len(real_values))
                forecast_values = forecast_values[:min_length]
                real_values = real_values[:min_length]

                # Метрики
                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100

                # Оценка точности модели
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                # Прогноз на будущее
                last_date = df_well.index[-1]  # Последняя дата реальных данных
                future_dates = pd.date_range(last_date, periods=horizon, freq='D')  # Начинаем прогноз с последней точки

                # Добавляем регрессоры в df_future
                df_future = pd.DataFrame({'ds': future_dates})

                # Добавляем регрессоры из прошлого
                for col in significant_regressors:
                    df_future[col] = df_well[col].iloc[-1]

                # Добавляем измененный параметр как регрессор
                if parameter_change:
                    last_date = df_well.index[-1]
                    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
                    changed_values = np.linspace(df_well[parameter_change].iloc[-1],
                                                 df_well[parameter_change].iloc[-1] + value_change, horizon)
                    df_future[parameter_change] = changed_values

                # Генерируем прогноз
                forecast = model.predict(df_future)
                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index()
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                # Добавляем показ регрессоров на графике
                if show_regressors:

                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')
                
            elif selected_model == "Prophet":
                model = Prophet(
                    seasonality_mode=seasonality_mode,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality
                )
                df_train_prophet = train.rename(columns={"ds": "ds", target_parameter: "y"})
                df_test_prophet = test.rename(columns={"ds": "ds"})
                model.fit(df_train_prophet)
                forecast_test = model.predict(df_test_prophet[["ds"]])

                forecast_values = forecast_test['yhat'].values
                real_values = test[target_parameter].values

                # Синхронизация длин
                min_length = min(len(forecast_values), len(real_values))
                forecast_values = forecast_values[:min_length]
                real_values = real_values[:min_length]

                # Метрики
                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100

                # Оценка точности модели
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                # Прогноз на будущее
                last_date = df_well.index[-1]  # Последняя дата реальных данных
                future_dates = pd.date_range(last_date, periods=horizon, freq='D')  # Начинаем прогноз с последней точки

                # Добавляем регрессоры в df_future
                df_future = pd.DataFrame({'ds': future_dates})

                # Добавляем регрессоры из прошлого
                for col in significant_regressors:
                    df_future[col] = df_well[col].iloc[-1]

                # Добавляем измененный параметр как регрессор
                if parameter_change:
                    last_date = df_well.index[-1]
                    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
                    changed_values = np.linspace(df_well[parameter_change].iloc[-1],
                                                 df_well[parameter_change].iloc[-1] + value_change, horizon)
                    df_future[parameter_change] = changed_values

                # Генерируем прогноз
                forecast = model.predict(df_future)
                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index()
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                # Добавляем показ регрессоров на графике
                if show_regressors:

                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')
                
            elif selected_model == "Chronos":
                # Инициализация ChronosPipeline
                pipeline = ChronosPipeline.from_pretrained(
                    chronos_model_name,
                    device_map=device_map_chronos,
                    torch_dtype=torch.bfloat16
                )
                model = pipeline
                # Прогноз на реальных данных
                context_full = torch.tensor(train[target_parameter].values, dtype=torch.float32)  # Весь контекст доступных данных
                prediction_length = len(test)  # Длина прогноза по всем тестовым данным
                forecast_test = pipeline.predict(context_full, prediction_length)
                
                low, median, high = np.quantile(forecast_test[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                forecast_test = pd.DataFrame({
                    'ds': test['ds'], 
                    'yhat': median,
                    'yhat_lower': low,
                    'yhat_upper': high
                })

                forecast_values = forecast_test['yhat'].values
                real_values, forecast_values = synchronize_forecast_and_test(test[target_parameter], forecast_values)

                # Метрики
                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100

                # Оценка точности модели
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                # Прогноз на будущее
                last_date = df_well.index[-1]  # Последняя дата реальных данных
                future_dates = pd.date_range(last_date, periods=horizon, freq='D')  # Начинаем прогноз с последней точки

                # Добавляем регрессоры в df_future
                df_future = pd.DataFrame({'ds': future_dates})

                # Добавляем регрессоры из прошлого
                for col in significant_regressors:
                    df_future[col] = df_well[col].iloc[-1]

                # Добавляем измененный параметр как регрессор
                if parameter_change:
                    last_date = df_well.index[-1]
                    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
                    changed_values = np.linspace(df_well[parameter_change].iloc[-1],
                                                 df_well[parameter_change].iloc[-1] + value_change, horizon)
                    df_future[parameter_change] = changed_values

                # Прогноз на будущее
                prediction_length = horizon
                forecast_future = pipeline.predict(context_full, prediction_length)
                
                low_future, median_future, high_future = np.quantile(forecast_future[0].numpy(), [0.1, 0.5, 0.9], axis=0)
                forecast = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': median_future,
                    'yhat_lower': low_future,
                    'yhat_upper': high_future
                })

                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index()
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                # Добавляем показ регрессоров на графике
                if show_regressors:

                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')
                

            elif selected_model == "LSTM":
                # Подготовка данных для LSTM
                def to_supervised(train, n_input, n_out=7):
                    data = train.values
                    X, y = [], []
                    for i in range(len(data)):
                        end_ix = i + n_input
                        if end_ix + n_out <= len(data):
                            X.append(data[i:end_ix])
                            y.append(data[end_ix:end_ix + n_out])
                    return np.array(X), np.array(y)

                def build_lstm_model(n_input, n_output):
                    model = Sequential()
                    model.add(LSTM(100, activation='relu', input_shape=(n_input, 1)))
                    model.add(RepeatVector(n_output))
                    model.add(LSTM(100, activation='relu', return_sequences=True))
                    model.add(TimeDistributed(Dense(1)))
                    model.compile(optimizer='adam', loss='mse')
                    return model

                n_input = lstm_n_steps * lstm_n_length
                X, y = to_supervised(train[[target_parameter]], n_input)
                X = X.reshape((X.shape[0], X.shape[1], 1))
                y = y.reshape((y.shape[0], y.shape[1], 1))

                # Построение и обучение модели
                model = build_lstm_model(n_input, y.shape[1])
                model.fit(X, y, epochs=lstm_epochs, batch_size=lstm_batch_size, verbose=0)

                # Прогноз на реальных данных
                X_test, y_test = to_supervised(test[[target_parameter]], n_input)
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                yhat = model.predict(X_test, verbose=0)
                
                # Вычисление стандартного отклонения ошибки
                forecast_errors = y_test[:, -1, 0] - yhat[:, -1, 0]
                std_error = np.std(forecast_errors)

                forecast_test = pd.DataFrame({
                    'ds': test['ds'].iloc[:len(yhat)],  # Привязка оси x к длине прогнозируемых значений
                    'yhat': yhat[:, -1, 0][:len(test)],
                    'yhat_lower': yhat[:, -1, 0] - 1.96 * std_error,
                    'yhat_upper': yhat[:, -1, 0] + 1.96 * std_error
                })

                forecast_values = forecast_test['yhat'].values
                real_values, forecast_values = synchronize_forecast_and_test(test[target_parameter], forecast_values)

                # Метрики
                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100
                
                # Оценка точности модели
                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                # Прогноз на будущее
                last_date = df_well.index[-1]  # Последняя дата реальных данных
                future_dates = pd.date_range(last_date, periods=horizon, freq='D')  # Начинаем прогноз с последней точки

                # Добавляем регрессоры в df_future
                df_future = pd.DataFrame({'ds': future_dates})

                # Добавляем регрессоры из прошлого
                for col in significant_regressors:
                    df_future[col] = df_well[col].iloc[-1]

                # Добавляем измененный параметр как регрессор
                if parameter_change:
                    last_date = df_well.index[-1]
                    forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
                    changed_values = np.linspace(df_well[parameter_change].iloc[-1],
                                                 df_well[parameter_change].iloc[-1] + value_change, horizon)
                    df_future[parameter_change] = changed_values

                # Прогноз на будущее
                context = df_well[target_parameter].values[-n_input:]  # Последние n_input значений
                predictions = []
                
                for _ in range(horizon):
                    context_reshaped = context.reshape((1, n_input, 1))
                    yhat = model.predict(context_reshaped, verbose=0)
                    next_value = yhat[0, -1, 0]  # Достаем последний предсказанный шаг
                    predictions.append(next_value)
                    context = np.append(context[1:], next_value)  # Добавляем прогноз в контекст

                forecast = pd.DataFrame({
                    'ds': future_dates,
                    'yhat': predictions,
                    'yhat_lower': np.array(predictions) - 1.96 * std_error,
                    'yhat_upper': np.array(predictions) + 1.96 * std_error
                })
                
                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index()
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                # Добавляем показ регрессоров на графике
                if show_regressors:

                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')
    
            elif selected_model == "TFT":
                
                # Убедимся, что столбец 'time_idx' добавлен
                if 'time_idx' not in df_well.columns:
                    df_well['time_idx'] = range(len(df_well))  # Добавляем индекс времени

                df_well['well_id'] = str(well_name)  # Уникальный идентификатор группы

                # Проверка на наличие целевой переменной
                if target_parameter not in df_well.columns:
                    st.error(f"Целевой параметр '{target_parameter}' не найден в данных.")
                    st.stop()
                    
                # Приведение типов к float32
                df_well = df_well.astype({col: 'float32' for col in df_well.select_dtypes(include=['float64']).columns})

                # Создание TimeSeriesDataSet для тренировки
                training = TimeSeriesDataSet(
                    df_well,
                    time_idx="time_idx",
                    target=target_parameter,
                    group_ids=["well_id"],
                    time_varying_known_reals=["time_idx"],
                    time_varying_unknown_reals=[target_parameter],
                )

                # Validation на всем диапазоне теста (predict=False)
                validation = TimeSeriesDataSet.from_dataset(
                    training, df_well, predict=False
                )

                train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
                val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

                # Создание и обучение модели
                tft = TemporalFusionTransformer.from_dataset(
                    training,
                    learning_rate=tft_learning_rate,
                    hidden_size=tft_hidden_size,
                    attention_head_size=tft_attention_head_size,
                    dropout=tft_dropout,
                    hidden_continuous_size=tft_hidden_continuous_size,
                    loss=QuantileLoss(),
                ).float() 

                trainer = Trainer(max_epochs=tft_max_epochs, accelerator="auto")
                trainer.fit(tft, train_dataloader, val_dataloader)

                predictions = tft.predict(validation, mode="prediction")
                min_length = min(len(test['ds']), len(predictions.flatten()))
                forecast_test = pd.DataFrame({
                    'ds': test['ds'].iloc[:min_length],
                    'yhat': predictions.flatten().numpy()[:min_length]
                })

                # Вызываем модель в режиме квантилей
                quantiles_test = tft.predict(validation, mode="quantiles")

                # Приведём к удобной форме (N, Q), чтобы N совпадало с длиной forecast_test['yhat']
                if quantiles_test.ndim == 3:
                    # Если batch_size = 1, то возьмём quantiles_test[0] => форма (N, Q)
                    if quantiles_test.shape[0] == 1:
                        quantiles_test = quantiles_test[0]  # теперь форма (N, Q)
                    else:
                        # Если batch_size > 1, то придётся аккуратно объединять. Обычно batch=1 для валидации.
                        quantiles_test = quantiles_test.reshape(-1, quantiles_test.shape[2])
                        # Теперь (B*N, Q)

                # Теперь предполагаем, что quantiles_test.shape == (N, Q)
                # Где N >= min_length, Q >= 3  (например, 0.1, 0.5, 0.9)
                if quantiles_test.shape[1] < 3:
                    raise ValueError("Квантильных значений меньше трёх, не можем построить доверительный интервал.")

                lower_test = quantiles_test[:min_length, 0]
                median_test = quantiles_test[:min_length, 1]
                upper_test = quantiles_test[:min_length, 2]

                # Добавляем колонки доверительных интервалов в forecast_test
                forecast_test['yhat_lower'] = lower_test
                forecast_test['yhat'] = median_test
                forecast_test['yhat_upper'] = upper_test

                forecast_values = forecast_test['yhat'].values
                real_values, forecast_values = synchronize_forecast_and_test(test[target_parameter], forecast_values)

                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100

                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                from datetime import timedelta
                max_encoder_length = training.max_encoder_length if hasattr(training, 'max_encoder_length') else 24

                last_time_idx = df_well['time_idx'].max()
                last_date = df_well['ds'].iloc[-1]

                encoder_data = df_well[df_well['time_idx'] > last_time_idx - max_encoder_length].copy()

                future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
                future_time_indices = [last_time_idx + i for i in range(1, horizon + 1)]

                last_data = df_well[df_well['time_idx'] == last_time_idx].copy()

                decoder_rows = []
                for time_idx, date in zip(future_time_indices, future_dates):
                    new_row = last_data.copy()
                    new_row['time_idx'] = time_idx
                    new_row['ds'] = date
                    # При необходимости скорректируйте известные ковариаты
                    decoder_rows.append(new_row)

                decoder_data = pd.concat(decoder_rows, ignore_index=True)

                new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
                if 'ds' in new_prediction_data.columns:
                    new_prediction_data = new_prediction_data.drop(columns=['ds'])

                # Квантильный прогноз на будущее
                future_quantiles = tft.predict(new_prediction_data, mode="quantiles")
                print("Форма массива квантилей (будущее):", future_quantiles.shape)

                # Аналогичная логика "разворачивания" размерностей
                if future_quantiles.ndim == 3 and future_quantiles.shape[0] == 1:
                    future_quantiles = future_quantiles[0]  # -> (N, Q)

                if future_quantiles.shape[1] < 3:
                    raise ValueError("Недостаточно квантильных значений для будущего (Q < 3).")

                # N - число временных точек, которое спрогнозировано
                N = future_quantiles.shape[0]
                # На случай, если модель по каким-то причинам возвращает меньше точек, чем horizon
                n_future = min(horizon, N)

                lower_future = future_quantiles[:n_future, 0]
                median_future = future_quantiles[:n_future, 1]
                upper_future = future_quantiles[:n_future, 2]

                forecast = pd.DataFrame({
                    'ds': future_dates[:n_future],
                    'yhat': median_future,
                    'yhat_lower': lower_future,
                    'yhat_upper': upper_future
                })

                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index(drop=True)
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                if show_regressors:
                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')
                
            elif selected_model == "N-BEATS":

                # Подготовка данных
                df_well['well_id'] = str(well_name)
                df_well['time_idx'] = range(len(df_well))

                if target_parameter not in df_well.columns:
                    st.error(f"Целевой параметр '{target_parameter}' не найден в данных.")
                    st.stop()

                df_well = df_well.astype({col: 'float32' for col in df_well.select_dtypes(include=['float64']).columns})

                training = TimeSeriesDataSet(
                    df_well,
                    time_idx="time_idx",
                    target=target_parameter,
                    group_ids=["well_id"],
                    time_varying_unknown_reals=[target_parameter],
                )

                validation = TimeSeriesDataSet.from_dataset(training, df_well, predict=False)

                train_dataloader = training.to_dataloader(train=True, batch_size=128, num_workers=0)
                val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

                nbeats = NBeats.from_dataset(
                    training,
                    learning_rate=nbeats_learning_rate,
                    widths = nbeats_widths,
                    backcast_loss_ratio = nbeats_backcast_loss_ratio, 
                    log_interval = nbeats_log_interval,
                    log_val_interval = nbeats_log_val_interval,
                    weight_decay = nbeats_weight_decay,                         
                ).float()

                trainer = Trainer(max_epochs=nbeats_max_epochs, accelerator="auto", gradient_clip_val=None)
                trainer.fit(nbeats, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

                predictions = nbeats.predict(validation, mode="prediction")
                min_length = min(len(test['ds']), len(predictions.flatten()))
                forecast_test = pd.DataFrame({
                    'ds': test['ds'].iloc[:min_length].values,
                    'yhat': predictions.flatten().numpy()[:min_length]
                })

                # Попытка получить квантильные предсказания
                try:
                    quantiles_test = nbeats.predict(validation, mode="quantiles")
                    print("Форма массива квантилей (тест, N-BEATS):", quantiles_test.shape)

                    if quantiles_test.ndim == 3 and quantiles_test.shape[0] == 1:
                        quantiles_test = quantiles_test[0]  # Преобразуем форму (1, N, Q) -> (N, Q)

                    lower_test = quantiles_test[:min_length, 0]
                    median_test = quantiles_test[:min_length, 1]
                    upper_test = quantiles_test[:min_length, 2]

                    forecast_test['yhat_lower'] = lower_test
                    forecast_test['yhat_upper'] = upper_test
                except Exception as e:
                    forecast_test['yhat_lower'] = None
                    forecast_test['yhat_upper'] = None

                forecast_values = forecast_test['yhat'].values
                real_values, forecast_values = synchronize_forecast_and_test(test[target_parameter], forecast_values)

                rmse = mean_squared_error(real_values, forecast_values, squared=False)
                mae = mean_absolute_error(real_values, forecast_values)
                mape = np.mean(np.abs((real_values - forecast_values) / real_values)) * 100

                st.write(f"RMSE: {rmse:.2f}")
                st.write(f"MAE: {mae:.2f}")
                st.write(f"MAPE: {mape:.2f}%")

                from datetime import timedelta

                horizon = 30  # Горизонт прогнозирования в днях
                max_encoder_length = training.max_encoder_length if hasattr(training, 'max_encoder_length') else 24

                last_time_idx = df_well['time_idx'].max()
                last_date = df_well['ds'].iloc[-1]

                encoder_data = df_well[df_well['time_idx'] > last_time_idx - max_encoder_length].copy()

                future_dates = [last_date + timedelta(days=i) for i in range(1, horizon + 1)]
                future_time_indices = [last_time_idx + i for i in range(1, horizon + 1)]

                last_data = df_well[df_well['time_idx'] == last_time_idx].copy()

                decoder_rows = []
                for time_idx, date in zip(future_time_indices, future_dates):
                    new_row = last_data.copy()
                    new_row['time_idx'] = time_idx
                    new_row['ds'] = date
                    decoder_rows.append(new_row)

                decoder_data = pd.concat(decoder_rows, ignore_index=True)

                new_prediction_data = pd.concat([encoder_data, decoder_data], ignore_index=True)
                if 'ds' in new_prediction_data.columns:
                    new_prediction_data = new_prediction_data.drop(columns=['ds'])

                # Попытка квантильного прогноза на будущее
                try:
                    future_quantiles = nbeats.predict(new_prediction_data, mode="quantiles")
                    print("Форма массива квантилей (будущее, N-BEATS):", future_quantiles.shape)

                    if future_quantiles.ndim == 3 and future_quantiles.shape[0] == 1:
                        future_quantiles = future_quantiles[0]  # -> (N, Q)

                    N = future_quantiles.shape[0]
                    n_future = min(horizon, N)

                    lower_future = future_quantiles[:n_future, 0]
                    median_future = future_quantiles[:n_future, 1]
                    upper_future = future_quantiles[:n_future, 2]
                except Exception as e:
                    # При отсутствии квантилей, используем обычное предсказание без интервалов
                    future_predictions = nbeats.predict(new_prediction_data, mode="prediction")
                    n_future = min(horizon, len(future_predictions.flatten()))
                    median_future = future_predictions.flatten().numpy()[:n_future]
                    lower_future = np.full_like(median_future, np.nan)
                    upper_future = np.full_like(median_future, np.nan)

                n_final = len(median_future)
                forecast = pd.DataFrame({
                    'ds': future_dates[:n_final],
                    'yhat': median_future,
                    'yhat_lower': lower_future,
                    'yhat_upper': upper_future
                })

                forecast_combined = pd.concat([forecast_test, forecast], ignore_index=True)
                forecast_combined = forecast_combined.reset_index(drop=True)
                forecast_combined = forecast_combined.rename(columns={'ds': 'Дата'})

                if show_regressors:
                    for col in significant_regressors:
                        forecast_combined[col] = df_well[col].fillna(method='ffill')

        # Создаём информативные графики
        if selected_model == "Prophet":
            fig_original = go.Figure()
            fig_original.add_trace(go.Scatter(x=df_well_original.index, y=df_well_original[target_parameter], mode='lines', name='Реальные данные (Исходные)', line=dict(color=graph_color_actual, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], mode='lines', name='Прогноз на реальных данных', line=dict(color=graph_color_test, width=graph_line_width)))
            # Добавление prediction_5 и prediction_95 и закрашиваем область
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_upper'], mode='lines',
                                              line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_lower'], fill='tonexty',
                                              name='Доверительный интервал', fillcolor='rgba(255, 255, 208, 0.7)',
                                              line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз в будущее',
                           line=dict(color=graph_color, width=1)))
            # Добавление prediction_5 и prediction_95 и закрашиваем область
            fig_original.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines',
                                              line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', name='Доверительный интервал',
                           fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))

        elif selected_model == "Chronos":
            fig_original = go.Figure()
            fig_original.add_trace(go.Scatter(x=df_well_original.index, y=df_well_original[target_parameter], mode='lines', name='Реальные данные (Исходные)', line=dict(color=graph_color_actual, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], mode='lines', name='Прогноз на реальных данных', line=dict(color=graph_color_test, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз на будущее', line=dict(color=graph_color, width=1)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_lower'], fill='tonexty', fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.update_layout(
                title=f"Прогноз {target_parameter} для скважины {well_name} (Модель: Chronos. Способ заполнения: {fill_method}.)",
                xaxis_title="Дата",
                yaxis_title=target_parameter,
                height=600,
                width=1000,
                legend=dict(x=0, y=1.0),
                plot_bgcolor=background_color,
                hovermode='x',
                font=dict(size=graph_font_size),
                showlegend=True
            )
            
        elif selected_model == "LSTM":
            fig_original = go.Figure()
            fig_original.add_trace(go.Scatter(x=df_well_original.index, y=df_well_original[target_parameter], mode='lines', name='Реальные данные (Исходные)', line=dict(color=graph_color_actual, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], mode='lines', name='Прогноз на реальных данных', line=dict(color=graph_color_test, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз на будущее', line=dict(color=graph_color, width=1)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_lower'], fill='tonexty', fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.update_layout(
                title=f"Прогноз {target_parameter} для скважины {well_name} (Модель: LSTM. Способ заполнения: {fill_method}.)",
                xaxis_title="Дата",
                yaxis_title=target_parameter,
                height=600,
                width=1000,
                legend=dict(x=0, y=1.0),
                plot_bgcolor=background_color,
                hovermode='x',
                font=dict(size=graph_font_size),
                showlegend=True
            )

        elif selected_model == "TFT":
            fig_original = go.Figure()
            fig_original.add_trace(go.Scatter(x=df_well_original.index, y=df_well_original[target_parameter], mode='lines', name='Реальные данные (Исходные)', line=dict(color=graph_color_actual, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], mode='lines', name='Прогноз на реальных данных', line=dict(color=graph_color_test, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз на будущее', line=dict(color=graph_color, width=1)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_lower'], fill='tonexty', fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.update_layout(
                title=f"Прогноз {target_parameter} для скважины {well_name} (Модель: TFT. Способ заполнения: {fill_method}.)",
                xaxis_title="Дата",
                yaxis_title=target_parameter,
                height=600,
                width=1000,
                legend=dict(x=0, y=1.0),
                plot_bgcolor=background_color,
                hovermode='x',
                font=dict(size=graph_font_size),
                showlegend=True
            )
            
        elif selected_model == "N-BEATS":
            fig_original = go.Figure()
            fig_original.add_trace(go.Scatter(x=df_well_original.index, y=df_well_original[target_parameter], mode='lines', name='Реальные данные (Исходные)', line=dict(color=graph_color_actual, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat'], mode='lines', name='Прогноз на реальных данных', line=dict(color=graph_color_test, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Прогноз на будущее', line=dict(color=graph_color, width=1)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['yhat_lower'], fill='tonexty', fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.update_layout(
                title=f"Прогноз {target_parameter} для скважины {well_name} (Модель: N-BEATS. Способ заполнения: {fill_method}.)",
                xaxis_title="Дата",
                yaxis_title=target_parameter,
                height=600,
                width=1000,
                legend=dict(x=0, y=1.0),
                plot_bgcolor=background_color,
                hovermode='x',
                font=dict(size=graph_font_size),
                showlegend=True
            )
            
        else:
            fig_original = go.Figure()
            fig_original.add_trace(go.Scatter(x=df_well_original.index, y=df_well_original[target_parameter], mode='lines', name='Реальные данные (Исходные)', line=dict(color=graph_color_actual, width=graph_line_width)))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['prediction'], mode='lines', name='Прогноз на реальных данных', line=dict(color=graph_color_test, width=graph_line_width)))
            # Добавление prediction_5 и prediction_95 и закрашиваем область
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['prediction_95'], mode='lines',
                                              line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(go.Scatter(x=forecast_test['ds'], y=forecast_test['prediction_5'], fill='tonexty',
                                              name='Доверительный интервал', fillcolor='rgba(255, 255, 208, 0.7)',
                                              line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['prediction'], mode='lines', name='Прогноз в будущее',
                           line=dict(color=graph_color, width=1)))
            # Добавление prediction_5 и prediction_95 и закрашиваем область
            fig_original.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['prediction_95'], mode='lines',
                                              line=dict(color='rgba(255, 255, 255, 0)')))
            fig_original.add_trace(
                go.Scatter(x=forecast['ds'], y=forecast['prediction_5'], fill='tonexty', name='Доверительный интервал',
                           fillcolor='rgba(205, 51, 51, 0.7)', line=dict(color='rgba(255, 255, 255, 0)')))

        if parameter_change:
            last_date = df_well.index[-1]
            forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=horizon)
            changed_values = np.linspace(df_well[parameter_change].iloc[-1], df_well[parameter_change].iloc[-1] + value_change, horizon)

            df_changed = pd.DataFrame({
                'ds': forecast_dates,
                parameter_change: changed_values
            })

            fig_original.add_trace(go.Scatter(
                x=df_changed['ds'],
                y=df_changed[parameter_change],
                mode='lines',
                name=f'Изменённый {parameter_change}',
                line=dict(color='red', width=graph_line_width, dash='dash')
            ))

        if show_regressors:
            for i, col in enumerate(significant_regressors):
                fig_original.add_trace(go.Scatter(x=df_well.index, y=df_well[col], mode='lines', name=col, line=dict(width=3, dash='dash'), yaxis="y2"))

        # Настройки осей и оформления
        fig_original.update_layout(
            title=f"Прогноз {target_parameter} для скважины {well_name} (Модель: {selected_model}. Способ заполнения: {fill_method}.)",
            xaxis_title="Дата",
            yaxis_title=target_parameter,
            yaxis2=dict(
                title="Регрессоры",
                overlaying="y",
                side="right"
            ),
            height=600,
            width=1000,
            legend=dict(x=0, y=1.0),
            plot_bgcolor=background_color,
            hovermode='x',
            hoverdistance=100,
            spikedistance=-1,
            font=dict(size=graph_font_size),
            showlegend=True
        )

        # Отключаем шаблон hover и добавляем свои подписи
        fig_original.update_traces(hovertemplate=None)
        fig_original.update_traces(hovertemplate="<b>Дата:</b> %{x}<br><b>Значение:</b> %{y}<br>")

        # Добавляем график в историю прогнозов и выводим его на экран
        add_to_history(
            well_name, horizon, forecast_combined,
            forecast_test, forecast,
            selected_model, rmse, mae, mape, fig_original, 
            target_parameter, estimator_dlt, estimator_lgt, estimator_ets, estimator_ktr, global_trend_option_dlt, 
            n_bootstrap_draws_dlt, n_bootstrap_draws_lgt, n_bootstrap_draws_ets, n_bootstrap_draws_ktr, 
            regression_penalty_dlt, regression_penalty_lgt, num_steps_ktr, fill_method, seasonality_mode, 
            chronos_model_name, device_map_chronos, lstm_epochs, lstm_batch_size, lstm_n_steps, lstm_n_length,
            tft_learning_rate, tft_hidden_size, tft_attention_head_size, tft_dropout, tft_hidden_continuous_size,
            tft_max_epochs, nbeats_learning_rate, nbeats_log_interval, nbeats_log_val_interval,
            nbeats_weight_decay, nbeats_widths, nbeats_backcast_loss_ratio, nbeats_max_epochs,
        )

        # Выводим график
        st.plotly_chart(fig_original)

        # Таблица для прогноза в будущее
        st.markdown("## Прогноз:")
        st.dataframe(forecast_test)
        st.dataframe(forecast)

# Кнопка перезапуска
if st.button("Перезапустить"):
    st.rerun()

# Просмотр истории прогнозов
if st.button("Просмотреть историю прогнозов"):
    if st.session_state.history:
        st.markdown("## История прогнозов")
        for item in st.session_state.history:
            st.markdown(f"**Дата и время:** {item['datetime']}")
            st.markdown(f"**Скважина:** {item['well_name']}")
            st.markdown(f"**Горизонт прогноза:** {item['horizon']} дней")
            st.markdown(f"**Модель:** {item['model_name']}")
            if item['model_name'] == "DLT":
                st.markdown(f"**Estimator:** {item['estimator_dlt']}")
                st.markdown(f"**Global_trend_option:** {item['global_trend_option_dlt']}")
                st.markdown(f"**n bootstrap draws:** {item['n_bootstrap_draws_dlt']}")
                st.markdown(f"**Regression penalty:** {item['regression_penalty_dlt']}")
            if item['model_name'] == "LGT":
                st.markdown(f"**Estimator:** {item['estimator_lgt']}")
                st.markdown(f"**n bootstrap draws:** {item['n_bootstrap_draws_lgt']}")
                st.markdown(f"**Regression penalty:** {item['regression_penalty_lgt']}")
            if item['model_name'] == "ETS":
                st.markdown(f"**Estimator:** {item['estimator_ets']}")
                st.markdown(f"**n bootstrap draws:** {item['n_bootstrap_draws_ets']}")
            if item['model_name'] == "KTR":
                st.markdown(f"**Estimator:** {item['estimator_ktr']}")
                st.markdown(f"**n bootstrap draws:** {item['n_bootstrap_draws_ktr']}")
                st.markdown(f"**Num steps:** {item['num_steps_ktr']}")
            if item['model_name'] == "Prophet":
                st.markdown(f"**Seasonality Mode:** {item['seasonality_mode']}")
                st.markdown(f"**Yearly Seasonality:** {item.get('yearly_seasonality', True)}")
                st.markdown(f"**Weekly Seasonality:** {item.get('weekly_seasonality', True)}")
            if item['model_name'] == "Chronos":
                st.markdown(f"**Модель Chronos:** {item['chronos_model_name']}")
                st.markdown(f"**Устройство:** {item['device_map_chronos']}")            
            elif item['model_name'] == "LSTM":
                st.markdown(f"**Количество эпох:** {item.get('lstm_epochs', 20)}")
                st.markdown(f"**Размер батча:** {item.get('lstm_batch_size', 16)}")
                st.markdown(f"**Количество шагов:** {item.get('lstm_n_steps', 2)}")
                st.markdown(f"**Длина подпоследовательности:** {item.get('lstm_n_length', 7)}")
            st.markdown(f"**Способ заполнения пропусков:** {item['fill_method']}") 
            st.markdown(f"**RMSE:** {item['rmse']:.2f}")
            st.markdown(f"**MAE:** {item['mae']:.2f}")
            st.markdown(f"**MAPE:** {item['mape']:.2f}")
            st.markdown(f"**Прогнозируемый параметр:** {item['target_parameter']}")
            st.plotly_chart(item['fig_original'])
            st.dataframe(item['forecast_combined'])
    else:
        st.markdown("История прогнозов пуста.")

# Кнопка для анализа пропущенных данных
if st.button("Анализ пропущенных данных"):
    if selected_wells:
        df_well = create_well_dataframe(selected_wells[0], data_sheets, file_path)
        analyze_missing_data(df_well, target_parameter)
    else:
        st.warning("Выберите скважину(ы) для анализа пропущенных данных.")

# Просмотр исходных данных
if st.button("Просмотреть исходные данные"):

    if uploaded_file is not None:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names

        for sheet_name in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
            st.markdown(f"## Исходные данные - Лист: {sheet_name}")
            st.dataframe(df)

# Просмотр данных по выбранным скважинам
if st.button("Посмотреть данные по выбранным скважинам"):
    if uploaded_file is not None:
        st.markdown("## Данные по выбранным скважинам")
        if selected_wells:
            for well_name in selected_wells:
                df_well = create_well_dataframe(well_name, data_sheets, file_path)
                st.markdown(f"### Скважина {well_name}")
                st.dataframe(df_well)

                # Обновляем структуру данных для графика с точками, соединенными линиями
                df_well_plot = df_well.rename(columns={'ds': 'Даты'})  

                # Создаем график с точками, соединенными линиями
                fig_line = go.Figure()
                for column_name in df_well_plot.columns[1:]:
                    fig_line.add_trace(go.Scatter(x=df_well_plot['Даты'], y=df_well_plot[column_name], mode='lines', name=column_name))

                fig_line.update_layout(
                    hovermode='x unified',  
                    yaxis_title='Параметры скважины',  
                )

                st.plotly_chart(fig_line) 

                # Создаем график с распределением данных точками
                fig_scatter = go.Figure()
                for column_name in df_well_plot.columns:
                    fig_scatter.add_trace(go.Scatter(x=df_well_plot['Даты'], y=df_well_plot[column_name], mode='markers', name=column_name))

                fig_scatter.update_layout(
                    hovermode='x unified', 
                    yaxis_title='Параметры скважины',  
                )

                st.plotly_chart(fig_scatter)  

        else:
            st.markdown("Выберите скважину для просмотра данных.")
    else:
        st.markdown("Загрузите файл с данными.")
