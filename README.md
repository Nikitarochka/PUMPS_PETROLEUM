# PUMPS_PETROLEUM
Репозиторий для хранения проекта по оптимизации режимов работы скважин
# Прогноз параметров скважины

Веб-приложение для прогнозирования параметров скважины с использованием нескольких моделей временных рядов, построенное на `Streamlit`.

## Описание
Приложение предоставляет возможность загружать данные, настраивать параметры модели, запускать оптимизацию гиперпараметров с использованием библиотеки `optuna` и визуализировать результаты прогноза.

## Установка

### Зависимости
Перед началом работы необходимо установить все зависимости, используемые в проекте. Они включают:

- `pandas` для работы с данными
- `orbit-ml` для построения моделей временных рядов (DLT, LGT, ETS, KTR)
- `chronos` для построения моделей временных рядов (Amazon Chronos)
- `keras` для построения моделей временных рядов (LSTM)
- `pytorch_forecasting` для построения моделей временных рядов (TFT, N-BEATS)
- `scikit-learn` для вычисления метрик точности
- `numpy` и `scipy` для научных вычислений
- `plotly` для построения графиков
- `streamlit` для создания интерфейса
- `optuna` для оптимизации гиперпараметров

### Установка зависимостей
Для установки зависимостей выполните следующие команды:
pip install pandas orbit-ml chronos keras pytorch_forecasting scikit-learn numpy scipy plotly streamlit optuna

## Запуск
Для запуска приложения вам потребуется открыть консоль разработчика. Сочетание клавиш win + R, ввести cmd, после установить все необходимые зависимости в окружении.


### Открытие веб-интерфейса Streamlit
В открытой консоли вбиваем команду streamlit run "Абсолютный путь к файлу на вашем компьютере". После этого веб-интерфейс откроется в вашем браузере.

### При загрузке файл он должен быть строго в формате Дата - 1 столбец, Cкважины - 1 строка, не считая ячейки A1. Количество листов (параметров скважины - не ограничено).
