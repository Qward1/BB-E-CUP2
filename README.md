# 🛡️ Система обнаружения контрафакта на маркетплейсе OZON

<div align="center">

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)
![License](https://img.shields.io/badge/license-Internal-green.svg)

*Комплексное ML-решение для автоматического выявления контрафактных товаров*

</div>

## 📋 Описание проекта

Интеллектуальная система машинного обучения для платформы OZON, использующая ансамблевый подход с глубоким анализом:
- 🔤 **Текстовых описаний** товаров
- 📊 **Метаданных** и статистики продаж  
- 👤 **Поведенческих паттернов** продавцов

---

## 🏗️ Архитектура решения

<details>
<summary><strong>1️⃣ Модуль извлечения текстовых эмбеддингов</strong> <code>extract_emb.py</code></summary>

- 🌐 Использование трансформера **XLM-RoBERTa** для мультиязычных текстов
- 🎯 Fine-tuning модели на задаче классификации контрафакта
- 🧠 Извлечение семантических представлений описаний и названий товаров

</details>

<details>
<summary><strong>2️⃣ Модуль обработки метаданных</strong> <code>feature_eng.py</code></summary>

- ⚙️ Создание **40+** инженерных признаков из статистики продаж
- 📈 CatBoost encoding категориальных переменных
- 🔍 Анализ поведенческих паттернов продавцов

</details>

<details>
<summary><strong>3️⃣ Модуль текстовых признаков</strong> <code>extract_text_feature.py</code></summary>

- 🚨 Детекция подозрительных ключевых слов
- 🌍 Анализ языковых пропорций и структуры текста
- 📊 Извлечение статистических характеристик

</details>

<details>
<summary><strong>4️⃣ Ансамблевая модель</strong> <code>pred.py</code></summary>

- 🎰 **5 CatBoost моделей** с различными random seeds
- ⚖️ Взвешенное усреднение предсказаний
- 🎯 Оптимизированный порог классификации: **0.617**

</details>

<details>
<summary><strong>5️⃣ Web-интерфейс</strong> <code>web.py</code></summary>

- 🚀 **FastAPI** сервис для загрузки и обработки данных
- 🤖 Автоматическая генерация результатов

</details>

---

## 🚀 Быстрый старт

### Требования к системе

| Компонент | Требование |
|-----------|------------|
| 🐍 Python | 3.8+ |
| 🎮 GPU | CUDA-совместимая (рекомендуется) |
| 💾 RAM | Минимум 8GB |

### Установка зависимостей

```bash
# Основные библиотеки
pip install pandas numpy scikit-learn
pip install catboost lightgbm xgboost

# NLP и трансформеры
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
pip install nltk beautifulsoup4
pip install pymorphy3 pymorphy3-dicts-ru

# Для кодирования
pip install category_encoders

# Web-сервис
pip install fastapi uvicorn python-multipart
```

---

## 📂 Структура проекта

```
project/
├── 🤖 models/                 # Предобученные модели
│   ├── model1.pkl            # CatBoost модель (seed=42)
│   ├── model2.pkl            # CatBoost модель (seed=123)
│   ├── model3.pkl            # CatBoost модель (seed=456)
│   ├── model4.pkl            # CatBoost модель (seed=789)
│   ├── model5.pkl            # CatBoost модель (seed=2024)
│   ├── catboost_encoder.pkl  # Энкодер для категориальных признаков
│   └── PCA.pkl               # Редьюсер размерности для эмбеддингов
│
├── 📚 resources/              # Fine-tuned трансформер
│   ├── tokenizer/            # Токенизатор
│   ├── encoder/              # Энкодер модели
│   ├── classifier/           # Классификационная голова
│   ├── preprocessor.pkl      # Препроцессор текста
│   └── metadata.pkl
│
├── 📤 output/                 # Результаты предсказаний
├── 📁 workdir/in/             # Временные файлы
│
├── 🎯 main.py                 # Основной пайплайн
├── 🧠 extract_emb.py          # Работа с эмбеддингами
├── ⚙️ feature_eng.py          # Инженерия признаков
├── 📝 extract_text_feature.py # Текстовые фичи
├── 🎪 pred.py                 # Ансамблевые предсказания
└── 🌐 web.py                  # Web-сервис (FastAPI)
```

---

## 💻 Использование

### 🌐 Запуск через Web-интерфейс

```bash
# Запуск сервера
uvicorn web:app --reload --host 0.0.0.0 --port 8000

# Открыть в браузере http://localhost:8000
# Загрузить CSV файл через интерфейс
# Скачать результат
```

### 💻 Запуск из командной строки

```python
from main import final

# Путь к тестовому файлу
test_file = "ml_ozon_counterfeit_test.csv"

# Запуск предсказаний
output_path = final(test_file)
print(f"Результаты сохранены в: {output_path}")
```

### 🎓 Обучение собственной модели

```python
import extract_emb as ee

# Обучение и сохранение модели эмбеддингов
ee.train_and_save_model(
    df_train=train_data,
    save_dir="./my_model",
    fine_tune_epochs=3,
    fine_tune_lr=5e-5,
    fine_tune_sample_size=10000
)

# Использование обученной модели
embeddings = ee.inference_embeddings(
    df_test=test_data,
    model_dir="./my_model"
)
```

---

## 🔧 Конфигурация

### ⚖️ Настройка порога классификации

```python
# В файле main.py
submission = pd.DataFrame({
    'id': df_test['id'],
    'prediction': preds > 0.4  # <- настройка порога
})
```

### 🎪 Настройка ансамбля

```python
# В файле pred.py
weights = {
    'catboost_seed_42': 0.2,
    'catboost_seed_123': 0.2,
    'catboost_seed_456': 0.2,
    'catboost_seed_789': 0.2,
    'catboost_seed_2024': 0.2
}
```

---

## 🔍 Ключевые особенности решения

<table>
<tr>
<td align="center">🎯</td>
<td><strong>Мультимодальный подход</strong><br>Комбинирование текстовых, числовых и категориальных признаков</td>
</tr>
<tr>
<td align="center">🎓</td>
<td><strong>Fine-tuning на целевой задаче</strong><br>Адаптация языковой модели под специфику контрафакта</td>
</tr>
<tr>
<td align="center">🛡️</td>
<td><strong>Робастность через ансамблирование</strong><br>Устойчивость к выбросам и шуму в данных</td>
</tr>
<tr>
<td align="center">🔬</td>
<td><strong>Оптимизация малых категорий</strong><br>Группировка редких брендов и категорий</td>
</tr>
<tr>
<td align="center">📈</td>
<td><strong>Анализ временных трендов</strong><br>Учет динамики продаж и возвратов</td>
</tr>
</table>

---

## 📈 Результаты

> **Оптимизированный порог классификации:** `0.617`  
> **Количество моделей в ансамбле:** `5`  
> **Инженерных признаков:** `40+`

---

## 📄 Лицензия

Проект разработан для внутреннего использования **OZON**.

---

## 👥 Команда разработки

<div align="center">

*Решение создано в рамках хакатона по обнаружению контрафакта на маркетплейсе OZON*

**Сделано с ❤️ 

</div>

---

<div align="center">

### 🌟 Если проект был полезен, поставьте звезду!

</div>
