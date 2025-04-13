import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Загружаем данные
@st.cache_data
def load_data():
    data = pd.read_csv("Price.csv")
    return data

data = load_data()

# Предобработка данных
data = data.dropna()
X = data[['median_income', 'total_rooms']]
y = data['median_house_value']

# Разделяем данные на тренировочные и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаем и обучаем модель
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозируем и оцениваем модель
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Заголовок приложения
st.title("Прогнозирование цен на жилье")

# Отображаем метрики модели
st.write(f"Среднеквадратичная ошибка модели: {mse}")

# Прогнозирование на основе пользовательского ввода
income = st.number_input("Введите средний доход населения:", value=3.0)
rooms = st.number_input("Введите общее количество комнат:", value=2000)

user_data = pd.DataFrame({'median_income': [income], 'total_rooms': [rooms]})
user_prediction = model.predict(user_data)

st.write(f"Прогнозируемая стоимость дома: ${user_prediction[0]:,.2f}")