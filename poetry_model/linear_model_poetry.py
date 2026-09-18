import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def main():
    x = np.array([1, 2, 3, 4, 5, 6], dtype=float).reshape(-1, 1)
    y = np.array([5, 8, 11, 14, 17, 20], dtype=float)

    model = LinearRegression()
    model.fit(x, y)

    predictions = model.predict(x)

    print("Окружение Poetry")
    print("Версия Python проверена через Poetry")
    print("Коэффициент:", model.coef_[0])
    print("Свободный член:", model.intercept_)
    print("Прогноз для x=7:", model.predict([[7]])[0])
    print("MSE:", mean_squared_error(y, predictions))
    print("R2:", r2_score(y, predictions))


if __name__ == "__main__":
    main()
