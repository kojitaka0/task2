import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    x = np.array([30, 40, 50, 60, 70, 80, 90, 100]).reshape(-1, 1)
    y = np.array([60, 80, 100, 120, 140, 160, 180, 200])

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    print("Окружение Anaconda")
    print("Тестовые значения:", y_test)
    print("Предсказания:", predictions)
    print("Коэффициент:", model.coef_[0])
    print("Свободный член:", model.intercept_)
    print("MSE:", mean_squared_error(y_test, predictions))
    print("R2:", r2_score(y_test, predictions))


if __name__ == "__main__":
    main()
