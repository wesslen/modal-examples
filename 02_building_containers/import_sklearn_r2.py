import time
import modal

app = modal.App(
    "import-sklearn",
    image=modal.Image.debian_slim()
    .apt_install("libgomp1")
    .pip_install("scikit-learn"),
)

with app.image.imports():
    import numpy as np
    from sklearn import datasets, linear_model
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score

@app.function()
def fit():
    print("Inside run!")
    X, y = datasets.load_diabetes(return_X_y=True)
    X = X[:, np.newaxis, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    regr = linear_model.LinearRegression()
    regr.fit(X_train, y_train)
    predict = regr.predict(X_test)
    
    return r2_score(predict, y_test)


if __name__ == "__main__":
    t0 = time.time()
    with app.run():
        t = fit.remote()
        print("R Squared is:", t)
    print("Full time spent:", time.time() - t0)
