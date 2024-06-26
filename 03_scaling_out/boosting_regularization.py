import io
import os

import modal

app = modal.App(
    "example-boosting-regularization",
    image=modal.Image.debian_slim()
    .pip_install("scikit-learn~=1.2.2")
    .pip_install("matplotlib~=3.9.0"),
)


@app.function()
def fit_boosting(n):
    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn import datasets, ensemble
    from sklearn.metrics import log_loss
    from sklearn.model_selection import train_test_split

    X, y = datasets.make_hastie_10_2(n_samples=n, random_state=1)

    # map labels from {-1, 1} to {0, 1}
    labels, y = np.unique(y, return_inverse=True)

    # note change from 0.8 to 0.2 test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    original_params = {
        "n_estimators": 500,
        "max_leaf_nodes": 4,
        "max_depth": None,
        "random_state": 2,
        "min_samples_split": 5,
    }

    plt.figure()

    for label, color, setting in [
        ("No shrinkage", "orange", {"learning_rate": 1.0, "subsample": 1.0}),
        ("learning_rate=0.2", "turquoise", {"learning_rate": 0.2, "subsample": 1.0}),
        ("subsample=0.5", "blue", {"learning_rate": 1.0, "subsample": 0.5}),
        (
            "learning_rate=0.2, subsample=0.5",
            "gray",
            {"learning_rate": 0.2, "subsample": 0.5},
        ),
        (
            "learning_rate=0.2, max_features=2",
            "magenta",
            {"learning_rate": 0.2, "max_features": 2},
        ),
    ]:
        params = dict(original_params)
        params.update(setting)

        clf = ensemble.GradientBoostingClassifier(**params)
        clf.fit(X_train, y_train)

        # compute test set deviance
        test_deviance = np.zeros((params["n_estimators"],), dtype=np.float64)

        for i, y_proba in enumerate(clf.staged_predict_proba(X_test)):
            test_deviance[i] = 2 * log_loss(y_test, y_proba[:, 1])

        plt.plot(
            (np.arange(test_deviance.shape[0]) + 1)[::5],
            test_deviance[::5],
            "-",
            color=color,
            label=label,
        )

    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Test Set Deviance")

        # Dump the chart to .png and return the bytes
    with io.BytesIO() as buf:
        plt.savefig(buf, format="png", dpi=300)
        return buf.getvalue()

OUTPUT_DIR = "/tmp/modal"


@app.local_entrypoint()
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for n in [1000,5000,10000,20000,50000]:
        plot = fit_boosting.remote(n)
        filename = os.path.join(OUTPUT_DIR, f"boosting_{n}.png")
        print(f"saving data to {filename}")
        with open(filename, "wb") as f:
            f.write(plot)