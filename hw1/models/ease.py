import numpy as np
from scipy import linalg


class EASE:
    def __init__(self, reg=500.0):
        self.reg = reg

    def fit(self, ui):
        self.ui = ui.tocsr().astype(np.float32)

        gram = (self.ui.T @ self.ui).toarray()
        diag = np.arange(gram.shape[0])
        gram[diag, diag] += self.reg

        inv = linalg.inv(
            gram, overwrite_a=True, check_finite=False
        )
        self.weights = -inv / np.diag(inv)
        self.weights[diag, diag] = 0.0
        self.weights = self.weights.astype(np.float32, copy=False)
        return self

    def recommend(self, users, top_k=10, batch=256):
        users = np.asarray(users)
        recs = []

        for start in range(0, len(users), batch):
            batch_u = users[start : start + batch]
            hist = self.ui[batch_u]
            scores = hist @ self.weights
            scores = np.asarray(scores)

            rows, cols = hist.nonzero()
            scores[rows, cols] = -np.inf

            top = np.argpartition(
                scores, -top_k, axis=1
            )[:, -top_k:]
            top_vals = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_vals, axis=1)
            recs.append(np.take_along_axis(top, order, axis=1))

        return np.vstack(recs)
