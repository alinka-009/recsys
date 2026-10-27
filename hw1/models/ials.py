import numpy as np


class IALS:
    def __init__(
        self,
        factors=16,
        reg=0.1,
        alpha=20.0,
        iters=5,
        seed=42,
    ):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iters = iters
        self.seed = seed
        self.ui = None
        self.user_vecs = None
        self.item_vecs = None

    def _update(self, data, fixed):
        data = data.tocsr()
        gram = fixed.T @ fixed
        reg_eye = self.reg * np.eye(
            self.factors, dtype=np.float32
        )
        result = np.empty(
            (data.shape[0], self.factors), dtype=np.float32
        )

        for row in range(data.shape[0]):
            start, end = data.indptr[row], data.indptr[row + 1]
            idx = data.indices[start:end]
            vals = data.data[start:end]
            vecs = fixed[idx]
            conf = 1.0 + self.alpha * vals

            lhs = gram + reg_eye + (
                vecs.T * (conf - 1.0)
            ) @ vecs
            rhs = vecs.T @ conf

            result[row] = np.linalg.solve(lhs, rhs)

        return result

    def fit(self, ui):
        self.ui = ui.astype(np.float32).tocsr()
        rng = np.random.default_rng(self.seed)
        n_users, n_items = self.ui.shape
        self.user_vecs = rng.normal(
            0, 0.01, size=(n_users, self.factors)
        ).astype(np.float32)
        self.item_vecs = rng.normal(
            0, 0.01, size=(n_items, self.factors)
        ).astype(np.float32)

        iu = self.ui.T.tocsr()
        for _ in range(self.iters):
            self.user_vecs = self._update(
                self.ui, self.item_vecs
            )
            self.item_vecs = self._update(
                iu, self.user_vecs
            )

        return self

    def recommend(self, users, top_k=10, batch=256):
        users = np.asarray(users, dtype=int)
        n_items = self.item_vecs.shape[0]

        recs = []
        for start in range(0, len(users), batch):
            batch_u = users[start : start + batch]
            scores = self.user_vecs[batch_u] @ self.item_vecs.T

            hist = self.ui[batch_u]
            seen_r, seen_i = hist.nonzero()
            scores[seen_r, seen_i] = -np.inf

            split = n_items - top_k
            top = np.argpartition(scores, split, axis=1)[:, split:]
            top_vals = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_vals, axis=1)
            recs.append(np.take_along_axis(top, order, axis=1))

        return np.vstack(recs)
