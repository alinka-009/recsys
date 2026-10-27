import numpy as np


class FM:
    def __init__(
        self,
        factors=8,
        lr=0.05,
        reg=1e-5,
        neg=1,
    ):
        self.factors_n = factors
        self.lr = lr
        self.reg = reg
        self.neg = neg

    def _features(self, users, items):
        users = np.asarray(users, dtype=np.int32)
        items = np.asarray(items, dtype=np.int32)
        n = len(users)

        idx = np.empty((n, 7), dtype=np.int32)
        idx[:, 0] = users
        idx[:, 1] = self.item_off + items
        idx[:, 2] = self.fmt_off + self.item_fmt[items]
        idx[:, 3:] = self.num_off + np.arange(4)

        vals = np.ones((n, 7), dtype=np.float32)
        vals[:, 3:5] = self.user_feat[users]
        vals[:, 5:7] = self.item_feat[items]
        return idx, vals

    def _score(self, idx, vals):
        linear = self.bias + (self.weights[idx] * vals).sum(axis=1)
        weighted = self.vecs[idx] * vals[:, :, None]
        summed = weighted.sum(axis=1)
        pairwise = 0.5 * (
            summed**2 - (weighted**2).sum(axis=1)
        ).sum(axis=1)
        return linear + pairwise

    def predict(self, users, items):
        idx, vals = self._features(users, items)
        return self._score(idx, vals)

    def fit(self, ui, item_fmt, user_feat, item_feat):
        self.ui = ui.tocsr()
        self.n_users, self.n_items = self.ui.shape
        self.item_fmt = np.asarray(item_fmt, dtype=np.int32)
        self.user_feat = np.asarray(user_feat, dtype=np.float32)
        self.item_feat = np.asarray(item_feat, dtype=np.float32)

        self.item_off = self.n_users
        self.fmt_off = self.item_off + self.n_items
        self.num_off = self.fmt_off + self.item_fmt.max() + 1
        self.n_feat = self.num_off + 4

        rng = np.random.default_rng(42)

        pos_u, pos_i = self.ui.nonzero()
        neg_u = np.repeat(pos_u, self.neg)
        neg_i = rng.integers(0, self.n_items, len(neg_u))

        seen = np.asarray(
            self.ui[neg_u, neg_i]
        ).ravel() > 0
        while seen.any():
            neg_i[seen] = rng.integers(0, self.n_items, seen.sum())
            seen = np.asarray(
                self.ui[neg_u, neg_i]
            ).ravel() > 0

        users = np.concatenate([pos_u, neg_u]).astype(np.int32)
        items = np.concatenate([pos_i, neg_i]).astype(np.int32)
        labels = np.concatenate([
            np.ones(len(pos_u), dtype=np.float32),
            np.zeros(len(neg_u), dtype=np.float32),
        ])

        self.bias = np.float32(0.0)
        self.weights = np.zeros(self.n_feat, dtype=np.float32)
        self.vecs = rng.normal(
            0, 0.01, size=(self.n_feat, self.factors_n)
        ).astype(np.float32)

        for _ in range(5):
            order = rng.permutation(len(labels))

            for start in range(0, len(labels), 2048):
                rows = order[start : start + 2048]
                idx, vals = self._features(users[rows], items[rows])
                target = labels[rows]
                pred = self._score(idx, vals)
                prob = 1 / (1 + np.exp(-np.clip(pred, -20, 20)))
                err = prob - target
                scale = 1.0 / len(rows)

                summed = (
                    self.vecs[idx] * vals[:, :, None]
                ).sum(axis=1)
                grad = err[:, None, None] * vals[:, :, None] * (
                    summed[:, None, :]
                    - self.vecs[idx] * vals[:, :, None]
                )

                self.bias -= self.lr * err.mean()
                shrink = 1.0 - self.lr * self.reg
                self.weights *= shrink
                self.vecs *= shrink
                np.add.at(
                    self.weights,
                    idx,
                    -self.lr * scale * err[:, None] * vals,
                )
                np.add.at(
                    self.vecs,
                    idx,
                    -self.lr * scale * grad,
                )

        return self

    def recommend(self, users, top_k=10, batch=32):
        users = np.asarray(users, dtype=np.int32)
        all_items = np.arange(self.n_items, dtype=np.int32)
        recs = []

        for start in range(0, len(users), batch):
            batch_u = users[start : start + batch]
            pairs_u = np.repeat(batch_u, self.n_items)
            pairs_i = np.tile(all_items, len(batch_u))
            scores = self.predict(pairs_u, pairs_i).reshape(
                len(batch_u), self.n_items
            )

            hist = self.ui[batch_u]
            rows, cols = hist.nonzero()
            scores[rows, cols] = -np.inf

            top = np.argpartition(
                scores, -top_k, axis=1
            )[:, -top_k:]
            top_vals = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_vals, axis=1)
            recs.append(np.take_along_axis(top, order, axis=1))

        return np.vstack(recs)
