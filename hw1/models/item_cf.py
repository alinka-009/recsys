import numpy as np
import scipy.sparse as sp


def keep_top_k(mat, k):
    mat = mat.tocsr()
    rows = []
    cols = []
    data = []

    for row in range(mat.shape[0]):
        start, end = mat.indptr[row], mat.indptr[row + 1]
        col = mat.indices[start:end]
        val = mat.data[start:end]

        if len(val) > k:
            keep = np.argpartition(val, -k)[-k:]
            col = col[keep]
            val = val[keep]

        rows.append(np.full(len(val), row, dtype=np.int32))
        cols.append(col)
        data.append(val)

    return sp.csr_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=mat.shape,
    )


class ItemCF:
    def __init__(self, neighbors=100):
        self.neighbors = neighbors
        self.ui = None
        self.sim = None

    def fit(self, ui):
        self.ui = ui.astype(np.float32).tocsr()
        cooc = (self.ui.T @ self.ui).tocsr()

        norms = np.sqrt(cooc.diagonal())
        inv_norms = np.zeros_like(norms, dtype=np.float32)
        nonzero = norms > 0
        inv_norms[nonzero] = 1.0 / norms[nonzero]

        norm = sp.diags(inv_norms)
        sim = norm @ cooc @ norm
        sim.setdiag(0)
        sim.eliminate_zeros()

        self.sim = keep_top_k(sim, self.neighbors)
        return self

    def recommend(self, users, top_k=10, batch=256):
        users = np.asarray(users, dtype=int)
        n_items = self.ui.shape[1]

        recs = []
        for start in range(0, len(users), batch):
            batch_u = users[start : start + batch]
            hist = self.ui[batch_u]
            scores = (hist @ self.sim).toarray()

            seen_r, seen_i = hist.nonzero()
            scores[seen_r, seen_i] = -np.inf

            split = n_items - top_k
            top = np.argpartition(scores, split, axis=1)[:, split:]
            top_vals = np.take_along_axis(scores, top, axis=1)
            order = np.argsort(-top_vals, axis=1)
            recs.append(np.take_along_axis(top, order, axis=1))

        return np.vstack(recs)
