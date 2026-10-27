import numpy as np

from models.ials import IALS


class CustomIALS(IALS):
    def _update(self, targets, fixed):
        targets = targets.tocsr()
        gram = fixed.T @ fixed
        reg_eye = self.reg * np.eye(
            self.factors, dtype=np.float32
        )
        result = np.empty(
            (targets.shape[0], self.factors), dtype=np.float32
        )

        for row in range(targets.shape[0]):
            start, end = targets.indptr[row], targets.indptr[row + 1]
            idx = targets.indices[start:end]
            vals = targets.data[start:end]
            vecs = fixed[idx]

            lhs = gram + reg_eye + self.alpha * (
                vecs.T @ vecs
            )
            rhs = vecs.T @ (
                (1.0 + self.alpha) * vals
            )

            result[row] = np.linalg.solve(lhs, rhs)

        return result
