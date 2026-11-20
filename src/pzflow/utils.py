"""Repeated PDF sampling, PIT, and central-interval coverage calculations."""
import numpy as np


class PDFSampler:
    """Continuous piecewise-linear PDF sampler; cache CDFs once per parent."""
    def __init__(self, grid, pdf):
        self.grid = np.asarray(grid, dtype=float)
        if np.any(np.diff(self.grid) <= 0):
            raise ValueError("Redshift grid must increase strictly")
        self.p = np.asarray(pdf, dtype=np.float32).copy()
        self.c = np.empty_like(self.p)
        for start in range(0, len(pdf), 2048):
            p = self.p[start:start+2048]
            if not np.isfinite(p).all() or (p < 0).any():
                raise ValueError("PDFs must be finite and nonnegative")
            mass = (p[:, :-1] + p[:, 1:]) * np.diff(self.grid) / 2
            c = np.c_[np.zeros(len(p)), np.cumsum(mass, axis=1)]
            norm = c[:, -1]
            if (norm <= 0).any():
                raise ValueError("Empty posterior")
            self.p[start:start+len(p)] /= norm[:, None]
            self.c[start:start+len(p)] = c / norm[:, None]

    def draw(self, indices, rng, n_samples=1):
        indices = np.asarray(indices)[:, None]
        u = rng.random((len(indices), n_samples))
        lo = np.zeros(u.shape, dtype=int)
        hi = np.full(u.shape, len(self.grid)-1, dtype=int)
        for _ in range(int(np.ceil(np.log2(len(self.grid))))+1):
            mid = (lo+hi)//2
            right = self.c[indices, mid] <= u
            lo, hi = np.where(right, mid, lo), np.where(right, hi, mid)
        j = np.minimum(lo, len(self.grid)-2)
        a = self.p[indices, j]
        slope = (self.p[indices, j+1]-a) / np.diff(self.grid)[j]
        mass = np.maximum(u-self.c[indices, j], 0)
        den = a + np.sqrt(np.maximum(a*a+2*slope*mass, 0))
        dt = np.divide(2*mass, den, out=np.zeros_like(mass), where=den > 0)
        z = (self.grid[j]+np.clip(dt, 0, np.diff(self.grid)[j])).astype(np.float32)
        return z[:, 0] if n_samples == 1 else z

    def pit(self, z, indices=None):
        z = np.asarray(z)
        row = np.arange(len(z)) if indices is None else np.asarray(indices)
        j = np.clip(np.searchsorted(self.grid, z, side="right")-1, 0, len(self.grid)-2)
        dt = np.clip(z, self.grid[0], self.grid[-1])-self.grid[j]
        a = self.p[row, j]
        slope = (self.p[row, j+1]-a) / np.diff(self.grid)[j]
        return np.clip(self.c[row, j]+a*dt+.5*slope*dt**2, 0, 1)


def coverage(pit, levels=(.68, .95)):
    """Empirical coverage of equal-tailed intervals from PIT values."""
    rank = np.abs(2*np.asarray(pit)-1)
    return np.array([np.mean(rank <= level) for level in np.atleast_1d(levels)])
