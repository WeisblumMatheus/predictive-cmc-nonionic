import numpy as np


class AmoebaRoutine:
    def __init__(self):
        self.ITMAX = 5000
        self.TINY = 1.0e-10

    def swap_array(self, a, b):
        a[:], b[:] = b.copy(), a.copy()

    def swap_integers(self, a, b):
        return b, a

    def nrerror(self, string):
        raise Exception(f'nrerror: {string}')

    def assert_eq(self, n1, n2, n3, string):
        if n1 == n2 and n2 == n3:
            return n1
        else:
            self.nrerror(f'an assert_eq failed with this tag: {string}')

    def get_max_indices(self, arr):
        return np.argmax(arr)

    def get_min_indices(self, arr):
        return np.argmin(arr)

    def amoeba(self, p, y, ftol, func, f_data):
        ndim = self.assert_eq(p.shape[1], p.shape[0] - 1, len(y) - 1, 'amoeba')
        iter = 0
        psum = np.sum(p, axis=0)

        while True:
            ilo = self.get_min_indices(y)
            ihi = self.get_max_indices(y)
            ytmp = y[ihi]
            y[ihi] = y[ilo]
            inhi = self.get_max_indices(y)
            y[ihi] = ytmp
            rtol = 2.0 * abs(y[ihi] - y[ilo]) / (abs(y[ihi]) + abs(y[ilo]) + self.TINY)

            if rtol < ftol:
                y[0], y[ilo] = self.swap_integers(y[0], y[ilo])
                self.swap_array(p[0, :], p[ilo, :])
                return

            if iter >= self.ITMAX:
                self.nrerror('ITMAX exceeded in amoeba')

            ytry = self.amotry(p, y, psum, -1.0, ihi, func, f_data)
            iter += 1

            if ytry <= y[ilo]:
                ytry = self.amotry(p, y, psum, 2.0, ihi, func, f_data)
                iter += 1
            elif ytry >= y[inhi]:
                ysave = y[ihi]
                ytry = self.amotry(p, y, psum, 0.5, ihi, func, f_data)
                iter += 1
                if ytry >= ysave:
                    p = 0.5 * (p + np.outer(np.ones((p.shape[0],)), p[ilo, :]))
                    for i in range(ndim + 1):
                        if i != ilo:
                            y[i] = func(p[i, :], f_data)
                    iter += ndim
                    psum = np.sum(p, axis=0)

    def amotry(self, p, y, psum, fac, ihi, func, f_data):
        ndim = p.shape[1]
        fac1 = (1.0 - fac) / ndim
        fac2 = fac1 - fac
        ptry = psum * fac1 - p[ihi, :] * fac2
        ytry = func(ptry, f_data)
        if ytry < y[ihi]:
            y[ihi] = ytry
            psum -= p[ihi, :]
            psum += ptry
            p[ihi, :] = ptry

        return ytry
