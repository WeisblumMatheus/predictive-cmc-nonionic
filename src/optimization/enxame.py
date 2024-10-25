import numpy as np


class PSO:
    def __init__(self, Npt, Niter, Npar, C1, C2, Wo, Wf, Plim, func, f_data, seed=None):
        self.Npt = Npt  # Número de partículas
        self.Niter = Niter  # Número de iterações
        self.Npar = Npar  # Número de parâmetros
        self.C1 = C1  # Peso individual
        self.C2 = C2  # Peso global
        self.Wo = Wo  # Peso de inércia inicial
        self.Wf = Wf  # Peso de inércia final
        self.Plim = np.array(Plim)  # Limites dos parâmetros
        self.func = func  # Função objetivo
        self.f_data = f_data  # Dados para a função objetivo
        np.random.seed(seed)
        self.optimize()

    def optimize(self):
        # Inicializar posições e velocidades das partículas
        P = np.random.uniform(low=self.Plim[:, 0], high=self.Plim[:, 1], size=(self.Npt, self.Npar))
        Vmax = (self.Plim[:, 1] - self.Plim[:, 0]) / 2.0
        Pvel = np.random.uniform(low=-Vmax, high=Vmax, size=(self.Npt, self.Npar))

        # Inicializar melhores posições pessoais e melhor global
        Ppt = np.copy(P)
        Potm = np.zeros(self.Npar)
        Fotm = np.inf
        Fpt = np.full(self.Npt, np.inf)

        # Loop de otimização por enxame de partículas
        for it in range(self.Niter):
            F = np.apply_along_axis(self.func, 1, P, self.f_data)  # Avaliar aptidão de cada partícula
            best_idx = np.argmin(F)

            # Atualizar melhor global se uma solução melhor for encontrada
            if F[best_idx] < Fotm:
                Fotm = F[best_idx]
                Potm = P[best_idx, :]

            # Atualizar melhores pessoais
            mask = F < Fpt
            Fpt[mask] = F[mask]
            Ppt[mask, :] = P[mask, :]

            # Atualizar peso de inércia
            W = self.Wo + (self.Wf - self.Wo) * it / (self.Niter - 1)

            # Atualizar velocidades e posições
            for i in range(self.Npt):
                r1 = np.random.random(self.Npar)
                r2 = np.random.random(self.Npar)
                Pvel[i, :] = (W * Pvel[i, :] +
                              self.C1 * r1 * (Ppt[i, :] - P[i, :]) +
                              self.C2 * r2 * (Potm - P[i, :]))

                # Limitação de velocidade
                Pvel[i, :] = np.clip(Pvel[i, :], -Vmax, Vmax)

                # Atualizar posições
                P[i, :] += Pvel[i, :]

                # Aplicar condições de contorno
                P[i, :] = np.clip(P[i, :], self.Plim[:, 0], self.Plim[:, 1])

        # Retornar a melhor solução encontrada e seu valor objetivo
        return Potm, Fotm