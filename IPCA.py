import numpy as np


def kron_3d(x, y):
    assert x.shape[0] == y.shape[0]

    t = x.shape[0]
    a = x.shape[1]
    b = x.shape[2]
    n = y.shape[1]
    m = y.shape[2]

    return np.einsum('tab,tnm->tanbm', x, y).reshape(t, a * n, b * m)


class IPCA:
    def __init__(self, r, z, K, restricted=True, epsilon=1e-7):
        """
        :param r: return data. size TxNx1
        :param z: observable characteristics. size TxNxL
        :param K: dim of latent.
        :param epsilon: to prevent a singular matrix appearing
        """

        self.epsilon = epsilon

        # params
        self.r = r
        self.z = z

        # hparams
        self.N = self.r.shape[1]    # num of items
        self.K = K                  # reduced dims
        self.L = self.z.shape[2]    # original dims
        self.T = self.r.shape[0]    # time horizon used in fitting

        # should be estimated
        self.restricted = restricted
        if self.restricted:
            self.f = np.random.normal(size=(self.T, self.K, 1))   # randomly initialized.

            self.gamma_alpha = np.zeros(shape=(self.L, 1))
            self.gamma_beta = np.random.normal(size=(self.L, K))
            self.gamma = self.gamma_beta  # randomly initialized.
        else:
            self.f = np.concatenate([np.ones(shape=(self.T, 1, 1)),
                                     np.random.normal(size=(self.T, self.K, 1))],
                                     axis=1)  # randomly initialized.

            self.gamma_alpha = np.random.normal(size=(self.L, 1))
            self.gamma_beta = np.random.normal(size=(self.L, K))
            self.gamma = np.concatenate([self.gamma_alpha, self.gamma_beta],
                                        axis=1)  # randomly initialized.

        self.r_hat = self.predict()

    def predict(self):
        return self.z @ self.gamma @ self.f

    def loss(self):
        Q = self.r - self.r_hat
        QQ = np.transpose(Q, (0, 2, 1)) @ Q

        return np.sum(QQ)

    def est_f(self):
        gamma_prime = np.transpose(self.gamma, (1, 0))
        z_prime = np.transpose(self.z, (0, 2, 1))

        A = gamma_prime @ z_prime @ self.z @ self.gamma
        if self.restricted:
            B = gamma_prime @ z_prime @ self.r
        else:
            B = gamma_prime @ z_prime @ (self.r - (self.z @ self.gamma_alpha))

        eI = np.array([np.identity(A.shape[1])] * A.shape[0]) * self.epsilon
        F = np.linalg.inv(A + eI) @ B

        return F

    def est_gamma(self):
        z_prime = np.transpose(self.z, (0, 2, 1))  # T L N
        f_prime = np.transpose(self.f, (0, 2, 1))  # T 1 K

        A = kron_3d(z_prime @ self.z, self.f @ f_prime)
        B = np.transpose(kron_3d(self.z, f_prime), (0, 2, 1)) @ self.r

        A = np.sum(A, axis=0)
        B = np.sum(B, axis=0)

        eI = np.identity(A.shape[0]) * self.epsilon
        vec = np.linalg.inv(A + eI) @ B

        if self.restricted:
            gamma = np.array(np.split(vec, self.L)).reshape(self.L, self.K)
        else:
            gamma = np.array(np.split(vec, self.L)).reshape(self.L, self.K+1)

        return gamma

    def fit(self, epochs=100, verbose_interval=50):
        loss_list = []
        gamma_hat_list = []
        f_hat_list = []

        for i in range(0, epochs):
            try:
                self.gamma = self.est_gamma()
                if self.restricted:
                    self.gamma_beta = self.gamma
                else:
                    self.gamma_alpha = self.gamma[:, :1]
                    self.gamma_beta = self.gamma[:, 1:]
                self.f = self.est_f()

                self.r_hat = self.predict()
                error = self.loss()

                # save estimators
                gamma_hat_list.append(self.gamma)
                f_hat_list.append(self.f)
                loss_list.append(error)

                if i % verbose_interval == 0:
                    print('EPOCHS:', i)
                    print('error:', error)

            except np.linalg.LinAlgError:
                print('*' * 30)
                print('Singular Matrix appeared')
                print('EPOCHS:', i)
                print('error:', error)
                break

        idx = np.argmin(loss_list)
        self.gamma = gamma_hat_list[idx]
        self.f = f_hat_list[idx]

        self.r_hat = self.z @ self.gamma @ self.f

        print('*' * 30)
        print('Ideal Epochs:', idx)
        print('Ideal error:', loss_list[idx])

        return {'predict': self.r_hat,
                'loss': loss_list,
                'ideal_epochs': idx,
                'ideal_error': loss_list[idx],
                'estimators': {'gamma': self.gamma,
                               'gamma_alpha': self.gamma_alpha,
                               'gamma_beta': self.gamma_beta,
                               'f': self.f}}

    def forecast(self, z_post):
        assert z_post.shape[0] == 1  # only supports one-period forecasting
        assert z_post.shape[1:] == self.z.shape[1:]
        self.f_post = self.f[-1]
        self.r_fore = z_post @ self.gamma @ self.f_post

        return self.r_fore