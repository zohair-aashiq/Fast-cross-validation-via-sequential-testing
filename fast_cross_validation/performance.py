import statsmodels.stats.api as ap
import numpy as np
import scipy.stats as st


class Performance:

    def similar_performance(ts, alpha):
        """
        :param ts:
        :param alpha:
        :return:
        """

        pw = ap.cochrans_q(ts)
        return pw[1]

    def select_winner(ps, isActive, wstop, s):
        """
        :param isActive:
        :param wstop:
        :param s:
        :return:
        """
        Rx = []
        p = ps.shape
        Rs = np.empty(p)
        Rs[:] = np.NAN
        print("isActive[c]", isActive)
        for i in range(0, p[1]):
            Rs[:, i] = st.rankdata(ps[:, i])  # Gather the rank of c in step i
        print("Rs", Rs)
        Ms = np.zeros(p[0])
        print("S", s)
        for c in range(p[0]):
            if isActive[c] == 1:
                Rx = Rs[c, s - wstop + 1:s]
                print("Rx", Rx)
                Rx = sum(Rx)
                print("Rx", Rx)
                Ms[c] = Rx / wstop  # Mean rank for the last wstop steps
        print("Ms", Ms)
        return np.argmax(Ms)  # Return configuration with minimal mean rank
