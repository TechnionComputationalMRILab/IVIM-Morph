import numpy as np
from scipy.optimize import curve_fit

class IVIM_LS():
    def __init__(self,b_vector, bounds):
        '''
        :param b_vector:
        :param bounds:
        All fit method return D,f,DStar,s0 with same length as input signal.
        fitting done pixel by pixel. If the optimization fail ffor some pixels, D,f, DStar and s0 will set to np.nan in
        the corresponding index
        '''
        self.b_vector = b_vector
        self.bounds = bounds #[D,f,Ds,s0]
        eps = 10e-3
        self.D_factor = 1 / np.sqrt(bounds[1][0] - bounds[0][0] + eps)
        self.f_factor = 1 / np.sqrt(bounds[1][1] - bounds[0][1] + eps)
        self.DStar_factor  =  1 / np.sqrt(bounds[1][2] - bounds[0][2]+ eps)
        self.s0_factor = 1 / np.sqrt(bounds[1][3] - bounds[0][3] + eps)
        self.p0 = [(bounds[0][0] + bounds[1][0]) / 2,
                   (bounds[0][1] + bounds[1][1]) / 2,
                   (bounds[0][2] + bounds[1][2]) / 2,
                   (bounds[0][3] + bounds[1][3]) / 2]
        self.bounds_factor = ([bounds[0][0] * self.D_factor,
                          bounds[0][1]*self.f_factor,
                          bounds[0][2]*self.DStar_factor,
                          bounds[0][3] * self.s0_factor],
                         [bounds[1][0] * self.D_factor,
                          bounds[1][1]*self.f_factor,
                          bounds[1][2]*self.DStar_factor,
                          bounds[1][3] * self.s0_factor])
    def ivimN(self, b_vector, D, f,DStar, s0):
        # IVIM function with equal variance in the different IVIM parameters
        # D - fitted D (D_hat)
        # s0 - fitted s0 (s0_hat)
        D = D / self.D_factor
        f = f / self.f_factor
        DStar = DStar / self.DStar_factor
        s0 = s0 / self.s0_factor
        return s0 * (f * np.exp(-b_vector * (D + DStar)) + (1 - f) * np.exp(-b_vector * D))

    def fit_least_squares_lm(self, si):
        '''

        :param b_vector:
        :param si: (Num of pixels, b_val)
        :param p0:
        :param D_factor:
        :param DStar_factor:
        :param f_factor:
        :param s0_factor:
        :return:
        '''
        # the algorithm uses the Levenberg-Marquardt algorithm
        p0 = self.p0
        p0[0] *= self.D_factor
        p0[1] *= self.f_factor
        p0[2] *= self.DStar_factor
        p0[3] *= self.s0_factor

        D = np.array([])
        f = np.array([])
        DStar = np.array([])
        s0 = np.array([])

        N = si.shape[0]
        for i in range(N):
            s = np.squeeze(si[i, :])
            try:
                params,_ = curve_fit(self.ivimN, self.b_vector, s, p0, maxfev=2000)
                D = np.append(D, params[0] / self.D_factor)
                f = np.append(f, params[1] / self.f_factor)
                DStar = np.append(DStar, params[2] / self.DStar_factor)
                s0 = np.append(s0, params[3] / self.s0_factor)
            except:
                print(f"{i} lm fit failed. removing sample from DS")
                D = np.append(D, 0)
                f = np.append(f, 0)
                DStar = np.append(DStar, 0)
                s0 = np.append(s0, 0)

        return D, f, DStar, s0

    def fit_least_squares_trf(self, si):
        # the algorithm uses the Trust Region Reflective algorithm
        p0 = self.p0
        p0[0] *= self.D_factor
        p0[1] *= self.f_factor
        p0[2] *= self.DStar_factor
        p0[3] *= self.s0_factor

        D = np.array([])
        f = np.array([])
        DStar = np.array([])
        s0 = np.array([])

        N = si.shape[0]
        for i in range(N):
            s = np.squeeze(si[i, :])
            try:
                params,_ = curve_fit(self.ivimN, self.b_vector, s, p0, bounds=self.bounds_factor, maxfev=4000)
                D = np.append(D, params[0] / self.D_factor)
                f = np.append(f, params[1] / self.f_factor)
                DStar = np.append(DStar, params[2] / self.DStar_factor)
                s0 = np.append(s0, params[3] / self.s0_factor)
            except:
                print(f"{i} trf fit failed")
                D = np.append(D, 0)
                f = np.append(f, 0)
                DStar = np.append(DStar, 0)
                s0 = np.append(s0, 0)
        return D,f, DStar, s0

    def IVIM_fit_sls(self, si, min_bval_high=200):
        # for one smaple
        # first estimate D,S0 for mono-exp:

        s_high = si[:, self.b_vector >= min_bval_high]
        b_vector_high = self.b_vector[self.b_vector >= min_bval_high]
        s0_diffusion, D = fitMonoExpModel(s_high.T, b_vector_high)

        s0 = si[:, 0]
        f = (s0 - s0_diffusion) / (s0+1e-6)
        # find D* by NLLS of IVIM:
        bounds_Ds = (self.bounds[0][2], self.bounds[1][2])
        p0_Ds = self.p0[2]

        DStar = np.array([])
        N = si.shape[0]
        for i in range(N):
            s = np.squeeze(si[i, :])
            try:
                params, _ = curve_fit(lambda b, DStar: s0[i] * (
                        f[i] * np.exp(-self.b_vector * (D[i] + DStar)) + (1 - f[i]) * np.exp(-self.b_vector * D[i])), self.b_vector, s,
                                      p0=p0_Ds, bounds=bounds_Ds)
                DStar = np.append(DStar, params[0])
            except:
                print(f"{i} sls fit failed. removing sample from DS")
                DStar = np.append(DStar, np.nan)

        return D, f,DStar, s0

    def IVIM_fit_sls_trf(self, si, eps=0.00001, min_bval_high=200):
        D_sls, f_sls,DStar_sls,s0_sls = self.IVIM_fit_sls(si, min_bval_high)

        D = np.array([])
        DStar = np.array([])
        f = np.array([])
        s0 = np.array([])

        N =  si.shape[0]
        for i in range(N):
            p0_old = self.p0
            p0 = [D_sls[i], f_sls[i],DStar_sls[i], s0_sls[i]]
            # if p0 out of bound, take extrime bound as p0:
            sls_bounds = np.array(self.bounds)
            for j in range(len(p0)):
                if p0[j] < sls_bounds[0, j]:
                    p0[j] = sls_bounds[0, j] + sls_bounds[0, j] * eps
                elif p0[j] > sls_bounds[1, j]:
                    p0[j] = sls_bounds[1, j] - sls_bounds[0, j] * eps
            self.p0 = p0

            try:
                D_fit, f_fit, DStar_fit, s0_fit = self.fit_least_squares_trf(si[None, i, :])
                D = np.append(D, D_fit)
                f = np.append(f, f_fit)
                DStar = np.append(DStar, DStar_fit)
                s0 = np.append(s0, s0_fit)
            except:
                print('using original p0')
                self.p0 = p0_old
                D_fit, f_fit, DStar_fit, s0_fit = self.fit_least_squares_trf(si[None, i, :])
                D = np.append(D, D_fit)
                f = np.append(f, f_fit)
                DStar = np.append(DStar, DStar_fit)
                s0 = np.append(s0, s0_fit)

        return D, f,DStar, s0

    def IVIM_fit_sls_lm(self, si, eps=0.00001, min_bval_high=200):
        D_sls, f_sls,DStar_sls,s0_sls = self.IVIM_fit_sls(si, min_bval_high)

        D = np.array([])
        DStar = np.array([])
        f = np.array([])
        s0 = np.array([])

        N =  si.shape[0]
        for i in range(N):
            p0 = [D_sls[i], f_sls[i],DStar_sls[i], s0_sls[i]]
            # if p0 out of bound, take extrime bound as p0:
            sls_bounds = np.array(self.bounds)
            for j in range(len(p0)):
                if p0[j] < sls_bounds[0, j]:
                    p0[j] = sls_bounds[0, j] + sls_bounds[0, j] * eps
                elif p0[j] > sls_bounds[1, j]:
                    p0[j] = sls_bounds[1, j] - sls_bounds[0, j] * eps
            self.p0 = p0

            try:
                D_fit, f_fit, DStar_fit, s0_fit = self.fit_least_squares_lm(si[None, i, :])
                D = np.append(D, D_fit)
                f = np.append(f, f_fit)
                DStar = np.append(DStar, DStar_fit)
                s0 = np.append(s0, s0_fit)
            except:
                print('sls-lm failed')
                D = np.append(D, np.nan)
                f = np.append(f, np.nan)
                DStar = np.append(DStar, np.nan)
                s0 = np.append(s0, np.nan)

        return D, f,DStar, s0

def fitMonoExpModel(s, b_vector):
    A = np.matrix(
        np.concatenate((np.ones((len(b_vector), 1), order='C'), -np.reshape(b_vector, (len(b_vector), 1))), axis=1))
    # s = np.log(s)
    # s[np.isinf(s)] = 0.0
    s = np.where(s == 0, 1e-6, s)

    s = np.log(s)
    x = np.linalg.lstsq(A, s, rcond=None)
    # x =  np.linalg.inv(A.T @ A) @ A.T @ s
    s0 = np.exp(x[0][0])
    ADC = x[0][1]

    return s0, ADC


def parmNRMSE(org_param, fit_param):
    NanId = np.isnan(fit_param)
    fit_param = fit_param[~NanId]
    org_param = org_param[~NanId]

    return (np.linalg.norm(org_param - fit_param) / np.sqrt(len(org_param))) / np.mean(org_param)

def IVIM_model(b_vector, D, f,DStar, s0):
    si = s0 * (f * np.exp(-b_vector * (D + DStar)) + (1 - f) * np.exp(-b_vector * D))
    return si