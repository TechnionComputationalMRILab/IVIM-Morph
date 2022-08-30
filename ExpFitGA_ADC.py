import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from scipy.stats import pearsonr, spearmanr

from config import config

config = config()
case_list_valid = config['case_list_valid']


# case_list_blow_25 = config['case_list_below_25']
# case_list_above_25 = config['case_list_above_25']


# def monoExp(GA, A, B):
# return A*(1- np.exp(-GA * B))

def monoExp(GA, A, B, C):
    return A - B * np.exp(-GA * C)


def ADC_GA_corr(path, fit):
    for list, name in [
        (case_list_valid, 'All')]:  # , (case_list_blow_25, 'below 25'), (case_list_above_25, 'above 25')]:
        GA_vector = np.array([fit[case]['GA'] for case in list])
        ADC_IRLS = np.array([fit[case]['ADC_IRLS_AVG'] for case in list])
        R_2_vector = np.array([fit[case]['R2_AVG'] for case in list])

        ind = np.argsort(GA_vector)
        GA_vector = np.take_along_axis(GA_vector, ind, 0)
        ADC_IRLS = np.take_along_axis(ADC_IRLS, ind, 0)
        R_2_vector = np.take_along_axis(R_2_vector, ind, 0)

        pears_corr_IRLS, _ = pearsonr(GA_vector, ADC_IRLS)
        spear_corr_IRLS, _ = spearmanr(GA_vector, ADC_IRLS)
        print(
            f'{name}: ADC fit by IRLS.\n Pearson’s Correlation = {pears_corr_IRLS:.2f}\n Spearman’s Correlation = {spear_corr_IRLS:.2f}')

        if not os.path.exists(path):
            os.makedirs(path)
        if name == 'All':
            p0 = (0.003, 0.005, 0.1)
            params, cv = scipy.optimize.curve_fit(monoExp, GA_vector, ADC_IRLS, p0)
            A, B, C = params

            # p0 = (2.63*10**(-3), 0.21)
            # params, cv = scipy.optimize.curve_fit(monoExp, GA_vector, ADC_IRLS, p0)
            # A, B = params

            squaredDiffs = np.square(ADC_IRLS - monoExp(GA_vector, A, B, C))
            # squaredDiffs = np.square(ADC_IRLS - monoExp(GA_vector, A, B))
            squaredDiffsFromMean = np.square(ADC_IRLS - np.mean(ADC_IRLS))
            rSquared_all = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
            print(f"R²-{name} = {rSquared_all:.2f}")
            print(f'{A=}, {B=}, {C=}')

            fig, ax = plt.subplots()
            sc = ax.scatter(GA_vector, ADC_IRLS, c=R_2_vector, label='Cases',
                            cmap='Blues', vmin=0.65, vmax=1)
            plt.plot(GA_vector, monoExp(GA_vector, A, B, C), label="fit curve")
            # plt.plot(GA_vector, monoExp(GA_vector, A, B), label="fit curve")

            plt.ylim(0.001, 0.0045)
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ax.set_aspect(abs(x1 - x0) / abs(y1 - y0))
            ax.grid(b=True, which='major', color='k', linestyle='--')
            plt.xlabel('Gestational age in weeks')
            plt.ylabel('ADC $(mm^2/s)$')
            plt.title(f'Avereged ADC in Lung vs. Gestational age\n $R^2$ = {rSquared_all:.2f}')
            plt.colorbar(sc)
            plt.savefig(os.path.join(path, f'ADC_GA_RegEst_{name}.pdf'), format='pdf')
            plt.close(fig)
    return rSquared_all
