from typing import Dict
import sympy

import torch.nn.functional as F
import numpy as np


def mix_weights(beta):
    beta1m_cumprod = (1. - beta.double()).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


def mix_weights_numpy(beta):
    beta_cumprod = (1. - beta).cumprod(-1)

    return np.pad(beta, (0, 1), mode='constant', constant_values=(1, 1)) * np.pad(beta_cumprod, (1, 0),
                                                                                  mode='constant',
                                                                                  constant_values=(1, 1))
def prob(x, y, x_val, y_val, pdf, distrs, param_name, curr_res, params_left):
    hist, bin_edges = np.histogram(distrs[param_name])
    hist = np.divide(hist, len(distrs[param_name]))

    if len(params_left) == 0:
        for i, h in enumerate(hist):
            curr_res += h * pdf.subs([(param_name, (bin_edges[i] + bin_edges[i+1])/2),
                                      (y, y_val), (x, x_val)]).evalf()

        return curr_res

    else:
        for i, h in enumerate(hist):
            curried_pdf  = pdf.subs(param_name, (bin_edges[i] + bin_edges[i+1])/2)
            curr_res =+ h * prob(x, y, x_val, y_val, curried_pdf, distrs, params_left[0], curr_res, params_left[1:])

        return curr_res




