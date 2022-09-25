import numpy as np
from numpy.random import normal
from scipy.stats import gmean
from utils.etimodel import ETIModel
import matplotlib.pyplot as plt

def generate_censored_data(n_players, n_beatmaps, censor_rate):
    rng = normal(0, 1, (n_players, n_beatmaps))
    censor = np.random.uniform(0, 1, (n_players, n_beatmaps)) < censor_rate
    rng[censor] = np.nan

    return rng

model = ETIModel()

def simulate(n_players, rc_parameters, hb_parameters, ln_parameters, n_simulation=100):
    simulated_expected = []
    simulated_rc = []
    simulated_hb = []
    simulated_ln = []
    simulated_std = []
    for _ in range(n_simulation):
        rc_censored_data = generate_censored_data(n_players, *rc_parameters)
        ln_censored_data = generate_censored_data(n_players, *ln_parameters)
        hb_censored_data = generate_censored_data(n_players, *hb_parameters)
        eti_rc = model.fit(rc_censored_data).reshape(-1, 1)
        eti_hb = model.fit(hb_censored_data).reshape(-1, 1)
        eti_ln = model.fit(ln_censored_data).reshape(-1, 1)
        eti_simulated = gmean(np.hstack((eti_rc, eti_hb, eti_ln)), axis=1)
        simulated_expected.append(np.mean(eti_simulated))
        simulated_rc.append(np.mean(eti_rc))
        simulated_hb.append(np.mean(eti_hb))
        simulated_ln.append(np.mean(eti_ln))
        simulated_std.append(np.var(eti_simulated, ddof=1))
    
    return np.mean(simulated_expected), (np.mean(simulated_rc), np.mean(simulated_hb), np.mean(simulated_ln)), np.mean(simulated_std)

def simulate_distribution(n_players, rc_parameters, hb_parameters, ln_parameters):
    rc_censored_data = generate_censored_data(n_players, *rc_parameters)
    ln_censored_data = generate_censored_data(n_players, *ln_parameters)
    hb_censored_data = generate_censored_data(n_players, *hb_parameters)
    eti_rc = model.fit(rc_censored_data).reshape(-1, 1)
    eti_hb = model.fit(hb_censored_data).reshape(-1, 1)
    eti_ln = model.fit(ln_censored_data).reshape(-1, 1)
    eti_simulated = gmean(np.hstack((eti_rc, eti_hb, eti_ln)), axis=1)
    plt.hist(eti_simulated)

if __name__ == "__main__":
    expected, (rc, hb, ln), variance = simulate(200, (37, 0.7984597500726534), (19, 0.7530099759201926), (17, 0.7900669249735822), 1000)
    print(expected, np.sqrt(variance))
    print(rc, hb, ln)
    # simulate_distribution(200, (37, 0.7984597500726534), (19, 0.7530099759201926), (17, 0.7900669249735822))
    # plt.show()
