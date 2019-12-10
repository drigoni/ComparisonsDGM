import numpy as np
from matplotlib import pyplot as plt
from scipy import stats


dataQM9 = [('Sol.', 0.28, 0.14),
           ('SAS', 0.22, 0.23),
           ('QED', 0.46, 0.08),
           ('NP', 0.89, 0.18)
           ]

dataZINC = [('Sol.', 0.56, 0.17),
        ('SAS', 0.56, 0.23),
        ('QED', 0.73, 0.14),
        ('NP', 0.42, 0.18)
        ]


def plot(dataset, data):
    xval = []
    plt.figure()
    for i in data:
        mu = i[1]
        var = i[2]
        name = i[0]
        x = sorted(np.random.normal(mu, var, 10000))
        pdf = stats.norm.pdf(x, mu, var)
        xval.extend(x)
        plt.plot(x, pdf, '-', label=name)
        plt.fill_between(x, pdf, alpha=0.3)

    axis = plt.gca()
    # axis.set_ylim(0, 1)
    print((min(xval), max(xval)))
    axis.set_xlim(min(xval), max(xval))
    plt.legend()
    plt.savefig('./' + dataset + '/plot.png')


plot("qm9", dataQM9)
plot("zinc", dataZINC)