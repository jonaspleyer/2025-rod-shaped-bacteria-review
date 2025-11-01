import matplotlib.pyplot as plt
import numpy as np


def studies_scatterplots():
    data = np.random.random((40, 2)) ** 2
    data[:, 0] *= 8
    data[:, 1] **= 2

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(data[:, 0], data[:, 1], marker="x", color="k")

    ax.set_xlabel("Studied Effects (Biology)")
    ax.set_ylabel("Parameter Estimation (Quantification)")

    fig.savefig("figures/studies-scatterplots.png")
    fig.savefig("figures/studies-scatterplots.pdf")
    fig.savefig("figures/studies-scatterplots.svg")


def studies_over_time():
    data = 1 - np.random.random(40) ** 1.7
    data *= 30
    data += 1995

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.hist(data)

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Studies")

    fig.savefig("figures/studies-over-time.png")
    fig.savefig("figures/studies-over-time.pdf")
    fig.savefig("figures/studies-over-time.svg")


if __name__ == "__main__":
    studies_scatterplots()
    studies_over_time()
