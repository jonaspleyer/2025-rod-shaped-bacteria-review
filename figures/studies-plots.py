import matplotlib.pyplot as plt
import numpy as np
import requests
import bibtexparser
from matplotlib import colors


COLOR1 = "#585123"
COLOR2 = "#EEC170"
COLOR3 = "#F2A65A"
COLOR4 = "#F58549"
COLOR5 = "#772F1A"

cmap = colors.LinearSegmentedColormap.from_list(
    "mymap",
    [
        (0.00, colors.hex2color(COLOR3)),
        (0.25, colors.hex2color(COLOR2)),
        (0.50, colors.hex2color(COLOR1)),
        (0.75, colors.hex2color(COLOR4)),
        (1.00, colors.hex2color(COLOR5)),
    ],
)


def set_mpl_rc_params():
    plt.rcParams.update(
        {
            "font.family": "Courier New",  # monospace font
            "font.size": 20,
            "axes.titlesize": 20,
            "axes.labelsize": 20,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "legend.fontsize": 20,
            "figure.titlesize": 20,
        }
    )


def configure_ax(ax):
    ax.grid(True, which="major", linestyle="-", linewidth=0.75, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle="-", linewidth=0.25, alpha=0.15)
    ax.set_axisbelow(True)


def get_citation_count_crossref(doi: str) -> int | None:
    """
    Use the Crossref REST API to get the citation count.
    Note: This returns the “is-referenced-by‐count” field, which is the number of times the work is cited by other works that Crossref knows about. :contentReference[oaicite:1]{index=1}
    """
    url = f"https://api.crossref.org/works/{doi}"
    resp = requests.get(url, headers={"Accept": "application/json"})
    resp.raise_for_status()
    data = resp.json()
    count = data["message"].get("is-referenced-by-count", None)
    return count


def get_citation_count_opencitations(doi: str) -> int | None:
    """
    Use the OpenCitations REST API to get the number of incoming citations.
    Their endpoint: /citation-count/{id} where id = doi:10.xxxx/… :contentReference[oaicite:3]{index=3}
    """
    # encode doi for URL (just simple replacement of slash, etc)
    encoded = doi  # note: for DOIs with "/" this may need urllib.parse.quote
    url = f"https://opencitations.net/index/api/v2/citation-count/doi:{encoded}"
    resp = requests.get(url, headers={"Accept": "application/json"})
    resp.raise_for_status()
    data = resp.json()
    return int(data[0]["count"])


def studies_scatterplots():
    data = np.random.random((40, 2)) ** 2
    data[:, 0] *= 8
    data[:, 1] **= 2

    fig, ax = plt.subplots(figsize=(8, 8))
    configure_ax(ax)

    ax.scatter(data[:, 0], data[:, 1], marker="o", color=COLOR5)

    ax.set_xlabel("Studied Effects (Biology)")
    ax.set_ylabel("Parameter Estimation (Quantification)")

    fig.savefig("figures/studies-scatterplots.png")
    fig.savefig("figures/studies-scatterplots.pdf")
    fig.savefig("figures/studies-scatterplots.svg")


def studies_over_time(data):
    """
    Data has shape (N_entries, 3) and contains (year, citecount1, citecount2)
    """
    data1 = data[:108]
    data2 = data[108:]
    data1 = data1[data1[:, 0] >= 1980]
    data2 = data2[data2[:, 0] >= 1980]
    years1 = data1[:, 0]
    years2 = data2[:, 0]

    fig, ax = plt.subplots(figsize=(8, 8))
    configure_ax(ax)

    ax.hist(
        [years1, years2],
        np.arange(np.min(years1), np.max(years2))[::4],
        stacked=True,
        color=[COLOR1, COLOR4],
        label=["Biological", "Computational"],
    )
    ax.legend()

    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Studies")

    fig.savefig("figures/studies-over-time.png")
    fig.savefig("figures/studies-over-time.pdf")
    fig.savefig("figures/studies-over-time.svg")


def biblatex_to_json(bib_file_path: str) -> dict:
    """
    Convert a BibLaTeX (.bib) file to a JSON structure.

    Args:
        bib_file_path (str): Path to the input .bib file.
        json_file_path (str, optional): Path to save the resulting JSON.
                                        If None, returns JSON as a Python dict.
    Returns:
        dict: Parsed BibLaTeX data if json_file_path is None.
    """
    with open(bib_file_path, "r", encoding="utf-8") as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    # Convert BibLaTeX entries to structured dict
    entries = []
    for entry in bib_database.entries:
        entry_data = {
            "entry_type": entry.get("ENTRYTYPE", ""),
            "citation_key": entry.get("ID", ""),
            "fields": {k: v for k, v in entry.items() if k not in ["ENTRYTYPE", "ID"]},
        }
        entries.append(entry_data)

    return {"entries": entries}


def load_or_obtain_data(bib_json, datafile="figures/data.csv"):
    try:
        data = np.genfromtxt(datafile, delimiter=",")
        print(f"Loaded datafile {datafile}")
        return data
    except:
        print(f'No datafile "{datafile}" found. Creating datafile now.')

    data = []
    for entry in bib_json["entries"]:
        print(f"Processing {entry['citation_key']}")
        if "doi" in entry["fields"] and "year" in entry["fields"]:
            doi = entry["fields"]["doi"]
            year = entry["fields"]["year"]
            count1 = get_citation_count_crossref(doi)
            count2 = get_citation_count_opencitations(doi)
            data.append((int(year), count1, count2))
    np.savetxt(datafile, data, delimiter=",")
    return np.array(data)


if __name__ == "__main__":
    set_mpl_rc_params()

    # Example usage
    bib_json = biblatex_to_json("references.bib")
    doi = bib_json["entries"][0]["fields"]["doi"]

    data = load_or_obtain_data(bib_json)
    studies_scatterplots()
    studies_over_time(data)
