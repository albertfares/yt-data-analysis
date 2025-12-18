# The Hidden Web of YouTube: How Comments Connect Content and Communities
 <p align="center">
    <img width="518" height="337" alt="Capture d’écran 2025-11-05 à 14 06 08" src="https://github.com/user-attachments/assets/bc958119-fce3-4110-957b-121419a43282" />
</p>


## Abstract :page_facing_up::
YouTube's recommendation algorithm is a closely guarded secret. We seek to circumvent this "black box" by mapping the social structure of the platform from the bottom up. Using a massive dataset including 8.6 billion comments, we construct a network where channels are connected solely by the users who comment on them.

This approach ignores standard metrics such as view counts in order to reveal organic communities formed by genuine human interactions. The result is a transparent recommendation engine that prioritizes users' active interests —what makes them react— rather than passive metadata that may result from a simple trend. By shifting the focus from static categories (simple views) to dynamic user behavior, we offer a new way to discover content based on where communities are actually active.

## Research Questions

- **User Level (Behavior):** Is there any tendency in user commenting behavior? Is it possible to construct a network based on it?
- **Channel Level (Structure):** What organic structures emerge when we connect channels based on human behavior (comments) rather than algorithms? Which metric could be used to design a meaningful network based on comments?
- **Application (Recommendation):** Can we build a transparent recommendation engine based on comments that bypasses the "Rich-Get-Richer" cycle? Can we predict a user's next favorite channel simply by knowing who their "digital neighbors" are?


## Dataset :books::
Considering the size of YouNiverse, we chose not to explore another dataset.

For detailed documentation and methodology, see the original YouNiverse paper: 
[YouNiverse: Large-Scale Channel and Video Metadata from English-Speaking YouTube](https://arxiv.org/pdf/2012.10378)

The dataset is available on [Zenodo](https://zenodo.org/records/4650046).


## Methods

We chose to focus on the large-scale structure of the provided YouNiverse dataset.
We used the following methods:
- **Louvain Community Detection** (for identifying organic clusters)
- **Pointwise Mutual Information (PMI)** (for edge weighting)
- **PageRank & Betweenness Centrality** (for identifying Hubs and Bridges)
- **Interactive Visualization** (Chord Diagrams, Sankey Diagrams)

### Preprocessing & User Profiling (The Signal)
To build a robust graph, we first had to define the "Signal" amidst the noise.
- **High-Fidelity Filter:** We removed bots, spam, and "drive-by" commenters to isolate **"Super Users"**—authors who are consistently active and socially validated (receiving likes).
- **Behavioral Analysis:** We analyzed commenting tendencies to distinguish between "Tourists" (one-off visitors) and "Residents" (loyal community members), ensuring our network edges reflect genuine engagement.

### Network Construction (The Map)
We aggregated billions of interactions to reconstruct the map of YouTube.
- **The Interaction Score:** To design a meaningful network, we developed a custom edge weight: $Score = PMI \times \log(Shared Users)$. This balances **volume** (popularity) with **specificity** (PMI), identifying significant connections rather than just trending ones.
- **Topology Analysis:** We applied the **Louvain Algorithm** to detect organic clusters ("Hidden Tribes") and calculated centrality metrics to identify **Hubs** (anchors) and **Bridges** (gateways), revealing how the ecosystem organizes itself beyond official categories.

### The Recommendation Engine (The Tool)
Finally, we operationalized the network structure.
- **Proximity-Based Logic:** We built a tool that suggests channels based on **network proximity**. By locating a user within a specific behavioral cluster, the engine recommends the strongest neighboring nodes ("digital neighbors") that they haven't visited yet.
- **Value over Views:** This topology-based approach prioritizes **Appreciation** (strong social links) over raw **Views**, effectively bypassing the "Rich-Get-Richer" loop of traditional algorithms.

## Repository Structure

├── data/                                   # Data folder (not tracked by git)
│   ├── raw/                                # Original datasets
│   └── processed/                          # Cleaned data and network edge lists
├── src/                                    # Source code for analysis
│   ├── network_construction.py             # Script to build edges and calculate PMI/Scores
│   ├── community_detection.py              # Logic for clustering algorithms
│   └── visualization.py                    # Scripts for Chord diagrams and interactive plots
├── results.ipynb                           # Main Jupyter notebook presenting the findings
├── pip_requirements.txt                    # Required Python packages
└── README.md                               # Project documentation

## How to Run the Code 💻

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/epfl-ada/ada-2024-project-ooohfada.git](https://github.com/epfl-ada/ada-2024-project-ooohfada.git)
    cd ada-2024-project-ooohfada
    ```

2.  **Set up the environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r pip_requirements.txt
    ```

3.  **Data Acquisition:**
    * Download the dataset from the [ADA 2024 YouNiverse Kaggle page](https://zenodo-org.translate.goog/records/4650046) and place it in `data/raw/`.

4.  **Execution:**
    * Run the notebook `results.ipynb` to view the analysis pipeline and visualizations.

## Contributions 👥

| Team Member | Contribution Focus |
| :--- | :--- |
| **[SltMatteo](https://github.com/SltMatteo)** | **Network Construction:** Handled the crawling of edge data, implementation of the PMI/Score metric, and efficient handling of large dataframes. |
| **[Tkemper2](https://github.com/Tkemper2)** | **Visualization & Story:** Created the Chord diagrams, Gephi network exports, and led the design and implementation of the Data Story website. |
| **[albertfares](https://github.com/albertfares)** | **Algorithm & Analysis:** Implemented community detection (Louvain), defined the specific metrics for "Hubs" and "Bridges," and managed the repository structure. |
| **[jeanninhugo](https://github.com/jeanninhugo)** | **User Analysis & Report:** Working on the User-level metrics to position users within the network and synthesizing findings into the final textual report/README. |




## Acknowledgments & AI Usage :ballot_box_with_check::
**External Tools:**
- **Gephi:** Used for large-scale network visualization to validate our community detection results visually.

**AI Assistants:**
- AI coding assistants were used to assist with code implementation, debugging, data visualization, and technical documentation.
- All analytical decisions, research design, and interpretations were made by the team.
- The introductory image was created using ChatGPT.

### Contributors :busts_in_silhouette::
[SltMatteo](https://github.com/SltMatteo), [Tkemper2](https://github.com/Tkemper2), [albertfares](https://github.com/albertfares), [jeanninhugo](https://github.com/jeanninhugo), [frossardr](https://github.com/frossardr)

