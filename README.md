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

We focused on the large-scale structure of the YouNiverse dataset, employing the following methods:
- **Louvain Community Detection** (for identifying organic clusters)
- **Pointwise Mutual Information (PMI)** (for edge weighting)
- **PageRank & Betweenness Centrality** (for identifying Hubs and Bridges)
- **Interactive Visualization** (Chord Diagrams, Sankey Diagrams)

### 1. Preprocessing & User Profiling (The Signal)
To build a robust graph, we defined the "Signal" by filtering for **"Super Users"** ($U_{super}$). A user $u$ is retained only if they satisfy the following engagement thresholds:

$$u \in U_{super} \iff (N_{videos}(u) \ge 24) \land (N_{likes}(u) \ge 5)$$

Where:
* $N_{videos}(u)$: The number of unique videos user $u$ commented on (ensuring consistency).
* $N_{likes}(u)$: The total likes received on their comments (ensuring social validation).
* **Bot Removal:** We strictly removed the top $1\%$ of most active accounts to eliminate non-human behavior.

### 2. Network Construction (The Map)
We aggregated billions of interactions into a graph where nodes $i$ and $j$ represent Channels (aggregated by category).

* **The Interaction Score ($W_{ij}$):**
    We developed a custom edge weight that balances **specificity** (PMI) with **volume** (raw shared count). The weight $W_{ij}$ between two channels is defined as:

    $$W_{ij} = \text{PMI}(i, j) \times \log(|U_i \cap U_j|)$$

    Where the **Pointwise Mutual Information (PMI)** is calculated as:

    $$\text{PMI}(i, j) = \log\left(\frac{P(i, j)}{P(i)P(j)}\right) = \log\left(\frac{N \cdot |U_i \cap U_j|}{|U_i| \cdot |U_j|}\right)$$

    * $|U_i \cap U_j|$: Number of shared commentators between channel $i$ and $j$.
    * $N$: Total number of users in the network.
    * **Logic:** PMI penalizes generic links between massive channels, while the $\log$ term prevents statistically high PMI values from insignificant niche channels (e.g., 2 users sharing 2 channels) from dominating the graph.

* **Topology Analysis:**
    * We applied the **Louvain Algorithm** to maximize the modularity $Q$, partitioning the network into communities $C_1, ..., C_k$ where internal density is maximized.
    * We calculated **Degree** to identify Hubs (anchors) and **Betweenness Centrality** ($C_B$) to identify Bridges.

### 3. The Recommendation Engine (The Tool)
Finally, we operationalized the network structure.
- **Proximity-Based Logic:** We built a tool that suggests channels based on **network proximity**. By locating a user within a specific behavioral cluster, the engine recommends the strongest neighboring nodes ("digital neighbors") that they haven't visited yet.
- **Value over Views:** This topology-based approach prioritizes **Appreciation** (strong social links $W_{ij}$) over raw **Views**, effectively bypassing the "Rich-Get-Richer" loop of traditional algorithms.
## Repository Structure
```
ADA-2024-PROJECT-OOOHFADA/
├── data/                                           # Data folder (will contain all the data used and created in the project)
├── extras/                                         # Extra files
│   ├── dataset_description.md                      # Description of the dataset
│   └── dataset_presenation.pdf                     # Presentation of the dataset            
├── plot_data/                                      # Folder for plots (will contain the plots generated by the scripts 
│                                                   # and used in the data story when generated by the results notebook)
├── src/                            
│   ├── data/                                       # Data processing scripts
│   │   ├── bbdataset_preprocessing.py              # Preprocessing of the bad buzz dataset
│   │   ├── dataloader_functions.py                 # Dataloader for the dataset
│   │   ├── final_dataset_creation.py               # Creation of the final dataset
│   │   ├── preprocessing.py                        # Preprocessing of the dataset
│   │   ├── reduce_metadata.py                      # Reduction of the metadata dataset
│   │   └── video_extraction.py                     # Extraction of the videos
│   ├── models/                                     # LLM related scripts
│   │   └── llm_call_helpers.py                     # LLM call helpers
│   ├── scripts/                                    # Utility scripts
│   │   └──  preprocessing_pipeline.sh              # Preprocessing pipeline script
│   ├── utils/                                      # Helper functions
│   │   ├── 1M_plus_utils.py                        # Helper functions for the 1M+ analysis
│   │   ├── find_video_categories.py                # Helper functions for the video categories analysis
│   │   ├── plots.py                                # Helper functions for the plots
│   │   ├── recovery_analysis_utils.py              # Helper functions for the recovery analysis
│   │   └── results_utils.py                        # Helper functions for the results
├── venv/                                           # Virtual environment
├── .gitignore
├── pip_requirements.txt                            # Required packages
├── README.md                                       # Project description and instructions
└── results.ipynb                                   # Jupyter notebook with the results
```
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

## Contributions

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

