# The Hidden Web of YouTube: How Comments Connect Content and Communities
 <p align="center">
    <img width="518" height="337" alt="Capture d’écran 2025-11-05 à 14 06 08" src="https://github.com/user-attachments/assets/bc958119-fce3-4110-957b-121419a43282" />
</p>


## Abstract :page_facing_up:
YouTube's recommendation algorithm is a closely guarded secret. We seek to circumvent this "black box" by mapping the social structure of the platform from the bottom up. Using a massive dataset including 8.6 billion comments, we construct a network where channels are connected solely by the users who comment on them.

This approach ignores standard metrics such as view counts in order to reveal organic communities formed by genuine human interactions. The result is a transparent recommendation engine that prioritizes users' active interests —what makes them react— rather than passive metadata that may result from a simple trend. By shifting the focus from static categories (simple views) to dynamic user behavior, we offer a new way to discover content based on where communities are actually active.

Alongside the structural analysis of channels and videos, we also study the humans behind the network. Using streaming methods on billions of comments, we analyze how users actually behave on YouTube: how active they are, how concentrated their attention is, and wether meaningful participation patterns exist at all. 

We show that commenter behavior is not random: most users leave weak traces, while a smaller population exhibits stable and informative engagement. This behavioral backbone justifies the selection of "signal users", and ensures that the resulting content network reflects genuine community structure rather than statistical noise. 

## Research Questions :thought_balloon: 

- **User Level (Behavior):** Is there any tendency in user commenting behavior? Is it possible to construct a network based on it?
- **Content Level (Structure):** What organic structures emerge when we connect channels/videos/categories based on human behavior (comments) rather than algorithms? Which metric could be used to design a meaningful network based on comments?
- **Application (Recommendation):** Can we build a transparent recommendation engine based on comments that bypasses the "Rich-Get-Richer" cycle? Can we predict a user's next favorite channel simply by knowing who their "digital neighbors" are?


## Dataset :books:
Considering the size of YouNiverse, we chose not to explore another dataset.

For detailed documentation and methodology, see the original YouNiverse paper: 
[YouNiverse: Large-Scale Channel and Video Metadata from English-Speaking YouTube](https://arxiv.org/pdf/2012.10378)

The dataset is available on [Zenodo](https://zenodo.org/records/4650046).


## Methods :hammer_and_wrench:

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
    * We calculated **Degree** to identify Hubs and **Betweenness Centrality** ($C_B$) to identify Bridges.

### 3. The Recommendation Engine (The Tool) 
Finally, we operationalized the network structure.
- **Proximity-Based Logic:** We built a tool that suggests channels based on **network proximity**. By locating a user within a specific behavioral cluster, the engine recommends the strongest neighboring nodes ("digital neighbors") that they haven't visited yet.
- **Value over Views:** This topology-based approach prioritizes **Appreciation** (strong social links $W_{ij}$) over raw **Views**, effectively bypassing the "Rich-Get-Richer" loop of traditional algorithms.

### 4. User Behavior Analysis 
We studied wether user behavior is structured, stable and meaningful or wether it's mostly random noise. 
Using streaming computation over billions of comments, we compute for every user: 
- **Activity:** total number of comments
- **Breadth:** number of distinct channels interacted with
- **Focus/Concentration:** wether engagement is centered on a few channels or widely scattered

### 5. Participation Regimes & Signal Extraction 
From this analysis we identify **Participation Regimes**, separating users according to the strength and stability of their behavior: 
- casual, low-signal users
- moderately engaged users
- highly committed users with stable preferences

Only the later categories provide enough behavioral evidences to meaningfully infer relationships between channels. Thus justifies filtering not as arbitrary tresholds, but as an empirically grounded decision supported by the data.

### 6. Diagnostics & Robustness
To ensure that our framework truly reflects human behavior rather than artifacts, we perform a series of diagnostic checks:

- **Noise and Singletons:** evaluate the influence of users appearing only once or in extremely sparse contexts
- **Stability Checks:** verify that behavioral summaries remain consistent under thresholding
- **Distributional Sanity Checks:** ensure results match expected large-scale engagement behavior

These diagnostics confirm that the network we build is supported by reliable human signal rather than statistical chance, reinforcing the robustness of the final model.

## Data Story
Dive into the visual side of our analysis. Our [data story](https://epfl-ada.github.io/ada-2025-project-radatouille/) moves beyond the code to visualize the full network of 449 million users, featuring interactive chord diagrams and a deep dive into the "Hubs" and "Bridges" that define the platform. It also shows you how user behavior emerges from the chaotic sea of YouTube comments, featuring graphs about user profiles and their characterization. 

## Repository Structure :file_folder:
```
ada-2025-project-radatouille/
├── data/
│   ├── raw/
│   ├── models/                           # network modeling dataset, with additional values than juste filtered files
│   └── filtered/                             # first filtering files
│
├── utils/                                         
│   ├── __init__.py                      
│   ├── network_helper.py       #utils methods file for VIDEO-level part
│   └── community_helper.py     #utils methods file for USER-level part
│
├── web/
│                                           
├── .gitignore
├── README.md                                       # Project description and instructions
├── requirements.txt                           
├── results_content_network.ipynb                   # Jupyter notebook with the results for USER-level part
└── results_user_community.ipynb                    # Jupyter notebook with the results for VIDEO-level part
```
## How to Run the Code :computer:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/epfl-ada/ada-2025-project-radatouille.git
    cd ada-2025-project-radatouille
    ```

2.  **Set up the environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Data Acquisition:**
    * Download the dataset from the [Zenodo](https://zenodo.org/records/4650046) and place it in `data/raw/`.

4.  **Execution:**
    * Run the `results_[...].ipynb` notebooks to view the analysis pipeline and visualizations.
    * :warning:The notebook is long, and some cells may take a long time and be memory-intensive to run.

## Contributions :memo:

| Team Member | Contribution Focus |
| :--- | :--- |
| **[Romain](https://github.com/frossardr)** | **Network Construction:** Handled the crawling of edge data, implementation of the PMI/Score metric, and efficient handling of large dataframes. |
| **[Albert](https://github.com/albertfares)** | **Visualization & Story:** Created the Chord diagrams, Gephi network exports, and led the design and implementation of the Data Story website. |
| **[Hugo](https://github.com/jeanninhugo)** | **Algorithm & Analysis:** Implemented community detection (Louvain), defined the specific metrics for "Hubs" and "Bridges," and managed the repository structure. |
| **[Thomas](https://github.com/Tkemper2)** | **User Analysis & Report:** Built the user-level analysis pipeline, created and analyzed the user space and clustered the groups into communities |
| **[Matteo](https://github.com/SltMatteo)** | **User Analysis & Report:**  Developed user-level diagnostics, demonstrated the existence of meaningful behavioral structure and applied filtering strategies |




## Acknowledgments & AI Usage :ballot_box_with_check:
- AI coding assistants were used to assist with code implementation, debugging, data visualization, and technical documentation.
- All analytical decisions, research design, and interpretations were made by the team.
- The introductory image was created using ChatGPT.

### Contributors :busts_in_silhouette:
[SltMatteo](https://github.com/SltMatteo), [Tkemper2](https://github.com/Tkemper2), [albertfares](https://github.com/albertfares), [jeanninhugo](https://github.com/jeanninhugo), [frossardr](https://github.com/frossardr)

