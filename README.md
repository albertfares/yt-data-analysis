# The Hidden Web of YouTube: How Comments Connect Content and Communities
 <p align="center">
    <img width="518" height="337" alt="Capture d’écran 2025-11-05 à 14 06 08" src="https://github.com/user-attachments/assets/bc958119-fce3-4110-957b-121419a43282" />
</p>



## Abstract :page_facing_up::
YouTube hosts billions of videos, but how do audiences navigate this vast landscape? This project maps the hidden social structures connecting content and communities by analyzing 8.6 billion comments from 449 million users across 72.9 million videos (2016-2019). We construct an audience network where videos connect through shared commenters, revealing how content clusters emerge from viewing behavior rather than official categories. By detecting communities at multiple scales, we examine how these groups interact, overlap, and evolve over time. We investigate what characterizes community members, which videos serve as bridges between different audience clusters, and how engagement quality relates to network position. Since we cannot access YouTube's recommendation algorithm, we infer content relationships directly from audience behavior, providing a bottom-up view of the platform's social structure. Our findings reveal the organic organization of YouTube's ecosystem and offer insights into how communities form and connect across this massive platform.

## Research Questions :thought_balloon:: 
### User Communities (User level):
- How to define and create communities of users from comment behavior?
- Which channels are popular within each community and are they shared between different communities?
- What characterizes members of different communities: comment frequency, content diversity, channel loyalty, engagement intensity, new metrics?

### Content Network (Video level):
- Can we construct a meaningful video network from comment behavior and overlapping users, and how does it compare to self-defined categories?
- Which videos serve as "pathways" connecting different parts of the network?
- How do user communities behave inside the video network?

### Engagement Dynamics (Comments impact on channel) (Additional):
- To what extent do comment volume and quality correlate with channel stats in subscribers and views?
- Do highly engaged comment sections indicate stronger audience loyalty than channels with passive audiences?


## Dataset :books::
Considering the size of YouNiverse, we chose not to explore another dataset.

For detailed documentation and methodology, see the original YouNiverse paper: 
[YouNiverse: Large-Scale Channel and Video Metadata from English-Speaking YouTube](https://arxiv.org/pdf/2012.10378)

The dataset is available on [Zenodo](https://zenodo.org/records/4650046).

## Methods :hammer_and_wrench:: 
The provided dataset is already cleaned. However, to further reduce noise and improve computational efficiency while preserving meaningful structures (i.e. communities and network), we decided to apply additional filtering steps, as described below.

### User Communities (User level):
For this part our aim is to explore community dynamics on a user level. We want to do in order to discuss our research question: 
- Filter our user dataset to lower noise. Indeed, we want to spot patterns based on where, how and how much some users post comments. But we noticed that a lot of users comment very little. In order to achieve meaningful results, and because the dataset is huge, we keep only users that have commented more than a given threshold.
- Build a modified table that maps comment authors to the channel they commented on and the number of comments they posted. This serves as the foundational dataset for all subsequent user-level analyses.
- Choose and extract meaningful features that describe user commenting profiles. 
- Apply similarity measure and clustering techniques (KNN with varying k, DBScan, HDBScan, Louvain) to see what type of communities of similar users emerge.
- Characterize and visualize clusters; from the emerging communities from the earlier point, we analyze their behavior with feature analysis and visualize the results (PCA to grpah clusters, etc)
- Apply all the aforementionned methods to subsets of the original dataset and progressively scale it in optimized pipelines.

### Content Network (Video level):
Here are the different steps we are considering to explore the video level: 
- Threshold users with a minimum number of videos commented and videos with a minimum number of users in its comment section.
- Construct a network where nodes represent videos and edges indicate shared commenters. Keep only strong connections using Jaccard similarity.
- Use NetworkX to output a graph, initially with ~72k videos and ~654k edges, where edge weight reflects audience overlap strength.
- Detect communities (Louvain/Leiden) to find clusters of overlapping audiences.
- Characterize clusters: dominant channels, common content themes (metadata keywords), and cluster overlaps (Jaccard similarity of shared commenters).
- Identify "pathway" videos: nodes with high betweeness bridging clusters; compare audience-driven clusters with official categories using purity or normalized mutual information.

The example below illustrates the local ego-network of a video of interest (in red), which has the highest degree within the most connected component. It is connected to other videos (in blue) through overlapping commenting users.
 <p align="center">
    <img width="545" height="393" alt="Capture d’écran 2025-11-05 à 11 29 54" src="https://github.com/user-attachments/assets/81b8a4a0-b1e4-4036-b04e-eb1b9fbb1250" />
</p>

Additional:
Run an LLM analysis on video metadata (titles and descriptions) for high degree and betweenness-similarity videos to have an insight on he type of language that convince people to comment.  

### Engagement Dynamics (Comments impact on channel) (Additional):
We define loyalty metrics to distinguish active from passive audiences, then we conduct Pearson and Spearman correlation analyses between comment metrics and channel performance indicators to assess:
- The strength of association between comment volume and subscriber count.
- The relationship between unique commenter counts and channel size.
- The difference in loyalty metrics between high-engagement (top quartile by comment rate) and low-engagement channels (bottom quartile).

## Proposed Timeline :calendar::

   | Week | Deadline | Tasks | Responsible |
|-----------|-----------|-----------|-----------|
|  0  |  05.11  | Define a Project proposal and run intial analysis   | Matteo, Thomas, Albert, Romain, Hugo  |
|  1  |  12.11  |Prepare and build the network (video-level): Select the most relevant videos and users using filtering thresholds, then capture relationships between them based on shared commenters and compute Jaccard similarity to quantify audience overlap.   |  Albert, Romain, Hugo  |
|     |         | Continue to run experiments on small datasets to extract early results and think about important features for the next steps | Thomas, Matteo  |
|  2  |  19.11  | Reveal natural content communities by detecting groups of related videos using Louvain or Leiden algorithms (video-level)  | Albert, Romain, Hugo   |
|     |         | Build a pipeline to make the early results scale on the full dataset and define the features and parameters useful for it | Thomas, Matteo   |
|  3  |  26.11  | Understand audience structure by examining dominant channels, common content themes, and overlaps between communities (video-level)   | Albert, Romain, Hugo   |
|     |         | Run our analysis with the previously mentionned robust pipeline  | Thomas, Matteo  |
|  4  |  03.12  | Highlight videos bridging different clusters using betweenness centrality; compare audience-driven clusters to official categories to evaluate alignment. (video-level)  | Albert, Romain, Hugo   |
|     |         | cont. objectives of earlier week   | Thomas, Matteo  |
|  5  |  10.12  | Engagement Dynamics analysis: Examine how commenting activity relates to channel performance by analyzing correlations between engagement metrics (volume, diversity, loyalty) and audience size or growth.   | Albert, Romain, Hugo   |
|     |         | Use an LLM to analyze titles/descriptions of central videos to identify language patterns that encourages commenting behavior.   | Albert, Romain, Hugo   |
|     |         | Compare results between user network and video network and interpret and explain our full results| Thomas, Matteo |
|  6  |  17.12  | Complete project, document results, finalize report and clean up repository    | Matteo, Thomas, Albert, Romain, Hugo   |




## Questions for TAs :question:: 
  - Can we use external librairies/softwares to visualize the video network such as Gephi?

## Acknowledgments :ballot_box_with_check::
AI coding assistants were used to assist with code implementation, debugging, data visualization, and technical documentation. All analytical decisions, research design, and interpretations were made by the team.

In addition, the introductory image was created using ChatGPT.

### Contributors :busts_in_silhouette::
[SltMatteo](https://github.com/SltMatteo), [Tkemper2](https://github.com/Tkemper2), [albertfares](https://github.com/albertfares), [jeanninhugo](https://github.com/jeanninhugo), [frossardr](https://github.com/frossardr)

