import HistFromJson from "../components/HistFromJson";
import LineFromJson from "../components/LineFromJson";
import MultiHistFromJson from "../components/MultiHistFromJson";
import HeatmapFromJson from "../components/HeatmapFromJson";
import KMeansScatter from "../components/KMeansScatter";
import KMeansExplorer from "../components/KMeansExplorer";
import KMeansMedianExplorer from "../components/KMeansMedianExplorer";
import ProfileCards from "../components/ProfileCards";

import { useState } from "react";

function KSlider({ label = "K", K, setK, min = 3, max = 15, note }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", margin: "10px 0 8px" }}>
      <label><strong>{label}</strong> = {K}</label>
      <input
        type="range"
        min={min}
        max={max}
        value={K}
        onChange={(e) => setK(Number(e.target.value))}
        style={{ width: 260 }}
      />
      {note && <span style={{ opacity: 0.85 }}>{note}</span>}
    </div>
  );
}

export default function User_community() {
  // ✅ one K per plot
  const [Kexplorer, setKexplorer] = useState(12);
  const [Kmedian, setKmedian] = useState(12);
  const [Kcards, setKcards] = useState(12);

  return (
    <div className="page">
      <h1>Entering the Youniverse</h1>

      <p>
        Between 2014 and 2019, millions of people commented on YouTube videos.
        The <strong>Youniverse</strong> dataset captures this activity at an
        unprecedented scale. Our goal today is to understand this activity, 
        analyze users profiles and understand which user <strong>YOU</strong> are ! 
      </p>

      <p>In total, it contains approximately:</p>
      <ul>
        <li><strong>8.6 billion</strong> comments</li>
        <li><strong>448 million</strong> authors</li>
        <li><strong>153,550</strong> channels mentioned</li>
      </ul>

      <p>
        At this scale, analyzing individual users directly is neither practical
        nor meaningful. Instead, we adopt a different perspective.
        Our adventure starts by abstracting away the individuals users. 
      </p>

      <h2>From Users to Groups</h2>
      <p>
        We group together authors who exhibit <strong>exactly the same commenting behavior</strong>. 
        Imagine you and your friend create YouTube accounts and comment on the same video. Congrats! 
        You are now part of the same group.
      </p>

      <p>
        A <strong>group</strong> is defined as a set of authors who commented on the same channels, with the same number
        of comments per channel. Two authors belogn to the same group if their commenting habits are identical.
      </p>

      <p>
        These groups are not social communities; they are <strong>behavioral patterns</strong>: recurring ways 
        of interacting with Youtube through comments.
        Applying this definition to the entire dataset results in <strong>216,979,127 distinct groups</strong>.
      </p>

      <h2>How Large Are These Groups?</h2>

      <p> 
        Most groups are extremely small. In fact, the vast majority of groups contain only a single author. 
        A much smaller number of groups gather many people who behaved in exactly the same way. 
      </p>
      <HistFromJson
        jsonPath="data/group_size_hist.json"
        title="Distribution of Group Sizes (Number of Authors)"
        xTitle="Group size (num_authors)"
        logX={false}
        logY={true}
      />

      <p> 
        This extreme imbalance reveals an important property of online behavior: 
        most commenting patterns are either unique or shared by very few people, 
        while only a handful of behaviors are widely shared. 
      </p>

      <h2 style={{ marginTop: 16 }}>Zooming in on Inequality</h2>

      <p>
        Group size distributions already suggest strong imbalance, but we can
        make this even clearer by looking at how participation accumulates when
        we sort groups from smallest to largest.
      </p>

      <p>
        The <strong>Lorenz curve</strong> below answers the question:
        <em> “What fraction of all authors is contained in the smallest X% of groups?” </em>
        If participation were perfectly equal, the curve would be a straight diagonal.
        The more it bends, the more unequal the system is.
      </p>

      <LineFromJson
              jsonPath="data/matteo_lorenz_curve.json"
              title="Lorenz curve of authors across groups"
              xName="Cumulative fraction of groups"
              yName="Cumulative fraction of authors"
            />


      <p> 
        The curve shows an extremely unbalanced structure; half of all authors are
        contained in less than 10% of all groups! In other words, most behavioral 
        patterns are rare, but a few are massively shared. 
      </p>

      <h2>How Active Are Groups?</h2>

      <p>
        Earlier, we got to see how large the groups were. Group size tells us
        <em> how many people</em> share a behavior. But for
        understanding habits, another quantity matters even more:
        <strong> how many comments a group produces in total</strong>.
      </p>

      <p>
        Total comments is a proxy for <strong>activity</strong>: it distinguishes
        one-off participation from stable, repeated habits.
      </p>

      <HistFromJson
        jsonPath="data/groups_total_comments_hist.json"
        title="Distribution of Total Comments per Group"
        xTitle="Total comments"
        logX={true}
        logY={true}
      />

      <p style={{ marginTop: 16 }}>
        Pay attention to that graph's scale; it's log-scaled!
        The distribution is extremely skewed: most groups are low-activity, while
        a small fraction of groups generates a disproportionate amount of
        commenting activity.
      </p>

      <h2>How Many Channels Do Groups Interact With?</h2>

      <p>
        Finally, we can ask how broad these behaviors are. Some groups focus on a
        single channel, while others spread their activity across many channels,
        revealing more complex habits.
      </p>

      <HistFromJson
        jsonPath="data/groups_num_channels_hist.json"
        title="Distribution of Number of Channels per Group"
        xTitle="Number of channels"
        logX={false}
        logY={true}
      />

      <h2 style={{ marginTop: 44 }}>Filtering Out Noise</h2>
      
            <p>
              Not all groups carry the same amount of information. Many extremely small
              groups correspond to one-off or accidental behavior: a single comment, or
              a handful of comments scattered across unrelated channels.
            </p>
      
            <p>
              To focus on meaningful habits, we progressively raise the minimum number
              of comments required for a group to be kept. The plot below overlays the
              group-size distributions obtained after removing groups below several
              activity thresholds.
            </p>
      
             <MultiHistFromJson
              jsonPath="data/matteo_noise_removed_hist_overlay.json"
              title="Group-size distributions under different minimum-activity thresholds"
              xName="Group size (num_authors)"
              logX={false}
              logY={true}
            />
      
             <p style={{ marginTop: 12 }}>
              When we include every group, the distribution is dominated by extremely small behaviors. As we raise the minimum activity threshold, only groups with more 
              meaningful behaviors remain. This motivates our need to filter out low-activity groups.
            </p>


     {/* ===================== ACTIVITY REGIMES ===================== */}

      <h2 style={{ marginTop: 44 }}>From Activity to Profiles</h2>

      <p>
        From now on, we use <strong>total comments per group</strong> to separate
        participation regimes. The intuition is simple: below a certain activity
        level, behavior is too sparse to reveal stable preferences. As we said earlier, 
        we will filter out groups that don't reveal much.
      </p>

      <h3>👻 Ghosts: one comment, then silence</h3>

      <p>
        Groups with <strong>exactly one comment</strong> represent a single
        interaction with YouTube: one message, then nothing.
      </p>

      <ul>
        <li><strong>Definition:</strong> total_comments = 1</li>
        <li><strong>How much:</strong> <strong>131,167</strong> groups are ghost groups.</li>
        <li><strong>Authors:</strong> <strong>TODO</strong></li>
        <li><strong>Share of all groups:</strong> <strong>TODO%</strong></li>
      </ul>

      <p>
        These ghost groups are meaningful as a <em>participation profile</em>,
        but they do not contain enough signal to analyze preferences. 
        We’ll just leave them alone to get on with their ghostly business.
      </p>

      <h3>🌫️ Sparse activity: too little signal</h3>

      <p>
        We also group together low-activity behaviors with fewer than 10 total
        comments. Even if they are slightly more active than ghosts, their
        behavior is still dominated by chance: a few isolated comments are not
        enough to define a consistent pattern.
      </p>

      <ul>
        <li><strong>Definition:</strong> 2 ≤ total_comments &lt; 10</li>
        <li><strong>How much:</strong> <strong>TODO</strong></li>
        <li><strong>Authors:</strong> <strong>TODO</strong></li>
        <li><strong>Share of all groups:</strong> <strong>TODO%</strong></li>
      </ul>

      <p>
        From here onward, we focus on the regime where behavior becomes
        interpretable.
      </p>

      <h3>⚖️ Two activity regimes</h3>

      <p>
        For groups with at least 10 comments, stable habits start to emerge. We
        split them into two regimes:
      </p>

      <ul>
        <li>
          <strong>Moderately active:</strong> 10 ≤ total_comments &lt; 1000
          <br />
          <span style={{ opacity: 0.9 }}>
            → enough signal to compare behaviors using compact, interpretable features
          </span>
        </li>
        <li style={{ marginTop: 10 }}>
          <strong>Highly active:</strong> total_comments ≥ 1000
          <br />
          <span style={{ opacity: 0.9 }}>
            → richer patterns, where we later use stronger similarity methods (top channels, k-NN, etc.)
          </span>
        </li>
      </ul>


      <h2 style={{ marginTop: 44 }}>Behavior Features (10–1000 comments)</h2>

      <p>
        In the <strong>10–1000</strong> regime, groups are active enough to show
        structure, but not so active that every behavior becomes unique. To
        summarize habits, we compute two interpretable features:
      </p>

      <ul>
        <li>
          <strong>Fidelity</strong> (channel entropy): how concentrated activity is across channels.
          <br />
          <span style={{ opacity: 0.9 }}>
            Low fidelity → focused on a few channels. High fidelity → spread across many channels.
          </span>
        </li>
        <li style={{ marginTop: 10 }}>
          <strong>Category entropy</strong>: how diverse the commented topics are.
          <br />
          <span style={{ opacity: 0.9 }}>
            Low entropy → mostly one category. High entropy → many categories.
          </span>
        </li>
      </ul>

      <p>
        These two dimensions form a “behavior space” where we can compare groups
        and look for recurring profiles.
      </p>

      <HistFromJson
        jsonPath="data/category_entropy_hist.json"
        title="Distribution of Category Entropy (10–1000 comments)"
        xTitle="Category entropy"
        logX={false}
        logY={false}
      />

      <HistFromJson
        jsonPath="data/fidelity_hist.json"
        title="Distribution of Channel diversity (10–1000 comments)"
        xTitle="Channel diversity)"
        logX={false}
        logY={true}
      />

      <p style={{ marginTop: 16 }}>
        This distribution shows that topic diversity varies substantially: many
        groups are specialized, while others spread attention across multiple
        categories.
      </p>

      <HeatmapFromJson
        jsonPath="data/fidelity_vs_category_entropy_heatmap.json"
        title="Channel VS Category diversity (density, 10–1000 comments)"
        xName="Channel diversity"
        yName="Category diversity"
      />

      <p style={{ marginTop: 16 }}>
        The dense regions in this map correspond to recurring styles of behavior:
        for example, some groups are <em>focused and specialized</em>, while others are
        <em> exploratory and diverse</em>. This motivates clustering: rather than
        describing hundreds of millions of groups individually, we can identify a
        small number of stable profiles.
      </p>

      {/* ===================== CLUSTERING SECTION ===================== */}

      <h2 style={{ marginTop: 44 }}>From Behavior Space to Profiles (K-means)</h2>

      <p>
        To transform this continuous behavior space into a small set of{" "}
        <strong>interpretable commenter profiles</strong>, we apply{" "}
        <strong>k-means clustering</strong>.
      </p>

      <p>
        Importantly, these clusters are not “true” natural categories with sharp
        borders. Commenting behavior varies continuously. K-means helps us{" "}
        <em>summarize</em> the space by placing a small number of representative
        profiles across the main dense regions.
      </p>

      <p style={{ marginTop: 12 }}>
        The plot below shows a sample of groups (each dot is a group), colored by
        its assigned profile. You can already see that profiles correspond to
        different combinations of focus (fidelity) and topical diversity
        (category entropy).
      </p>

      <p style={{ marginTop: 16 }}>
        In the next steps, we will “open up” these profiles: for each one, we
        will examine typical activity levels, number of channels, and category
        preferences — and give them human-readable names (e.g.,{" "}
        <em>Focused Specialists</em>, <em>Exploratory Generalists</em>, etc.).
      </p>

      <p style={{ marginTop: 24 }}>
        Finally, once these profiles are established, we will move beyond groups
        and aggregate them into broader <strong>communities</strong> — capturing
        not only shared habits, but also how common each style of participation is.
      </p>

      <h2 style={{ marginTop: 44 }}>K-means Explorer (interactive)</h2>
      <p style={{ opacity: 0.9 }}>
        This plot shows one clustering result (the K used when you exported <code>kmeans_scatter_fid_cat.json</code>).
        The interactive explorers below let you vary K.
      </p>

      <KMeansExplorer basePath="data/kmeans_explorer" K={Kexplorer} />

      <h2 style={{ marginTop: 44 }}>Profile Summary Explorer (medians)</h2>

      <KMeansMedianExplorer basePath="data/kmeans_explorer" K={Kmedian} useAllKFile={false} />

      <h2 style={{ marginTop: 44 }}>Profiles as Characters</h2>
      <ProfileCards
        basePath="data/kmeans_explorer"
        peopleShareJsonPath="data/kmeans_cluster_people_share.json"
        Kfixed={10}
      />
    </div>
  );
}