import { useState } from "react";

import HistFromJson from "../components/HistFromJson";
import LineFromJson from "../components/LineFromJson";
import MultiHistFromJson from "../components/MultiHistFromJson";
import HeatmapFromJson from "../components/HeatmapFromJson";
import KMeansExplorer from "../components/KMeansExplorer";
import KMeansMedianExplorer from "../components/KMeansMedianExplorer";
import ProfileCards from "../components/ProfileCards";
import BackgroundNodes from "../components/BackgroundNodes";
import BackgroundGhostNetwork from "../components/BackgroundGhostNetwork";
import BackgroundCommentFlow from "../components/BackgroundCommentFlow";

function openRandomVideo(e) {
  e.preventDefault();
  const links = [
    "https://www.youtube.com/watch?v=XAu5MuTGBC8",
    "https://www.youtube.com/watch?v=oHg5SJYRHA0",
    "https://www.youtube.com/watch?v=L3tsYC5OYhQ",
    "https://www.youtube.com/watch?v=pqEodXPP3hI",
    "https://www.youtube.com/watch?v=TAeNlpUIlRs",
    "https://www.youtube.com/watch?v=F8dKVbP1Nzo",
    "https://www.youtube.com/watch?v=8cW6gPUzQ44",
    "https://www.youtube.com/watch?v=bpfKJo8aYd8",
    "https://www.youtube.com/watch?v=6Ti1g_P6u7s",
    "https://www.youtube.com/watch?v=__WPQkmxBY8",
    "https://www.youtube.com/watch?v=Dnx0z1cC8u8",
    "https://www.youtube.com/watch?v=5YuQQwLGTxA",
    "https://www.youtube.com/watch?v=uCNJOcgeZE8",
    "https://www.youtube.com/watch?v=UCqR3iNow4s",
    "https://www.youtube.com/watch?v=XeY0eOEEURM",
    "https://www.youtube.com/watch?v=xKi8KT2r0VI",
  ];
  const url = links[Math.floor(Math.random() * links.length)];
  window.open(url, "_blank");
}

function Section({ title, children, variant = "default" }) {
  return (
    <section
      className={`section section-${variant}`}
      style={{
        margin: "22px 0",
        padding: "22px 22px",
        borderRadius: 14,
        background: "rgba(255,255,255,0.92)",
        border: "1px solid rgba(0,0,0,0.06)",
        boxShadow: "0 1px 0 rgba(0,0,0,0.03)",
      }}
    >
      {title ? <h2 style={{ marginTop: 0 }}>{title}</h2> : null}
      {children}
    </section>
  );
}

export default function User_community() {
  const [Kexplorer, setKexplorer] = useState(12);
  const [Kmedian, setKmedian] = useState(12);

  return (
    <>
      <BackgroundNodes />

      <div className="page">
        {/* HERO */}
        <Section variant="hero">
          <h1 style={{ marginTop: 0 }}>Entering the Youniverse</h1>

          <p>
            You're chilling on YouTube, scrolling through{" "}
            <a href="#" onClick={openRandomVideo}>
              interesting videos
            </a>
            , when you notice the comments section. Some comments are funny, some are
            insightful, and some are just plain weird. But suddenly a thought hits you:
            <br />
            <br />
            <em style={{ display: "block", textAlign: "center" }}>
              who are the people behind these comments?
            </em>
            <br />
            Are they like you, or completely different? You see them here, under the same
            video you are watching, but do they hang around the same kind of content than
            you in general ? Do they behave in predictable ways, or is YouTube just a
            chaotic comment soup ? <br />
            At this point, curiosity takes over. You decide to investigate. <br />
            You embark on a journey to explore the hidden universe of YouTube commenters
            and discover what kind of user <strong>YOU</strong> are ! <br />
            And it all starts with finding the right dataset.
          </p>
        </Section>

        {/* DATASET */}
        <Section title="The Youniverse Dataset">
          <p>
            Between 2014 and 2019, millions of people commented on YouTube videos, leaving
            behind billions of messages. <br />
            Luckily for you, someone captured this activity at an incredible scale: the{" "}
            <strong>Youniverse</strong> dataset. <br />
            <br />
            You look around for something better. You don't find anything. <br />
            You look <em>again</em>, just in case. <br />
            Nope. This is it. This dataset is absolutely perfect. You roll up your sleeves
            and decide to use it.
          </p>

          <p>In total, the Youniverse dataset contains approximately:</p>
          <ul>
            <li>
              <strong>8.6 billion</strong> comments
            </li>
            <li>
              <strong>448 million</strong> authors
            </li>
            <li>
              <strong>153,550</strong> channels mentioned
            </li>
          </ul>

          <p>
            At this scale, analyzing individual users directly makes no sense. Even if you
            wanted to, looking at hundreds of millions of people one by one would tell you
            nothing meaningful. <br />
            You need a smarter strategy. But you have an idea.
          </p>
        </Section>

        {/* USERS -> GROUPS */}
        <Section title="From Users to Groups">
          <p>
            Instead of looking at people one by one, you decide to take a step back. What
            if, instead of focusing on <em>who</em> the users are, you looked at <em>how</em>{" "}
            they behave ? You come up with an idea:
            <br />
            Let&apos;s group together people who comment in exactly the same way. <br />
            Imagine you and your friend both go on YouTube, comment on the same channels,
            and leave the same number of comments on each of them. Congratulations! You're
            now behavior-twins: you belong to the same group.
          </p>

          <p>
            You define:
            <br />
            <br />
            <strong style={{ display: "block", textAlign: "center" }}>
              A group is a set of users who commented on the same channels, with the same
              number of comments per channel.
            </strong>
            <br />
            These groups are not social communities; they are <em>behavioral patterns</em>,
            recurring ways of interacting with YouTube. <br />
            <br />
            You apply this definition to the entire dataset, and you end up with <br />
            <br />
            <strong style={{ display: "block", textAlign: "center" }}>
              216,979,127 distinct groups
            </strong>
            <br />
            That's when you realize: YouTube commenters are much more diverse than you
            thought.
          </p>
        </Section>

        {/* GROUP SIZE */}
        <Section title="How Large Are These Groups?">
          <p>
            Now that you've grouped people by identical commenting behavior, a new question
            naturally pops into your mind: <em>Okay... but how big are these groups actually ?</em>
            <br />
            Do most YouTube commenters behave in totally unique ways? Or are many people
            following the same commenting patterns without even realizing it ? There's only
            one way to find out: You take all your groups and count how many authors belong
            to each of them.
          </p>

          <p>
            What you discover is pretty striking. <br />
            Most groups are incredibly small. In fact, the overwhelming majority of them
            contain only a single person. Just one lone commenter following a behavior
            pattern that nobody else on YouTube matches. <br />
            But you notice that some groups are different. A tiny fraction of behaviors are
            surprisingly popular, gathering huge numbers of people who all ended up behaving
            in exactly the same way. To see this clearly, you visualize the distribution of
            group sizes:
          </p>

          <HistFromJson
            jsonPath="data/group_size_hist.json"
            title="Distribution of Group Sizes (Number of Authors)"
            xTitle="Group size (num_authors)"
            logX={false}
            logY={true}
          />

          <p>
            This extreme imbalance reveals an important property of online behavior: most
            commenting patterns are either unique or shared by very few people, while only a
            handful of behaviors are widely shared.
          </p>
        </Section>

        {/* LORENZ */}
        <Section title="Zooming in on Inequality">
          <p>
            Seeing so many tiny groups and a few gigantic ones, you start wondering:{" "}
            <em>Just how unequal is this world of YouTube commenting behavior ?</em> You
            already know most patterns are rare, but you want to measure{" "}
            <strong>how extreme</strong> that imbalance really is. <br />
            <br />
            So you sort groups from the smallest to the largest and ask yourself a simple
            question: <em>As I move up this ranking, how quickly do i cover all YouTube authors ?</em>
            <br />
            <br />
            This leads you to one of the classic tools to measure inequality: the{" "}
            <strong>Lorenz curve</strong>.
          </p>

          <LineFromJson
            jsonPath="data/matteo_lorenz_curve.json"
            title="Lorenz curve of authors across groups"
            xName="Cumulative fraction of groups"
            yName="Cumulative fraction of authors"
          />

          <p>
            When you look at your curve, you see it bend dramatically. You notice that at
            the very beginning, just a tiny fraction of groups already captures a huge
            share of all authors; looking at the numbers you find out that half of all
            YouTube commenters belong to less than 10% of all groups! <br />
            <br />
            That's huge. <br />
            <br />
            It means the world of YouTube participation is not slightly unbalanced. It's
            deeply unequal: Most behavioral patterns are shared by almost nobody, and a
            small number of recurring habits dominate the platform.
          </p>

          <p>
            At this point, you've already uncovered somethign fundamental about how people
            behave online. Even if something as chaotic and personal as commenting on
            YouTube, crowd behavior organizes into a few massively popular patterns
            surrounded by an ocean of unique, individual quirks. <br />
            <br />
            And with that realization, you're ready to dig deeper.
          </p>
        </Section>

        {/* ACTIVITY */}
        <Section title="How Active Are Groups?">
          <p>
            So far, you got to see how large the groups were. But size alone doesn't tell
            the whole story. A new question starts buzzing in your mind:{" "}
            <em>Do all groups behave with the same intensity? Or are some far more talkative than others?</em>
            <br />
            <br />
            To find out, you stop looking at <strong>how many people</strong> are in each
            group and instead focus on <strong>how many comments</strong> each group
            produces in total.
          </p>

          <p>
            Most groups barely comment at all. A huge fraction of them generate only a
            handful of messages across their entire lifetime. These are fleeting presences:
            people who pass by, leave a small trace, then disappear. <br />
            But then the tail stretches... and stretches... and stretches. <br />
            A tiny subset of groups produces an enormous amount of activity, posting again
            and again, across time, across videos, shaping a huge part of the conversation
            happening on the platform. <br />
            <br />
            To see this clearly, you look at the dsitribution of total comments per group:
          </p>

          <HistFromJson
            jsonPath="data/groups_total_comments_hist.json"
            title="Distribution of Total Comments per Group"
            xTitle="Total comments"
            logX={true}
            logY={true}
          />

          <p>
            You immediatly notice the scale: the plot has to be log-scaled because the
            differences are so extreme. Just like group sizes, activity is wildly
            imbalanced.
          </p>

          <p>
            As you're looking at this distribution, another thought hits you:{" "}
            <em>
              I see how much these groups comment, but where do they comment ? Are they focused on a few channels, or do they spread their activity everywhere?
            </em>
            <br />
            <br />
            You now want to understand how many different channels each group interacts
            with.
          </p>
        </Section>

        {/* CHANNELS */}
        <Section title="How Many Channels Do Groups Interact With?">
          <p>
            You decide to zoom in on this question. If a group represents a shared way of
            experiencing YouTube, then the number of channels it touches tells you
            something important about its scope. <br />
            <br />
            Some behaviors might belong to loyal fans who stay in a single corner of the
            platform. Others may reflect curious wanderers who jump from channel to channel,
            constantly exploring new territory.
          </p>

          <HistFromJson
            jsonPath="data/groups_num_channels_hist.json"
            title="Distribution of Number of Channels per Group"
            xTitle="Number of channels"
            logX={false}
            logY={true}
          />

          <p>
            Most groups barely move. They focus on just one channel, or a very small
            handful. These are tightly anchored behaviors; deep attachments to specific
            creators or niches. But then, just like before, a long tail appears: a much
            smaller number of groups spread their attention widely.
          </p>
        </Section>

        {/* NOISE */}
        <Section title="Filtering Out Noise">
          <p>
            Now that you’ve explored how big groups are, how active they are, and how widely
            they spread, a clear realization forms in your mind:
            <br />
            <br />
            <strong style={{ display: "block", textAlign: "center" }}>
              Not every behavioral pattern deserves the same attention.
            </strong>
            <br />
            Some groups barely exist: a single comment, a small trace of activity. They
            technically count as behavior, but they don't really <em>say</em> anything.
          </p>

          <p>
            You gradually increase the minimum number of comments a group must have to
            remain in your analysis, and you watch what happens to the landscape:
          </p>

          <MultiHistFromJson
            jsonPath="data/matteo_noise_removed_hist_overlay.json"
            title="Group-size distributions under different minimum-activity thresholds"
            xName="Group size (num_authors)"
            logX={false}
            logY={true}
          />

          <p>
            You see that the YouTube world is flooded with tiny behaviors. But when you
            raise the treshold, you see that the picture becomes clearer. It motivates a
            natural breakdown of activity regimes.
          </p>
        </Section>

        {/* PROFILES */}
        <Section title="From Activity to Profiles">
          <p>
            From now on, you use <strong>total comments per group</strong> as your compass.
            It tells you wether a pattern is too faint to interpret or strong enough to
            reveal real preferences.
          </p>

          <h3>👻 Ghosts: one comment, then silence</h3>
          <p>
            Thse groups contain exactly one comment. <br />
            They represent a single interaction with YouTube: someone appeared, spoke once,
            and vanished.
          </p>
          <ul>
            <li>
              <strong>Definition:</strong> total_comments = 1
            </li>
            <li>
              <strong>How much:</strong> <strong>131,167</strong> groups are ghost groups.
            </li>
            <li>
              <strong>Authors:</strong> <strong>TODO</strong>
            </li>
            <li>
              <strong>Share of all groups:</strong> <strong>TODO%</strong>
            </li>
          </ul>

          <h3>🌫️ Sparse activity: too little signal</h3>
          <p>
            Next come groups with a handful of comments, but fewer than ten. <br />
            They exist, but their behavior is still dominated by randomness: a few scattered
            interactions that don’t yet form a clear habit.
          </p>
          <ul>
            <li>
              <strong>Definition:</strong> 2 ≤ total_comments &lt; 10
            </li>
            <li>
              <strong>How much:</strong> <strong>TODO</strong>
            </li>
            <li>
              <strong>Authors:</strong> <strong>TODO</strong>
            </li>
            <li>
              <strong>Share of all groups:</strong> <strong>TODO%</strong>
            </li>
          </ul>

          <h3>⚖️ Where behavior becomes meaningful</h3>
          <p>
            Then something changes. Once a group reaches at least ten total comments, its
            behavior starts to stabilize. Patterns stop looking like noise and start looking
            like <strong>habits</strong>.
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
        </Section>

        {/* FEATURES */}
        <Section title="Behavior Features (10–1000 comments)">
          <p>
            Now that you're focusing on groups with enough activity to reveal real habits,
            a new question appears:
            <br />
            <br />
            <em>
              Okay, these behaviors are meaningful.. but how do we actually describe them?
              What makes one group's way of using YouTube different from another's ?
            </em>
          </p>

         <section className="subsection">

          <h3>Channel Concentration (Fidelity)</h3>
          <p>
            The first question you ask is: <em>How concentrated is this group's attention across channels?</em>
          </p>
          <ul>
            <li>
              <strong>Low concentration</strong> : activity is focused on just a few channels
            </li>
            <li>
              <strong>High concentration</strong> : activity is spread across many channels
            </li>
          </ul>

          <HistFromJson
            jsonPath="data/fidelity_hist.json"
            title="Distribution of Channel diversity (10–1000 comments)"
            xTitle="Channel diversity)"
            logX={false}
            logY={true}
          />

          </section>

          <section className="subsection">

          <h3>Category Diversity</h3>
          <p>
            So you ask a second question: <em>How diverse are the topics this group engages with ?</em>
          </p>
          <ul>
            <li>
              <strong>Low diversity</strong> : mostly one topic
            </li>
            <li>
              <strong>High diversity</strong> : many different topics
            </li>
          </ul>

          <HistFromJson
            jsonPath="data/category_entropy_hist.json"
            title="Distribution of Category Entropy (10–1000 comments)"
            xTitle="Category entropy"
            logX={false}
            logY={false}
          />
          </section>

          <section className="subsection">

          <h3>A Behavior Map of YouTube</h3>
          <p>
            Together, these two dimensions for a <strong>behavior space</strong>. Each group becomes a point
            in this space:
          </p>
          <ul>
            <li>narrow vs broad channel focus</li>
            <li>specialized vs topic-diverse</li>
          </ul>

          <HeatmapFromJson
            jsonPath="data/fidelity_vs_category_entropy_heatmap.json"
            title="Channel VS Category diversity (density, 10–1000 comments)"
            xName="Channel diversity"
            yName="Category diversity"
          />

          <p style={{ marginTop: 16 }}>
            Dense regions in this map correspond to recurring “styles of being a YouTube commenter.”
            And that sparks an exciting idea: Instead of describing millions of groups individually,
            why not identify a small number of representative commenter profiles ?
          </p>
          </section>
        </Section>

        {/* CLUSTERING */}
        <Section title="From Behavior Space to Profiles">
          <p>
            You decide to summarize this continuous world into a small set of representative profiles.
            To do this, you will use <strong>K-means clustering</strong>.
          </p>

          <KMeansExplorer basePath="data/kmeans_explorer" K={Kexplorer} />

          <p>
            The cloudy mass turns into recognizable regions. These colorful shapes confirm what you suspected:
            YouTube doesn't contain one kind of commenter.
          </p>

          <KMeansMedianExplorer basePath="data/kmeans_explorer" K={Kmedian} useAllKFile={false} />

          <p>
            You think: <em>Hmmm... This is cool.. but it doesn't tell me who I am! That's what I wanted to see from the beginning!</em>
            <br />
            <br />
            And just like that, you finally <strong>meet the commenter profiles</strong>.
          </p>
        </Section>

        {/* CARDS */}
        <Section title="Profiles as Characters">
          <ProfileCards
            basePath="data/kmeans_explorer"
            peopleShareJsonPath="data/kmeans_cluster_people_share.json"
            Kfixed={10}
          />

          <p>
            Looking at these profiles, you can't help but smile. Each one feels alive, just like a friend.
            These profiles are not data, they're a way of moving through YouTube.
            <br />
            <br />
            And as you stand here, at the end of your journey, one final question remains:
            <br />
            <br />
            <strong style={{ display: "block", textAlign: "center" }}>
              Which commenter are YOU ?
            </strong>
            <br />
            You nod your head. You need to think about it. Maybe it’s time to open your YouTube profile and find out.
          </p>
        </Section>


      {/* JOURNEY NAVIGATION */}
<section
  style={{
    margin: "48px 0 80px",
    padding: "28px 22px",
    borderRadius: 16,
    background: "rgba(255,255,255,0.85)",
    border: "1px solid rgba(0,0,0,0.06)",
    textAlign: "center",
  }}
>
  <p style={{ marginTop: 0, fontSize: "1.05rem" }}>
    You’ve reached the end of this path.
    <br />
    Where would you like to go next?
  </p>

  <div
    style={{
      display: "flex",
      justifyContent: "center",
      gap: 24,
      marginTop: 24,
      flexWrap: "wrap",
    }}
  >
    <a className="cta-btn subtle" href="../index.html">
      Introduction
    </a>

    <a className="cta-btn accent" href="../viz/">
      Network Journey
    </a>

    <a className="cta-btn subtle" href="../conclusion.html">
      Conclusion
    </a>
  </div>
</section>
      </div>


    </>
  );
}