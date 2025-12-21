import { useEffect, useMemo, useState } from "react";

function clamp01(x) {
  if (x == null || Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

function normLog(value, minV, maxV) {
  // normalize log10(value+1) to 0..1
  const v = Math.log10((value ?? 0) + 1);
  const a = Math.log10(minV + 1);
  const b = Math.log10(maxV + 1);
  if (b <= a) return 0;
  return Math.max(0, Math.min(1, (v - a) / (b - a)));
}

function statLabelBar(label, value01) {
  const v = Math.round(100 * value01);
  return (
    <div style={{ marginTop: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12, opacity: 0.9 }}>
        <span>{label}</span>
        <span>{v}</span>
      </div>
      <div style={{ height: 10, borderRadius: 999, background: "rgba(0,0,0,0.10)", overflow: "hidden" }}>
        <div style={{ width: `${v}%`, height: "100%", borderRadius: 999, background: "rgba(0,0,0,0.55)" }} />
      </div>
    </div>
  );
}

function archetypeFrom(p) {
  const fid = clamp01(p.fidelity_median);
  const cat = clamp01(p.catent_median);
  const nch = p.nch_median ?? 0;

  const focus = fid < 0.35 ? "Fidel" : fid > 0.7 ? "Explorer" : "Balanced";
  const topic = cat < 0.35 ? "Specialist" : cat > 0.7 ? "Generalist" : "Mixed";
  const breadth = nch < 5 ? "Narrow" : nch > 25 ? "Wide" : "Moderate";

  return { focus, topic, breadth };
}

function titleFrom(id, a) {
  const base = ["Nimbus", "Echo", "Vortex", "Lumen", "Quartz", "Nova", "Orbit", "Mosaic", "Fable", "Drift", "Pulse", "Glint", "Atlas", "Kite", "Rune"];
  const word = base[id % base.length];
  return `${word} — ${a.focus} ${a.topic}`;
}

function taglineFrom(a) {
  if (a.focus === "Focused" && a.topic === "Specialist") return "Lives in one corner of YouTube and knows it deeply.";
  if (a.focus === "Explorer" && a.topic === "Generalist") return "Wanders everywhere, comments across many channels and topics.";
  if (a.focus === "Balanced" && a.topic === "Mixed") return "Has a stable routine but still likes variety.";
  if (a.focus === "Focused") return "Only comments on a few channels but across various topics.";
  if (a.topic === "Generalist") return "Enjoys many topics and channels without a strong preference.";
  return "A consistent commenter with a recognizable pattern.";
}

const CUSTOM_DESCRIPTIONS_10 = {
  0: "Biggest group, focus on a small set of channel with low activity.",
  1: "Moderate",
  2: "Balanced users with stable habits who occasionally branch out to discover new creators.",
  3: "Topic specialists who comment deeply within a narrow set of themes across a few channels.",
  4: "Wide-ranging generalists who participate broadly but without strong attachment to any channel.",
  5: "Low-activity but consistent commenters who return to familiar channels over long periods.",
  6: "Highly active explorers commenting across many channels with strong topical diversity.",
  7: "Focused power users who comment frequently within a tightly defined interest space.",
  8: "Casual participants engaging sporadically across different topics and creators.",
  9: "Hybrid profiles mixing loyalty and exploration depending on content type."
};

export default function ProfileCards({
  basePath = "data/kmeans_explorer",
  peopleShareJsonPath = "data/kmeans_cluster_people_share.json",
  Kfixed = 10,
  minTc = 10,
  maxTc = 1000,
  minNch = 1,
  maxNch = 1000,
}) {
  const [profilesAllK, setProfilesAllK] = useState(null);  // profiles_allK.json
  const [peopleShare, setPeopleShare] = useState(null);    // Map(cluster -> share)

  // 1) Load profiles_allK.json once
  useEffect(() => {
    fetch(import.meta.env.BASE_URL + `${basePath}/profiles_allK.json`)
      .then((r) => r.json())
      .then(setProfilesAllK);
  }, [basePath]);

  // 2) Load people share JSON once
  useEffect(() => {
    fetch(import.meta.env.BASE_URL + peopleShareJsonPath)
      .then((r) => r.json())
      .then((d) => {
        // expected shape:
        // { "K": 10, "people_share": { "0": 0.12, "1": 0.03, ... } }
        const m = new Map();
        const obj = d?.people_share ?? {};
        for (const [k, v] of Object.entries(obj)) {
          const cid = Number(k);
          const share = Number(v);
          if (!Number.isNaN(cid) && !Number.isNaN(share)) m.set(cid, share);
        }
        setPeopleShare(m);
      });
  }, [peopleShareJsonPath]);

  const cards = useMemo(() => {
    const prof = profilesAllK?.profiles?.[String(Kfixed)] ?? [];
    if (!prof.length) return [];

    // merge people_share into profiles
    const merged = prof.map((p) => ({
      ...p,
      people_share: peopleShare?.get(p.id) ?? null,
    }));

    // Sort: by people share if available, else by sample size
    merged.sort((a, b) => {
      const ap = a.people_share ?? -1;
      const bp = b.people_share ?? -1;
      if (bp !== ap) return bp - ap;
      return (b.n_points ?? 0) - (a.n_points ?? 0);
    });

    return merged;
  }, [profilesAllK, peopleShare, Kfixed]);

  if (!profilesAllK || !peopleShare) return <div>Loading profiles…</div>;

  return (
    <div style={{ marginTop: 18 }}>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "baseline" }}>
        <h3 style={{ margin: 0 }}>Meet the {Kfixed} Commenter Profiles</h3>
        <span style={{ opacity: 0.8, fontSize: 13 }}>
          (K = {Kfixed}, regime: {minTc}–{maxTc} comments)
        </span>
      </div>

      <div
        style={{
          marginTop: 14,
          marginBottom: 48,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 16,
        }}
      >
        {cards.map((p) => {
          const a = archetypeFrom(p);
          const title = `${a.focus} · ${a.topic}`;
          const tagline = taglineFrom(a);

          const tcN = normLog(p.tc_median ?? 0, minTc, maxTc);
          const nchN = normLog(p.nch_median ?? 0, minNch, maxNch);
          const channelDivN = clamp01(p.fidelity_median);  // you may rename in UI
          const catDivN = clamp01(p.catent_median);

          const samplePct = ((p.share ?? 0) * 100);
          const peoplePct = (p.people_share == null) ? null : (p.people_share * 100);

          return (
            <div
              key={p.id}
              style={{
                borderRadius: 18,
                padding: 16,
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.10)",
                boxShadow: "0 8px 24px rgba(0,0,0,0.15)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                <div>
                  <div style={{ fontSize: 12, opacity: 0.75 }}>Profile</div>
                  <div style={{ fontSize: 18, fontWeight: 700, lineHeight: 1.2 }}>
                    {title}
                  </div>
                </div>

                <div
                  style={{
                    fontSize: 12,
                    fontWeight: 700,
                    padding: "6px 10px",
                    borderRadius: 999,
                    background: "rgba(0,0,0,0.25)",
                    border: "1px solid rgba(255,255,255,0.12)",
                    height: "fit-content",
                    whiteSpace: "nowrap",
                  }}
                >
                  C{p.id}
                </div>
              </div>

              <div style={{ marginTop: 10, fontSize: 13, opacity: 0.92 }}>
                {tagline}
              </div>

              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 10 }}>
                <span style={badgeStyle}>{a.focus}</span>
                <span style={badgeStyle}>{a.topic}</span>
                <span style={badgeStyle}>{a.breadth}</span>

                <span style={badgeStyle}>{samplePct.toFixed(2)}% of sample</span>

                {peoplePct != null && (
                  <span style={{ ...badgeStyle, background: "rgba(0,0,0,0.28)" }}>
                    {peoplePct.toFixed(2)}% of people
                  </span>
                )}
              </div>

              <div style={{ marginTop: 8, fontSize: 12, opacity: 0.82 }}>
                <div>
                  <strong>Median activity:</strong>{" "}
                  {Math.round(p.tc_median ?? 0).toLocaleString()} comments
                </div>
                <div>
                  <strong>Median breadth:</strong>{" "}
                  {Math.round(p.nch_median ?? 0).toLocaleString()} channels
                </div>
              </div>

              {statLabelBar("Total comments (log)", tcN)}
              {statLabelBar("Number of channels (log)", nchN)}
              {statLabelBar("Channel diversity", channelDivN)}
              {statLabelBar("Category diversity", catDivN)}
            </div>
          );
        })}
      </div>
    </div>
  );
}

const badgeStyle = {
  fontSize: 12,
  padding: "6px 10px",
  borderRadius: 999,
  background: "rgba(0,0,0,0.18)",
  border: "1px solid rgba(255,255,255,0.12)",
};