import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

function clamp01(x) {
  if (x == null || Number.isNaN(x)) return 0;
  return Math.max(0, Math.min(1, x));
}

export default function KMeansMedianExplorer({
  basePath = "data/kmeans_explorer",
  height = 520,
  Kmin = 1,
  Kmax = 15,
}) {
  const [K, setK] = useState(12);
  const [allK, setAllK] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(import.meta.env.BASE_URL + `${basePath}/profiles_allK.json`, {
      signal: controller.signal,
      cache: "force-cache",
    })
      .then((r) => r.json())
      .then(setAllK)
      .catch((e) => {
        if (e.name !== "AbortError") console.error(e);
      });

    return () => controller.abort();
  }, [basePath]);

  const profiles = useMemo(() => {
    const key = String(K);
    return allK?.profiles?.[key] ?? [];
  }, [allK, K]);

  const option = useMemo(() => {
    if (!profiles.length) return null;

    const values = profiles.map((p) => {
      const tc = p.tc_median ?? 0;
      const nch = p.nch_median ?? 0;

      return [
        Math.log10(tc + 1),              // dim 0
        Math.log10(nch + 1),             // dim 1
        clamp01(p.fidelity_median),      // dim 2
        clamp01(p.catent_median),        // dim 3
        p.id ?? null,                    // dim 4 tooltip
        p.share ?? null,                 // dim 5 tooltip
        p.n_points ?? null,              // dim 6 tooltip
      ];
    });

    const tcLogs = values.map((v) => v[0]);
    const nchLogs = values.map((v) => v[1]);

    const tcMin = Math.min(...tcLogs);
    const tcMax = Math.max(...tcLogs);
    const nchMin = Math.min(...nchLogs);
    const nchMax = Math.max(...nchLogs);

    return {
      title: { text: `Median profile explorer (K=${K})` },

      animation: true,
      animationDurationUpdate: 320,
      animationEasingUpdate: "cubicOut",

      tooltip: {
        formatter: (p) => {
          const v = p.value;
          const logTc = v[0],
            logNch = v[1],
            fid = v[2],
            cat = v[3],
            id = v[4],
            share = v[5],
            nPoints = v[6];

          const tc = Math.round(Math.pow(10, logTc) - 1);
          const nch = Math.round(Math.pow(10, logNch) - 1);

          const lines = [
            `<b>Profile C${id}</b>`,
            `tc_median: ${tc.toLocaleString()}`,
            `nch_median: ${nch.toLocaleString()}`,
            `channel diversity (median): ${Number(fid).toFixed(3)}`,
            `category diversity (median): ${Number(cat).toFixed(3)}`,
          ];
          if (share != null) lines.push(`share (sample): ${(share * 100).toFixed(2)}%`);
          if (nPoints != null) lines.push(`points: ${Number(nPoints).toLocaleString()}`);
          return lines.join("<br/>");
        },
      },

      parallelAxis: [
        { dim: 0, name: "total comments (log10)", min: tcMin, max: tcMax, realtime: false },
        { dim: 1, name: "number of channel (log10)", min: nchMin, max: nchMax, realtime: false },
        { dim: 2, name: "channel diversity", min: 0, max: 1, realtime: false },
        { dim: 3, name: "category diversity", min: 0, max: 1, realtime: false },
      ],

      parallel: {
        left: 70,
        right: 70,
        top: 90,
        bottom: 30,
        parallelAxisDefault: {
          nameLocation: "end",
          nameGap: 14,
          realtime: false,
        },
      },

      series: [
        {
          name: "profiles",
          type: "parallel",
          data: values,
          lineStyle: { width: 2, opacity: 0.55 },
          emphasis: { lineStyle: { opacity: 0.95, width: 3 } },
        },
      ],
    };
  }, [profiles, K]);

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10, flexWrap: "wrap" }}>
        <label><strong>K</strong> = {K}</label>
        <input
          type="range"
          min={Kmin}
          max={Kmax}
          value={K}
          onChange={(e) => setK(Number(e.target.value))}
          style={{ width: 260 }}
        />
        <span style={{ opacity: 0.85 }}>
          Each line is one profile (medians). Drag K to see profiles split/merge.
        </span>
      </div>

      <div style={{ position: "relative" }}>
        <ReactECharts
          option={option || {}}
          style={{ height, width: "100%" }}
          opts={{ renderer: "canvas" }}
          notMerge={true}
          replaceMerge={["series"]}
          lazyUpdate={true}
        />

        {!option && (
          <div style={{
            position: "absolute", inset: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(0,0,0,0.04)"
          }}>
            Loading profiles…
          </div>
        )}
      </div>
    </div>
  );
}