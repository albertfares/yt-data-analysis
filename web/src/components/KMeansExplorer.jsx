import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function KMeansExplorer({
  basePath = "data/kmeans_explorer",
  height = 560,
}) {
  const [points, setPoints] = useState(null);
  const [allLabels, setAllLabels] = useState(null);
  const [K, setK] = useState(12);
  const [showCentroids, setShowCentroids] = useState(true);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + `${basePath}/points.json`)
      .then((r) => r.json())
      .then(setPoints);

    fetch(import.meta.env.BASE_URL + `${basePath}/labels_allK.json`)
      .then((r) => r.json())
      .then(setAllLabels);
  }, [basePath]);

  const option = useMemo(() => {
    if (!points || !allLabels) return null;

    const pack = allLabels.data?.[String(K)];
    if (!pack) return null;

    const n = points.n;

    // Build per-cluster arrays (like KMeansScatter)
    const byCluster = Array.from({ length: K }, () => []);

    for (let i = 0; i < n; i++) {
      const fid = points.fidelity[i];
      const cat = points.catent[i];
      const cid = pack.labels[i];

      // Safety: ignore any label outside 0..K-1
      if (cid >= 0 && cid < K) {
        byCluster[cid].push([
          fid,                      // x
          cat,                      // y
          cid,                      // cluster id (tooltip)
          points.nch?.[i] ?? null,  // num channels
          points.tc?.[i] ?? null,   // total comments (if you stored it)
        ]);
      }
    }

    const centroidSeries = showCentroids
      ? [
          {
            name: "Centroids",
            type: "scatter",
            data: pack.centroids.map((c) => [
              c.fidelity,
              c.catent,
              c.id,
              c.nch,
              c.n_points,
            ]),
            symbol: "diamond",
            symbolSize: 18,
            z: 10,
            tooltip: {
              formatter: (p) => {
                const [x, y, id, nch, np] = p.data;
                return [
                  `<b>Centroid C${id}</b>`,
                  `Channel diversity: ${Number(x).toFixed(3)}`,
                  `Category diversity: ${Number(y).toFixed(3)}`,
                  `#Channels (center): ${Number(nch).toFixed(1)}`,
                  `Points: ${Number(np).toLocaleString()}`,
                ].join("<br/>");
              },
            },
          },
        ]
      : [];

    return {
      title: { text: `K-means explorer (K=${K}, n=${n.toLocaleString()})` },

      // Smooth-ish updates
      animation: true,
      animationDurationUpdate: 350,
      animationEasingUpdate: "cubicOut",

      tooltip: {
        trigger: "item",
        formatter: (p) => {
          const d = p.data;
          // centroid has a different shape; let centroidSeries override tooltip
          const x = d[0], y = d[1], c = d[2], nch = d[3], tc = d[4];
          const lines = [
            `Cluster: C${c}`,
            `Channel diversity: ${Number(x).toFixed(3)}`,
            `Category diversity: ${Number(y).toFixed(3)}`,
          ];
          if (nch != null) lines.push(`#Channels: ${Number(nch).toLocaleString()}`);
          if (tc != null) lines.push(`Total comments: ${Number(tc).toLocaleString()}`);
          return lines.join("<br/>");
        },
      },

      legend: {
        type: "scroll",
        top: 35,
      },

      grid: { left: 70, right: 20, top: 90, bottom: 60 },

      xAxis: {
        type: "value",
        name: "Channel diversity",
        min: 0,
        max: 1,
        axisLabel: { formatter: (v) => Number(v).toFixed(2) },
      },

      yAxis: {
        type: "value",
        name: "Category diversity",
        min: 0,
        max: 1,
        axisLabel: { formatter: (v) => Number(v).toFixed(2) },
      },

      series: [
        ...byCluster.map((arr, cid) => ({
          name: `C${cid}`,
          type: "scatter",
          data: arr,
          symbolSize: 2,
          large: true,
          progressive: 60000,
          emphasis: { focus: "series" },
        })),
        ...centroidSeries,
      ],
    };
  }, [points, allLabels, K, showCentroids]);

  return (
    <div style={{ marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 10,
          flexWrap: "wrap",
        }}
      >
        <label>
          <strong>K</strong> = {K}
        </label>
        <input
          type="range"
          min={1}
          max={15}
          value={K}
          onChange={(e) => setK(Number(e.target.value))}
          style={{ width: 260 }}
        />
        <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <input
            type="checkbox"
            checked={showCentroids}
            onChange={(e) => setShowCentroids(e.target.checked)}
          />
          Show centroids
        </label>
      </div>

      <div style={{ position: "relative" }}>
        <ReactECharts
            option={option || {}}
            style={{ height, width: "100%" }}
            notMerge={true}                 // ✅ IMPORTANT: remove old series
            lazyUpdate={true}
            replaceMerge={["series", "legend"]} // ✅ extra safety
        />
        {(!points || !allLabels) && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "rgba(0,0,0,0.04)",
            }}
          >
            Loading…
          </div>
        )}
      </div>
    </div>
  );
}