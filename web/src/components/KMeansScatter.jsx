import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function KMeansScatter({
  jsonPath = "data/kmeans_scatter_fid_cat.json",
  height = 520,
}) {
  const [d, setD] = useState(null);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + jsonPath)
      .then((r) => r.json())
      .then(setD);
  }, [jsonPath]);

  const option = useMemo(() => {
    if (!d) return null;

    // Build [x,y,cluster] rows
    const points = d.x.map((x, i) => [d.x[i], d.y[i], d.c[i]]);
    const K = d.k ?? (Math.max(...d.c) + 1);

    // Split by cluster (so legend + colors are stable)
    const byCluster = Array.from({ length: K }, () => []);
    for (const p of points) byCluster[p[2]].push(p);

    return {
      title: { text: `K-means clusters in behavior space (n=${d.n.toLocaleString()})` },
      tooltip: {
        trigger: "item",
        formatter: (p) => {
          const [x, y, c] = p.data;
          return [
            `Cluster: ${c}`,
            `Fidelity: ${Number(x).toFixed(3)}`,
            `Category entropy: ${Number(y).toFixed(3)}`
          ].join("<br/>");
        }
      },
      legend: {
        type: "scroll",
        top: 35,
      },
      grid: { left: 70, right: 20, top: 90, bottom: 60 },
      xAxis: {
        type: "value",
        name: "Fidelity",
        min: 0,
        max: 1,
        axisLabel: { formatter: (v) => Number(v).toFixed(2) },
      },
      yAxis: {
        type: "value",
        name: "Category entropy",
        min: 0,
        max: 1,
        axisLabel: { formatter: (v) => Number(v).toFixed(2) },
      },
      series: byCluster.map((arr, cid) => ({
        name: `C${cid}`,
        type: "scatter",
        data: arr,
        symbolSize: 2,
        large: true,
        progressive: 50000,
        emphasis: { focus: "series" },
      })),
    };
  }, [d]);

  if (!option) return <div>Loading…</div>;
  return <ReactECharts option={option} style={{ height, width: "100%" }} />;
}