import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function HeatmapFromJson({
  jsonPath,
  title,
  xName,
  yName,
  xIsLog10 = false,   // if true, x axis labels show 10^value
  height = 460,
}) {
  const [h, setH] = useState(null);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + jsonPath)
      .then((r) => r.json())
      .then(setH);
  }, [jsonPath]);

  const option = useMemo(() => {
    if (!h) return null;

    const xCenters = h.x_edges.slice(0, -1).map((l, i) => (l + h.x_edges[i + 1]) / 2);
    const yCenters = h.y_edges.slice(0, -1).map((l, i) => (l + h.y_edges[i + 1]) / 2);

    const maxV = h.cells.reduce((m, c) => Math.max(m, c[2]), 1);

    return {
      title: { text: title },
      tooltip: {
        position: "top",
        formatter: (p) => {
          const [xi, yi, v] = p.data;
          const xVal = xCenters[xi];
          const yVal = yCenters[yi];
          const xShown = xIsLog10 ? Math.round(10 ** xVal) : xVal;
          return [
            `${xName}: ${Number(xShown).toLocaleString()}`,
            `${yName}: ${Number(yVal).toFixed(3)}`,
            `Count: ${Number(v).toLocaleString()}`
          ].join("<br/>");
        }
      },
      grid: { left: 80, right: 60, top: 60, bottom: 60 },
      xAxis: {
        type: "category",
        name: xName,
        data: xCenters,
        axisLabel: {
          formatter: (v) => {
            const xv = Number(v);
            return xIsLog10 ? Number(Math.round(10 ** xv)).toLocaleString() : xv.toFixed(2);
          }
        }
      },
      yAxis: {
        type: "category",
        name: yName,
        data: yCenters,
        axisLabel: { formatter: (v) => Number(v).toFixed(2) }
      },
      visualMap: {
        min: 1,
        max: maxV,
        calculable: true,
        orient: "vertical",
        right: 10,
        top: 70,
      },
      series: [
        {
          name: "density",
          type: "heatmap",
          data: h.cells,
          progressive: 5000,
          emphasis: { itemStyle: { borderColor: "#333", borderWidth: 1 } }
        }
      ]
    };
  }, [h, title, xName, yName, xIsLog10]);

  if (!option) return <div>Loading…</div>;
  return <ReactECharts option={option} style={{ height, width: "100%" }} />;
}