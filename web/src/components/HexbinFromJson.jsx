import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function HexbinFromJson({
  jsonPath,
  title,
  xName,
  yName,
  logX = false,
  height = 460,
}) {
  const [h, setH] = useState(null);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + jsonPath)
      .then((r) => r.json())
      .then(setH);
  }, [jsonPath]);

  const points = useMemo(() => {
    if (!h) return [];
    return h.x.map((xv, i) => [Number(xv), Number(h.y[i]), Number(h.counts[i])]);
  }, [h]);

  if (!h) return <div>Loading…</div>;
  if (!points.length) return <div>No data (empty hexbin)</div>;

  const maxC = Math.max(...points.map((p) => p[2]));

  const option = {
    title: { text: title },
    tooltip: {
      trigger: "item",
      formatter: (p) => {
        const [x, y, c] = p.data;
        return [
          `${xName}: ${x.toLocaleString()}`,
          `${yName}: ${y.toLocaleString()}`,
          `Count: ${c.toLocaleString()}`,
        ].join("<br/>");
      },
    },
    xAxis: {
      type: logX ? "log" : "value",
      name: xName,
      min: logX ? 1 : null,
      axisLabel: { formatter: (v) => Number(v).toLocaleString() },
    },
    yAxis: {
      type: "value",
      name: yName,
      axisLabel: { formatter: (v) => Number(v).toLocaleString() },
    },
    visualMap: {
      min: 1,
      max: maxC,
      dimension: 2,
      calculable: true,
      right: 10,
      top: 60,
      inRange: {}, // let echarts pick default gradient
    },
    series: [
      {
        type: "scatter",
        data: points,
        large: true,
        itemStyle: { opacity: 0.65 },
        symbolSize: (val) => {
          const c = val[2];
          return 2 + Math.log10(1 + c) * 6; // good for heavy tails
        },
      },
    ],
    grid: { left: 80, right: 80, bottom: 60, top: 60 },
  };

  return <ReactECharts option={option} style={{ height, width: "100%" }} />;
}