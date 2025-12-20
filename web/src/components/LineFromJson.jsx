import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function LineFromJson({
  jsonPath,
  title,
  xName,
  yName,
  logX = false,
  logY = false,
  height = 420,
}) {
  const [d, setD] = useState(null);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + jsonPath)
      .then((r) => r.json())
      .then(setD);
  }, [jsonPath]);

  const data = useMemo(() => {
    if (!d) return [];
    return d.x.map((x, i) => ({ x, y: d.y[i] }));
  }, [d]);

  if (!d) return <div>Loading…</div>;

  const option = {
    title: { text: title },
    tooltip: {
      trigger: "axis",
      formatter: (params) => {
        const p = params[0].data;
        const fmt = (v) => Number(v).toLocaleString(undefined, { maximumFractionDigits: 4 });
        return [`${xName}: ${fmt(p.x)}`, `${yName}: ${fmt(p.y)}`].join("<br/>");
      },
    },
    dataset: { source: data },
    xAxis: {
      type: logX ? "log" : "value",
      name: xName,
      min: logX ? 1e-4 : 0,
      max: 1,
      logBase: logX ? 10 : undefined,
      axisLabel: { formatter: (v) => Number(v).toLocaleString() },
    },
    yAxis: {
      type: logY ? "log" : "value",
      name: yName,
      min: logY ? 1e-4 : 0,
      max: 1,
      logBase: logY ? 10 : undefined,
      axisLabel: { formatter: (v) => Number(v).toLocaleString() },
    },
    series: [
      {
        type: "line",
        encode: { x: "x", y: "y" },
        showSymbol: false,
        smooth: true,
      },
    ],
    grid: { left: 80, right: 20, bottom: 60, top: 60 },
  };

  return <ReactECharts option={option} style={{ height, width: "100%" }} />;
}

