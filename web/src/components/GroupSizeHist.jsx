import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function GroupSizeHist() {
  const [h, setH] = useState(null);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + "data/group_size_hist.json")
      .then((r) => r.json())
      .then(setH);
  }, []);

  if (!h) return <div>Loading…</div>;

 const labels = h.bin_left.map(
  (l, i) => `${Math.round(l)}–${Math.round(h.bin_right[i])}`
);
  const option = {
    title: { text: "Distribution of Group Sizes" },
    tooltip: { trigger: "axis" },
    xAxis: {
      type: "category",
      data: labels,
      axisLabel: { interval: "auto", rotate: 45 },
    },
    yAxis: {
    type: "log",
    name: "Number of groups",
    min: 1,                 // IMPORTANT for log scale
    logBase: 10,
    axisLabel: {
        formatter: (v) => v.toLocaleString()
    }
    },
    series: [{ type: "bar", data: h.counts }],
    grid: { left: 60, right: 20, bottom: 90, top: 60 },
  };

  return (
  <ReactECharts
    option={option}
    theme="dark"
    style={{ height: 420, width: "100%" }}
  />
);
}