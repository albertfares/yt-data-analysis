import { useEffect, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function App() {
  const [payload, setPayload] = useState(null);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + "data/test.json")
      .then((r) => r.json())
      .then(setPayload);
  }, []);

  if (!payload) return <div style={{ padding: 24 }}>Loading…</div>;

  const option = {
    title: { text: "Test chart" },
    tooltip: { trigger: "axis" },
    xAxis: { type: "category", data: payload.labels },
    yAxis: { type: "value" },
    series: [{ type: "bar", data: payload.values }],
  };

  return (
    <div style={{ padding: 24 }}>
      <h1>Community Explorer</h1>
      <ReactECharts option={option} style={{ height: 420 }} />
    </div>
  );
}