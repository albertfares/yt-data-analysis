import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function HistFromJson({
  jsonPath,
  title,
  xTitle,
  logX = false,
  logY = false,
  height = 420,
}) {
  const [h, setH] = useState(null);

  useEffect(() => {
    fetch(import.meta.env.BASE_URL + jsonPath)
      .then((r) => r.json())
      .then(setH);
  }, [jsonPath]);

  const option = useMemo(() => {
    if (!h) return null;

    // build [x_center, count, left, right] rows
    const data = h.bin_left.map((l, i) => {
      const r = h.bin_right[i];
      const x = (l + r) / 2;
      return [x, h.counts[i], l, r];
    });

    return {
      title: { text: title },
      grid: { left: 70, right: 20, top: 60, bottom: 60 },

      tooltip: {
        trigger: "axis",
        axisPointer: {
          type: "line",     // ✅ always a thin line
          snap: true,
        },
        formatter: (params) => {
          const p = params[0];
          const [x, cnt, l, r] = p.data;
          return [
            `Bin: ${Number(l).toLocaleString()} – ${Number(r).toLocaleString()}`,
            `Count: ${Number(cnt).toLocaleString()}`,
          ].join("<br/>");
        },
      },

      xAxis: {
        type: logX ? "log" : "value",     // ✅ numeric axis, not category
        name: xTitle,
        min: logX ? 1 : null,
        axisLabel: { formatter: (v) => Number(v).toLocaleString() },
      },

      yAxis: {
        type: logY ? "log" : "value",
        name: "Number of groups",
        min: logY ? 1 : 0,
        axisLabel: { formatter: (v) => Number(v).toLocaleString() },
      },

      series: [
        {
          type: "bar",
          data,
          // make bars match bin width on continuous axis
          barWidth: "99%",
          large: true,
          encode: { x: 0, y: 1 },
        },
      ],
    };
  }, [h, title, xTitle, logX, logY]);

  if (!option) return <div>Loading…</div>;
  return <ReactECharts option={option} style={{ height, width: "100%" }} />;
}