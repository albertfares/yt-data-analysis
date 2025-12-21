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

    const n = h.counts?.length ?? 0;
    if (!n) return null;

    // Clamp x-range to histogram edges (prevents last bar spilling out)
    const xMin = h.bin_left[0];
    const xMax = h.bin_right[n - 1];

    // Build raw rows: [x_center, count, left, right]
    const raw = h.bin_left.map((l, i) => {
      const r = h.bin_right[i];
      const x = (l + r) / 2;
      return [x, h.counts[i], l, r];
    });

    // ---- logY trick: plot log10(count+1) on a linear axis ----
    const log10p1 = (c) => Math.log10(c + 1);

    const maxCnt = Math.max(0, ...h.counts);
    const yMax = Math.max(1, Math.ceil(Math.log10(maxCnt + 1))); // decades

    // If logY: data becomes [x, yLog, left, right, originalCount]
    const data = logY
      ? raw.map(([x, cnt, l, r]) => [x, log10p1(cnt), l, r, cnt])
      : raw;

    return {
      title: { text: title },
      grid: { left: 70, right: 20, top: 60, bottom: 60 },

      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line", snap: true },
        formatter: (params) => {
          const p = params?.[0];
          if (!p?.data) return "";
          if (logY) {
            const [, , l, r, cnt] = p.data;
            return [
              `Bin: ${Number(l).toLocaleString()} – ${Number(r).toLocaleString()}`,
              `Count: ${Number(cnt).toLocaleString()}`,
            ].join("<br/>");
          } else {
            const [, cnt, l, r] = p.data;
            return [
              `Bin: ${Number(l).toLocaleString()} – ${Number(r).toLocaleString()}`,
              `Count: ${Number(cnt).toLocaleString()}`,
            ].join("<br/>");
          }
        },
      },

      xAxis: {
        type: logX ? "log" : "value",
        name: xTitle,
        min: logX ? Math.max(1, xMin) : xMin,
        max: xMax,
        boundaryGap: [0, 0],
        axisLabel: { formatter: (v) => Number(v).toLocaleString() },
      },

      yAxis: logY
        ? {
            type: "value",
            name: "Number of groups",
            min: 0,
            max: yMax,
            interval: 1, // ticks at 0,1,2,3... (decades)
            axisLabel: {
              formatter: (v) => {
                if (v <= 0) return "0";
                return Number(Math.pow(10, v)).toLocaleString(); // 10, 100, 1 000...
              },
            },
          }
        : {
            type: "value",
            name: "Number of groups",
            min: 0,
            axisLabel: { formatter: (v) => Number(v).toLocaleString() },
          },

      series: [
        {
          type: "bar",
          data,
          large: true,
          clip: true,
          barWidth: "95%",
          barMinWidth: 1,
          encode: { x: 0, y: 1 },
        },
      ],
    };
  }, [h, title, xTitle, logX, logY]);

  if (!option) return <div>Loading…</div>;
  return <ReactECharts option={option} style={{ height, width: "100%" }} />;
}