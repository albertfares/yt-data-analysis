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

    // Build [x_center, count, left, right]
    const raw = h.bin_left.map((l, i) => {
      const r = h.bin_right[i];
      const x = (l + r) / 2;
      return [x, h.counts[i], l, r];
    });

    // log10(count+1) so 0 is representable
    const log10p1 = (c) => Math.log10(c + 1);

    const maxCnt = Math.max(0, ...h.counts);
    const yMax = Math.max(1, Math.ceil(Math.log10(maxCnt + 1))); // at least 1 decade

    const data = logY
      ? raw.map(([x, cnt, l, r]) => [x, log10p1(cnt), l, r, cnt]) // keep cnt for tooltip
      : raw;

    return {
      title: { text: title },
      grid: { left: 70, right: 20, top: 60, bottom: 60 },

      tooltip: {
        trigger: "axis",
        axisPointer: { type: "line", snap: true },
        formatter: (params) => {
          const p = params[0];
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
        min: logX ? 1 : null,
        axisLabel: { formatter: (v) => Number(v).toLocaleString() },
      },

      yAxis: logY
        ? {
            type: "value",
            name: "Number of groups",
            min: 0,
            max: yMax,
            interval: 1, // decades
            axisLabel: {
              formatter: (v) => {
                if (v === 0) return "0";
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