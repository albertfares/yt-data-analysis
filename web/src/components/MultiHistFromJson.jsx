import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";

export default function MultiHistFromJson({
  jsonPath,
  title,
  xName,
  yName = "Number of groups",
  logX = false,
  logY = true,
  height = 420,
}) {
  const [h, setH] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    setError(null);
    fetch(import.meta.env.BASE_URL + jsonPath)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setH)
      .catch((err) => {
        console.error("Error loading JSON for MultiHistFromJson:", err);
        setError(err.message);
      });
  }, [jsonPath]);

  const option = useMemo(() => {
    if (!h || error) return null;

    const countsArray = h.counts ?? h.count_groups;

    const ok =
      Array.isArray(h.bin_left) &&
      Array.isArray(h.bin_right) &&
      Array.isArray(h.thresholds) &&
      Array.isArray(countsArray);

    if (!ok) {
      console.error("MultiHistFromJson: invalid JSON shape", h);
      setError("Invalid data format for histogram overlay.");
      return null;
    }

    const xCenters = h.bin_left.map((l, i) => (l + h.bin_right[i]) / 2);

    const series = h.thresholds.map((th, idx) => {
      const counts = countsArray[idx] || [];
      const data = xCenters.map((x, i) => {
        const raw = counts[i] ?? 0;
        const y = logY ? Math.max(raw, 1) : raw; // avoid 0 on log axis
        return [x, y];
      });
      return {
        name: `≥ ${th} comments`,
        type: "line",
        data,
        showSymbol: false,
      };
    });

    return {
      title: { text: title },
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "shadow" },
        formatter: (params) => {
          if (!params.length) return "";
          const p = params[0];
          const x = p.data[0];
          const y = p.data[1];
          const th = p.seriesName.replace("≥ ", "").replace(" comments", "");
          const fmt = (v) => Math.round(v).toLocaleString();
          return [
            `Bin center: ${fmt(x)}`,
            `${yName}: ${fmt(y)}`,
            `Threshold: ≥ ${th} comments`,
          ].join("<br/>");
        },
      },
      legend: { top: 40 },
      xAxis: {
        type: logX ? "log" : "value",
        name: xName,
        min: logX ? 1 : 0,
        logBase: logX ? 10 : undefined,
        axisLabel: { formatter: (v) => Number(v).toLocaleString() },
      },
      yAxis: {
        type: logY ? "log" : "value",
        name: yName,
        min: logY ? 1 : 0,
        logBase: logY ? 10 : undefined,
        axisLabel: { formatter: (v) => Number(v).toLocaleString() },
      },
      series,
      grid: { left: 80, right: 20, bottom: 60, top: 80 },
    };
  }, [h, error, title, xName, yName, logX, logY]);

  if (error) {
    return <div style={{ color: "red" }}>Error: {error}</div>;
  }
  if (!h || !option) {
    return <div>Loading…</div>;
  }

  return <ReactECharts option={option} style={{ height, width: "100%" }} />;
}

