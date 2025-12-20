import { useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";

function clamp(v, a, b) {
  return Math.max(a, Math.min(b, v));
}

export default function KMeansMedianExplorer({
  basePath = "data/kmeans_explorer",
  height = 520,
  Kmin = 1,
  Kmax = 15,
  useAllKFile = false, // if true: load profiles_allK.json once
}) {
  const [K, setK] = useState(12);

  const [allK, setAllK] = useState(null);               // { profiles: { "12": [...] } }
  const [profilesCache, setProfilesCache] = useState({}); // { "12": {K, profiles:[...]} }

  // Used to ignore stale fetch responses when user drags slider quickly
  const latestReqId = useRef(0);

  // ---------- Load all-K file (optional) ----------
  useEffect(() => {
    if (!useAllKFile) return;

    const controller = new AbortController();
    fetch(import.meta.env.BASE_URL + `${basePath}/profiles_allK.json`, {
      signal: controller.signal,
    })
      .then((r) => r.json())
      .then(setAllK)
      .catch((e) => {
        if (e.name !== "AbortError") console.error(e);
      });

    return () => controller.abort();
  }, [basePath, useAllKFile]);

  // ---------- Load per-K file (default) with cancellation + stale-response guard ----------
  useEffect(() => {
    if (useAllKFile) return;

    const key = String(K);
    if (profilesCache[key]) return;

    const controller = new AbortController();
    const reqId = ++latestReqId.current;

    fetch(import.meta.env.BASE_URL + `${basePath}/profiles_K${K}.json`, {
      signal: controller.signal,
    })
      .then((r) => r.json())
      .then((payload) => {
        // Ignore stale response (arrived after K changed again)
        if (reqId !== latestReqId.current) return;

        setProfilesCache((prev) => {
          if (prev[key]) return prev;
          return { ...prev, [key]: payload };
        });
      })
      .catch((e) => {
        if (e.name !== "AbortError") console.error(e);
      });

    return () => controller.abort();
    // IMPORTANT: do NOT depend on profilesCache to avoid extra reruns
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [K, basePath, useAllKFile]);

  // ---------- Select payload ----------
  const payload = useMemo(() => {
    if (useAllKFile) {
      if (!allK) return null;
      return { K, profiles: allK.profiles?.[String(K)] ?? [] };
    }
    return profilesCache[String(K)] ?? null;
  }, [K, useAllKFile, allK, profilesCache]);

  // ---------- Build ECharts option ----------
  const option = useMemo(() => {
    if (!payload) return null;

    const prof = payload.profiles || [];
    if (!prof.length) return null;

    // Each profile becomes one polyline across parallel axes
    const values = prof.map((p) => {
      const tc = p.tc_median ?? 0;
      const nch = p.nch_median ?? 0;
      const fid = p.fidelity_median ?? 0;
      const cat = p.catent_median ?? 0;

      return [
        Math.log10(tc + 1),        // dim 0: log10(tc+1)
        Math.log10(nch + 1),       // dim 1: log10(nch+1)
        clamp(fid, 0, 1),          // dim 2
        clamp(cat, 0, 1),          // dim 3
        p.id ?? null,              // dim 4: id (tooltip)
        p.share ?? null,           // dim 5 optional
        p.n_points ?? null,        // dim 6 optional
      ];
    });

    const tcLogs = values.map((v) => v[0]);
    const nchLogs = values.map((v) => v[1]);

    const tcMin = Math.min(...tcLogs);
    const tcMax = Math.max(...tcLogs);
    const nchMin = Math.min(...nchLogs);
    const nchMax = Math.max(...nchLogs);

    return {
      title: { text: `Median profile explorer (K=${K})` },

      // ✅ subtle, smooth transitions
      animation: true,
      animationDuration: 220,
      animationDurationUpdate: 320,
      animationEasing: "cubicOut",
      animationEasingUpdate: "cubicOut",

      tooltip: {
        formatter: (p) => {
          const v = p.value;
          const logTc = v[0],
            logNch = v[1],
            fid = v[2],
            cat = v[3],
            id = v[4],
            share = v[5],
            nPoints = v[6];

          const tc = Math.round(Math.pow(10, logTc) - 1);
          const nch = Math.round(Math.pow(10, logNch) - 1);

          const lines = [
            `<b>Profile C${id}</b>`,
            `tc_median: ${tc.toLocaleString()}`,
            `nch_median: ${nch.toLocaleString()}`,
            `fidelity_median: ${Number(fid).toFixed(3)}`,
            `catent_median: ${Number(cat).toFixed(3)}`,
          ];
          if (share != null) lines.push(`share: ${(share * 100).toFixed(2)}%`);
          if (nPoints != null) lines.push(`points: ${Number(nPoints).toLocaleString()}`);
          return lines.join("<br/>");
        },
      },

      parallelAxis: [
        { dim: 0, name: "log10(tc_median+1)", min: tcMin, max: tcMax, realtime: false },
        { dim: 1, name: "log10(nch_median+1)", min: nchMin, max: nchMax, realtime: false },
        { dim: 2, name: "fidelity_median", min: 0, max: 1, realtime: false },
        { dim: 3, name: "catent_median", min: 0, max: 1, realtime: false },
      ],

      parallel: {
        left: 60,
        right: 60,
        top: 90,
        bottom: 30,
        parallelAxisDefault: {
          nameLocation: "end",
          nameGap: 14,
          realtime: false,
        },
      },

      series: [
        {
          id: "profiles", // ✅ stable id helps update transitions
          type: "parallel",
          data: values,

          // ✅ helps ECharts animate dataset changes
          universalTransition: true,

          lineStyle: { width: 2, opacity: 0.5 },
          emphasis: { lineStyle: { opacity: 0.95, width: 3 } },
        },
      ],
    };
  }, [payload, K]);

  // ---------- UI ----------
  return (
    <div style={{ marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 12,
          marginBottom: 10,
          flexWrap: "wrap",
        }}
      >
        <label>
          <strong>K</strong> = {K}
        </label>
        <input
          type="range"
          min={Kmin}
          max={Kmax}
          value={K}
          onChange={(e) => setK(Number(e.target.value))} // ✅ live updates while dragging
          style={{ width: 260 }}
        />
        <span style={{ opacity: 0.85 }}>
          Each line is one profile (median behavior). Drag K to see profiles split/merge.
        </span>
      </div>

      <div style={{ position: "relative" }}>
        <ReactECharts
          option={option || {}}
          style={{ height, width: "100%" }}
          lazyUpdate={true}
          notMerge={false}                // ✅ allow smooth transitions
          replaceMerge={["series"]}       // ✅ but replace the series cleanly
          opts={{ renderer: "canvas" }}   // ✅ consistent renderer
        />

        {!option && (
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              background: "rgba(0,0,0,0.04)",
            }}
          >
            Loading profiles…
          </div>
        )}
      </div>
    </div>
  );
}