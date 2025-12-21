import { useEffect, useRef } from "react";

/**
 * Comment Flow background:
 * - Sankey-ish flowing ribbons (bezier curves)
 * - gentle drift + subtle pulse
 * - optional particles moving along paths
 * - DPR-aware, resize-safe, RAF cleaned up
 */
export default function BackgroundCommentFlow() {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: true });

    let w = 0, h = 0, dpr = 1;
    const reduceMotion =
      window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

    const rand = (a, b) => a + Math.random() * (b - a);
    const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

    // Theme color (YouTube-red-ish). Change these 3 numbers to change color.
    const COLOR = { r: 231, g: 76, b: 60 };

    // --- Sankey-like lanes ---
    let sources = [];
    let sinks = [];
    let flows = [];
    let packets = [];

    function pickLaneYs(count) {
      // Spread evenly with jitter
      const ys = [];
      const top = h * 0.15;
      const bottom = h * 0.85;
      for (let i = 0; i < count; i++) {
        const t = count === 1 ? 0.5 : i / (count - 1);
        ys.push(top + t * (bottom - top) + rand(-h * 0.03, h * 0.03));
      }
      return ys;
    }

    function buildFlow() {
      flows = [];
      packets = [];

      // “bars” (like Sankey columns)
      const leftX = w * 0.06;
      const rightX = w * 0.94;

      const nSources = clamp(Math.round(w / 260), 4, 8);
      const nSinks = clamp(Math.round(w / 240), 4, 9);

      const srcYs = pickLaneYs(nSources);
      const sinkYs = pickLaneYs(nSinks);

      sources = srcYs.map((y, i) => ({
        x: leftX,
        y,
        width: 6,
        height: rand(22, 44),
        phase: rand(0, Math.PI * 2),
      }));

      sinks = sinkYs.map((y, i) => ({
        x: rightX,
        y,
        width: 6,
        height: rand(22, 44),
        phase: rand(0, Math.PI * 2),
      }));

      // Create connections: each source connects to 2–4 sinks
      for (let i = 0; i < sources.length; i++) {
        const k = clamp(Math.floor(rand(2, 5)), 2, 4);
        const targets = new Set();
        while (targets.size < k) targets.add(Math.floor(rand(0, sinks.length)));

        for (const j of targets) {
          const weight = rand(0.6, 2.2); // ribbon thickness basis
          flows.push({
            a: i,
            b: j,
            w: weight,
            // curve control offsets
            c1x: rand(w * 0.25, w * 0.45),
            c2x: rand(w * 0.55, w * 0.75),
            yJitter: rand(-h * 0.02, h * 0.02),
            phase: rand(0, Math.PI * 2),
          });

          // add a couple of packets per flow (moving dots)
          if (!reduceMotion && Math.random() < 0.7) {
            const nPackets = Math.random() < 0.5 ? 1 : 2;
            for (let p = 0; p < nPackets; p++) {
              packets.push({
                flowIndex: flows.length - 1,
                t: Math.random(), // 0..1 along curve
                speed: rand(0.0012, 0.0022),
                size: rand(1.2, 2.2),
                phase: rand(0, Math.PI * 2),
              });
            }
          }
        }
      }
    }

    function resize() {
      dpr = window.devicePixelRatio || 1;
      w = Math.floor(window.innerWidth);
      h = Math.floor(window.innerHeight);

      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      buildFlow();
    }

    // cubic bezier point
    function bezierPoint(P0, P1, P2, P3, t) {
      const u = 1 - t;
      const tt = t * t;
      const uu = u * u;
      const uuu = uu * u;
      const ttt = tt * t;
      return {
        x: uuu * P0.x + 3 * uu * t * P1.x + 3 * u * tt * P2.x + ttt * P3.x,
        y: uuu * P0.y + 3 * uu * t * P1.y + 3 * u * tt * P2.y + ttt * P3.y,
      };
    }

    function drawBar(x, y, height, time, phase) {
      // subtle pulsing opacity
      const pulse = 0.5 + 0.5 * Math.sin(time * 0.9 + phase);
      const alpha = 0.035 + 0.035 * pulse;

      ctx.fillStyle = `rgba(${COLOR.r}, ${COLOR.g}, ${COLOR.b}, ${alpha})`;
      ctx.beginPath();
      ctx.roundRect(x - 3, y - height / 2, 6, height, 6);
      ctx.fill();
    }

    function drawRibbon(flow, time) {
      const A = sources[flow.a];
      const B = sinks[flow.b];

      const y0 = A.y + flow.yJitter;
      const y3 = B.y - flow.yJitter;

      const P0 = { x: A.x, y: y0 };
      const P3 = { x: B.x, y: y3 };
      const P1 = { x: flow.c1x, y: y0 + rand(-2, 2) };
      const P2 = { x: flow.c2x, y: y3 + rand(-2, 2) };

      // thickness + pulse
      const pulse = 0.6 + 0.4 * Math.sin(time * 0.8 + flow.phase);
      const thickness = flow.w * (1.6 + pulse); // px

      // gradient-ish stroke
      const alpha = 0.012 + 0.03 * pulse; // super subtle
      ctx.strokeStyle = `rgba(${COLOR.r}, ${COLOR.g}, ${COLOR.b}, ${alpha})`;
      ctx.lineWidth = thickness;
      ctx.lineCap = "round";

      ctx.beginPath();
      ctx.moveTo(P0.x, P0.y);
      ctx.bezierCurveTo(P1.x, P1.y, P2.x, P2.y, P3.x, P3.y);
      ctx.stroke();

      return { P0, P1, P2, P3 };
    }

    function step(ts) {
      const time = ts * 0.001;

      // clear
      ctx.clearRect(0, 0, w, h);

      // very faint “paper” wash (optional)
      ctx.fillStyle = `rgba(${COLOR.r}, ${COLOR.g}, ${COLOR.b}, 0.015)`;
      ctx.fillRect(0, 0, w, h);

      // bars (left/right)
      for (const s of sources) drawBar(s.x, s.y, s.height, time, s.phase);
      for (const t of sinks) drawBar(t.x, t.y, t.height, time, t.phase);

      // ribbons
      const curves = flows.map((f) => drawRibbon(f, time));

      // packets (moving along ribbons)
      if (!reduceMotion) {
        for (const pk of packets) {
          const curve = curves[pk.flowIndex];
          pk.t += pk.speed;
          if (pk.t > 1) pk.t -= 1;

          const pos = bezierPoint(curve.P0, curve.P1, curve.P2, curve.P3, pk.t);

          const pulse = 0.5 + 0.5 * Math.sin(time * 2.0 + pk.phase);
          const alpha = 0.06 + 0.08 * pulse;

          ctx.beginPath();
          ctx.arc(pos.x, pos.y, pk.size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${COLOR.r}, ${COLOR.g}, ${COLOR.b}, ${alpha})`;
          ctx.fill();
        }
      }

      rafRef.current = requestAnimationFrame(step);
    }

    resize();
    rafRef.current = requestAnimationFrame(step);
    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return <canvas id="background-canvas" ref={canvasRef} />;
}