import { useEffect, useRef } from "react";

/**
 * Ghost Network background:
 * - faint nodes + faint edges
 * - slow drift (almost imperceptible)
 * - no mouse interaction (so it stays calm while reading)
 * - DPR-aware, resize-safe, and RAF cleaned up
 */
export default function BackgroundGhostNetwork() {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: true });
    let w = 0, h = 0, dpr = 1;

    // Respect "reduced motion"
    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches;

    // --- helpers ---
    const rand = (a, b) => a + Math.random() * (b - a);
    const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

    // --- network state ---
    let nodes = [];
    let edges = [];

    // Make a "user-group" vibe via clustered neighborhoods
    function buildNetwork() {
      nodes = [];
      edges = [];

      // density is mild; adjust if you want more/less
      const baseCount = Math.floor((w * h) / 22000); // ~40-70 on typical screens
      const count = clamp(baseCount, 28, 85);

      // Create a few cluster centers
      const clusterCount = clamp(Math.round(count / 12), 3, 7);
      const clusters = Array.from({ length: clusterCount }, () => ({
        x: rand(0.15, 0.85) * w,
        y: rand(0.15, 0.85) * h,
        r: rand(90, 190),
      }));

      // Node generation (clustered + some roamers)
      for (let i = 0; i < count; i++) {
        const useCluster = Math.random() < 0.82;
        const c = clusters[Math.floor(Math.random() * clusters.length)];

        const x = useCluster ? c.x + rand(-c.r, c.r) : rand(0, w);
        const y = useCluster ? c.y + rand(-c.r, c.r) : rand(0, h);

        const size = rand(1.0, 2.2);
        const vx = rand(-0.12, 0.12);
        const vy = rand(-0.12, 0.12);

        nodes.push({
          x: clamp(x, 0, w),
          y: clamp(y, 0, h),
          vx,
          vy,
          r: size,
          // mild individuality for pulsing
          phase: rand(0, Math.PI * 2),
        });
      }

      // Build edges by connecting to nearest neighbors
      // (keeps it network-y without turning into a hairball)
      const maxLinksPerNode = 3;
      const maxDist2 = (Math.min(w, h) * 0.23) ** 2;

      for (let i = 0; i < nodes.length; i++) {
        // gather distances
        const dists = [];
        for (let j = 0; j < nodes.length; j++) {
          if (i === j) continue;
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const d2 = dx * dx + dy * dy;
          if (d2 < maxDist2) dists.push({ j, d2 });
        }
        dists.sort((a, b) => a.d2 - b.d2);

        const links = dists.slice(0, maxLinksPerNode);
        for (const { j, d2 } of links) {
          // avoid duplicates by enforcing i < j
          if (i < j) {
            edges.push({
              a: i,
              b: j,
              d2,
              // edge pulse timing
              phase: rand(0, Math.PI * 2),
            });
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
      buildNetwork();
    }

    function step(t) {
      // clear
      ctx.clearRect(0, 0, w, h);

      // background wash (optional, extremely subtle)
      // Comment out if you want *only* nodes.
      ctx.fillStyle = "rgba(231, 76, 60, 0.02)";
      ctx.fillRect(0, 0, w, h);

      // animate nodes (very slow drift)
      if (!reduceMotion) {
        for (const n of nodes) {
          n.x += n.vx;
          n.y += n.vy;

          // bounce softly
          if (n.x < 0 || n.x > w) n.vx *= -1;
          if (n.y < 0 || n.y > h) n.vy *= -1;

          n.x = clamp(n.x, 0, w);
          n.y = clamp(n.y, 0, h);
        }
      }

      // draw edges (faded)
      const time = t * 0.001;
      for (const e of edges) {
        const A = nodes[e.a];
        const B = nodes[e.b];
        const dx = A.x - B.x;
        const dy = A.y - B.y;
        const d2 = dx * dx + dy * dy;

        // fade with distance
        const distFade = 1 - clamp(d2 / (Math.min(w, h) * 0.28) ** 2, 0, 1);

        // gentle pulse (very slow)
        const pulse = 0.5 + 0.5 * Math.sin(time * 0.8 + e.phase);

        // final alpha is tiny (ghost)
        const alpha = 0.015 + 0.05 * distFade * pulse;

        ctx.strokeStyle = `rgba(231, 76, 60, ${alpha})`; // YouTube-red-ish
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(A.x, A.y);
        ctx.lineTo(B.x, B.y);
        ctx.stroke();
      }

      // draw nodes
      for (const n of nodes) {
        const pulse = 0.6 + 0.4 * Math.sin(time * 1.1 + n.phase);
        const alpha = 0.06 + 0.06 * pulse; // still subtle

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(231, 76, 60, ${alpha})`;
        ctx.fill();
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