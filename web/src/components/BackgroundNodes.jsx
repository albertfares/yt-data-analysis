import { useEffect, useRef } from "react";

export default function BackgroundNodes() {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d", { alpha: true });

    let particles = [];
    const mouse = { x: null, y: null, r: 120 };

    function resize() {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const w = window.innerWidth;
      const h = window.innerHeight;

      // CSS size
      canvas.style.width = `${w}px`;
      canvas.style.height = `${h}px`;

      // Drawing buffer size
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);

      // Draw in CSS pixels
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      mouse.r = Math.min(180, Math.max(80, (h / 80) * (w / 80)));
      init();
    }

    class Particle {
      constructor(x, y, vx, vy, size) {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.size = size;
      }
      draw() {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(231, 76, 60, 0.6)';
        ctx.fill();
        ctx.strokeStyle = `rgba(231, 76, 60, 0.9})`;
      }
      update(w, h) {
        // bounce
        if (this.x > w || this.x < 0) this.vx *= -1;
        if (this.y > h || this.y < 0) this.vy *= -1;

        // mouse repulsion
        if (mouse.x != null && mouse.y != null) {
          const dx = mouse.x - this.x;
          const dy = mouse.y - this.y;
          const dist = Math.hypot(dx, dy);
          if (dist < mouse.r + this.size) {
            const force = (mouse.r - dist) / mouse.r;
            const fx = (dx / (dist || 1)) * force * 5;
            const fy = (dy / (dist || 1)) * force * 5;
            this.x -= fx;
            this.y -= fy;
          }
        }

        this.x += this.vx;
        this.y += this.vy;
        this.draw();
      }
    }

    function init() {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const count = Math.floor((w * h) / 9000);

      particles = [];
      for (let i = 0; i < count; i++) {
        const size = Math.random() * 2.5 + 1; // a bit subtler
        const x = Math.random() * (w - size * 2) + size;
        const y = Math.random() * (h - size * 2) + size;
        const vx = (Math.random() - 0.5) * 0.7;
        const vy = (Math.random() - 0.5) * 0.7;
        particles.push(new Particle(x, y, vx, vy, size));
      }
    }

    function connect() {
      const w = window.innerWidth;
      const h = window.innerHeight;
      const threshold = (w / 7) * (h / 7);

      for (let a = 0; a < particles.length; a++) {
        for (let b = a + 1; b < particles.length; b++) {
          const dx = particles[a].x - particles[b].x;
          const dy = particles[a].y - particles[b].y;
          const d2 = dx * dx + dy * dy;

          if (d2 < threshold) {
            const opacity = 1 - d2 / 20000;
            ctx.strokeStyle = `rgba(231, 76, 60, ${Math.max(0, Math.min(0.35, opacity))})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(particles[a].x, particles[a].y);
            ctx.lineTo(particles[b].x, particles[b].y);
            ctx.stroke();
          }
        }
      }
    }

    function animate() {
      const w = window.innerWidth;
      const h = window.innerHeight;

      ctx.clearRect(0, 0, w, h);
      for (const p of particles) p.update(w, h);
      connect();

      rafRef.current = requestAnimationFrame(animate);
    }

    // events
    const onMove = (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
    };
    const onOut = () => {
      mouse.x = null;
      mouse.y = null;
    };
    const onVis = () => {
      // if browser throttled/stopped, restart cleanly
      cancelAnimationFrame(rafRef.current);
      rafRef.current = requestAnimationFrame(animate);
    };

    window.addEventListener("resize", resize);
    window.addEventListener("mousemove", onMove, { passive: true });
    window.addEventListener("mouseout", onOut);
    document.addEventListener("visibilitychange", onVis);

    resize();
    rafRef.current = requestAnimationFrame(animate);

    return () => {
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", resize);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseout", onOut);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return <canvas id="background-canvas" ref={canvasRef} aria-hidden="true" />;
}