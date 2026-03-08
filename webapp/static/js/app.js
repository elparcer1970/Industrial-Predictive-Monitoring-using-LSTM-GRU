/* static/js/app.js — SensorAI global helpers */

/* ── Navbar scroll effect ──────────────────────────────────── */
(function () {
  const nav = document.getElementById("mainNav");
  if (!nav) return;
  window.addEventListener("scroll", () => {
    if (window.scrollY > 40) {
      nav.style.boxShadow = "0 4px 24px rgba(0,0,0,0.5)";
    } else {
      nav.style.boxShadow = "none";
    }
  }, { passive: true });
})();

/* ── Smooth in-view animation ─────────────────────────────── */
(function () {
  if (!("IntersectionObserver" in window)) return;
  const style = document.createElement("style");
  style.textContent = `
    .reveal { opacity: 0; transform: translateY(24px); transition: opacity 0.5s ease, transform 0.5s ease; }
    .reveal.visible { opacity: 1; transform: none; }
  `;
  document.head.appendChild(style);

  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add("visible"); observer.unobserve(e.target); }
    });
  }, { threshold: 0.12 });

  document.querySelectorAll(
    ".feature-card, .metric-card, .pipe-block, .kpi-card, .scenario-card, .about-section"
  ).forEach(el => { el.classList.add("reveal"); observer.observe(el); });
})();
