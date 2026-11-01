const motion = window.matchMedia('(prefers-reduced-motion: reduce)');

// Duplicate each strip once so the last card meets the first without a jump.
document.querySelectorAll('.film-row').forEach(row => {
  const set = row.querySelector('.film-set');
  const duplicate = set.cloneNode(true);
  duplicate.setAttribute('aria-hidden', 'true');
  row.append(duplicate);
  const measure = () => row.style.setProperty('--film-width', `${set.getBoundingClientRect().width + parseFloat(getComputedStyle(row).gap)}px`);
  new ResizeObserver(measure).observe(set);
  measure();
});

function syncMotion() {
  document.querySelectorAll('.film-row').forEach(row => row.style.animationPlayState = document.hidden || motion.matches ? 'paused' : 'running');
}
motion.addEventListener('change', syncMotion);
document.addEventListener('visibilitychange', syncMotion);
syncMotion();
