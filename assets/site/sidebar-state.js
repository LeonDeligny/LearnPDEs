// Set the saved layout before painting, including when storage is unavailable.
(() => {
  const narrow = matchMedia('(max-width: 760px)').matches;
  let mode = narrow ? 'collapsed' : 'expanded';
  try {
    const saved = localStorage.getItem('learnpdes-sidebar-mode');
    if (['collapsed', 'hover', 'expanded'].includes(saved)) mode = saved;
  } catch {
    // Private browsing and disabled storage still get a usable layout.
  }
  document.documentElement.dataset.sidebarPreference = mode;
  document.documentElement.dataset.sidebarMode = narrow ? 'collapsed' : mode;
})();
