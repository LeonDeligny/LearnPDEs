const sidebar = document.querySelector('#site-sidebar');
const modeTrigger = document.querySelector('.sidebar-mode-trigger');
const modeMenu = document.querySelector('#sidebar-modes');
const modeLabel = document.querySelector('#sidebar-mode-label');
const modeNote = document.querySelector('#sidebar-mode-note');
const narrowScreen = matchMedia('(max-width: 760px)');
const modes = { collapsed: 'Collapsed', hover: 'Expand on hover', expanded: 'Expanded' };
let preferredMode;
let touchExpandPending = false;

try {
  const saved = localStorage.getItem('learnpdes-sidebar-mode');
  if (Object.hasOwn(modes, saved)) preferredMode = saved;
} catch {
  // A storage restriction must not disable navigation or search.
}

// Figures stack their comparison panels on narrow screens. Match the frame to
// its content, accepting messages only from one of our own figure windows.
window.addEventListener('message', event => {
  if (event.origin !== location.origin || event.data?.type !== 'learnpdes:figure-height') return;
  const height = event.data.height;
  if (!Number.isFinite(height) || height < 300 || height > 3000) return;
  for (const frame of document.querySelectorAll('iframe.training-plot')) {
    if (frame.contentWindow === event.source) frame.style.height = `${Math.ceil(height)}px`;
  }
});

function setSidebarMode(mode, remember = false) {
  if (!Object.hasOwn(modes, mode)) return;
  const autoCollapsed = narrowScreen.matches && mode !== 'collapsed';
  const effectiveMode = narrowScreen.matches ? 'collapsed' : mode;
  document.documentElement.dataset.sidebarPreference = mode;
  document.documentElement.dataset.sidebarMode = effectiveMode;
  modeLabel.textContent = autoCollapsed ? 'Collapsed (auto)' : modes[effectiveMode];
  modeTrigger.title = autoCollapsed
    ? `Collapsed to fit; preference: ${modes[mode]}`
    : `Sidebar mode: ${modes[mode]}`;
  modeNote.hidden = !autoCollapsed;
  for (const radio of modeMenu.querySelectorAll('input')) radio.checked = radio.value === mode;
  sidebar.removeAttribute('data-hover-open');
  if (remember) {
    preferredMode = mode;
    try { localStorage.setItem('learnpdes-sidebar-mode', mode); } catch { /* Optional preference. */ }
  }
}

function closeModeMenu(restoreFocus = false) {
  modeMenu.hidden = true;
  modeTrigger.setAttribute('aria-expanded', 'false');
  if (restoreFocus) modeTrigger.focus();
}

modeTrigger.addEventListener('click', () => {
  if (!modeMenu.hidden) {
    closeModeMenu();
    return;
  }
  modeMenu.hidden = false;
  modeTrigger.setAttribute('aria-expanded', 'true');
  modeMenu.querySelector(':checked').focus();
});
modeMenu.addEventListener('change', event => {
  setSidebarMode(event.target.value, true);
});
modeMenu.addEventListener('focusout', event => {
  // Pressing a label briefly blurs its radio before the forwarded click selects
  // the new one. Do not hide the menu during that pointer interaction.
  if (!event.relatedTarget) return;
  requestAnimationFrame(() => {
    if (!modeMenu.contains(document.activeElement) && document.activeElement !== modeTrigger) closeModeMenu();
  });
});
document.addEventListener('pointerdown', event => {
  if (!modeMenu.contains(event.target) && !modeTrigger.contains(event.target)) closeModeMenu();
  if (!sidebar.contains(event.target) && !modeMenu.contains(event.target)) sidebar.removeAttribute('data-hover-open');
});
sidebar.addEventListener('pointerdown', event => {
  touchExpandPending = event.pointerType === 'touch' &&
    document.documentElement.dataset.sidebarMode === 'hover' &&
    sidebar.getBoundingClientRect().width < 100;
});
sidebar.addEventListener('click', event => {
  if (touchExpandPending && event.target.closest('.nav-item')) {
    event.preventDefault();
    sidebar.setAttribute('data-hover-open', '');
  }
  touchExpandPending = false;
});
sidebar.addEventListener('pointerleave', () => {
  const focused = document.activeElement;
  if (sidebar.contains(focused) && !focused.matches(':focus-visible')) focused.blur();
});
narrowScreen.addEventListener('change', () => {
  setSidebarMode(preferredMode || (narrowScreen.matches ? 'collapsed' : 'expanded'));
});
setSidebarMode(document.documentElement.dataset.sidebarPreference);

const dialog = document.querySelector('#site-search');
const input = document.querySelector('#search-input');
const results = document.querySelector('#search-results');
const status = document.querySelector('.search-status');
let indexPromise;

function loadSearchIndex() {
  if (!indexPromise) {
    indexPromise = fetch(dialog.dataset.indexUrl)
      .then(response => {
        if (!response.ok) throw new Error('Search index unavailable');
        return response.json();
      })
      .then(index => index.docs)
      .catch(error => {
        indexPromise = undefined;
        throw error;
      });
  }
  return indexPromise;
}

function openSearch() {
  closeModeMenu();
  sidebar.removeAttribute('data-hover-open');
  if (!dialog.open) dialog.showModal();
  input.focus();
}

document.querySelector('[data-open-search]').addEventListener('click', openSearch);
document.querySelector('[data-close-search]').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  const rect = dialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right ||
      event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    if (!modeMenu.hidden) {
      event.preventDefault();
      closeModeMenu(true);
    }
    sidebar.removeAttribute('data-hover-open');
    if (dialog.open) {
      event.preventDefault();
      dialog.close();
    }
  }
  const typing = event.target.closest('input, textarea, [contenteditable="true"]');
  if ((!typing && event.key === '/') || ((event.ctrlKey || event.metaKey) && event.key === 'k')) {
    event.preventDefault();
    openSearch();
  }
});

input.addEventListener('input', async () => {
  results.replaceChildren();
  if (!input.value.trim()) {
    status.textContent = '';
    return;
  }
  status.textContent = 'Searching…';
  try {
    const entries = await loadSearchIndex();
    const terms = input.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    if (!terms.length) {
      status.textContent = '';
      return;
    }
    const matches = entries.map(entry => {
      const title = entry.title.toLowerCase();
      const text = `${title} ${entry.text}`.toLowerCase();
      const score = terms.every(term => text.includes(term))
        ? 1 + terms.filter(term => title.includes(term)).length * 3 : 0;
      return { ...entry, score };
    }).filter(entry => entry.score).sort((a, b) => b.score - a.score).slice(0, 10);
    results.replaceChildren();
    status.textContent = matches.length
      ? `${matches.length} matching ${matches.length === 1 ? 'section' : 'sections'}`
      : 'No matches.';
    for (const entry of matches) {
      const link = document.createElement('a');
      link.className = 'search-result';
      link.href = new URL(entry.location, new URL(dialog.dataset.baseUrl, location.href));
      const title = document.createElement('strong');
      title.textContent = entry.title;
      const excerpt = document.createElement('p');
      const text = entry.text.replace(/\s+/g, ' ').trim();
      const position = text.toLowerCase().indexOf(terms[0]);
      const start = Math.max(0, position - 55);
      excerpt.textContent = `${start ? '…' : ''}${text.slice(start, start + 175)}${text.length > start + 175 ? '…' : ''}`;
      link.append(title, excerpt);
      link.addEventListener('click', () => dialog.close());
      results.append(link);
    }
  } catch {
    status.textContent = 'Search unavailable. Try again.';
  }
});

document.querySelectorAll('.article pre').forEach(pre => {
  const wrapper = document.createElement('div');
  wrapper.className = 'code-block';
  pre.before(wrapper);
  wrapper.append(pre);
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'copy-button';
  button.textContent = 'Copy';
  button.setAttribute('aria-label', 'Copy code');
  button.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.textContent);
      button.textContent = 'Copied';
    } catch {
      button.textContent = 'Copy unavailable';
    }
    setTimeout(() => { button.textContent = 'Copy'; }, 1800);
  });
  wrapper.append(button);
});
