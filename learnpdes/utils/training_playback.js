// Inline in each export so playback also works offline.
const plot = document.getElementById('{plot_id}');
const frames = plot._transitionData._frames;
const frameIndices = new Map(frames.map((frame, index) => [frame.name, index]));
let current = plot.layout.sliders[0].active || 0;
let playing = false;
let revision = 0;
let timer;
let rendering = Promise.resolve();

function pause() {
    playing = false;
    revision++;
    clearTimeout(timer);
}

// Playback, seeking, field selection, and resizing share one rendering queue.
// A superseded seek is skipped before it can paint an obsolete checkpoint.
function enqueue(action) {
    rendering = rendering.then(action).catch(error => {
        pause();
        console.error('Unable to update the training figure:', error);
    });
    return rendering;
}

function showFrame(index, expectedRevision) {
    return enqueue(async () => {
        if (expectedRevision !== revision) return;
        const frame = frames[index];
        const keys = new Set(frame.data.flatMap(trace => Object.keys(trace)));
        keys.delete('type');
        // Undefined entries leave the corresponding trace attribute alone.
        // Update retains the existing heatmap images; animate with redraw
        // removes them, leaving a white flash while replacements are painted.
        const data = Object.fromEntries([...keys].map(key => [
            key, frame.data.map(trace => trace[key])
        ]));
        await Plotly.update(plot, data, {
            'title.text': frame.layout.title.text,
            'sliders[0].active': index
        }, frame.traces);
        current = index;
    });
}

async function advance(index, expectedRevision, duration) {
    if (!playing || expectedRevision !== revision) return;
    const started = performance.now();
    await showFrame(index, expectedRevision);
    if (!playing || expectedRevision !== revision) return;
    if (index === frames.length - 1) {
        playing = false;
        return;
    }
    // Slow devices finish the current render before scheduling the next one.
    timer = setTimeout(() => advance(index + 1, expectedRevision, duration),
        Math.max(0, duration - (performance.now() - started)));
}

plot.on('plotly_buttonclicked', event => {
    const button = event.button;
    if (button.method === 'update') {
        const [data, layout = {}, traces] = button.args;
        const active = event.menu.buttons.indexOf(button);
        enqueue(() => Plotly.update(plot, data, {
            ...layout, [`updatemenus[${event.menu._index}].active`]: active
        }, traces));
    } else if (button.label === 'Pause') {
        pause();
    } else if (button.label === 'Play' && !playing) {
        playing = true;
        const duration = button.args[1].frame.duration;
        advance(current === frames.length - 1 ? 0 : current + 1, ++revision, duration);
    }
});
plot.on('plotly_sliderchange', event => {
    if (!event.interaction) return;
    const index = frameIndices.get(event.step.args[0][0]);
    if (index === undefined) return;
    pause();
    showFrame(index, revision);
});

const compact = window.matchMedia('(max-width: 700px)');
function reportHeight() {
    if (window.parent !== window) window.parent.postMessage({
        type: 'learnpdes:figure-height', height: Math.ceil(plot.getBoundingClientRect().bottom + window.scrollY) + 8
    }, window.location.origin);
}
function alignFields() {
    const update = {};
    const bars = { 'colorbar.x': [], 'colorbar.y': [], 'colorbar.len': [] };
    const indices = [];
    plot.data.forEach((trace, index) => {
        if (trace.type !== 'heatmap') return;
        const x = plot._fullLayout['xaxis' + trace.xaxis.slice(1)].domain;
        const y = plot._fullLayout['yaxis' + trace.yaxis.slice(1)].domain;
        if (trace.coloraxis) {
            update[`${trace.coloraxis}.colorbar.x`] = x[1] + .01;
            update[`${trace.coloraxis}.colorbar.y`] = (y[0] + y[1]) / 2;
            update[`${trace.coloraxis}.colorbar.len`] = y[1] - y[0];
        } else {
            indices.push(index);
            bars['colorbar.x'].push(x[1] + .01);
            bars['colorbar.y'].push((y[0] + y[1]) / 2);
            bars['colorbar.len'].push(y[1] - y[0]);
        }
        if (index < 3) {
            update[`annotations[${index}].x`] = (x[0] + x[1]) / 2;
            update[`annotations[${index}].y`] = y[1];
            update[`annotations[${index}].yshift`] = 14;
        }
    });
    return indices.length ? Plotly.update(plot, bars, update, indices) : Plotly.relayout(plot, update);
}
let scheduled = false;
function fitFigure() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
        enqueue(async () => {
            scheduled = false;
            const layouts = plot.layout.meta?.responsive_layout;
            if (!layouts) return;
            await Plotly.relayout(plot, {
                ...layouts[compact.matches ? 'compact' : 'desktop'],
                width: plot.clientWidth
            });
            await alignFields();
            reportHeight();
        });
    });
}
window.addEventListener('resize', fitFigure);
fitFigure();
