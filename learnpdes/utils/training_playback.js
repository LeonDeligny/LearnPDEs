// Inline in each export so playback also works offline.
(function () {
    if (typeof window === 'undefined' || !window.document || !window.Plotly) return;
    var plot = window.document.getElementById('{plot_id}');
    if (!plot || !plot._transitionData || !plot._transitionData._frames.length) return;

    var plotly = window.Plotly;
    var frames = plot._transitionData._frames;
    var current = plot.layout.sliders[0].active || 0;
    var playing = false;
    var revision = 0;
    var timer;
    var rendering = false;
    var actions = [];

    function pause() {
        playing = false;
        revision++;
        window.clearTimeout(timer);
    }

    function failed(error) {
        pause();
        window.console.error('Unable to update the training figure:', error);
        renderNext();
    }

    // Plotly supplies the asynchronous result; no global Promise constructor is
    // needed. Playback, seeking, field selection, and resizing share this queue.
    function renderNext() {
        var action = actions.shift();
        if (!action) {
            rendering = false;
            return;
        }
        rendering = true;
        try {
            var result = action();
            if (result && typeof result.then === 'function') {
                result.then(renderNext, failed);
            } else {
                renderNext();
            }
        } catch (error) {
            failed(error);
        }
    }

    function enqueue(action) {
        actions.push(action);
        if (!rendering) renderNext();
    }

    function showFrame(index, expectedRevision, done) {
        if (typeof index !== 'number' || index % 1 !== 0 || index < 0 || index >= frames.length) return;
        enqueue(function () {
            // Skip a superseded seek before it can paint an obsolete checkpoint.
            if (expectedRevision !== revision) return;
            var frame = frames.slice(index, index + 1)[0];
            var traces = frame.data.slice();
            // These are the attributes that vary in exported checkpoints,
            // including resampled collocation points. Never copy arbitrary keys.
            var data = { x: [], y: [], z: [], name: [], customdata: [], meta: [] };
            while (traces.length) {
                var trace = traces.shift();
                data.x.push(trace.x);
                data.y.push(trace.y);
                data.z.push(trace.z);
                data.name.push(trace.name);
                data.customdata.push(trace.customdata);
                data.meta.push(trace.meta);
            }
            // Undefined entries leave the corresponding trace attribute alone.
            // Update retains heatmap images while replacements are painted.
            return plotly.update(plot, data, {
                'title.text': frame.layout.title.text,
                'sliders[0].active': index
            }, frame.traces).then(function () {
                current = index;
                if (done) done();
            });
        });
    }

    function advance(index, expectedRevision, duration) {
        if (!playing || expectedRevision !== revision) return;
        var started = new Date().getTime();
        showFrame(index, expectedRevision, function () {
            if (!playing || expectedRevision !== revision) return;
            if (index === frames.length - 1) {
                playing = false;
                return;
            }
            // Finish the current render before scheduling the next one.
            timer = window.setTimeout(function () {
                advance(index + 1, expectedRevision, duration);
            }, Math.max(0, duration - (new Date().getTime() - started)));
        });
    }

    plot.on('plotly_buttonclicked', function (event) {
        var button = event.button;
        if (button.method === 'update') {
            var buttons = event.menu.buttons.slice();
            var active = 0;
            while (buttons.length && buttons.shift() !== button) active++;
            var menuIndex = event.menu._index;
            if (typeof menuIndex !== 'number' || menuIndex % 1 !== 0 || menuIndex < 0) return;
            enqueue(function () {
                var layout = button.args[1] || {};
                layout['updatemenus[' + menuIndex + '].active'] = active;
                return plotly.update(plot, button.args[0], layout, button.args[2]);
            });
        } else if (button.label === 'Pause') {
            pause();
        } else if (button.label === 'Play' && !playing) {
            playing = true;
            var duration = button.args[1].frame.duration;
            advance(current === frames.length - 1 ? 0 : current + 1, ++revision, duration);
        }
    });
    plot.on('plotly_sliderchange', function (event) {
        if (!event.interaction) return;
        var name = event.step.args[0][0];
        var candidates = frames.slice();
        var index = 0;
        while (candidates.length) {
            if (candidates.shift().name === name) {
                pause();
                showFrame(index, revision);
                return;
            }
            index++;
        }
    });

    var compact = window.matchMedia('(max-width: 700px)');
    function reportHeight() {
        if (window.parent !== window) window.parent.postMessage({
            type: 'learnpdes:figure-height',
            height: Math.ceil(plot.getBoundingClientRect().bottom + window.scrollY) + 8
        }, window.location.origin);
    }
    function alignFields() {
        var update = {};
        var bars = { 'colorbar.x': [], 'colorbar.y': [], 'colorbar.len': [] };
        var indices = [];
        var traces = plot.data.slice();
        var index = -1;
        while (traces.length) {
            var trace = traces.shift();
            index++;
            if (trace.type !== 'heatmap') continue;
            // Only Plotly axis identifiers may form layout paths.
            if (!/^x([2-9]|[1-9][0-9]+)?$/.test(trace.xaxis) ||
                !/^y([2-9]|[1-9][0-9]+)?$/.test(trace.yaxis)) continue;
            var x = plot._fullLayout['xaxis' + trace.xaxis.slice(1)].domain;
            var y = plot._fullLayout['yaxis' + trace.yaxis.slice(1)].domain;
            if (trace.coloraxis) {
                if (!/^coloraxis([2-9]|[1-9][0-9]+)?$/.test(trace.coloraxis)) continue;
                update[trace.coloraxis + '.colorbar.x'] = x[1] + 0.01;
                update[trace.coloraxis + '.colorbar.y'] = (y[0] + y[1]) / 2;
                update[trace.coloraxis + '.colorbar.len'] = y[1] - y[0];
            } else {
                indices.push(index);
                bars['colorbar.x'].push(x[1] + 0.01);
                bars['colorbar.y'].push((y[0] + y[1]) / 2);
                bars['colorbar.len'].push(y[1] - y[0]);
            }
            if (index < 3) {
                update['annotations[' + index + '].x'] = (x[0] + x[1]) / 2;
                update['annotations[' + index + '].y'] = y[1];
                update['annotations[' + index + '].yshift'] = 14;
            }
        }
        return indices.length ? plotly.update(plot, bars, update, indices) : plotly.relayout(plot, update);
    }
    var scheduled = false;
    function fitFigure() {
        if (scheduled) return;
        scheduled = true;
        window.requestAnimationFrame(function () {
            enqueue(function () {
                scheduled = false;
                var layouts = plot.layout.meta && plot.layout.meta.responsive_layout;
                if (!layouts) return;
                var layout = compact.matches ? layouts.compact : layouts.desktop;
                layout.width = plot.clientWidth;
                return plotly.relayout(plot, layout).then(alignFields).then(reportHeight);
            });
        });
    }
    window.addEventListener('resize', fitFigure);
    fitFigure();
}());
