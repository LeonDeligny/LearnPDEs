"""Exercise the inline playback runtime with controlled Plotly completions."""

import shutil

# Node runs only repository-owned test scripts with fixed arguments and no shell.
import subprocess  # nosec B404
import unittest
from pathlib import Path

PLAYBACK = Path(__file__).resolve().parents[2] / 'learnpdes/utils/training_playback.js'
NODE = shutil.which('node')
HARNESS = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const source = require('node:fs').readFileSync(process.argv[2], 'utf8');
const events = {}, calls = [], pending = [], animationFrames = [], errors = [];
const timers = new Map();
let timerId = 0;
const frames = [0, 1, 2, 3].map(index => ({
    name: String(index), data: [{x: [index], y: [index * 2]}], traces: [2],
    layout: {title: {text: 'Step ' + index}}
}));
const plot = {
    _transitionData: {_frames: frames}, data: [], _fullLayout: {},
    layout: {sliders: [{active: 0}]}, clientWidth: 900,
    on(name, handler) { events[name] = handler; },
    getBoundingClientRect() { return {bottom: 800}; }
};
function record(kind, data, layout, traces) {
    calls.push({kind, data, layout, traces});
    return new Promise((resolve, reject) => pending.push({resolve, reject}));
}
const media = {matches: false};
const browser = {
    document: {getElementById: () => plot},
    Plotly: {
        update: (target, data, layout, traces) => record('update', data, layout, traces),
        relayout: (target, layout) => record('relayout', null, layout)
    },
    console: {error: (...args) => errors.push(args)},
    setTimeout: action => { timers.set(++timerId, action); return timerId; },
    clearTimeout: id => timers.delete(id),
    requestAnimationFrame: action => animationFrames.push(action),
    addEventListener: (name, action) => { events[name] = action; },
    matchMedia: () => media, scrollY: 0, location: {origin: 'https://example.test'}
};
browser.parent = browser;
const context = vm.createContext({window: browser});
vm.runInContext(`
    Promise = undefined; Map = undefined; Set = undefined;
    Object.keys = undefined; Object.fromEntries = undefined;
    Array.prototype.map = undefined; Array.prototype.flatMap = undefined;
    Array.prototype.forEach = undefined; Array.prototype.indexOf = undefined;
`, context);
vm.runInContext(source, context);
function seek(name, interaction = true) {
    events.plotly_sliderchange({interaction, step: {args: [[name]]}});
}
function play(label = 'Play') {
    events.plotly_buttonclicked({button: {label, args: [null, {frame: {duration: 180}}]}});
}
function choose() {
    const button = {method: 'update', args: [{visible: [false]}, {}, [2]]};
    events.plotly_buttonclicked({button, menu: {_index: 1, buttons: [{}, button]}});
}
async function complete(error) {
    assert.ok(pending.length, 'A Plotly render must be pending');
    const task = pending.shift();
    if (error) task.reject(error); else task.resolve();
    for (let tick = 0; tick < 12; tick++) await Promise.resolve();
}
function tick() {
    assert.equal(timers.size, 1);
    const [id, action] = timers.entries().next().value;
    timers.delete(id);
    action();
}
const plain = value => JSON.parse(JSON.stringify(value));
"""


@unittest.skipUnless(NODE, 'Node.js is required to execute the playback runtime')
class TestPlayback(unittest.TestCase):
    def run_script(self, script: str) -> None:
        assert NODE is not None
        # The executable is absolute; stdin contains only the static test harness.
        result = subprocess.run(  # nosec B603
            [str(Path(NODE).resolve()), '-', str(PLAYBACK)],
            input=HARNESS
            + '\nasync function check() {\n'
            + script
            + '\n}\ncheck().catch(error => { console.error(error); process.exitCode = 1; });',
            text=True,
            shell=False,
            capture_output=True,
            timeout=15,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_no_browser_missing_plot_or_empty_frames_is_harmless(self) -> None:
        self.run_script("""
            vm.runInNewContext(source, {});
            vm.runInNewContext(source, {window: {}});
            vm.runInNewContext(source, {window: {document: {}}});
            const missing = {...browser, document: {getElementById: () => null}};
            vm.runInNewContext(source, {window: missing});
            frames.length = 0;
            vm.runInNewContext(source, {window: browser});
            assert.equal(calls.length, 0);
        """)

    def test_checkpoint_fields_are_explicit_and_preserve_mixed_trace_positions(
        self,
    ) -> None:
        self.run_script("""
            const heatmap = {z: [[1, 2]], type: 'heatmap'};
            Object.defineProperty(heatmap, '__proto__', {
                enumerable: true, get() { throw Error('Unsafe property read'); }
            });
            Object.defineProperty(heatmap, 'unexpected', {
                enumerable: true, get() { throw Error('Unknown property read'); }
            });
            frames[1].data = [heatmap, frames[1].data[0], {
                x: [8], y: [9], name: 'PDE points (1)', customdata: ['interior'],
                meta: {role: 'collocation', count: 1}
            }];
            frames[1].traces = [0, 2, 5];
            seek('1');
            const call = calls[0];
            assert.deepEqual(plain(call.data.x), [null, [1], [8]]);
            assert.deepEqual(plain(call.data.z), [[[1, 2]], null, null]);
            assert.equal(call.data.x[0], undefined);
            assert.equal(call.data.name[2], 'PDE points (1)');
            assert.deepEqual(plain(call.data.customdata[2]), ['interior']);
            assert.equal(call.data.meta[2].count, 1);
            assert.deepEqual(call.traces, [0, 2, 5]);
            assert.equal(call.layout['title.text'], 'Step 1');
            assert.deepEqual(Object.keys(call.data).sort(),
                ['customdata', 'meta', 'name', 'x', 'y', 'z']);
            await complete();
            assert.deepEqual(errors, []);
        """)

    def test_unknown_names_and_programmatic_slider_events_do_not_seek(self) -> None:
        self.run_script("""
            for (const name of ['__proto__', 'constructor', 'toString', '-1', '9', 1]) seek(name);
            seek('1', false);
            assert.equal(calls.length, 0);
            frames[1].name = '__proto__';
            seek('__proto__');
            assert.equal(calls[0].layout['sliders[0].active'], 1);
        """)

    def test_rapid_seeks_skip_obsolete_queued_frames(self) -> None:
        self.run_script("""
            seek('1'); seek('2'); seek('3');
            assert.equal(calls.length, 1);
            await complete();
            assert.equal(calls.length, 2);
            assert.equal(calls[1].layout['sliders[0].active'], 3);
            await complete();
            assert.equal(pending.length, 0);
        """)

    def test_play_waits_for_render_completion_and_restarts_at_the_end(self) -> None:
        self.run_script("""
            play(); play();
            assert.equal(calls.length, 1);
            assert.equal(timers.size, 0);
            for (const index of [1, 2, 3]) {
                assert.equal(calls.at(-1).layout['sliders[0].active'], index);
                await complete();
                if (index < 3) tick();
            }
            assert.equal(timers.size, 0);
            play();
            assert.equal(calls.at(-1).layout['sliders[0].active'], 0);
        """)

    def test_pause_cancels_timers_and_inflight_playback_continuations(self) -> None:
        self.run_script("""
            play(); play('Pause');
            await complete();
            assert.equal(timers.size, 0);
            play();
            assert.equal(calls.at(-1).layout['sliders[0].active'], 2);
            await complete();
            assert.equal(timers.size, 1);
            play('Pause');
            assert.equal(timers.size, 0);
            play(); seek('0');
            await complete();
            assert.equal(calls.at(-1).layout['sliders[0].active'], 0);
            await complete();
            assert.equal(timers.size, 0);
        """)

    def test_sync_and_async_failures_pause_playback_and_release_the_queue(self) -> None:
        self.run_script("""
            play(); choose();
            await complete(Error('Rejected render'));
            assert.equal(errors.length, 1);
            assert.equal(timers.size, 0);
            assert.deepEqual(plain(calls[1].data), {visible: [false]});
            await complete();
            const update = browser.Plotly.update;
            browser.Plotly.update = () => { throw Error('Synchronous failure'); };
            seek('2');
            browser.Plotly.update = update;
            seek('3');
            assert.equal(errors.length, 2);
            assert.equal(calls.at(-1).layout['sliders[0].active'], 3);
            await complete();
        """)

    def test_resize_and_controls_wait_for_playback_and_align_colorbars(self) -> None:
        self.run_script("""
            plot.layout.meta = {responsive_layout: {
                desktop: {height: 900}, compact: {height: 1450}
            }};
            plot.data = [
                {type: 'heatmap', xaxis: 'x', yaxis: 'y', coloraxis: 'coloraxis'},
                {type: 'heatmap', xaxis: 'x2', yaxis: 'y2'},
                {type: 'heatmap', xaxis: '__proto__', yaxis: 'y'},
                {type: 'heatmap', xaxis: 'x', yaxis: 'y', coloraxis: '__proto__'}
            ];
            plot._fullLayout = {
                xaxis: {domain: [0, 0.4]}, yaxis: {domain: [0.5, 1]},
                xaxis2: {domain: [0.6, 0.9]}, yaxis2: {domain: [0, 0.4]}
            };
            const messages = [];
            browser.parent = {postMessage: (...args) => messages.push(args)};
            play(); choose();
            events.resize(); events.resize();
            assert.equal(animationFrames.length, 1);
            animationFrames.shift()();
            assert.equal(calls.length, 1);
            await complete();
            assert.equal(calls[1].layout['updatemenus[1].active'], 1);
            await complete();
            assert.deepEqual(plain(calls[2].layout), {height: 900, width: 900});
            await complete();
            assert.equal(calls[3].layout['coloraxis.colorbar.x'], 0.4 + 0.01);
            assert.deepEqual(plain(calls[3].traces), [1]);
            assert.deepEqual(plain(calls[3].data['colorbar.len']), [0.4]);
            await complete();
            assert.equal(messages[0][0].height, 808);
            assert.equal(messages[0][1], 'https://example.test');
            media.matches = true;
            plot.clientWidth = 390;
            events.resize(); animationFrames.shift()();
            assert.deepEqual(plain(calls[4].layout), {height: 1450, width: 390});
            await complete(); await complete();
            assert.deepEqual(errors, []);
        """)


if __name__ == '__main__':
    unittest.main()
