const {chromium} = require('playwright');
const fs = require('fs');
const path = require('path');
const assert = require('assert');

const base = process.env.HOMEEASE_TEST_URL || 'http://127.0.0.1:8012';
const suffix = '';  // the interface is English only
const out = path.resolve(__dirname, '../artifacts/browser');
fs.mkdirSync(out, {recursive: true});

async function json(response) {
  assert(response.ok(), `${response.status()} ${response.url()}: ${await response.text()}`);
  return response.json();
}

async function waitState(page, sid, predicate) {
  const deadline = Date.now() + 90000;
  while (Date.now() < deadline) {
    const state = await json(await page.request.get(`${base}/api/sessions/${sid}`));
    if (predicate(state)) return state;
    await page.waitForTimeout(200);
  }
  throw new Error(`Session did not reach expected state: ${sid}`);
}

async function layout(page) {
  const result = await page.evaluate(() => ({
    overflow: document.documentElement.scrollWidth > innerWidth + 1,
    clipped: [...document.querySelectorAll('#exerciseActive button,#exerciseDiagnostics span,#exerciseQuality,#exerciseCycle li,#exerciseCue,.exercise-guide p,.exercise-guide li,.exercise-guide figcaption span,.exercise-angle-notice strong,.exercise-angle-notice span')]
      .filter(el => el.getBoundingClientRect().width && el.scrollWidth > el.clientWidth + 2)
      .map(el => el.textContent),
    pixels: [...document.getElementById('poseCanvas').getContext('2d').getImageData(0, 0, 100, 100).data]
      .some((value, index) => index % 4 === 3 && value > 0),
  }));
  assert(!result.overflow, 'horizontal overflow');
  assert.deepStrictEqual(result.clipped, [], 'clipped diagnostic text');
  assert(result.pixels, 'blank pose canvas');
  return result;
}

(async () => {
  const browser = await chromium.launch({headless: true});
  const page = await browser.newPage({viewport: {width: 1440, height: 1000}});
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const results = {};
  try {
    await page.goto(`${base}/#exercise`);
    await page.locator('#page-exercise').waitFor({state: 'visible'});
    const angleNotice = page.locator('.exercise-angle-notice');
    assert((await angleNotice.innerText()).includes('45'));
    assert(Number(await angleNotice.locator('strong').evaluate(el => getComputedStyle(el).fontWeight)) >= 700);
    assert(!/[\u4e00-\u9fff]/.test(await angleNotice.innerText()));
    assert.strictEqual(await page.locator('#exerciseGuide').getAttribute('data-exercise-guide'), 'chair_sit_to_stand');
    for (const [exercise, count] of Object.entries({chair_sit_to_stand: 9, squat: 10, standing_leg_raise: 10, hurdle_step: 10})) {
      await page.locator('#exerciseSource').selectOption('replay');
      await page.locator(`[data-exercise="${exercise}"]`).click();
      assert.strictEqual(await page.locator('#exerciseGuide').getAttribute('data-exercise-guide'), exercise);
      assert.strictEqual(await page.locator('#exerciseGuide ol li').count(), 3);
      await page.waitForFunction(() => document.querySelector('#exerciseGuide img').naturalWidth === 780);
      assert((await page.locator('#exerciseGuide .exercise-camera-guide').textContent()).includes('45'));
      await page.waitForTimeout(100);
      assert(!/[\u4e00-\u9fff]/.test(await page.locator('#exerciseGuide').innerText()), 'untranslated exercise guide');
      if (exercise === 'standing_leg_raise') {
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({path: path.join(out, `exercise-guide-desktop${suffix}.png`), fullPage: true});
        await page.setViewportSize({width: 390, height: 844});
        await page.waitForTimeout(300);
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({path: path.join(out, `exercise-guide-mobile${suffix}.png`), fullPage: true});
        assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        await page.setViewportSize({width: 1440, height: 1000});
      }
      const created = page.waitForResponse(response => response.url() === `${base}/api/sessions` && response.request().method() === 'POST');
      await page.locator('#startExercise').click();
      const {session_id: sid} = await json(await created);
      await waitState(page, sid, state => state.count > 0);
      await page.locator('#exerciseDiagnostics').waitFor({state: 'visible'});
      await page.waitForFunction(() => Number(document.getElementById('repCount').textContent) > 0);
      assert.strictEqual(await page.locator('#activeExerciseGuide').getAttribute('data-exercise-guide'), exercise);
      assert.strictEqual(await page.locator('#exerciseCycle li').count(), 5);
      assert(await page.locator('#exerciseCue').textContent());
      if (exercise === 'squat') {
        await page.locator('#exerciseDiagnostics').evaluate(el => el.open = true);
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({path: path.join(out, `exercise-desktop${suffix}.png`), fullPage: true});
        results.desktop = await layout(page);
        await page.setViewportSize({width: 390, height: 844});
        await page.waitForTimeout(300);
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.screenshot({path: path.join(out, `exercise-mobile${suffix}.png`), fullPage: true});
        results.mobile = await layout(page);
        await page.setViewportSize({width: 1440, height: 1000});
      }
      const final = await waitState(page, sid, state => state.status === 'ended');
      assert.strictEqual(final.action_scope, 'open');
      assert.strictEqual(final.inference_views, 'original');
      assert.strictEqual(final.count, count, exercise);
      const trace = await json(await page.request.get(`${base}/api/sessions/${sid}/diagnostics.json`));
      assert.strictEqual(trace.state.count, count);
      assert.strictEqual(trace.frames.length, final.counting_diagnostics.observed_frames);
      assert.strictEqual(trace.truncated, false);
      assert.strictEqual(trace.frames[0].keypoints.length, 17);
      assert(Object.keys(final.capture_diagnostics.joint_confidences).length === 8);
      fs.writeFileSync(path.join(out, `${exercise}-input.json`), JSON.stringify(trace));
      results[exercise] = {count, frames: trace.frames.length, traceAfterEnd: true};
      await page.waitForFunction(() => document.getElementById('exerciseCompleted').hidden === false);
      await page.locator('#newExercise').click();
    }
    // Exercise client mode lets a known brief dropout reach the real API.
    const created = await json(await page.request.post(`${base}/api/sessions`, {data: {exercise: 'squat', source: 'camera', capture: 'client'}}));
    assert.strictEqual(created.action_scope, 'training');
    const sid = created.session_id;
    const sample = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../algorithm2/examples/squat_replay.json'))).frames[0];
    const frames = [{...sample, timestamp_ms: 0, keypoints: sample.keypoints.map(point => [point[0], point[1], 0])}, {...sample, timestamp_ms: 33}];
    await json(await page.request.post(`${base}/api/sessions/${sid}/frames`, {data: {request_id: 'dropout', frames}}));
    const paused = await json(await page.request.post(`${base}/api/sessions/${sid}/control`, {data: {action: 'pause'}}));
    assert(paused.capture_diagnostics.joint_confidences.left_ankle > .3);
    assert(paused.counting_diagnostics.interruption_counts.insufficient_pose === 1);
    const trace = await json(await page.request.get(`${base}/api/sessions/${sid}/diagnostics.json`));
    assert(trace.frames.length === 2);
    await json(await page.request.post(`${base}/api/sessions/${sid}/control`, {data: {action: 'resume'}}));
    await json(await page.request.post(`${base}/api/sessions/${sid}/control`, {data: {action: 'end'}}));
    results.pauseResumeAndDropout = true;
    assert.deepStrictEqual(errors, []);
    fs.writeFileSync(path.join(out, `exercise-results${suffix}.json`), JSON.stringify(results, null, 2));
    console.log(JSON.stringify(results));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
