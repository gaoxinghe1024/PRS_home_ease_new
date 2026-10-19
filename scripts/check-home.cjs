const {chromium} = require('playwright');
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const base = process.env.HOMEEASE_TEST_URL || 'http://127.0.0.1:8013';
const out = path.resolve(__dirname, '../artifacts/browser');
fs.mkdirSync(out, {recursive:true});

async function json(response) {
  assert(response.ok(), `${response.status()} ${response.url()}: ${await response.text()}`);
  return response.json();
}
async function until(fn, predicate, timeout=90000) {
  const deadline=Date.now()+timeout;
  while(Date.now()<deadline) {
    const value=await fn();
    if(predicate(value))return value;
    await new Promise(resolve=>setTimeout(resolve,300));
  }
  throw new Error('State did not become ready');
}
async function layout(page, label) {
  await page.waitForTimeout(350);
  assert(await page.locator('.home-header .lang-toggle').isVisible(),`${label}: missing language control`);
  const result=await page.evaluate(()=>({
    overflow:document.documentElement.scrollWidth>innerWidth+1,
    images:Array.from(document.images).filter(el=>el.offsetWidth&&el.getAttribute('src')&&(!el.complete||!el.naturalWidth)).map(el=>el.src),
    clipped:Array.from(document.querySelectorAll('button,h1,h2,.badge,.home-header a')).filter(el=>el.offsetWidth&&el.scrollWidth>el.clientWidth+3).map(el=>el.textContent),
    chinese:document.documentElement.lang==='en'?Array.from(document.querySelectorAll('.home-header,.home-monitor-strip,.page:not([hidden]),.home-help-row')).flatMap(el=>Array.from(el.querySelectorAll('*'))).filter(el=>el.offsetWidth&&el.children.length===0&&!el.closest('[data-i18n-keep]')&&/[\u4e00-\u9fff]/.test(el.textContent)).map(el=>el.textContent):[]
  }));
  assert(!result.overflow, `${label}: overflow`);
  assert.deepStrictEqual(result.images, [], `${label}: missing images`);
  assert.deepStrictEqual(result.clipped, [], `${label}: clipped text`);
  assert.deepStrictEqual(result.chinese, [], `${label}: untranslated text`);
  await page.screenshot({path:path.join(out,`${label}.png`),fullPage:true});
  return {label,...result};
}

(async()=>{
  const browser=await chromium.launch({headless:true,args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
  const context=await browser.newContext({viewport:{width:1440,height:1000},permissions:['microphone']});
  const page=await context.newPage();
  const errors=[], results={layouts:[],workflows:{},errors};
  page.on('pageerror',e=>errors.push(e.message));
  const status=()=>page.request.get(`${base}/api/status`).then(json);
  const guard=enabled=>page.request.put(`${base}/api/home/monitoring`,{data:{enabled,camera_index:0}}).then(json);
  const profile=await json(await page.request.get(`${base}/api/settings`));
  const consent=value=>page.request.put(`${base}/api/settings`,{data:{...profile,audio_consent:value}}).then(json);
  try {
    await until(()=>page.request.get(`${base}/api/health`).then(json),s=>s.modules.every(m=>m.state==='ready'));
    await guard(false);await consent(false);
    for(const event of await json(await page.request.get(`${base}/api/events?state=open`))) {
      await json(await page.request.post(`${base}/api/events/${event.id}/action`,{data:{action:'resolve',outcome:'safe'}}));
    }
    await page.addInitScript(()=>{
      window.speechSynthesis.speak=phrase=>setTimeout(()=>phrase.onend?.(),100);
    });
    await page.goto(`${base}/home`);
    await page.locator('#page-home').waitFor({state:'visible'});
    results.layouts.push(await layout(page,'home-zh-desktop'));
    assert(!(await status()).home_monitoring.enabled,'Home chooser unexpectedly started monitoring');
    assert.deepStrictEqual((await json(await page.request.get(`${base}/api/sessions`))).filter(s=>s.status==='running'),[]);
    assert.strictEqual(await page.locator('.home-choices, .home-choice').count(),0,'Home page still shows self-service guard/exercise cards');
    await page.click('.home-header a[href="#safety"]');
    let running=await until(status,s=>s.home_monitoring.enabled&&s.video.state==='running');
    await page.locator('#homeAudioNotice').waitFor({state:'visible'});
    assert(!await page.evaluate(()=>!!app.audioStream),'Audio started without consent');
    await page.click('#homeAllowAudio');
    await page.waitForFunction(()=>app.audioContext?.state==='running'&&app.audioSocket?.readyState===1,null,{timeout:60000});
    await until(status,s=>s.coverage.some(c=>c.channel==='audio'&&c.state==='active'));
    await page.waitForFunction(()=>document.getElementById('homeFrame').naturalWidth>0,null,{timeout:30000});
    results.workflows.automaticVideoAndConsentedAudio=true;
    results.layouts.push(await layout(page,'safety-zh-desktop'));
    await page.click('.home-header a[href="#exercise"]');
    await page.locator('#page-exercise').waitFor({state:'visible'});
    assert(await page.evaluate(()=>!!app.audioStream),'Audio stopped on exercise navigation');
    assert.strictEqual((await status()).video.session,running.video.session);
    assert(!(await json(await page.request.get(`${base}/api/sessions`))).some(s=>s.status==='running'),'Exercise was automatically started');
    results.layouts.push(await layout(page,'exercise-home-zh-desktop'));
    await page.selectOption('#exerciseSource','replay');
    await page.click('#startExercise');
    await page.waitForFunction(()=>!document.getElementById('exerciseCompleted').hidden,null,{timeout:120000});
    assert(Number(await page.locator('#repCount').textContent())>0);
    assert.strictEqual((await status()).video.session,running.video.session);
    results.workflows.manualExerciseReplay=true;
    await page.click('#newExercise');await page.selectOption('#exerciseSource','camera');await page.click('#startExercise');
    await page.waitForFunction(()=>document.getElementById('exerciseCamera').naturalWidth>0,null,{timeout:60000});
    const session=await page.evaluate(()=>app.sid);
    const concurrent=await status();
    assert.strictEqual(concurrent.video.session,running.video.session);
    assert(concurrent.cameras.some(c=>c.users.includes('video')&&c.users.includes(session)),'Camera was not shared');
    await page.click('#endExercise');
    await page.waitForFunction(()=>!document.getElementById('exerciseCompleted').hidden);
    assert.strictEqual((await status()).video.state,'running');
    results.workflows.sharedCameraAndIndependentStop=true;
    await page.click('.home-header a[href="#safety"]');
    assert.strictEqual(await page.locator('#homePause, #homeRetry, .home-safety-actions').count(),0,'Older adult can still pause or restart the guard');
    // 守护的暂停与恢复由家属在后台完成，这里直接调用同一个接口。
    const guard=enabled=>page.request.put(`${base}/api/home/monitoring`,{data:{enabled,camera_index:running.home_monitoring.camera_index}}).then(json);
    await guard(false);
    await until(status,s=>!s.home_monitoring.enabled&&s.video.state!=='running');
    await page.waitForFunction(()=>!app.audioStream&&!app.audioStopping);
    await guard(true);
    running=await until(status,s=>s.home_monitoring.enabled&&s.video.state==='running');
    await page.waitForFunction(()=>app.audioSocket?.readyState===1,null,{timeout:60000});
    await page.reload();
    await page.waitForFunction(()=>app.audioSocket?.readyState===1,null,{timeout:60000});
    assert.strictEqual((await status()).video.session,running.video.session);
    const oldSocket=await page.evaluate(()=>{window.oldHomeSocket=app.audioSocket;app.audioSocket.close();return true;});
    await page.waitForFunction(()=>app.audioSocket&&app.audioSocket!==window.oldHomeSocket&&app.audioSocket.readyState===1,null,{timeout:45000});
    results.workflows.pauseResumeRefreshReconnect=oldSocket;
    await page.click('#elderHelp');
    await page.waitForFunction(()=>!document.getElementById('inquiryProgress').hidden,null,{timeout:15000});
    const help=await until(()=>page.request.get(`${base}/api/events?state=open&mode=live`).then(json),events=>events.some(e=>e.channel==='manual'&&e.inquiry.status==='listening'));
    const event=help.find(e=>e.channel==='manual');
    await until(()=>page.request.get(`${base}/api/events/${event.id}`).then(json),e=>!!e.inquiry.result,90000);
    await page.locator('#elderWithdraw').waitFor({state:'visible',timeout:15000});
    await page.click('#elderWithdraw');
    const withdrawn=await until(()=>page.request.get(`${base}/api/events/${event.id}`).then(json),e=>e.withdrawn);
    assert.strictEqual(withdrawn.state,'open');
    results.workflows.automaticInquiryAndWithdrawal=true;
    await page.waitForFunction(()=>app.audioSocket?.readyState===1,null,{timeout:45000});
    await page.request.post(`${base}/api/events/${event.id}/action`,{data:{action:'resolve',outcome:'safe'}});
    for(const language of ['zh','en']) {
      await page.evaluate(lang=>localStorage.setItem('homeease.lang',lang),language);
      await page.reload();await page.waitForFunction(()=>app.ready);
      for(const width of [1440,820,390,320]) {
        await page.setViewportSize({width,height:width===390?844:1000});
        for(const view of ['home','safety','exercise']) {
          await page.evaluate(v=>{location.hash=v;},view);
          await page.locator(`#page-${view}`).waitFor({state:'visible'});
          results.layouts.push(await layout(page,`${view}-${language}-${width}`));
        }
      }
    }
    await consent(false);
    await page.waitForFunction(()=>!app.audioStream&&!app.audioStopping,null,{timeout:30000});
    await page.waitForTimeout(3000);assert(!await page.evaluate(()=>!!app.audioStream));
    results.workflows.consentRevocation=true;
    await page.close();
    assert.strictEqual((await context.request.get(`${base}/api/status`).then(json)).video.state,'running');
    results.workflows.videoSurvivesPageClose=true;
    const caregiver=await context.newPage();
    await caregiver.goto(`${base}/#reports`);await caregiver.locator('#page-reports').waitFor({state:'visible'});
    assert(await caregiver.locator('.sidebar').isVisible());
    results.workflows.caregiverReportsUnchanged=true;
    await caregiver.request.put(`${base}/api/settings`,{data:{...profile,audio_consent:true}});
    const denied=await context.newPage();
    denied.on('pageerror',e=>errors.push(e.message));
    await denied.addInitScript(()=>{window.permissionAttempts=0;navigator.mediaDevices.getUserMedia=async()=>{window.permissionAttempts++;throw new DOMException('Microphone permission denied','NotAllowedError');};});
    await denied.goto(`${base}/home#safety`);
    await denied.waitForFunction(()=>window.permissionAttempts===1);
    await denied.waitForTimeout(16500);
    assert.strictEqual(await denied.evaluate(()=>window.permissionAttempts),1,'Repeated microphone permission prompts');
    await denied.setViewportSize({width:390,height:844});
    assert(await denied.evaluate(()=>{
      const copy=document.querySelector('#homeAudioNotice>div').getBoundingClientRect();
      const button=document.getElementById('homeAllowAudio').getBoundingClientRect();
      return button.top>=copy.bottom;
    }),'Microphone notice overlaps its control');
    results.layouts.push(await layout(denied,'safety-microphone-denied'));
    results.workflows.permissionDeniedDoesNotLoop=true;
    assert.deepStrictEqual(errors,[]);
    fs.writeFileSync(path.join(out,'home-results.json'),JSON.stringify(results,null,2));
    console.log(JSON.stringify(results,null,2));
  } finally {
    await context.request.put(`${base}/api/home/monitoring`,{data:{enabled:false,camera_index:0}}).catch(()=>{});
    await context.request.put(`${base}/api/settings`,{data:profile}).catch(()=>{});
    await browser.close();
  }
})().catch(error=>{console.error(error);process.exitCode=1;});
