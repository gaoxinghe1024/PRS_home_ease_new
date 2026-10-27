"use strict";
const $ = (id) => document.getElementById(id);
const household = document.body.dataset.household === "true";
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const icon = (name) => `<i data-lucide="${name}"></i>`;
const icons = () => window.lucide?.createIcons();
const names = {chair_sit_to_stand:"Sit-to-stand",squat:"Squat",standing_leg_raise:"Standing leg raise",hurdle_step:"Hurdle step"};
const exerciseGuides = {
  chair_sit_to_stand: {
    title:"Chair Sit-to-Stand", endpoints:["Sit Steady","Stand Upright","Sit Again +1"], cycle:["Sit Steady","Rising","Stand Upright","Sit Down","Seated +1"],
    count:"Sit steady, stand fully upright, then sit steady again: 1 rep. Standing up alone does not complete a rep.",
    steps:["Use a stable chair. Start seated with both feet flat on the floor and settle for about 2 seconds.", "Lean slightly forward and stand up. Straighten your hips and knees, then steady yourself briefly.", "Bend your knees slowly and sit back onto the chair. Settle before starting the next rep."],
    note:"This exercise requires sitting on a real chair. Position the chair side-on and keep its back and any table clear of your hips, knees, and ankles.",
    cues:["Start seated on the chair with both feet flat. Settle for about 2 seconds.", "Seated start confirmed. Please stand up from the chair.", "Rising: stand fully upright and steady yourself briefly.", "Standing confirmed. Please sit back down slowly.", "Settle into the chair to complete 1 rep."]
  },
  squat: {
    title:"Squat", endpoints:["Stand Steady","Descending","Stand Again +1"], cycle:["Stand Steady","Descending","Squat","Rising","Standing +1"],
    count:"Stand steady, squat down, then stand steady again: 1 rep. Staying in the squat does not complete a rep.",
    steps:["Place your feet about shoulder-width apart and stand steady for about 2 seconds.", "Move your hips back and bend your knees. Pause briefly at a comfortable low position.", "Stand up and straighten your hips and knees. Settle into standing before starting the next rep."],
    note:"Stay in place with both feet visible throughout. Use a comfortable, stable range of motion.",
    cues:["Start standing naturally and settle for about 2 seconds.", "Standing start confirmed. Please squat down.", "Squatting: pause briefly at a comfortable low position.", "Squat confirmed. Please stand up.", "Stand fully upright and steady to complete 1 rep."]
  },
  standing_leg_raise: {
    animation:true,
    title:"Standing Straight-Leg Forward Raise", endpoints:["Stand Steady","Straight-Leg Raise","Feet Down +1"], cycle:["Stand Steady","Raise Forward","Raised","Lower","Standing +1"],
    count:"Stand steady, raise one straight leg forward, then lower it and stand steady: 1 rep. Each complete leg movement counts separately.",
    steps:["Raise your upper arms out to shoulder height, bend your elbows, and point your hands upward. Keep both legs straight and stand steady for about 2 seconds.", "Keep your supporting leg steady. Raise the other leg forward with its knee as straight as comfortable, then pause. The reference leg approaches horizontal.", "Lower the same leg slowly. Place both feet down and stand steady before repeating or changing legs."],
    note:"Raise the whole straight leg forward. Keep your arms in the reference position. Small movements may be missed; stay comfortable and stable, and do not force a higher lift for a count.",
    cues:["Start with both feet on the floor. Stand steady for about 2 seconds.", "Standing start confirmed. Raise one straight leg in front of your body.", "Waiting for the raised position: keep your knee as straight as comfortable and pause briefly after raising forward.", "Raised leg confirmed. Lower the same leg slowly.", "Place both feet down and stand steady to complete 1 rep."]
  },
  hurdle_step: {
    animation:true,
    title:"Bent-Knee Hurdle Step", endpoints:["Stand Steady","Bent-Knee Lift","Return and Stand +1"], cycle:["Stand Steady","Lift Bent Knee","Raised","Return","Standing +1"],
    count:"Stand steady, lift one bent knee forward, then return the foot to its starting place and stand steady: 1 rep. One complete rep on each leg totals 2 reps.",
    steps:["Move your arms slightly out to the sides, bend your elbows, and hold your hands around shoulder height. Start with both feet down and stand steady for about 2 seconds.", "Lift one bent knee forward with your lower leg hanging naturally, then pause. The knee in the reference approaches hip height.", "Return the same foot to its starting place. Put both feet down and stand steady before repeating or changing legs."],
    note:"Keep your arms in the reference position and lift your bent knee in place; no obstacle is needed. Small movements may be missed. Stay comfortable and stable without forcing a higher lift. When alternating legs, place both feet down and settle between reps.",
    cues:["Start with both feet on the floor. Stand steady for about 2 seconds.", "Standing start confirmed. Bend one knee and lift the leg forward.", "Waiting for the hurdle-step peak: lift your bent knee forward and pause at a comfortable height.", "Hurdle-step peak confirmed. Return the same foot to its starting place.", "Return both feet to their starting places and stand steady to complete 1 rep."]
  }
};
const phases = {seated:"Seated",sitting:"Seated",rising:"Rising",standing:"Standing",lowering:"Lowering",bottom:"Lowest point",descending:"Descending",ascending:"Ascending",baseline:"Starting posture",lifting:"Leg raise",raised:"Raised",stepping:"Hurdle step",returning:"Returning",other:"Waiting for movement",up:"Raised",down:"Down",start:"Start",peak:"Peak"};
const qualityNames = {good:"Observation valid",waiting_for_input:"Waiting for image",insufficient_pose:"Not enough keypoints; counting is unavailable",multiple_people:"More than one person in the image; counting is unavailable",no_person:"No person detected",timestamp_gap:"The gap between frames is too large; waiting for a complete movement",person_changed:"Target changed; waiting for a complete movement",degenerate_pose:"Abnormal keypoints; counting is unavailable",paused:"Paused",module_error:"Module error; session interrupted",service_restarted:"Service restarted; session interrupted"};
const channels = {vision:"Video",accel:"Acceleration",audio:"Audio",manual:"Asked for help"};
const outcomes = {confirmed_fall:"Confirmed fall",false_alarm:"False alarm",safe:"Verified safe",unresolved:"Undetermined"};
const stateNames = {ready:"Ready",loading:"Loading",not_initialized:"Not initialized",unavailable:"Unavailable"};
const moduleNames = {video:"Algorithm 1 · Video fall detection",signal:"Algorithm 1 · Accelerometer fall detection",exercise:"Algorithm 2 · Four-exercise counting",audio:"Algorithm 3 · Sound and voice"};
const modelNames = {video:"YOLOv8n + P1 + A4",signal:"A2 · RF + XGB",exercise:"opt_tcn_b4_w48",audio:"Silero / YAMNet + LR / Whisper tiny / rule_v2"};
const soundNames = {HELP:"Help-related sounds",IMPACT:"Impact sounds",ABNORMAL:"Abnormal sounds",DAILY:"Daily sounds",UNKNOWN:"Undetermined"};
const dateTime = (v) => v ? new Date(v).toLocaleString("en-US",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}) : "--";
const timeOnly = (v) => v ? new Date(v).toLocaleTimeString("en-US",{hour12:false}) : "--";
const duration = (v) => v >= 3600 ? `${(v/3600).toFixed(1)} h` : v >= 60 ? `${Math.floor(v/60)} min ${Math.round(v%60)} s` : `${Math.round(v || 0)} s`;
const modeBadge = (v) => `<span class="badge ${v === "live" ? "green" : "blue"}">${v === "live" ? "Live" : "Sample / replay"}</span>`;
const levelBadge = (e) => `<span class="badge ${e.state === "resolved" ? "green" : e.level === "EMERGENCY" ? "red" : "amber"}">${e.state === "resolved" ? "Resolved" : e.level === "EMERGENCY" ? "Emergency call for help" : "Pending verification"}</span>${e.withdrawn ? '<span class="badge blue">Withdrawn by the older adult · to be confirmed</span>' : ""}`;
const empty = (title, sym="inbox", subtitle="", compact=false) => `<div class="empty-state ${compact?"compact":""}">${icon(sym)}<strong>${esc(title)}</strong>${subtitle?`<p>${esc(subtitle)}</p>`:""}</div>`;
const voiceBadge = (voice) => {
  if (!voice) return '<span class="badge neutral">Not asked</span>';
  const tone = voice.css === "ok" ? "green" : voice.css === "risk" ? "red" : "neutral";
  const note = voice.reason && !voice.answered ? `<br><span class="muted small">${esc(voice.reason)}</span>` : "";
  return `<span class="badge ${tone}">Voice: ${esc(voice.label.toLowerCase())}${voice.detail?` · ${esc(voice.detail)}`:""}</span>${note}`;
};
const voiceChip = (voice) => {
  if (!voice) return '<span class="badge neutral">Voice: not asked</span>';
  const tone = voice.css === "ok" ? "green" : voice.css === "risk" ? "red" : "neutral";
  return `<span class="badge ${tone}">Voice: ${esc(voice.label.toLowerCase())}${voice.detail?` · ${esc(voice.detail)}`:""}</span>`;
};
const app = {page:"overview",settings:null,status:null,events:[],sessions:[],notices:[],sounds:[],report:null,eventFilter:"open",eventMode:"live",reportDays:7,reportMode:"live",selectedExercise:"chair_sit_to_stand",sid:null,exercise:null,camera:null,capturing:false,exerciseSource:null,audioStream:null,audioSocket:null,audioContext:null,audioRecorder:null,recordedBlob:null,videoHistory:[],selectedEvent:null,noticeIds:new Set(),ready:false,overlay:{box:true,skeleton:true},pose:null,poseBusy:false,algorithms:null,plan:null};
let toastTimer, refreshBusy=false, exercisePollBusy=false;

async function api(path, options={}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeout || 180000);
  const settings = {...options, signal:controller.signal};
  delete settings.timeout;
  if (settings.body && !(settings.body instanceof FormData)) {
    settings.headers = {...settings.headers,"Content-Type":"application/json"};
    settings.body = JSON.stringify(settings.body);
  }
  try {
    const response = await fetch(path,settings);
    const result = await response.json();
    if (!response.ok) {
      let message = result.error || result.detail || `Request failed (${response.status})`;
      if (Array.isArray(message)) message = message.map(x=>`${x.loc?.slice(1).join(".") || "Parameter"}: ${x.msg}`).join("; ");
      if (typeof message !== "string") message = JSON.stringify(message);
      throw new Error(message);
    }
    return result;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("The request timed out. Check the module status and try again");
    throw e;
  } finally {clearTimeout(timer);}
}
function toast(message,error=false){$("toast").textContent=message;$("toast").className=`toast ${error?"error":""}`;$("toast").hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("toast").hidden=true,error?7500:4000);}
async function busy(button, fn){if(button.disabled)return;const wasDisabled=button.disabled;button.disabled=true;button.classList.add("loading");try{return await fn();}catch(e){toast(e.message,true);}finally{button.disabled=wasDisabled;button.classList.remove("loading");}}
function metric(title,value,unit,sym,note="",cls=""){return `<div class="metric${cls?` ${cls}`:""}"><div class="metric-header">${icon(sym)}${esc(title)}</div><div class="metric-value">${esc(value)}<small>${esc(unit)}</small></div><div class="metric-note">${esc(note)}</div></div>`;}
function channelRow(row){const active=row?.state==="active";return `<div class="channel-item">${icon({vision:"video",accel:"activity",audio:"mic"}[row.channel])}<div><strong>${channels[row.channel]}</strong><small><span class="dot ${active?"green":"amber"}"></span>${active?"Input valid":row.state==="reduced"?"Coverage limited":"Not connected"}</small></div></div>`;}

function navigate(){const requested=location.hash.slice(1).split("?")[0]||(household?"home":"overview");app.page=household?(["home","safety","exercise"].includes(requested)?requested:"home"):($("page-"+requested)?requested:"overview");document.querySelectorAll(".page").forEach(x=>x.hidden=x.id!==`page-${app.page}`);document.querySelectorAll("nav [data-page]").forEach(x=>x.classList.toggle("active",x.dataset.page===app.page));const label=household?({home:"Safe at home",safety:"Home Monitoring",exercise:"Exercise"})[app.page]:document.querySelector(`nav [data-page="${app.page}"]`)?.textContent.replace(/\d+$/,"")||"Care overview";$("pageCrumb").textContent=label;document.title=`${label} · HomeEase`;$("sidebar").classList.remove("open");$("navBackdrop").hidden=true;renderCurrent();if(app.page==="reports")loadReport();if(app.page==="settings")loadDiagnostics();if(app.page==="algorithms"||app.page==="plan")loadAlgorithms();else if(app.page==="plan"){renderPlan(app.plan);renderPlanProgress(app.plan?.progress);}window.HomeMonitoring?.navigate();window.scrollTo(0,0);}

async function refresh(){if(refreshBusy)return;refreshBusy=true;try{const results=await Promise.allSettled([api("/api/status",{timeout:12000}),api("/api/events"),api("/api/sessions"),api("/api/notifications"),app.page==="audio"?api("/api/sounds"):Promise.resolve(app.sounds),api("/api/settings")]);if(results[0].status==="rejected")throw results[0].reason;app.status=results[0].value;if(results[5].status==="fulfilled"){app.settings=results[5].value;if(!app.settings.audio_consent){if(app.audioStream)stopAudioLive();if(app.audioRecorder?.state==="recording")app.audioRecorder.stop();app.recordingStream?.getTracks().forEach(t=>t.stop());app.recordedBlob=null;}}if(results[1].status==="fulfilled")app.events=results[1].value;if(results[2].status==="fulfilled")app.sessions=results[2].value;if(results[3].status==="fulfilled")app.notices=results[3].value;if(results[4].status==="fulfilled")app.sounds=results[4].value;$("connectionState").innerHTML='<span class="dot green"></span>System connected';const n=app.events.filter(e=>e.state==="open"&&e.mode==="live").length;$("navEventCount").textContent=n;$("navEventCount").hidden=!n;$("noticeDot").hidden=!app.notices.some(n=>n.status!=="seen"&&n.kind==="inapp");syncExerciseSwitches();if(app.page==="algorithms"||app.page==="plan")loadAlgorithms();for(const notice of app.notices){if(app.ready&&!app.noticeIds.has(notice.id)&&notice.kind==="inapp"&&notice.mode==="live"){toast(`${notice.title} · ${notice.message}`);if("Notification" in window&&Notification.permission==="granted")new Notification(notice.title,{body:notice.message,tag:notice.event_id});}app.noticeIds.add(notice.id);}app.ready=true;renderCurrent();if(app.selectedEvent&&$("eventDialog").open){const latest=app.events.find(e=>e.id===app.selectedEvent.id);if(latest&&latest.revision!==app.selectedEvent.revision&&!$("eventDetail").contains(document.activeElement)){renderEventDetail(latest);}}}catch(e){$("connectionState").innerHTML='<span class="dot red"></span>System connection lost';}finally{refreshBusy=false;}}
function renderCurrent(){if(!app.status)return;if(!["overview","monitor"].includes(app.page)){for(const id of ["overviewFrame","monitorFrame"]){const image=$(id);if(image.dataset.streaming){image.removeAttribute("src");delete image.dataset.streaming;}}}renderModuleIndicators();if(app.page==="overview")renderOverview();if(app.page==="monitor")renderMonitor();if(app.page==="exercise")renderExerciseHistory();if(app.page==="audio")renderAudioHistory();if(app.page==="events")renderEvents();window.HomeMonitoring?.render();icons();}
function renderModuleIndicators(){for(const m of app.status.modules){if(m.id==="exercise"){$("exerciseModelState").textContent=`Exercise module ${(stateNames[m.state]||m.state).toLowerCase()}`;$("exerciseModelState").className=`badge ${m.state==="ready"?"green":"amber"}`;}}const consent=app.settings?.audio_consent;$("audioConsentBadge").textContent=consent?"Local audio authorized":"Audio not authorized";$("audioConsentBadge").className=`badge ${consent?"green":"neutral"}`;$("audioConsentBanner").hidden=!!consent;if(app.page==="settings")renderModules();syncExerciseSwitches();}
/* 设置页关掉的动作，在工作台上标出来并禁止开始，避免「练了却不计数」。 */
function syncExerciseSwitches(){
  // /api/status 给的是开关表本身，/api/algorithms 套了一层 switches，这里两种都要认。
  const status=app.status?.algorithms;
  const switches=status?.switches||(status&&!Array.isArray(status)?status:app.algorithms?.switches);
  if(!switches||!$("exerciseOptions"))return;
  document.querySelectorAll("[data-switch-card]").forEach(card=>{
    const closed=switches[card.dataset.switchCard]===false;
    card.classList.toggle("is-disabled",closed);
    const badge=card.querySelector(".exercise-off-badge");if(badge)badge.hidden=!closed;
  });
  const chooseVisible=$("exerciseChooser")&&!$("exerciseChooser").hidden;
  if(chooseVisible&&$("startExercise"))$("startExercise").disabled=switches[app.selectedExercise]===false;
  if($("exerciseGuide"))$("exerciseGuide").classList.toggle("is-disabled",chooseVisible&&switches[app.selectedExercise]===false);
}
function renderOverview(){const s=app.status;const coverage={active:["Monitoring is on","Video and acceleration inputs are valid"],reduced:["Monitoring coverage limited",s.coverage.filter(x=>x.channel!=="audio"&&x.state!=="active").map(x=>x.reason).join("; ")],unavailable:["Monitoring is not on yet","Video and acceleration inputs are not connected"]}[s.coverage_state];$("coverageTitle").textContent=coverage[0];$("coverageDescription").textContent=coverage[1];$("coverageBanner").classList.toggle("is-active",s.coverage_state==="active");const today=new Date().toLocaleDateString("en-CA");const live=app.events.filter(e=>e.mode==="live");const dayEvents=live.filter(e=>new Date(e.created_at).toLocaleDateString("en-CA")===today);const todaySessions=app.sessions.filter(s=>s.mode==="live"&&s.status==="ended"&&new Date(s.created_at).toLocaleDateString("en-CA")===today);const open=live.filter(e=>e.state==="open");$("dashboardMetrics").innerHTML=metric("Open events",open.length,"","bell-ring",open.some(e=>e.level==="EMERGENCY")?"An emergency call for help is waiting for a response":"Current live records")+metric("Confirmed falls today",dayEvents.filter(e=>e.outcome==="confirmed_fall").length,"reps","shield-check","The caregiver's resolution decides")+metric("Exercise today",todaySessions.reduce((sum,s)=>sum+s.count,0),"reps","activity",`${todaySessions.length} completed sessions`)+metric("Valid monitoring channels",s.coverage.filter(x=>x.state==="active"&&x.channel!=="audio").length,"/ 2","radio","Video / acceleration");$("dashboardChannels").innerHTML=s.coverage.sort((a,b)=>["vision","accel","audio"].indexOf(a.channel)-["vision","accel","audio"].indexOf(b.channel)).map(channelRow).join("");$("openEventCount").textContent=open.length;$("attentionList").innerHTML=open.length?open.slice(0,4).map(e=>`<div class="event-item" role="button" tabindex="0" data-event="${e.id}"><span class="event-icon ${e.level==="EMERGENCY"?"red":""}">${icon(e.level==="EMERGENCY"?"bell-ring":"triangle-alert")}</span><div class="event-body"><div class="event-title"><strong>${esc(e.room)} · ${e.level==="EMERGENCY"?"Emergency call for help":"Suspected fall"}</strong>${levelBadge(e)}${e.voice?.high_risk?voiceChip(e.voice):""}</div><p>${channels[e.channel]}${e.channel==="manual"?"":"detection"} · ${esc(e.resident)}</p><div class="event-meta"><span>${dateTime(e.occurred_at)}</span><span>#${e.id.slice(0,6)}</span></div></div>${icon("chevron-right")}</div>`).join(""):empty("No open events","circle-check","Live events will appear here");$("recentSessions").innerHTML=app.sessions.filter(s=>s.mode==="live").slice(0,3).map(s=>`<div class="session-row"><span class="session-icon">${icon("activity")}</span><div>${names[s.exercise]}<small>${dateTime(s.created_at)} · ${s.status==="ended"?"Completed":s.status==="interrupted"?"Interrupted":"In progress"}</small></div><span class="session-count">${s.count}<small> reps</small></span></div>`).join("")||empty("No exercise sessions today yet","activity","",true);const running=s.video.state==="running";$("overviewVideoMode").textContent=running?(s.video_mode==="live"?"Live monitoring":"Sample / replay"):"Off";$("overviewVideoMode").className=`badge ${running?"green":"neutral"}`;$("cameraSourceLabel").textContent=running?(s.video_mode==="live"?"Live image":"Sample / replay"):"Sample preview";$("cameraIdle").hidden=running;loadFrame("overviewFrame");if(app.report)drawBars($("overviewChart"),app.report.series,"reps","#579579");else loadReport(true);}
function renderMonitor(){
  const v=app.status.video,running=v.state==="running",sensor=app.status.sensor;
  const coverage=app.status.coverage.filter(c=>["vision","accel"].includes(c.channel));
  const active=coverage.filter(c=>c.state==="active").length;
  $("monitorChannels").innerHTML=coverage.map(channelRow).join("");
  $("monitorMode").textContent=active===2?"Both channels monitoring":active===1?"1 / 2 channels valid":"Waiting for live input";
  $("monitorMode").className=`badge ${active===2?"green":active===1?"amber":"neutral"}`;
  $("videoStateLabel").textContent=({idle:"Not started",running:app.status.video_mode==="live"?"Live video detection":"Video replay",finished:"This detection has finished",error:"Detection error"})[v.state]||v.state;
  $("videoFps").textContent=`Inference ${v.fps||0} FPS`;
  $("videoProbability").textContent=v.frames?`${(100*(v.prob||0)).toFixed(1)}%`:"--";
  $("videoFrames").textContent=v.frames||0;$("videoFalls").textContent=v.n_falls||0;
  $("videoVisibility").textContent=v.frames?(v.person_visible?"Person visible":"No person detected"):"Not detected";
  $("startVideo").disabled=running||app.status.modules.find(m=>m.id==="video").state!=="ready";
  $("stopVideo").disabled=!running;
  $("videoModuleInfo").innerHTML=`<div><span>Detection scheme</span><strong>A4</strong></div><div><span>Original threshold</span><strong>0.60</strong></div><div><span>Confirmed count</span><strong>2</strong></div><div><span>Target sampling frame rate</span><strong>15 FPS</strong></div>${v.error?`<div class="error-inline">${esc(v.error)}</div>`:""}`;
  if(v.frames){const last=app.videoHistory.at(-1);if(!last||last.frame!==v.frames)app.videoHistory.push({frame:v.frames,value:v.prob});app.videoHistory=app.videoHistory.slice(-120);}
  drawLines($("videoChart"),[{values:app.videoHistory.map(x=>x.value),color:"#47836a"}],{min:0,max:1,threshold:.6});
  const fresh=sensor.state==="connected"&&Date.now()/1000-(sensor.last_received||0)<5;
  $("sensorStatus").textContent=sensor.state==="connected"?(fresh?`Live data connected · ${sensor.windows||0} windows`:"Sensor data interrupted"):"Not connected";
  if(sensor.state==="connected"){
    $("signalState").textContent=fresh?"Monitoring live":"Data interrupted";
    $("signalState").className=`badge ${fresh?"green":"amber"}`;
    const samples=sensor.samples||[],recent=sensor.recent||[];
    drawLines($("signalChart"),[1,2,3].map((axis,i)=>({values:samples.map(s=>s[axis]),color:["#547bb4","#5a9877","#bf9459"][i]})));
    drawLines($("signalProbabilityChart"),[{values:recent.map(s=>s.p),color:"#52886d"}],{min:0,max:1,threshold:.5});
    $("signalResult").textContent=`Live-window fall probability: ${((sensor.prob||0)*100).toFixed(1)}%`;
    $("signalResult").dataset.live="true";
  }else if($("signalResult").dataset.live){
    $("signalState").textContent="Sensor disconnected";$("signalState").className="badge neutral";
  }else if(!$("signalChart").dataset.drawn){
    drawLines($("signalChart"),[],{min:-2,max:2});drawLines($("signalProbabilityChart"),[],{min:0,max:1,threshold:.5});
    $("signalChart").dataset.drawn="true";
  }
  loadFrame("monitorFrame");
}
function loadFrame(id){
  const image=$(id), visible=(id==="overviewFrame"&&app.page==="overview")||(id==="monitorFrame"&&app.page==="monitor");
  if(!visible)return;
  const other=$(id==="overviewFrame"?"monitorFrame":"overviewFrame");
  if(other.dataset.streaming){other.removeAttribute("src");delete other.dataset.streaming;}
  if(app.status.video.state==="running"){
    const key=String(app.status.video.session);
    if(image.dataset.streaming!==key){image.src=`/api/video/stream?session=${key}`;image.dataset.streaming=key;}
  }else if(image.dataset.streaming||!image.getAttribute("src")){
    image.src=app.status.video.frames?`/api/video/frame?t=${Date.now()}`:"/static/media/room-preview.jpg";
    delete image.dataset.streaming;
  }
  if(app.status.video.state!=="running")app.pose=null;
  paintOverlay(id);
}

/* ---------------------------------------------------------------------------
   画面叠加：目标检测框与骨架关键点。
   原算法目录不可修改，也不能单独开关它自带的两类绘制，因此引擎以「干净画面」
   输出，检测框与骨架由前端在画面之上独立绘制。开关只影响本机看到的画面，
   不改变原算法的检测、阈值与判定结果。
--------------------------------------------------------------------------- */
const OVERLAY_KEYS = {boxes:"box",skeleton:"skeleton"};
const OVERLAY_STORE = "homeease.videoOverlay";
const POSE_EDGES = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];
const POSE_MIN_SCORE = .3;

function overlayCanvasId(frameId){return frameId==="overviewFrame"?"overviewOverlay":"monitorOverlay";}
function activeCameraId(){return app.page==="monitor"?"monitorFrame":app.page==="overview"?"overviewFrame":null;}
function overlayVisible(){return app.overlay.box||app.overlay.skeleton;}

function restoreOverlay(){
  try{
    const saved=JSON.parse(localStorage.getItem(OVERLAY_STORE)||"null");
    if(saved)app.overlay={box:saved.box!==false,skeleton:saved.skeleton!==false};
  }catch(e){}
}
function paintOverlay(frameId){
  const target=frameId||activeCameraId();
  if(!target)return;
  const canvas=$(overlayCanvasId(target));
  if(!canvas)return;
  const setup=canvasContext(canvas);
  if(!setup)return;
  const {ctx,w,h}=setup;
  ctx.clearRect(0,0,w,h);
  const pose=app.pose;
  if(!pose||!overlayVisible())return;
  const image=$(target),frameW=pose.width||image?.naturalWidth,frameH=pose.height||image?.naturalHeight;
  const scale=frameW&&frameH?Math.min(w/frameW,h/frameH):1;
  const area={x:(w-frameW*scale)/2,y:(h-frameH*scale)/2,w:frameW*scale,h:frameH*scale};
  if(app.overlay.box&&Array.isArray(pose.box)&&pose.box.length===4){
    const [x1,y1,x2,y2]=pose.box;
    const left=area.x+x1*area.w,top=area.y+y1*area.h;
    const width=Math.max(2,(x2-x1)*area.w),height=Math.max(2,(y2-y1)*area.h);
    ctx.strokeStyle="#e08030";ctx.lineWidth=2;ctx.setLineDash([]);
    ctx.strokeRect(left,top,width,height);
    const label=`Suspected fall ${((pose.prob||0)*100).toFixed(0)}%`;
    ctx.font='11px "Segoe UI","Microsoft YaHei",sans-serif';ctx.textBaseline="middle";ctx.textAlign="left";
    const chipTop=Math.max(top-17,0),chipWidth=ctx.measureText(label).width+12;
    ctx.fillStyle="#e08030";ctx.fillRect(left,chipTop,chipWidth,17);
    ctx.fillStyle="#fff";ctx.fillText(label,left+6,chipTop+9);
  }
  if(app.overlay.skeleton&&pose.keypoints?.length){
    const points=pose.keypoints.map(p=>[area.x+p[0]*area.w,area.y+p[1]*area.h,p[2]]);
    ctx.lineCap="round";ctx.lineWidth=2;
    for(const [a,b] of POSE_EDGES){
      if(points[a]?.[2]<POSE_MIN_SCORE||points[b]?.[2]<POSE_MIN_SCORE)continue;
      ctx.strokeStyle=a%2?"#338567":"#5a84ba";
      ctx.beginPath();ctx.moveTo(points[a][0],points[a][1]);ctx.lineTo(points[b][0],points[b][1]);ctx.stroke();
    }
    for(const point of points){
      if(point[2]<POSE_MIN_SCORE)continue;
      ctx.fillStyle="#f9fcfa";ctx.strokeStyle="#4b876f";ctx.lineWidth=2;
      ctx.beginPath();ctx.arc(point[0],point[1],3.2,0,Math.PI*2);ctx.fill();ctx.stroke();
    }
  }
}
async function loadPose(){
  const target=activeCameraId();
  const running=app.status?.video?.state==="running";
  if(!target||!running||!overlayVisible()){app.pose=null;paintOverlay();return;}
  if(app.poseBusy)return;
  app.poseBusy=true;
  try{app.pose=await api("/api/video/pose",{timeout:8000});}
  catch(e){app.pose=null;}
  finally{app.poseBusy=false;}
  paintOverlay();
}
function setOverlay(key,on){
  if(!(key in OVERLAY_KEYS))return;
  app.overlay[key]=!!on;
  try{localStorage.setItem(OVERLAY_STORE,JSON.stringify(app.overlay));}catch(e){}
  for(const [id,name] of [["toggleBoxes","box"],["toggleSkeleton","skeleton"]]){
    const button=$(id);
    if(!button)continue;
    button.classList.toggle("active",app.overlay[name]);
    button.setAttribute("aria-pressed",String(app.overlay[name]));
  }
  const hint=$("overlayHint");
  if(hint){
    const shown=[app.overlay.box,app.overlay.skeleton].filter(Boolean).length;
    hint.textContent=shown===2?"Show all":shown===1?(app.overlay.box?"Detection box only":"Skeleton keypoints only"):"Off";
  }
  if(!overlayVisible())app.pose=null;
  paintOverlay();
}

function renderEvents(){let list=app.events.filter(e=>(app.eventFilter==="all"||e.state===app.eventFilter)&&(app.eventMode==="all"||e.mode===app.eventMode));const q=$("eventSearch").value.trim().toLowerCase();if(q)list=list.filter(e=>`${e.id} ${e.room} ${e.resident}`.toLowerCase().includes(q));$("eventsTotal").textContent=`${list.length} records`;$("eventList").innerHTML=list.length?list.map(e=>`<article class="event-record ${e.level==="EMERGENCY"?"emergency":""} ${e.state==="resolved"?"resolved":""}" data-event="${e.id}" role="button" tabindex="0"><div><h3>${esc(e.room)} · ${e.channel==="manual"?"Asked for help":e.channel==="audio"?"Spoken call for help":"Suspected fall"}${levelBadge(e)}${modeBadge(e.mode)}${voiceChip(e.voice)}</h3><p>${channels[e.channel]}${e.channel==="manual"?"":"detection"} · ${esc(e.resident)}${e.outcome?` · ${outcomes[e.outcome]}`:""}</p><footer><span>${dateTime(e.occurred_at)}</span><span>Event #${e.id}</span><span>${e.acknowledged_at?"A caregiver took over":""}</span></footer></div><span class="badge neutral">${e.tag?({single_channel:"Single channel",dual_channel:"Dual channel",triple_channel:"Triple channel"})[e.tag]:"Asked for help"}</span>${icon("chevron-right")}</article>`).join(""):empty("No events match the filter","inbox");icons();}
async function openEvent(id){try{renderEventDetail(await api(`/api/events/${id}`));if(!$("eventDialog").open)$("eventDialog").showModal();}catch(e){toast(e.message,true);}}
function evidenceText(e){const v=e.evidence;if(e.channel==="vision")return `The original video algorithm detected a suspected fall. Trigger probability ${((v.prob||0)*100).toFixed(1)}%, decision threshold ${v.threshold??.6}${v.peak?`, peak probability ${(v.peak*100).toFixed(1)}%`:""}.`;if(e.channel==="accel")return `The original accelerometer algorithm detected a suspected fall. Trigger probability ${((v.score||0)*100).toFixed(1)}%, decision threshold ${v.threshold??.5}${v.n_windows?`, hit ${v.n_windows} windows`:""}.`;if(e.channel==="audio")return "The original spoken-help listener recognized a clear call for help.";return "The older adult asked for help with the help button.";}
function renderEventDetail(e){app.selectedEvent=e;const settings=app.settings;const inquiry=e.inquiry;const inquiryText={not_started:"Not started",listening:"Waiting for reply",audio_unavailable:"Audio unavailable",completed:"Completed",button_safe:"The older adult says they are safe"}[inquiry.status]||inquiry.status;$("eventDetail").innerHTML=`<div class="event-detail-body"><div class="event-detail-header"><h3>${esc(e.room)} · ${esc(e.resident)}</h3><div>${levelBadge(e)} ${modeBadge(e.mode)}</div></div><div class="event-detail-meta"><div><span>Event ID</span>#${e.id}</div><div><span>Occurred at</span>${dateTime(e.occurred_at)}</div><div><span>Detection channel</span>${channels[e.channel]}</div><div><span>Voice question</span>${e.voice?voiceBadge(e.voice):esc(inquiryText)}</div><div><span>Detected at</span>${dateTime(e.detected_at)}</div><div><span>Input source</span>${esc(e.source)}</div></div>${e.withdrawn?`<div class="completion-banner">${icon("rotate-ccw")}<strong>The older adult withdrew the call for help. The event is kept; please contact them to confirm and then record the resolution.</strong></div>`:""}<div class="evidence-summary">${evidenceText(e)}${e.mode==="replay"?'<br><span class="badge blue">Sample / replay record</span>':""}</div>${e.evidence.clip?`<video class="event-video" controls preload="metadata" src="/api/video/clip/${e.id}"></video>`:""}${e.evidence.snippet?'<div class="section-heading"><h3>Acceleration around the alarm</h3><span class="muted small">−2 s to +1 s · g</span></div><div class="chart-wrap"><canvas id="eventSignalChart"></canvas></div>':""}${e.state!=="resolved"?`<div class="event-actions"><button class="button primary" data-action="acknowledge" ${e.acknowledged_at?"disabled":""}>${icon("user-check")}${e.acknowledged_at?"Taken over":"Take over"}</button><button class="button danger-outline" data-action="escalate">${icon("arrow-up-right")} Escalate response</button><a class="button secondary" href="/elder?event=${e.id}" target="_blank">${icon("audio-lines")} Older adult inquiry</a></div>`:""}<div>${(settings?.contacts||[]).map((c,i)=>`<div class="event-contact-row"><span class="badge neutral">${i+1}</span><strong>${esc(c.name)}</strong><span class="muted">${esc(c.relation)}</span><a class="button secondary" href="tel:${esc(c.phone.replace(/[^+0-9]/g,""))}" data-contact-call="true">${icon("phone")}${esc(c.phone)}</a></div>`).join("")||'<p class="muted small">No emergency contacts yet</p>'}</div><div class="section-heading spaced"><h3>Voice inquiry conclusion</h3><span class="muted small">${e.inquiry?.asked_at?`Started at ${dateTime(e.inquiry.asked_at)}`:"No inquiry started"}</span></div><div class="inquiry-conclusion">${e.voice?voiceBadge(e.voice):'<span class="badge neutral">Not asked</span>'}</div><div class="section-heading spaced"><h3>Response timeline</h3><a class="text-action" href="/reports/events/${e.id}" target="_blank">${icon("printer")} Event report</a></div><ol class="timeline">${e.timeline.map(t=>`<li><time>${dateTime(t.at)}</time><div>${esc(t.note)}<small>${esc(t.actor)}</small></div></li>`).join("")}</ol>${e.state==="resolved"?`<div class="completion-banner">${icon("circle-check")}<strong>${outcomes[e.outcome]}</strong><span>${e.response_s!==null?`Response time ${duration(e.response_s)}`:""}</span></div>${e.notes?`<p class="muted small" style="margin-top:15px">${esc(e.notes)}</p>`:""}`:`<form id="resolveForm" class="resolution-form"><h3>Resolution</h3><label>Final result<select id="resolveOutcome" required><option value="">Choose a resolution</option><option value="confirmed_fall">Confirmed fall</option><option value="false_alarm">False alarm</option><option value="safe">Verified safe</option><option value="unresolved">Undetermined</option></select></label><label>Notes<textarea id="resolveNote" rows="3" maxlength="2000" placeholder="Contact attempts, measures taken, etc."></textarea></label><button class="button primary" type="submit">${icon("check")} Record the result and close the event</button></form>`}</div>`;icons();if(e.evidence.snippet){requestAnimationFrame(()=>{const s=e.evidence.snippet;drawLines($("eventSignalChart"),[{values:s.ax,color:"#547bb4"},{values:s.ay,color:"#5a9877"},{values:s.az,color:"#bf9459"}],{labels:s.t?.map(t=>`${t.toFixed(1)}s`)});});}$("resolveForm")?.addEventListener("submit",ev=>{ev.preventDefault();busy(ev.submitter,()=>eventAction("resolve",{outcome:$("resolveOutcome").value,note:$("resolveNote").value}));});}
async function eventAction(action,extra={}){const e=app.selectedEvent;const result=await api(`/api/events/${e.id}/action`,{method:"POST",body:{action,revision:e.revision,...extra}});renderEventDetail(result);await refresh();toast(action==="resolve"?"Resolution recorded":"Action recorded");}

function renderExerciseHistory(){const sessions=app.sessions;$("exerciseHistory").innerHTML=sessions.length?`<table><thead><tr><th>Exercise time</th><th>Exercise</th><th>Reps / target</th><th>Valid observation</th><th>Source</th><th>Status</th><th>Report</th></tr></thead><tbody>${sessions.slice(0,30).map(s=>`<tr><td>${dateTime(s.created_at)}</td><td>${names[s.exercise]}</td><td>${s.count} / ${s.target_reps}</td><td>${duration(s.valid_duration_s)}</td><td>${modeBadge(s.mode)}</td><td>${({running:"In progress",paused:"Paused",ended:"Ended",interrupted:"Interrupted"})[s.status]||s.status}</td><td><a class="icon-button" title="Download exercise report" href="/api/sessions/${s.session_id}/export.json" aria-label="Download exercise report">${icon("download")}</a></td></tr>`).join("")}</tbody></table>`:empty("No exercise records","activity","",true);icons();}
function renderExerciseGuide(element,key){
  const guide=exerciseGuides[key];
  element.dataset.exerciseGuide=key;
  element.innerHTML=`<div class="exercise-guide-copy"><h3>${esc(guide.title)} · Movement and Rep Guide</h3><p class="exercise-count-rule">${esc(guide.count)}</p><ol>${guide.steps.map(step=>`<li>${esc(step)}</li>`).join("")}</ol><p class="exercise-motion-note">${esc(guide.note)}</p></div><figure><img src="/static/media/${key}-guide.png" alt="${esc(guide.title)}: Start, peak, and return illustration" width="780" height="252"><figcaption>${guide.endpoints.map(label=>`<span>${esc(label)}</span>`).join("")}</figcaption>${guide.animation?`<div class="exercise-guide-controls"><span>45-Degree Movement Reference</span><button type="button" class="icon-button" title="Play Movement Reference" aria-label="Play Movement Reference" aria-pressed="false">${icon("play")}</button></div>`:""}</figure><p class="exercise-camera-guide">${icon("camera")}<span><strong>Camera Position: </strong>Facing the camera is 0 degrees; fully side-on is 90 degrees. Turn about 45 degrees, with either side nearer the camera; no exact measurement is needed. Keep the camera level at about hip height. Fit your head and both feet in frame with space around you, and keep both legs visible.</span></p><p class="exercise-pace">Suggested pace: take about 1-2 seconds to rise or lift, pause briefly, then return over about 1-2 seconds. Settle for about 1 second before the next rep. Use a comfortable range of motion.</p>`;
  const playback=element.querySelector('.exercise-guide-controls button');
  if(playback)playback.onclick=()=>{
    const playing=playback.getAttribute('aria-pressed')!=='true';
    element.querySelector('figure img').src=`/static/media/${key}-guide.${playing?'gif':'png'}`;
    playback.setAttribute('aria-pressed',String(playing));
    playback.title=playback.ariaLabel=playing?"Stop Movement Reference":"Play Movement Reference";
    playback.innerHTML=icon(playing?'square':'play');
    icons();
  };
}
function chooseExercise(key){app.selectedExercise=key;document.querySelectorAll(".exercise-option").forEach(x=>{const selected=x.dataset.exercise===key;x.classList.toggle("selected",selected);x.setAttribute("aria-pressed",selected);});renderExerciseGuide($("exerciseGuide"),key);icons();syncExerciseSwitches();
  // 从长者端计划进来时，把计划里的次数先填好，别让人以为要按默认值做。
  const planned=household?app.plan?.items?.find(item=>item.exercise===key&&item.enabled):null;
  if(planned&&$("targetReps"))$("targetReps").value=planned.target_reps;}
function renderExerciseCycle(s){
  const guide=exerciseGuides[s.exercise];
  const previous=app.exerciseFeedback;
  const changed=previous?.sid===s.session_id&&s.count>previous.count;
  const justCounted=changed?Date.now():(previous?.sid===s.session_id?previous.justCounted:0);
  app.exerciseFeedback={sid:s.session_id,count:s.count,justCounted};
  const stage=Number.isInteger(s.counter_stage)?s.counter_stage:-1;
  const completed=Date.now()-justCounted<1800;
  const active=s.status==="running"?(completed?4:stage):-1;
  $("exerciseCycle").innerHTML=guide.cycle.map((label,index)=>`<li class="${index===active?"current":index<active?"confirmed":""}" ${index===active?'aria-current="step"':""}><b>${index+1}</b><span>${esc(label)}</span></li>`).join("");
  let cue=guide.cues[stage+1];
  if(s.status==="paused")cue="Paused. Return to the starting posture after resuming.";
  else if(["ended","interrupted"].includes(s.status))cue=s.status==="ended"?"Training ended.":"Training interrupted.";
  else if(s.quality!=="good")cue="Keep both shoulders, hips, knees, and ankles visible. Once visibility returns, restart from the starting posture.";
  else if(completed)cue="Complete rep counted. Settle before starting the next rep.";
  else if(s.exercise==="chair_sit_to_stand"&&s.counting_policy?.startsWith("pose_cycle_")&&stage<0){
    cue=s.setup_pose==="standing"?"Standing recognized. Sit on the chair and settle before starting.":"Sit steady briefly to confirm the starting position.";
  }
  if($("exerciseCue").dataset.cue!==cue){$("exerciseCue").dataset.cue=cue;$("exerciseCue").textContent=cue;}
}
async function startExercise(ev){
  ev.preventDefault();
  await busy($("startExercise"),async()=>{
    const source=$("exerciseSource").value;
    let uploadId=null;
    // 从长者端的计划卡片进来时，目标次数取计划里的设定。
    const planItem=household&&app.plan?.items?.find(item=>item.exercise===app.selectedExercise&&item.enabled);
    if(planItem)$("targetReps").value=planItem.target_reps;
    if(source==="upload"){
      const file=$("exerciseUpload").files[0];if(!file)throw new Error("Please choose an exercise video");
      const form=new FormData();form.append("file",file);
      uploadId=(await api("/api/uploads/video",{method:"POST",body:form})).upload_id;
    }
    const result=await api("/api/sessions",{method:"POST",body:{exercise:app.selectedExercise,target_reps:Number($("targetReps").value),source,upload_id:uploadId,camera_index:Number($("exerciseCameraIndex").value),capture:"shared"}});
    app.sid=result.session_id;app.exerciseSource=source;
    $("exerciseChooser").hidden=true;$("exerciseActive").hidden=false;$("exerciseCompleted").hidden=true;
    $("pauseExercise").disabled=false;$("endExercise").disabled=false;
    if(source==="camera"){
      $("exerciseCamera").hidden=false;$("exerciseCamera").src=`/api/sessions/${app.sid}/camera/stream`;
    }else $("exerciseCamera").hidden=true;
    renderExerciseState(result);
    if(household)app.plan=await api("/api/plan").catch(()=>app.plan);
    await refresh();
  });
}
async function controlExercise(action){
  if(!app.sid||app.exerciseControlling)return;
  app.exerciseControlling=true;
  $("pauseExercise").disabled=true;$("endExercise").disabled=true;
  try{
    renderExerciseState(await api(`/api/sessions/${app.sid}/control`,{method:"POST",body:{action}}));
    if(action==="end")await refresh();
  }finally{
    app.exerciseControlling=false;
    syncExerciseControls();
  }
}
function syncExerciseControls(){const finished=["ended","interrupted"].includes(app.exercise?.status);$("pauseExercise").disabled=finished||!!app.exerciseControlling;$("endExercise").disabled=finished||!!app.exerciseControlling;}
async function stopCamera(){ $("exerciseCamera").removeAttribute("src");$("exerciseCamera").hidden=true; }
function renderExerciseState(s){
  app.exercise=s;$("activeExerciseName").textContent=names[s.exercise];$("exerciseMode").textContent=s.mode==="live"?"Live exercise":"Sample / replay";
  if($("activeExerciseGuide").dataset.exerciseGuide!==s.exercise)renderExerciseGuide($("activeExerciseGuide"),s.exercise);
  renderExerciseCycle(s);
  $("repCount").textContent=s.count;$("repTarget").textContent=s.target_reps;$("repProgress").value=s.progress;
  $("exercisePhase").textContent=phases[s.phase]||s.phase;$("exerciseDuration").textContent=duration(s.valid_duration_s);
  const q=["multiple_people","no_person"].includes(s.person_id)?s.person_id:s.quality;
  const diagnostic=s.capture_diagnostics;
  const jointNames={left_shoulder:"Left shoulder",right_shoulder:"Right shoulder",left_hip:"Left hip",right_hip:"Right hip",left_knee:"Left knee",right_knee:"Right knee",left_ankle:"Left ankle",right_ankle:"Right ankle"};
  let quality=qualityNames[q]||q;
  const chairSetup=q==="good"&&s.exercise==="chair_sit_to_stand"&&s.counting_policy?.startsWith("pose_cycle_")&&s.counter_stage<0;
  const unrecognized=q==="good"&&s.phase==="other"&&!chairSetup;
  if(unrecognized)quality="Joints visible; selected exercise not yet recognized";
  if(chairSetup)quality=s.setup_pose==="standing"?"Standing recognized - sit steady to start":"Body visible - confirming seated start";
  if(diagnostic?.missing_joints?.length&&q==="insufficient_pose")quality+=`: ${diagnostic.missing_joints.map(n=>jointNames[n]||n).join(", ")}`;
  const inputGapLimit=s.max_input_gap_ms??250;
  if(diagnostic?.gap_ms>inputGapLimit&&q==="good")quality="The image was interrupted; the current movement is being observed again";
  $("exerciseQuality").textContent=quality;
  $("exerciseQuality").title=diagnostic?`Valid sampling ${diagnostic.observation_fps||0} FPS · latest gap ${diagnostic.gap_ms||0} ms`:"";
  $("exerciseQuality").classList.toggle("bad",unrecognized||!["good","waiting_for_input"].includes(q)||diagnostic?.gap_ms>inputGapLimit);
  const counts=s.counting_diagnostics,details=$("exerciseDiagnostics");
  details.hidden=!counts;
  if(counts){
    const joints=Object.entries(diagnostic?.joint_confidences||{});
    const interruptions=Object.values(counts.interruption_counts).reduce((sum,n)=>sum+n,0);
    const last=counts.last_interruption;
    details.innerHTML=`<summary>Joints and Input Status</summary><div class="exercise-input-stats"><span>Required joints visible <strong>${Math.round(counts.visible_pose_ratio*100)}%</strong></span><span>Uninterrupted input <strong>${counts.uninterrupted_s.toFixed(1)} s</strong></span><span>Sampling <strong>${diagnostic?.observation_fps||0} FPS</strong></span><span>Input interruptions <strong>${interruptions} reps</strong></span></div><div class="exercise-joints">${joints.map(([name,value])=>`<span class="${value<.3?"missing":"visible"}" title="${jointNames[name]||name} · ${(value*100).toFixed(1)}%"><i aria-hidden="true"></i>${jointNames[name]||name}<small>${Math.round(value*100)}%</small></span>`).join("")}</div>${last?`<p class="exercise-last-reset">Last interruption: ${esc(qualityNames[last.reason]||last.reason)}</p>`:""}`;
  }
  $("exerciseDiagnosticExport").href=`/api/sessions/${s.session_id}/diagnostics.json`;
  $("pauseExercise").innerHTML=icon(s.status==="paused"?"play":"pause")+`<span>${s.status==="paused"?"Resume":"Pause"}</span>`;
  if(["ended","interrupted"].includes(s.status)){
    $("exerciseCompleted").hidden=false;$("exerciseCompleted").querySelector("strong").textContent=s.status==="ended"?`Session ended · ${s.count} reps`:"Session interrupted";
    $("pauseExercise").disabled=true;$("endExercise").disabled=true;stopCamera();
  }
  drawPose($("poseCanvas"),s.keypoints,s.mode!=="live");icons();
}
async function pollExercise(){if(!app.sid||exercisePollBusy)return;const sid=app.sid;exercisePollBusy=true;try{const s=await api(`/api/sessions/${sid}`,{timeout:15000});if(app.sid===sid)renderExerciseState(s);}catch(e){}finally{exercisePollBusy=false;}}
function renderAudioHistory(){
  $("soundHistory").innerHTML=app.sounds.length?`<table><thead><tr><th>Time</th><th>Task</th><th>Result</th><th>Source</th></tr></thead><tbody>${app.sounds.map(s=>{
    const labels=(s.result.sounds?.events||s.result.events||[]).map(e=>soundNames[e.category]||e.category).join(", ")||"No confirmed sound events";
    const help=(s.result.help_result||s.result).help?"Call for help detected":"No clear call for help detected";
    return `<tr><td>${dateTime(s.created_at)}</td><td>${s.task==="monitor"?"Joint analysis":s.task==="help"?"Call-for-help listening":"Sound labels"}</td><td>${esc(s.task==="monitor"?`${labels} · ${help}`:s.task==="help"?help:labels)}</td><td>${modeBadge(s.mode)}</td></tr>`;
  }).join("")}</tbody></table>`:empty("No sound records","audio-lines","",true);
  const inquiries=app.events.filter(e=>e.mode==="live"&&e.state==="open").slice(0,8);
  $("inquiryQueue").innerHTML=inquiries.map(e=>`<div class="inquiry-row"><span>${esc(e.room)} · #${e.id.slice(0,6)}</span><span class="badge neutral">${({not_started:"Waiting to be asked",listening:"Waiting for the voice reply",processing:"Recognizing the reply",completed:"Inquiry completed",audio_unavailable:"Inquiry audio unavailable",button_safe:"The safe button was pressed"})[e.inquiry?.status]||"Waiting to be asked"}</span><button class="text-action" data-event="${e.id}">View event ${icon("arrow-right")}</button></div>`).join("")||empty("No events waiting for an inquiry","circle-check","",true);
}
function renderJointAudioResult(result){
  renderAudioResult(result.sounds,"sounds");const sounds=$("audioResult").innerHTML;
  renderAudioResult(result.help_result,"help");
  $("audioResult").innerHTML=`<h3>Ambient sounds</h3>${sounds}<div class="separator"></div><h3>Spoken calls for help</h3>${$("audioResult").innerHTML}`;
  $("audioResultDuration").textContent=`${Number(result.duration_s||0).toFixed(1)} s`;icons();
}
function renderLiveAudio(result){
  const degraded=result.pipeline?.status?.startsWith("degraded")||String(result.listener?.status||"").startsWith("degraded");
  $("audioLiveStatus").textContent=degraded?"Listening coverage is limited. Check the running status":"Microphone connected · listening for sounds and calls for help";
  for(const id of ["soundLiveState","helpLiveState"]){$(id).textContent=degraded?"Coverage limited":"Listening";$(id).className=`badge ${degraded?"amber":"green"}`;}
  $("liveSoundResult").innerHTML=(result.events||[]).slice().reverse().map(e=>`<div class="sound-event"><span class="sound-type">${icon("audio-lines")}</span><div><h3>${soundNames[e.category]||esc(e.category)}</h3><small>${dateTime(e.t_start_utc)} · ${e.n_windows||e.windows||0} windows</small></div><span class="badge blue">${(e.confidence*100).toFixed(1)}%</span></div>`).join("")||empty("No confirmed sound events yet","volume-1","",true);
  app.liveCalls=[...(app.liveCalls||[]),...(result.calls||[])].slice(-20);
  $("liveHelpResult").innerHTML=app.liveCalls.slice().reverse().map(c=>`<div class="sound-event"><span class="sound-type">${icon("bell-ring")}</span><div><h3>A spoken call for help was detected</h3><small>${dateTime(c.t_utc||new Date().toISOString())}</small></div><span class="badge red">Event created</span></div>`).join("")||empty("No spoken calls for help","speech","",true);
  icons();
}
function renderAudioResult(result,task){$("audioResultDuration").textContent=result.duration_s?`${result.duration_s} s`:"";if(task==="sounds"){$("audioResult").innerHTML=(result.events||[]).map(e=>`<div class="sound-event"><span class="sound-type">${icon("audio-lines")}</span><div><h3>${soundNames[e.category]||esc(e.category)}</h3><small>${e.start_s??dateTime(e.t_start_utc)}${e.end_s!==undefined?` – ${e.end_s} s`:""} · ${e.windows||e.n_windows||0} windows</small></div><span class="badge blue">${(e.confidence*100).toFixed(1)}%</span></div>`).join("")||empty("Unconfirmed sound events","volume-1");}else{$("audioResult").innerHTML=`<div class="help-result">${icon(result.help?"bell-ring":"circle-check")}<h3>${result.help?"A clear spoken call for help was detected":"No clear call for help detected"}</h3><p>${(result.segments||[]).length} speech segments</p>${(result.segments||[]).map(s=>`<div class="sound-event"><div><h3>${({help:"Clear call for help",not_help:"Not a call for help",not_a_call:"Not a clear call for help",cooldown:"In cooldown",too_short:"Speech too short",too_long:"Speech too long",low_confidence:"Recognition confidence too low",no_text:"No text recognized",too_many_words:"Utterance too long"})[s.verdict]||esc(s.verdict)}</h3><small>${s.start_s} s · ${s.duration_s?.toFixed(2)} s</small></div></div>`).join("")}</div>`;}icons();}
async function analyzeAudio(ev){ev.preventDefault();await busy($("analyzeAudio"),async()=>{if(!app.settings.audio_consent)throw new Error("Please obtain the audio-processing consent first");const task="monitor",source=$("audioSource").value;let response;if(source==="sample"){response=await api("/api/audio/sample",{method:"POST",body:{sample:$("audioSampleSelect").value,task}});}else{const file=source==="record"?app.recordedBlob:$("audioFile").files[0];if(!file)throw new Error(source==="record"?"Please finish recording first":"Please choose an audio file");const form=new FormData();form.append("file",file,"audio.webm");form.append("task",task);form.append("mode",source==="record"?"live":"replay");response=await api("/api/audio/analyze",{method:"POST",body:form});}renderJointAudioResult(response.result);await refresh();});}
async function recordAudio(){await busy($("recordAudio"),async()=>{
  if(!app.settings.audio_consent)throw new Error("Please obtain the audio-processing consent first");
  if(!navigator.mediaDevices?.getUserMedia)throw new Error("The microphone needs localhost or HTTPS access");
  const stream=await navigator.mediaDevices.getUserMedia({audio:true});let interval;
  app.recordingStream=stream;
  try{
    const recorder=new MediaRecorder(stream),chunks=[];app.audioRecorder=recorder;
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    await new Promise((resolve,reject)=>{
      recorder.onerror=()=>reject(new Error("Recording failed"));
      recorder.onstop=()=>{
        clearInterval(interval);
        if(app.settings.audio_consent){app.recordedBlob=new Blob(chunks,{type:recorder.mimeType});$("audioPreview").src=URL.createObjectURL(app.recordedBlob);}
        else app.recordedBlob=null;
        resolve();
      };
      recorder.start();let remaining=10;$("recordStatus").textContent=`Recording · ${remaining} s`;
      interval=setInterval(()=>{remaining--;$("recordStatus").textContent=`Recording · ${remaining} s`;if(remaining<=0||!app.settings.audio_consent){clearInterval(interval);if(recorder.state!=="inactive")recorder.stop();}},1000);
    });
    $("recordStatus").textContent=app.settings.audio_consent?"Recording is ready":"Audio consent withdrawn";
  }finally{clearInterval(interval);stream.getTracks().forEach(t=>t.stop());app.audioRecorder=null;app.recordingStream=null;}
});}
async function startAudioLive(){
  if(app.audioStream||app.audioStarting||app.audioStopping)return;
  app.audioStarting=true;
  try{
  await busy($("startAudioLive"),async()=>{
    let stream,context;
    try{
    if(!app.settings.audio_consent)throw new Error("Please obtain the audio-processing consent first");
    if(!navigator.mediaDevices?.getUserMedia)throw new Error("The microphone needs localhost or HTTPS access");
    stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:true}});
    context=new AudioContext({sampleRate:16000});
      if(context.sampleRate!==16000)throw new Error("This browser does not support 16 kHz audio input");
      await context.audioWorklet.addModule("/static/pcm-worklet.js");
      const socket=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/audio`);
      app.audioStream=stream;app.audioContext=context;app.audioSocket=socket;
      await new Promise((resolve,reject)=>{
        const timer=setTimeout(()=>fail(new Error("Sound connection timed out")),20000);
        const fail=error=>{clearTimeout(timer);reject(error);};
        socket.onmessage=event=>{const r=JSON.parse(event.data);if(r.error)fail(new Error(r.error));else if(r.state==="listening"){clearTimeout(timer);resolve();}};
        socket.onerror=()=>fail(new Error("Sound listening connection failed"));
        socket.onclose=()=>fail(new Error("Sound listening is not connected. Check the consent and module status"));
      });
      app.liveCalls=[];
      const queue=new AudioChunkQueue(message=>socket.send(JSON.stringify(message)));app.audioQueue=queue;
      const source=context.createMediaStreamSource(stream),processor=new AudioWorkletNode(context,"homeease-pcm");source.connect(processor);
      const silent=context.createGain();silent.gain.value=0;processor.connect(silent).connect(context.destination);
      processor.port.onmessage=event=>{
        const {samples,timestamp_ms}=event.data;
        const rms=Math.sqrt(samples.reduce((sum,v)=>sum+v*v,0)/samples.length);$("audioLevel").style.width=`${Math.min(100,rms*350)}%`;
        if(!app.settings.audio_consent||socket.readyState!==WebSocket.OPEN||app.audioStopping)return;
        const bytes=new Uint8Array(samples.buffer);let binary="";for(let i=0;i<bytes.length;i++)binary+=String.fromCharCode(bytes[i]);
        try{queue.push({pcm:btoa(binary),timestamp_ms});}catch(e){toast(e.message,true);queue.close();socket.close();stopAudioLive();}
      };
      socket.onmessage=event=>{
        const result=JSON.parse(event.data);
        if(result.error){toast(result.error,true);queue.close();stopAudioLive();return;}renderLiveAudio(result);
        queue.acknowledge();
      };
      socket.onclose=()=>{queue.close();stopAudioLive();};socket.onerror=()=>toast("Microphone connection lost",true);
      $("startAudioLive").disabled=true;$("stopAudioLive").disabled=false;renderLiveAudio({});await context.resume();window.HomeMonitoring?.audioReady();
    }catch(e){stream?.getTracks().forEach(t=>t.stop());if(context&&context.state!=="closed")await context.close();app.audioSocket?.close();app.audioSocket=null;app.audioStream=null;app.audioContext=null;window.HomeMonitoring?.audioFailed(e);throw e;}
  });
  if(app.audioStream)$("startAudioLive").disabled=true;
  }finally{app.audioStarting=false;}
}
async function stopAudioLive(){
  if(app.audioStopping)return app.audioStopping;
  app.audioStopping=(async()=>{
    const socket=app.audioSocket;app.audioSocket=null;
    $("startAudioLive").disabled=true;$("stopAudioLive").disabled=true;
    app.audioStream?.getTracks().forEach(t=>t.stop());app.audioStream=null;
    if(app.audioContext&&app.audioContext.state!=="closed")await app.audioContext.close();
    app.audioContext=null;
    if(socket?.readyState===WebSocket.OPEN){
      $("audioLiveStatus").textContent="Stopping listening";
      let timer;
      await Promise.race([app.audioQueue?.drain(),new Promise(resolve=>{timer=setTimeout(()=>{socket.close();resolve();},10000);})]);
      clearTimeout(timer);
    }
    if(socket?.readyState===WebSocket.OPEN){
      await new Promise(resolve=>{
        const timer=setTimeout(()=>{socket.close();resolve();},18000);
        const done=()=>{clearTimeout(timer);resolve();};
        socket.onclose=done;socket.onmessage=event=>{if(JSON.parse(event.data).state==="stopped"){socket.close();done();}};
        socket.send(JSON.stringify({type:"stop"}));
      });
    }else socket?.close();
    app.audioQueue?.close();app.audioQueue=null;$("audioLevel").style.width="0%";
    $("audioLiveStatus").textContent="Microphone not connected";$("startAudioLive").disabled=false;
    for(const id of ["soundLiveState","helpLiveState"]){$(id).textContent="Not listening";$(id).className="badge neutral";}
  })();
  try{await app.audioStopping;}finally{app.audioStopping=null;}
}

async function loadReport(overview=false){try{const report=await api(`/api/reports?days=${overview?7:app.reportDays}&mode=${overview?"live":app.reportMode}`);if(overview){app.report=report;if(app.page==="overview")drawBars($("overviewChart"),report.series,"reps","#579579");return;}app.report=report;renderReport();}catch(e){toast(e.message,true);}}
function renderReport(){const r=app.report;if(!r)return;const s=r.summary;$("reportRange").textContent=`${r.series[0].date} to ${r.series.at(-1).date} · ${r.mode==="live"?"Live records":"Sample / replay"}`;$("reportMetrics").innerHTML=metric("Confirmed fall",s.confirmed_falls,"reps","shield-alert",`${s.false_alarms} false alarms`)+metric("Unresolved events",s.open_events,"","bell-ring",`${s.unresolved_outcomes} results undetermined`)+metric("Completed sessions",s.sessions,"reps","activity",`${s.participation_days} days participated`)+metric("Valid exercise reps",s.reps,"reps","check-check","Total completed sessions")+metric("Voice answered",s.voice_answered,"","speech","The inquiry got a valid answer")+metric("High risk, no answer",s.voice_high_risk,"","shield-alert","The inquiry got no valid answer",s.voice_high_risk?"metric-risk":"")+metric("Not asked",s.voice_not_asked,"","circle-help","Not started or in progress");renderReportRecords(r);drawBars($("safetyChart"),r.series,"events","#bc8d58");drawBars($("exerciseReportChart"),r.series,"reps","#5d83bb");$("coverageSummary").innerHTML=r.mode==="live"?Object.entries(s.coverage_unavailable_s).map(([ch,n])=>`<div><span class="muted small">${channels[ch]} limited / unavailable time</span><strong>${duration(n)}</strong></div>`).join(""):"";$("coverageHistory").innerHTML=r.coverage.length?`<table><thead><tr><th>Channel</th><th>Start</th><th>End</th><th>Status</th><th>Reason</th></tr></thead><tbody>${r.coverage.slice(0,30).map(c=>`<tr><td>${channels[c.channel]}</td><td>${dateTime(c.started_at)}</td><td>${c.ended_at?dateTime(c.ended_at):"Ongoing"}</td><td><span class="badge ${c.state==="active"?"green":"amber"}">${c.state==="active"?"Valid":c.state==="reduced"?"Limited":"Unavailable"}</span></td><td>${esc(c.reason)}</td></tr>`).join("")}</tbody></table>`:empty(r.mode==="replay"?"Replays are not counted in live monitoring coverage":"No coverage records","radio","",true);$("exerciseSummary").innerHTML=`<div class="exercise-summary-grid">${Object.entries(s.reps_by_exercise).map(([key,n])=>`<div>${names[key]}<strong>${n}<small>reps</small></strong></div>`).join("")}</div>`;$("exportCsv").href=`/api/reports/export.csv?days=${app.reportDays}&mode=${app.reportMode}`;$("printReport").href=`/reports/print?days=${app.reportDays}&mode=${app.reportMode}`;icons();}
/* 危险操作确认框：样式与工作台一致，避免浏览器原生弹窗打断视觉。 */
function askConfirm(title,message,accept,cancel){
  return new Promise(resolve=>{
    const mask=document.createElement("div");
    mask.className="confirm-mask no-print";
    mask.innerHTML='<div class="confirm-box" role="dialog" aria-modal="true"><h3></h3><p></p><div class="confirm-actions"></div></div>';
    mask.querySelector("h3").textContent=title;
    mask.querySelector("p").textContent=message;
    const actions=mask.querySelector(".confirm-actions");
    const close=value=>{mask.remove();resolve(value);};
    if(cancel){
      const no=document.createElement("button");
      no.type="button";no.className="button secondary";no.textContent=cancel;
      no.addEventListener("click",()=>close(false));
      actions.append(no);
    }
    const yes=document.createElement("button");
    yes.type="button";yes.className=cancel?"button danger":"button primary";yes.textContent=accept;
    yes.addEventListener("click",()=>close(true));
    actions.append(yes);
    mask.addEventListener("click",event=>{if(event.target===mask)close(false);});
    document.body.append(mask);
    yes.focus();
  });
}
function recordSummary(result){return `Safety events ${result.events||0}, exercise sessions ${result.sessions||0}, notifications ${result.notifications||0}, video clips ${result.clips||0}`;}

/* 报告记录：逐条勾选删除，或清空整个报告范围。删除范围与报告范围共用同一时间窗口。 */
function reportRows(){return Array.from(document.querySelectorAll("#reportRecords .row-pick"));}
function reportPicked(){return reportRows().filter(row=>row.checked);}
function syncReportPicker(){
  const master=$("selectAllRecords"),hint=$("recordHint"),remove=$("deleteSelectedRecords");
  if(!master)return;
  const all=reportRows(),picked=all.filter(row=>row.checked);
  master.checked=all.length>0&&picked.length===all.length;
  master.indeterminate=picked.length>0&&picked.length<all.length;
  document.querySelectorAll("#reportRecords .pick-all").forEach(box=>{
    const group=box.dataset.scope==="all"?all:all.filter(row=>row.dataset.kind===box.dataset.scope);
    box.checked=group.length>0&&group.every(row=>row.checked);
    box.indeterminate=group.some(row=>row.checked)&&!group.every(row=>row.checked);
  });
  const events=picked.filter(row=>row.dataset.kind==="event").length;
  const sessions=picked.filter(row=>row.dataset.kind==="session").length;
  hint.textContent=picked.length?`Selected ${picked.length} (safety events ${events} · Exercise ${sessions})`:"No records selected";
  remove.disabled=picked.length===0;
}
function renderReportRecords(report){
  const events=report.events||[],sessions=report.sessions||[];
  const eventRows=events.map(e=>`<tr><td class="pick"><input type="checkbox" class="row-pick" data-kind="event" value="${esc(e.id)}" aria-label="Select safety events ${esc(e.id)}"></td><td><span class="badge neutral">Safety events</span></td><td>#${esc(e.id)}</td><td>${dateTime(e.created_at)}</td><td>${modeBadge(e.mode)}</td><td>${channels[e.channel]||esc(e.channel)} · ${e.level==="EMERGENCY"?"Emergency call for help":"Pending verification"}</td><td>${voiceBadge(e.voice)}</td><td>${({open:"Open",resolved:"Resolved"})[e.state]||esc(e.state)}</td></tr>`);
  const sessionRows=sessions.map(s=>`<tr><td class="pick"><input type="checkbox" class="row-pick" data-kind="session" value="${esc(s.session_id)}" aria-label="Select exercise sessions"></td><td><span class="badge blue">Exercise</span></td><td>${esc(String(s.session_id).slice(0,8))}</td><td>${dateTime(s.created_at)}</td><td>${modeBadge(s.mode)}</td><td>${esc(names[s.exercise]||s.exercise)} · ${s.count} / ${s.target_reps} reps</td><td><span class="muted small">Not applicable</span></td><td>${({running:"In progress",paused:"Paused",ended:"Ended",interrupted:"Interrupted"})[s.status]||esc(s.status)}</td></tr>`);
  const total=eventRows.length+sessionRows.length;
  $("reportRecords").innerHTML=total?`<table><thead><tr><th class="pick"><input type="checkbox" class="pick-all" data-scope="all" aria-label="Select all report records"></th><th>Type</th><th>Record ID</th><th>Time</th><th>Source</th><th>Level / action</th><th>Voice inquiry</th><th>Status</th></tr></thead><tbody>${eventRows.join("")}${sessionRows.join("")}</tbody></table>`:empty("No records in the current report range","inbox","",true);
  syncReportPicker();
  icons();
}
async function deleteSelectedRecords(){
  const picked=reportPicked();
  if(!picked.length)return;
  const confirmed=await askConfirm("Delete selected records",`This will permanently delete the selected ${picked.length} records, together with the notifications and video clips created with the events. This cannot be undone and the report statistics will change.`,"Delete permanently","Cancel");
  if(!confirmed)return;
  const button=$("deleteSelectedRecords");
  button.disabled=true;
  try{
    const result=await api("/api/records/delete",{method:"POST",body:{event_ids:picked.filter(row=>row.dataset.kind==="event").map(row=>row.value),session_ids:picked.filter(row=>row.dataset.kind==="session").map(row=>row.value)}});
    await askConfirm("Delete finished",`Deleted: ${recordSummary(result)}. The report has been recalculated from the latest records.`,"OK",null);
    await loadReport();await refresh();
  }catch(e){
    button.disabled=false;
    await askConfirm("Delete failed",e.message,"OK",null);
  }
}
async function clearReportScope(){
  const confirmed=await askConfirm("Clear the report range",`This will permanently delete everything in the current report range (${app.reportMode==="live"?"Live records":"Samples and replays"} · last ${app.reportDays} days) including notifications and video clips. This cannot be undone.`,"Clear and delete","Cancel");
  if(!confirmed)return;
  const button=$("clearReportScope");
  button.disabled=true;
  try{
    const result=await api("/api/records/clear",{method:"POST",body:{mode:app.reportMode,days:app.reportDays,events:true,sessions:true,confirm:"DELETE"}});
    await askConfirm("Clear finished",`Cleared: ${recordSummary(result)}.`,"OK",null);
    await loadReport();await refresh();
  }catch(e){
    button.disabled=false;
    await askConfirm("Clear failed",e.message,"OK",null);
  }
}

function contactEditor(c={name:"",relation:"Family",phone:""}){const div=document.createElement("div");div.className="contact-editor";div.innerHTML=`<label>Name<input class="contact-name" value="${esc(c.name)}" maxlength="60" required></label><label>Relationship<input class="contact-relation" value="${esc(c.relation)}" maxlength="30"></label><label>Phone<input class="contact-phone" type="tel" value="${esc(c.phone)}" maxlength="40" pattern="[+0-9 ()-]+" required></label><button class="icon-button remove-contact" type="button" title="Remove contact" aria-label="Remove contact">${icon("trash-2")}</button>`;$("contactsEditor").append(div);icons();}
function fillSettings(s){app.settings=s;$("residentName").value=s.resident.name;$("residentRoom").value=s.resident.room;$("residentNotes").value=s.resident.notes;$("audioConsent").checked=s.audio_consent;$("escalationSeconds").value=s.escalation_seconds;$("defaultTarget").value=s.exercise_target;$("targetReps").value=s.exercise_target;$("webhookUrl").value=s.webhook_url;$("contactsEditor").innerHTML="";s.contacts.forEach(contactEditor);$("consentStatusText").textContent=s.audio_consent?"Authorized":"Not authorized";$("notificationTestStatus").textContent=s.notification_test_at?`Last test ${dateTime(s.notification_test_at)}`:"Not tested";$("setupBanner").hidden=!!(s.resident.name&&s.contacts.length);$("overviewSubtitle").textContent=s.resident.name?`${s.resident.name} · Today's safety and activity record`:"Today's safety and activity record";$("overviewRoom").textContent=s.resident.room.toUpperCase()+" / HOME CARE";$("cameraRoomLabel").textContent=s.resident.room;}
async function saveSettings(ev){ev.preventDefault();await busy($("saveSettings"),async()=>{const contacts=Array.from(document.querySelectorAll(".contact-editor")).map(el=>({name:el.querySelector(".contact-name").value.trim(),relation:el.querySelector(".contact-relation").value.trim(),phone:el.querySelector(".contact-phone").value.trim()}));const value={resident:{name:$("residentName").value.trim(),room:$("residentRoom").value.trim(),notes:$("residentNotes").value.trim()},contacts,audio_consent:$("audioConsent").checked,exercise_target:Number($("defaultTarget").value),escalation_seconds:Number($("escalationSeconds").value),webhook_url:$("webhookUrl").value.trim()};if(!value.audio_consent)await stopAudioLive();const result=await api("/api/settings",{method:"PUT",body:value});fillSettings(result);toast("Settings saved");await refresh();});}
function renderModules(){$("moduleList").innerHTML=app.status.modules.map(m=>`<div class="module-row"><div><h3>${moduleNames[m.id]}</h3><small>${modelNames[m.id]}</small>${m.error?`<small style="color:var(--red)">${esc(m.error)}</small>`:""}</div><span class="badge ${m.state==="ready"?"green":m.state==="unavailable"?"red":"amber"}">${stateNames[m.state]||m.state}</span><button class="icon-button" type="button" data-init="${m.id}" title="Initialize the module again" aria-label="Initialize again ${moduleNames[m.id]}" ${m.state==="loading"?"disabled":""}>${icon("rotate-cw")}</button></div>`).join("");icons();}
async function loadDiagnostics(){try{const d=await api("/api/diagnostics");$("diagnosticsList").innerHTML=d.logs.length?d.logs.slice(0,30).map(x=>`<div class="diagnostic-item"><time>${dateTime(x.at)}</time><span>${esc(x.module)} · ${esc(x.action)}${x.payload.error?`<br>${esc(x.payload.error)}`:""}</span></div>`).join(""):empty("No run records","list", "",true);}catch(e){toast(e.message,true);}}

/* ---------------------------------------------------------------------------
   后台算法开关。开关只决定「原算法要不要跑」，模型、阈值与判定保持原样：
   - 后台常驻：服务在跑就一直跑，不依赖任何页面是否打开；
   - 视频跌倒检测：与家庭守护同一件事，两个入口同步；
   - 加速度跌倒检测：数据来自手机端 /sensor，后台只做接收与调度；
   - 四种运动：分别开关，关掉后既不再自动开始，也不再接受新的训练请求。
--------------------------------------------------------------------------- */
const algorithmCopy = {
  resident:{title:"Always on in background",note:"While the service runs, switched-on algorithms start by themselves; closing the page or locking the screen makes no difference"},
  vision:{title:"Video fall detection",note:"Runs continuously on the local camera; same switch as home monitoring"},
  accel:{title:"Accelerometer fall detection",note:"Receives the phone's three-axis stream; data stays on this machine"}
};
const residentStates={off:"Off",starting:"Starting",running:"Standing by",exited:"Stopped"};
function algorithmToggles(){
  return [["resident",$("algorithmResident")],["vision",$("algorithmVision")],["accel",$("algorithmAccel")]]
    .concat(Object.keys(names).map(key=>[key,document.querySelector(`[data-algorithm="${key}"]`)]))
    .filter(([,input])=>input);
}
/* 三个安全开关在模板里是静态节点，不像四种动作那样由 renderExerciseSwitches 现造，
   所以事件只在这里绑一次；重复绑定会让一次点击发两次请求。 */
function bindAlgorithmToggles(){
  for(const [key,input] of algorithmToggles()){
    if(input.dataset.bound)continue;
    input.dataset.bound="1";
    input.onchange=()=>setAlgorithm(key,input.checked);
  }
}
/* 「后台算法」页里四种动作的开关。单独成函数，初始化时先渲染一次，
   之后 refresh() 只改勾选状态，不会重复生成 DOM。 */
function renderExerciseSwitches(){
  const box=$("exerciseSwitches");
  if(!box||box.children.length)return;
  box.innerHTML=Object.keys(names).map(key=>{
    const guide=exerciseGuides[key]||{};
    return `<label class="toggle-row switch-card" data-switch="${key}"><span><strong>${esc(names[key])}</strong><small>${esc(guide.count||"")}</small></span><input type="checkbox" class="switch" role="switch" data-algorithm="${key}" aria-label="${esc(names[key])} detection switch"></label>`;
  }).join("");
  box.querySelectorAll("[data-algorithm]").forEach(input=>{
    input.onchange=()=>setAlgorithm(input.dataset.algorithm,input.checked);
  });
}
async function loadAlgorithms(){
  try{
    const [state,plan]=await Promise.all([api("/api/algorithms"),api("/api/plan")]);
    app.algorithms=state;app.plan=plan;renderAlgorithms();renderPlan(plan);renderPlanProgress(plan.progress);
  }catch(e){toast(e.message,true);}
}
function renderAlgorithms(){
  const state=app.algorithms;if(!state||!$("algorithmResident"))return;
  const effective=!!state.switches.resident;
  for(const [key,input] of algorithmToggles())input.checked=!!state.switches[key];
  document.querySelectorAll("#page-algorithms [data-switch]").forEach(card=>{
    const key=card.dataset.switch;
    card.classList.toggle("is-off",!state.switches[key]);
    card.classList.toggle("is-muted",!effective&&key!=="resident");
  });
  $("algorithmResident").disabled=!state.resident_ready;
  const channels=state.channels||[];
  const running=channels.filter(c=>c.active).length;
  $("algorithmsSummary").textContent=effective?`Standing by · ${running} running`:"Off";
  $("algorithmsSummary").className=`badge ${effective?running?"green":"amber":"neutral"}`;
  $("algorithmsBannerTitle").textContent=effective?"Background mode is on":"Background mode is off";
  $("algorithmsBannerText").textContent=effective
    ? (state.blockers[0]||`${running} background algorithms are running, with no page needed`)
    : "Once on, video, accelerometer and movement detection run on their own in the background";
  $("algorithmsBanner").classList.toggle("is-active",effective&&!state.blockers.length);
  $("algorithmChannels").innerHTML=channels.map(row=>{
    const on=!!row.active;
    return `<div class="channel-status-row"><span class="dot ${on?"green":"amber"}"></span><div><strong>${esc(names[row.key]||algorithmCopy[row.key]?.title||row.key)}</strong><small>${esc(row.detail||"")}</small></div><span class="badge ${on?"green":"neutral"}">${on?"Running":"Not running"}</span></div>`;
  }).join("");
  $("algorithmBlockers").innerHTML=(state.blockers||[]).map(text=>`<p><i data-lucide="circle-alert"></i><span>${esc(text)}</span></p>`).join("");
  icons();
}
let algorithmBusy=false;
async function setAlgorithm(key,value){
  if(algorithmBusy)return;
  algorithmBusy=true;
  const input=document.querySelector(`[data-algorithm="${key}"]`);
  if(input)input.disabled=true;
  try{
    app.algorithms=await api("/api/algorithms",{method:"PUT",body:{[key]:value}});
    renderAlgorithms();
    toast(`${algorithmCopy[key]?.title||names[key]} switched ${value?"on":"off"}`);
    await refresh();
    if(app.page==="algorithms")renderAlgorithms();
  }catch(e){
    if(input)input.checked=!value;
    toast(e.message,true);
  }finally{
    if(input)input.disabled=false;
    algorithmBusy=false;
  }
}

/* 运动计划：照护者设定，长者端展示并逐项完成。 */
let planDraft=null;
function planItemRow(item,index){
  const key=item.exercise,guide=exerciseGuides[key];
  return `<div class="plan-item" data-plan-item="${key}">
    <span class="plan-order">${index+1}</span>
    <img src="/static/media/${key}-guide.png" alt="${esc(guide.title)} movement guide">
    <div class="plan-item-copy"><strong>${esc(names[key])}</strong><small>${esc(guide.count)}</small></div>
    <label class="plan-count">reps<input type="number" min="1" max="10000" value="${item.target_reps}" data-plan-reps="${key}"></label>
    <label class="plan-include"><input type="checkbox" data-plan-enabled="${key}" ${item.enabled?"checked":""}><span>Include</span></label>
  </div>`;
}
function renderPlan(plan){
  if(!plan||!$("planEnabled"))return;
  const draft=planDraft||plan;
  $("planEnabled").checked=!!draft.enabled;
  $("planTitle").value=draft.title||"";
  $("planNote").value=draft.note||"";
  $("planTarget").value=draft.target_reps||10;
  const byKey=Object.fromEntries((draft.items||[]).map(item=>[item.exercise,item]));
  const items=Object.keys(names).map((key,index)=>byKey[key]||{exercise:key,target_reps:draft.target_reps||10,enabled:true});
  $("planItems").innerHTML=items.map(planItemRow).join("");
  icons();
}
function draftPlan(){
  const items=Array.from(document.querySelectorAll("[data-plan-item]")).map(row=>{
    const key=row.dataset.planItem;
    const reps=row.querySelector("[data-plan-reps]");
    return {exercise:key,target_reps:Math.min(10000,Math.max(1,Number(reps.value)||1)),enabled:row.querySelector("[data-plan-enabled]").checked};
  });
  planDraft={enabled:$("planEnabled").checked,title:$("planTitle").value.trim()||"Today's movement plan",note:$("planNote").value.trim(),target_reps:Math.min(10000,Math.max(1,Number($("planTarget").value)||10)),items};
  return planDraft;
}
function renderPlanProgress(progress){
  if(!progress||!$("planProgress"))return;
  $("planProgressDate").textContent=progress.date||"—";
  const rows=(progress.items||[]).filter(item=>item.enabled);
  $("planProgress").innerHTML=rows.length?rows.map(item=>{
    const percent=Math.min(100,Math.round(100*(item.reps||0)/Math.max(1,item.target_reps)));
    return `<div class="plan-progress-row ${item.done?"is-done":""}"><div><strong>${esc(names[item.exercise])}</strong><small>${item.done?"Done today":`${item.reps||0} / ${item.target_reps} reps · ${item.sessions||0} sessions`}${item.disabled?" · Detection is off":""}</small></div><div class="plan-progress-bar"><span style="width:${percent}%"></span></div><span class="badge ${item.done?"green":"neutral"}">${item.done?"Completed":"To do"}</span></div>`;
  }).join(""):empty("No items yet","clipboard-list","Tick the movements to include under Plan items",true);
}
async function savePlan(){
  const value=draftPlan();
  await busy($("savePlan"),async()=>{
    const result=await api("/api/plan",{method:"PUT",body:value});
    planDraft=null;app.plan=result;renderPlan(result);renderPlanProgress(result.progress);
    toast("Plan saved; the older adult sees it right away");
  });
}

function canvasContext(canvas){if(!canvas)return null;const rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return null;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);return {ctx,w:rect.width,h:rect.height};}
function chartBase(canvas,max,labels=[]){const setup=canvasContext(canvas);if(!setup)return null;const {ctx,w,h}=setup;const area={l:35,r:w-10,t:13,b:h-29};ctx.clearRect(0,0,w,h);ctx.font='10px "Segoe UI","Microsoft YaHei",sans-serif';ctx.textBaseline="middle";for(let i=0;i<4;i++){const y=area.t+(area.b-area.t)*i/3;ctx.strokeStyle="#e4ebe6";ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(area.l,y);ctx.lineTo(area.r,y);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle="#9ba99f";ctx.textAlign="right";const val=max*(1-i/3);ctx.fillText(max<=1?val.toFixed(1):String(Math.round(val)),area.l-9,y);}ctx.textAlign="center";if(labels.length){const step=Math.max(1,Math.ceil(labels.length/7));labels.forEach((label,i)=>{if(i%step===0||i===labels.length-1){const x=area.l+(area.r-area.l)*(i+.5)/labels.length;ctx.fillStyle="#98a89c";ctx.fillText(label,x,h-11);}});}return {...setup,area};}
function drawBars(canvas,series,key,color){const max=Math.max(3,...series.map(x=>x[key]||0));const setup=chartBase(canvas,max,series.map(x=>x.date.slice(5).replace("-","/")));if(!setup)return;const {ctx,area}=setup;const slot=(area.r-area.l)/series.length;series.forEach((s,i)=>{const n=s[key]||0;const bar=Math.max(2,n/max*(area.b-area.t));const width=Math.min(27,slot*.44);const x=area.l+slot*(i+.5)-width/2;ctx.fillStyle=n?color:"#e6ece8";ctx.beginPath();ctx.roundRect(x,area.b-bar,width,bar,[3,3,0,0]);ctx.fill();if(n){ctx.fillStyle=color;ctx.font="10px Segoe UI";ctx.fillText(String(n),x+width/2,area.b-bar-9);}});}
function drawLines(canvas,series,options={}){const all=series.flatMap(s=>s.values||[]).filter(Number.isFinite);let min=options.min??Math.min(0,...all),max=options.max??Math.max(1,...all);if(max===min)max=min+1;const setup=chartBase(canvas,max,options.labels||[]);if(!setup)return;const {ctx,area}=setup;const y=v=>area.b-(v-min)/(max-min)*(area.b-area.t);if(options.threshold!==undefined){ctx.strokeStyle="#c39c68";ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(area.l,y(options.threshold));ctx.lineTo(area.r,y(options.threshold));ctx.stroke();ctx.setLineDash([]);}for(const s of series){if(!s.values?.length)continue;ctx.strokeStyle=s.color;ctx.lineWidth=1.7;ctx.beginPath();s.values.forEach((v,i)=>{const x=area.l+(area.r-area.l)*i/Math.max(1,s.values.length-1);if(i===0)ctx.moveTo(x,y(v));else ctx.lineTo(x,y(v));});ctx.stroke();}if(!all.length){ctx.textAlign="center";ctx.fillStyle="#9aaba0";ctx.font='11px "Microsoft YaHei"';ctx.fillText("Waiting for data",(area.l+area.r)/2,(area.t+area.b)/2);}}
function drawPose(canvas,points,background=true){const setup=canvasContext(canvas);if(!setup)return;const {ctx,w,h}=setup;ctx.clearRect(0,0,w,h);if(background){ctx.fillStyle="#eaf1ed";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#dbe6de";ctx.lineWidth=1;for(let x=0;x<w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=0;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}}if(!points)return;const visible=points.filter(p=>p[2]>=.3);if(!visible.length)return;let minX,maxX,minY,maxY;if(background){minX=Math.min(...visible.map(p=>p[0]))-80;maxX=Math.max(...visible.map(p=>p[0]))+80;minY=Math.min(...visible.map(p=>p[1]))-60;maxY=Math.max(...visible.map(p=>p[1]))+60;}else{minX=0;maxX=app.exercise?.frame_shape?.[1]||640;minY=0;maxY=app.exercise?.frame_shape?.[0]||480;}const scale=Math.min(w/(maxX-minX),h/(maxY-minY));const dx=(w-(maxX-minX)*scale)/2-minX*scale,dy=(h-(maxY-minY)*scale)/2-minY*scale;const point=p=>[p[0]*scale+dx,p[1]*scale+dy];const edges=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];ctx.lineWidth=background?5:3;ctx.lineCap="round";for(const [a,b] of edges){if(points[a]?.[2]<.3||points[b]?.[2]<.3)continue;ctx.strokeStyle=a%2?"#338567":"#5a84ba";ctx.beginPath();ctx.moveTo(...point(points[a]));ctx.lineTo(...point(points[b]));ctx.stroke();}for(const p of visible){ctx.fillStyle="#f9fcfa";ctx.strokeStyle="#4b876f";ctx.lineWidth=2;ctx.beginPath();ctx.arc(...point(p),background?5:3.5,0,Math.PI*2);ctx.fill();ctx.stroke();}}

document.addEventListener("DOMContentLoaded",async()=>{
  icons();$("headerDate").textContent=new Date().toLocaleDateString("en-US",{month:"long",day:"numeric",weekday:"long"});
  $("menuButton").onclick=()=>{$("sidebar").classList.toggle("open");$("navBackdrop").hidden=!$("sidebar").classList.contains("open");};$("navBackdrop").onclick=()=>{$("sidebar").classList.remove("open");$("navBackdrop").hidden=true;};
  window.addEventListener("hashchange",navigate);document.querySelectorAll("[data-close]").forEach(b=>b.onclick=()=>$(b.dataset.close).close());document.querySelectorAll("dialog").forEach(d=>d.addEventListener("click",e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}}));
  document.addEventListener("click",async ev=>{const eventEl=ev.target.closest("[data-event]");if(eventEl)openEvent(eventEl.dataset.event);const action=ev.target.closest("[data-action]");if(action)busy(action,()=>eventAction(action.dataset.action));const contact=ev.target.closest("[data-contact-call]");if(contact&&app.selectedEvent?.state!=="resolved")api(`/api/events/${app.selectedEvent.id}/action`,{method:"POST",body:{action:"contact",note:"A phone call was started"}}).catch(e=>toast(e.message,true));const init=ev.target.closest("[data-init]");if(init)busy(init,async()=>{await api(`/api/modules/${init.dataset.init}/initialize`,{method:"POST"});toast("The module is being initialized again");await refresh();});const remove=ev.target.closest(".remove-contact");if(remove)remove.closest(".contact-editor").remove();});document.addEventListener("keydown",ev=>{if(["Enter"," "].includes(ev.key)&&ev.target.matches("[data-event]")){ev.preventDefault();openEvent(ev.target.dataset.event);}});
  $("notificationButton").onclick=async()=>{await refresh();$("notificationList").innerHTML=app.notices.length?app.notices.slice(0,30).map(n=>`<div class="notification-row" data-notice="${n.id}" ${n.event_id?`data-event="${n.event_id}"`:""}><strong>${esc(n.title)}</strong><p>${esc(n.message)}</p><footer><span>${dateTime(n.created_at)}</span><span>${n.kind==="webhook"?"Webhook":"In-app"} · ${({queued:"Queued",delivered:"Delivered",seen:"Seen",retry:"Retrying",failed:"Failed"})[n.status]||n.status}</span></footer></div>`).join(""):empty("No notifications","bell");$("notificationsDialog").showModal();$("notificationList").querySelectorAll("[data-notice]").forEach(el=>el.addEventListener("click",()=>{api(`/api/notifications/${el.dataset.notice}/seen`,{method:"POST"});$("notificationsDialog").close();}));icons();};
  $("videoSource").onchange=()=>{$("cameraIndexField").hidden=$("videoSource").value!=="camera";$("videoUploadField").hidden=$("videoSource").value!=="upload";};$("videoForm").onsubmit=ev=>{ev.preventDefault();busy($("startVideo"),async()=>{let upload_id=null;if($("videoSource").value==="upload"){const file=$("videoUpload").files[0];if(!file)throw new Error("Please choose a video file");const form=new FormData();form.append("file",file);$("videoUploadStatus").textContent="Uploading video…";try{const result=await api("/api/uploads/video",{method:"POST",body:form});upload_id=result.upload_id;$("videoUploadStatus").textContent=result.filename;}catch(e){$("videoUploadStatus").textContent="Upload failed";throw e;}}app.videoHistory=[];await api("/api/video/start",{method:"POST",body:{source:$("videoSource").value,camera_index:Number($("videoForm").elements.camera_index.value),upload_id}});await refresh();toast("Detection started");});};$("stopVideo").onclick=()=>busy($("stopVideo"),async()=>{await api("/api/video/stop",{method:"POST"});await refresh();});
  restoreOverlay();$("toggleBoxes").onclick=()=>setOverlay("box",!app.overlay.box);$("toggleSkeleton").onclick=()=>setOverlay("skeleton",!app.overlay.skeleton);setOverlay("box",app.overlay.box);setOverlay("skeleton",app.overlay.skeleton);
  $("signalSource").onchange=()=>$("signalUploadField").hidden=$("signalSource").value!=="upload";$("analyzeSignal").onclick=()=>busy($("analyzeSignal"),async()=>{let result;if($("signalSource").value==="sample")result=await api("/api/signal/sample",{method:"POST"});else{const file=$("signalFile").files[0];if(!file)throw new Error("Please choose a CSV file");const form=new FormData();form.append("file",file);result=await api("/api/signal/analyze",{method:"POST",body:form});}drawLines($("signalChart"),[{values:result.waveform.ax,color:"#547bb4"},{values:result.waveform.ay,color:"#5a9877"},{values:result.waveform.az,color:"#bf9459"}],{labels:result.waveform.t.map(t=>`${t.toFixed(1)}s`)});drawLines($("signalProbabilityChart"),[{values:result.info.probs,color:"#52886d"}],{min:0,max:1,threshold:.5,labels:result.info.t_end.map(t=>`${t.toFixed(1)}s`)});$("signalState").textContent="Replay analysis finished";$("signalResult").innerHTML=`${result.info.n_windows} windows · ${result.records.length} suspected fall events ${result.records.map(e=>`<button class="text-action" data-event="${e.id}">View #${e.id.slice(0,6)} ${icon("arrow-right")}</button>`).join(" ")}`;await refresh();icons();});
  $("exerciseOptions").innerHTML=Object.entries(names).map(([key,name],index)=>`<button type="button" class="exercise-option ${index===0?"selected":""}" data-exercise="${key}" aria-pressed="${index===0}" data-switch-card="${key}"><img src="/static/media/${key}-pose.png" alt="${name} skeleton movement"><span class="selection-mark">${icon("check")}</span><span class="exercise-off-badge" hidden>detection off</span><div><strong>${esc(exerciseGuides[key].title)}</strong><small>${["Complete sit / stand cycle","Complete squat / stand-up cycle","Complete leg raise / lowering cycle","Complete step / return cycle"][index]}</small></div></button>`).join("");document.querySelectorAll("[data-exercise]").forEach(b=>b.onclick=()=>chooseExercise(b.dataset.exercise));$("exerciseSource").onchange=()=>{$("exerciseUploadField").hidden=$("exerciseSource").value!=="upload";$("exerciseCameraField").hidden=$("exerciseSource").value!=="camera";};$("exerciseForm").onsubmit=startExercise;
  // 长者端点计划里某一项进来时，直接选中那个动作，目标次数在开始训练时套用计划值。
  const fromPlan=household?sessionStorage.getItem("homeease.plan"):null;
  if(fromPlan&&names[fromPlan])sessionStorage.removeItem("homeease.plan");
  chooseExercise(names[fromPlan]?fromPlan:app.selectedExercise);
  if(household&&!app.plan)api("/api/plan").then(plan=>{app.plan=plan;window.HomeMonitoring?.render();}).catch(()=>{});
  $("pauseExercise").onclick=async()=>{await busy($("pauseExercise"),()=>controlExercise(app.exercise?.status==="paused"?"resume":"pause"));syncExerciseControls();};$("endExercise").onclick=async()=>{await busy($("endExercise"),()=>controlExercise("end"));syncExerciseControls();};$("newExercise").onclick=()=>{app.sid=null;app.exercise=null;$("exerciseActive").hidden=true;$("exerciseChooser").hidden=false;};
  $("audioSource").onchange=()=>{const source=$("audioSource").value;$("audioSampleField").hidden=source!=="sample";$("audioFileField").hidden=source!=="upload";$("audioRecordField").hidden=source!=="record";$("audioPreview").removeAttribute("src");if(source==="sample")$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;};$("audioSampleSelect").onchange=()=>$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;$("audioFile").onchange=()=>{const file=$("audioFile").files[0];if(file)$("audioPreview").src=URL.createObjectURL(file);};$("audioForm").onsubmit=analyzeAudio;$("recordAudio").onclick=recordAudio;$("startAudioLive").onclick=startAudioLive;$("stopAudioLive").onclick=stopAudioLive;
  $("eventStateFilter").querySelectorAll("button").forEach(b=>b.onclick=()=>{app.eventFilter=b.dataset.value;$("eventStateFilter").querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));renderEvents();});$("eventModeFilter").onchange=()=>{app.eventMode=$("eventModeFilter").value;renderEvents();};$("eventSearch").oninput=renderEvents;$("reportDays").querySelectorAll("button").forEach(b=>b.onclick=()=>{app.reportDays=Number(b.dataset.value);$("reportDays").querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));loadReport();});$("reportMode").onchange=()=>{app.reportMode=$("reportMode").value;loadReport();};
  $("selectAllRecords").onchange=()=>{const checked=$("selectAllRecords").checked;reportRows().forEach(row=>{row.checked=checked;});syncReportPicker();};$("reportRecords").addEventListener("change",event=>{const box=event.target.closest(".pick-all");if(box){const all=reportRows();(box.dataset.scope==="all"?all:all.filter(row=>row.dataset.kind===box.dataset.scope)).forEach(row=>{row.checked=box.checked;});}if(box||event.target.closest(".row-pick"))syncReportPicker();});$("deleteSelectedRecords").onclick=deleteSelectedRecords;$("clearReportScope").onclick=clearReportScope;
  $("settingsForm").onsubmit=saveSettings;$("addContact").onclick=()=>{if(document.querySelectorAll(".contact-editor").length>=10){toast("At most 10 contacts can be added",true);return;}contactEditor();};$("audioConsent").onchange=()=>$("consentStatusText").textContent=$("audioConsent").checked?"Consent to be saved":"Withdrawal to be saved";$("testNotification").onclick=()=>busy($("testNotification"),async()=>{await api("/api/notifications/test",{method:"POST"});if("Notification" in window&&Notification.permission==="default")await Notification.requestPermission();toast("The test notification was added to the notification center");app.settings=await api("/api/settings");$("notificationTestStatus").textContent=`Last test ${dateTime(app.settings.notification_test_at)}`;await refresh();});
  if($("planForm"))$("planForm").onsubmit=ev=>{ev.preventDefault();savePlan();};
  if($("savePlan"))$("savePlan").onclick=ev=>{ev.preventDefault();savePlan();};
  renderExerciseSwitches();
  bindAlgorithmToggles();
  if($("planTargetFromDefault"))$("planTargetFromDefault").onclick=()=>{const value=Math.min(10000,Math.max(1,Number($("planTarget").value)||10));document.querySelectorAll("[data-plan-reps]").forEach(input=>{input.value=value;});toast(`Now ${value} reps`);};
  $("page-plan")?.addEventListener("change",()=>{if($("planForm"))draftPlan();});
  try{fillSettings(await api("/api/settings"));const samples=await api("/api/audio/samples");$("audioSampleSelect").innerHTML=[...(samples.scenes||[]),...(samples.replies||[])].map(s=>`<option value="${s.id}">${esc(s.title_en||s.title)}</option>`).join("");$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;}catch(e){toast(e.message,true);}navigate();await refresh();icons();setInterval(refresh,2500);setInterval(pollExercise,600);setInterval(loadPose,500);window.addEventListener("resize",()=>{if(app.page==="overview"&&app.report)drawBars($("overviewChart"),app.report.series,"reps","#579579");if(app.page==="monitor")renderMonitor();if(app.page==="reports")renderReport();if(app.exercise)drawPose($("poseCanvas"),app.exercise.keypoints,app.exercise.mode!=="live");});
  window.addEventListener("pagehide",()=>{app.camera?.getTracks().forEach(t=>t.stop());app.audioStream?.getTracks().forEach(t=>t.stop());app.recordingStream?.getTracks().forEach(t=>t.stop());if(app.sid&&["running","paused"].includes(app.exercise?.status)&&app.exerciseSource==="camera")fetch(`/api/sessions/${app.sid}/control`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:"end"}),keepalive:true});});
});
