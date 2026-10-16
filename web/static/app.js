"use strict";
const $ = (id) => document.getElementById(id);
const esc = (value) => String(value ?? "").replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const icon = (name) => `<i data-lucide="${name}"></i>`;
const icons = () => window.lucide?.createIcons();
const names = {chair_sit_to_stand:"坐站训练",squat:"深蹲训练",standing_leg_raise:"站立抬腿",hurdle_step:"跨步训练"};
const exerciseGuides = {
  chair_sit_to_stand: {
    title:"椅子坐站", endpoints:["坐稳","站直","再坐稳 +1"], cycle:["坐稳","起身","站直","坐下","坐稳 +1"],
    count:"坐稳 → 站起并站直 → 再坐稳，计 1 次。只站起来还不计数。",
    steps:["使用稳固椅子，先坐在椅子上，双脚平放，保持约 2 秒。", "身体稍前倾后站起，伸直髋部和膝盖，站稳片刻。", "缓慢屈膝，臀部重新坐到椅子上，坐稳后再开始下一次。"],
    note:"这是椅子坐站，需要实际坐到椅子上。椅子侧放，椅背和桌子不要挡住双髋、双膝和双踝。",
    cues:["请先坐在椅子上，双脚平放，坐稳约 2 秒。", "坐姿已确认，请从椅子上站起。", "正在起身，请站直并稳住片刻。", "站立已确认，请缓慢坐回椅子。", "请坐稳，回到坐姿后计 1 次。"]
  },
  squat: {
    title:"深蹲", endpoints:["站稳","下蹲","再站稳 +1"], cycle:["站稳","下蹲","蹲下","起身","站稳 +1"],
    count:"站稳 → 下蹲 → 起身站稳，计 1 次。停在蹲姿还不计数。",
    steps:["双脚约与肩同宽，先自然站稳约 2 秒。", "髋部向后，屈膝下蹲，在舒适的最低位置稍停。", "起身伸直髋部和膝盖，回到站姿，站稳后再开始下一次。"],
    note:"下蹲和起身都留在原地，双脚全程入镜。幅度以舒适、稳定为准。",
    cues:["请先自然站立，站稳约 2 秒。", "站姿已确认，请开始下蹲。", "正在下蹲，请到舒适的最低位置稍停。", "蹲姿已确认，请起身。", "请站直并站稳，回到站姿后计 1 次。"]
  },
  standing_leg_raise: {
    animation:true,
    title:"站立直腿前抬", endpoints:["站稳","直腿前抬","落脚站稳 +1"], cycle:["站稳","前抬","抬起","放下","站稳 +1"],
    count:"站稳 → 单腿伸直向前抬起 → 放下并站稳，计 1 次。每条腿单独完成一次算 1 次。",
    steps:["双臂向两侧抬至约肩高并屈肘，双手向上；双腿伸直，先站稳约 2 秒。", "支撑腿站稳，另一条腿膝盖尽量伸直，向身体正前方抬起后稍停；参考动作接近水平。", "缓慢放下同一条腿，双脚落地并站稳，再做下一次或换腿。"],
    note:"抬的是整条直腿，不是屈膝提大腿或向侧面抬腿。双臂保持参考姿势。小幅度动作可能漏计，以舒适稳定为限，不要为计数勉强抬高。",
    cues:["请先双脚落地，自然站稳约 2 秒。", "站姿已确认，请将一条直腿向身体正前方抬起。", "尚待确认抬腿顶点：膝盖尽量伸直，前抬后稍停。", "抬腿顶点已确认，请缓慢放下同一条腿。", "请双脚落地并站稳，复位后计 1 次。"]
  },
  hurdle_step: {
    animation:true,
    title:"屈膝跨步抬腿", endpoints:["站稳","屈膝前抬","回位站稳 +1"], cycle:["站稳","屈膝抬腿","抬起","回位","站稳 +1"],
    count:"站稳 → 单腿屈膝向前抬起 → 收回原位并站稳，计 1 次。左右腿各完成一次共 2 次。",
    steps:["双臂稍向两侧打开并屈肘，双手约肩高；双脚落地，先站稳约 2 秒。", "屈膝向前抬起一条腿，小腿自然下垂，顶点稍停；参考动作的膝盖接近髋高。", "把同一只脚收回原位，双脚落地并站稳，再做下一次或换腿。"],
    note:"双臂保持参考姿势，原地屈膝抬腿即可，不需要障碍物。小幅度动作可能漏计，以舒适稳定为限，不要为计数勉强抬高。左右交替时，每次先双脚落地站稳。",
    cues:["请先双脚落地，自然站稳约 2 秒。", "站姿已确认，请屈膝向前抬起一条腿。", "尚待确认跨步顶点：屈膝向前抬腿，在舒适高度稍停。", "跨步顶点已确认，请把同一只脚收回原位。", "请双脚回到原位并站稳，复位后计 1 次。"]
  }
};
const phases = {seated:"坐姿",sitting:"坐姿",rising:"起身",standing:"站立",lowering:"下落",bottom:"最低点",descending:"下蹲",ascending:"起立",baseline:"起始姿态",lifting:"抬腿",raised:"抬起",stepping:"跨步",returning:"回落",other:"等待动作",up:"抬起",down:"落下",start:"起始",peak:"顶点"};
const qualityNames = {good:"观察有效",waiting_for_input:"等待画面",insufficient_pose:"关键点不足，计数暂不可用",multiple_people:"画面中有多人，计数暂不可用",no_person:"未检测到人体",timestamp_gap:"画面间隔过大，等待完整动作",person_changed:"目标变化，等待完整动作",degenerate_pose:"关键点异常，计数暂不可用",paused:"已暂停",module_error:"模块异常，训练中断",service_restarted:"服务重启，训练中断"};
const channels = {vision:"视频",accel:"加速度",audio:"音频",manual:"主动求助"};
const outcomes = {confirmed_fall:"确认跌倒",false_alarm:"误报",safe:"核实安全",unresolved:"结果未确定"};
const stateNames = {ready:"已就绪",loading:"加载中",not_initialized:"未初始化",unavailable:"不可用"};
const moduleNames = {video:"算法一 · 视频跌倒",signal:"算法一 · 加速度跌倒",exercise:"算法二 · 四动作计数",audio:"算法三 · 声音与语音"};
const modelNames = {video:"YOLOv8n + P1 + A4",signal:"A2 · RF + XGB",exercise:"opt_tcn_b4_w48",audio:"Silero / YAMNet + LR / Whisper tiny / rule_v2"};
const soundNames = {HELP:"求助相关声音",IMPACT:"撞击声",ABNORMAL:"异常声音",DAILY:"日常声音",UNKNOWN:"未确定"};
const dateTime = (v) => v ? new Date(v).toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit",second:"2-digit",hour12:false}) : "--";
const timeOnly = (v) => v ? new Date(v).toLocaleTimeString("zh-CN",{hour12:false}) : "--";
const duration = (v) => v >= 3600 ? `${(v/3600).toFixed(1)} 小时` : v >= 60 ? `${Math.floor(v/60)} 分 ${Math.round(v%60)} 秒` : `${Math.round(v || 0)} 秒`;
const modeBadge = (v) => `<span class="badge ${v === "live" ? "green" : "blue"}">${v === "live" ? "实时" : "样例 / 回放"}</span>`;
const levelBadge = (e) => `<span class="badge ${e.state === "resolved" ? "green" : e.level === "EMERGENCY" ? "red" : "amber"}">${e.state === "resolved" ? "已处置" : e.level === "EMERGENCY" ? "紧急求助" : "待核实"}</span>${e.withdrawn ? '<span class="badge blue">\u957f\u8005\u5df2\u64a4\u56de \u00b7 \u5f85\u786e\u8ba4</span>' : ""}`;
const empty = (title, sym="inbox", subtitle="", compact=false) => `<div class="empty-state ${compact?"compact":""}">${icon(sym)}<strong>${esc(title)}</strong>${subtitle?`<p>${esc(subtitle)}</p>`:""}</div>`;
const voiceBadge = (voice) => {
  if (!voice) return '<span class="badge neutral">未问询</span>';
  const tone = voice.css === "ok" ? "green" : voice.css === "risk" ? "red" : "neutral";
  const note = voice.reason && !voice.answered ? `<br><span class="muted small">${esc(voice.reason)}</span>` : "";
  return `<span class="badge ${tone}">语音：${esc(voice.label)}${voice.detail?` · ${esc(voice.detail)}`:""}</span>${note}`;
};
const voiceChip = (voice) => {
  if (!voice) return '<span class="badge neutral">语音：未问询</span>';
  const tone = voice.css === "ok" ? "green" : voice.css === "risk" ? "red" : "neutral";
  return `<span class="badge ${tone}">语音：${esc(voice.label)}${voice.detail?` · ${esc(voice.detail)}`:""}</span>`;
};
const app = {page:"overview",settings:null,status:null,events:[],sessions:[],notices:[],sounds:[],report:null,eventFilter:"open",eventMode:"live",reportDays:7,reportMode:"live",selectedExercise:"chair_sit_to_stand",sid:null,exercise:null,camera:null,capturing:false,exerciseSource:null,audioStream:null,audioSocket:null,audioContext:null,audioRecorder:null,recordedBlob:null,videoHistory:[],selectedEvent:null,noticeIds:new Set(),ready:false,overlay:{box:true,skeleton:true},pose:null,poseBusy:false};
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
      let message = result.error || result.detail || `请求失败 (${response.status})`;
      if (Array.isArray(message)) message = message.map(x=>`${x.loc?.slice(1).join(".") || "参数"}: ${x.msg}`).join("；");
      if (typeof message !== "string") message = JSON.stringify(message);
      throw new Error(message);
    }
    return result;
  } catch (e) {
    if (e.name === "AbortError") throw new Error("请求超时，请检查模块状态后重试");
    throw e;
  } finally {clearTimeout(timer);}
}
function toast(message,error=false){$("toast").textContent=message;$("toast").className=`toast ${error?"error":""}`;$("toast").hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$("toast").hidden=true,error?7500:4000);}
async function busy(button, fn){if(button.disabled)return;const wasDisabled=button.disabled;button.disabled=true;button.classList.add("loading");try{return await fn();}catch(e){toast(e.message,true);}finally{button.disabled=wasDisabled;button.classList.remove("loading");}}
function metric(title,value,unit,sym,note="",cls=""){return `<div class="metric${cls?` ${cls}`:""}"><div class="metric-header">${icon(sym)}${esc(title)}</div><div class="metric-value">${esc(value)}<small>${esc(unit)}</small></div><div class="metric-note">${esc(note)}</div></div>`;}
function channelRow(row){const active=row?.state==="active";return `<div class="channel-item">${icon({vision:"video",accel:"activity",audio:"mic"}[row.channel])}<div><strong>${channels[row.channel]}</strong><small><span class="dot ${active?"green":"amber"}"></span>${active?"输入有效":row.state==="reduced"?"覆盖受限":"未连接"}</small></div></div>`;}

function navigate(){const requested=location.hash.slice(1).split("?")[0]||"overview";app.page=$("page-"+requested)?requested:"overview";document.querySelectorAll(".page").forEach(x=>x.hidden=x.id!==`page-${app.page}`);document.querySelectorAll("nav [data-page]").forEach(x=>x.classList.toggle("active",x.dataset.page===app.page));const label=document.querySelector(`nav [data-page="${app.page}"]`)?.textContent.replace(/\d+$/,"")||"照护总览";$("pageCrumb").textContent=label;document.title=`${label} · HomeEase`;$("sidebar").classList.remove("open");$("navBackdrop").hidden=true;renderCurrent();if(app.page==="reports")loadReport();if(app.page==="settings")loadDiagnostics();window.scrollTo(0,0);}

async function refresh(){if(refreshBusy)return;refreshBusy=true;try{const results=await Promise.allSettled([api("/api/status",{timeout:12000}),api("/api/events"),api("/api/sessions"),api("/api/notifications"),app.page==="audio"?api("/api/sounds"):Promise.resolve(app.sounds),api("/api/settings")]);if(results[0].status==="rejected")throw results[0].reason;app.status=results[0].value;if(results[5].status==="fulfilled"){app.settings=results[5].value;if(!app.settings.audio_consent){if(app.audioStream)stopAudioLive();if(app.audioRecorder?.state==="recording")app.audioRecorder.stop();app.recordingStream?.getTracks().forEach(t=>t.stop());app.recordedBlob=null;}}if(results[1].status==="fulfilled")app.events=results[1].value;if(results[2].status==="fulfilled")app.sessions=results[2].value;if(results[3].status==="fulfilled")app.notices=results[3].value;if(results[4].status==="fulfilled")app.sounds=results[4].value;$("connectionState").innerHTML='<span class="dot green"></span>系统已连接';const n=app.events.filter(e=>e.state==="open"&&e.mode==="live").length;$("navEventCount").textContent=n;$("navEventCount").hidden=!n;$("noticeDot").hidden=!app.notices.some(n=>n.status!=="seen"&&n.kind==="inapp");for(const notice of app.notices){if(app.ready&&!app.noticeIds.has(notice.id)&&notice.kind==="inapp"&&notice.mode==="live"){toast(`${notice.title} · ${notice.message}`);if("Notification" in window&&Notification.permission==="granted")new Notification(notice.title,{body:notice.message,tag:notice.event_id});}app.noticeIds.add(notice.id);}app.ready=true;renderCurrent();if(app.selectedEvent&&$("eventDialog").open){const latest=app.events.find(e=>e.id===app.selectedEvent.id);if(latest&&latest.revision!==app.selectedEvent.revision&&!$("eventDetail").contains(document.activeElement)){renderEventDetail(latest);}}}catch(e){$("connectionState").innerHTML='<span class="dot red"></span>系统连接中断';}finally{refreshBusy=false;}}
function renderCurrent(){if(!app.status)return;if(!["overview","monitor"].includes(app.page)){for(const id of ["overviewFrame","monitorFrame"]){const image=$(id);if(image.dataset.streaming){image.removeAttribute("src");delete image.dataset.streaming;}}}renderModuleIndicators();if(app.page==="overview")renderOverview();if(app.page==="monitor")renderMonitor();if(app.page==="exercise")renderExerciseHistory();if(app.page==="audio")renderAudioHistory();if(app.page==="events")renderEvents();icons();}
function renderModuleIndicators(){for(const m of app.status.modules){if(m.id==="exercise"){$("exerciseModelState").textContent=`运动模块${stateNames[m.state]||m.state}`;$("exerciseModelState").className=`badge ${m.state==="ready"?"green":"amber"}`;}}const consent=app.settings?.audio_consent;$("audioConsentBadge").textContent=consent?"已授权本地音频":"未授权音频";$("audioConsentBadge").className=`badge ${consent?"green":"neutral"}`;$("audioConsentBanner").hidden=!!consent;if(app.page==="settings")renderModules();}
function renderOverview(){const s=app.status;const coverage={active:["监测已开启","视频与加速度输入有效"],reduced:["监测覆盖受限",s.coverage.filter(x=>x.channel!=="audio"&&x.state!=="active").map(x=>x.reason).join("；")],unavailable:["监测尚未开启","视频与加速度输入未连接"]}[s.coverage_state];$("coverageTitle").textContent=coverage[0];$("coverageDescription").textContent=coverage[1];$("coverageBanner").classList.toggle("is-active",s.coverage_state==="active");const today=new Date().toLocaleDateString("en-CA");const live=app.events.filter(e=>e.mode==="live");const dayEvents=live.filter(e=>new Date(e.created_at).toLocaleDateString("en-CA")===today);const todaySessions=app.sessions.filter(s=>s.mode==="live"&&s.status==="ended"&&new Date(s.created_at).toLocaleDateString("en-CA")===today);const open=live.filter(e=>e.state==="open");$("dashboardMetrics").innerHTML=metric("待处理事件",open.length,"条","bell-ring",open.some(e=>e.level==="EMERGENCY")?"有紧急求助待响应":"当前实时记录")+metric("今日确认跌倒",dayEvents.filter(e=>e.outcome==="confirmed_fall").length,"次","shield-check","以照护者处置结果为准")+metric("今日运动",todaySessions.reduce((sum,s)=>sum+s.count,0),"次","activity",`${todaySessions.length} 次已完成训练`)+metric("有效监测通道",s.coverage.filter(x=>x.state==="active"&&x.channel!=="audio").length,"/ 2","radio","视频 / 加速度");$("dashboardChannels").innerHTML=s.coverage.sort((a,b)=>["vision","accel","audio"].indexOf(a.channel)-["vision","accel","audio"].indexOf(b.channel)).map(channelRow).join("");$("openEventCount").textContent=open.length;$("attentionList").innerHTML=open.length?open.slice(0,4).map(e=>`<div class="event-item" role="button" tabindex="0" data-event="${e.id}"><span class="event-icon ${e.level==="EMERGENCY"?"red":""}">${icon(e.level==="EMERGENCY"?"bell-ring":"triangle-alert")}</span><div class="event-body"><div class="event-title"><strong>${esc(e.room)} · ${e.level==="EMERGENCY"?"紧急求助":"疑似跌倒"}</strong>${levelBadge(e)}${e.voice?.high_risk?voiceChip(e.voice):""}</div><p>${channels[e.channel]}${e.channel==="manual"?"":"检测"} · ${esc(e.resident)}</p><div class="event-meta"><span>${dateTime(e.occurred_at)}</span><span>#${e.id.slice(0,6)}</span></div></div>${icon("chevron-right")}</div>`).join(""):empty("暂无待处理事件","circle-check","实时事件将显示在这里");$("recentSessions").innerHTML=app.sessions.filter(s=>s.mode==="live").slice(0,3).map(s=>`<div class="session-row"><span class="session-icon">${icon("activity")}</span><div>${names[s.exercise]}<small>${dateTime(s.created_at)} · ${s.status==="ended"?"已完成":s.status==="interrupted"?"已中断":"进行中"}</small></div><span class="session-count">${s.count}<small> 次</small></span></div>`).join("")||empty("今日还没有训练记录","activity","",true);const running=s.video.state==="running";$("overviewVideoMode").textContent=running?(s.video_mode==="live"?"实时监测":"样例 / 回放"):"未开启";$("overviewVideoMode").className=`badge ${running?"green":"neutral"}`;$("cameraSourceLabel").textContent=running?(s.video_mode==="live"?"实时画面":"样例 / 回放"):"样例预览";$("cameraIdle").hidden=running;loadFrame("overviewFrame");if(app.report)drawBars($("overviewChart"),app.report.series,"reps","#579579");else loadReport(true);}
function renderMonitor(){
  const v=app.status.video,running=v.state==="running",sensor=app.status.sensor;
  const coverage=app.status.coverage.filter(c=>["vision","accel"].includes(c.channel));
  const active=coverage.filter(c=>c.state==="active").length;
  $("monitorChannels").innerHTML=coverage.map(channelRow).join("");
  $("monitorMode").textContent=active===2?"双通道监测中":active===1?"1 / 2 通道有效":"等待实时输入";
  $("monitorMode").className=`badge ${active===2?"green":active===1?"amber":"neutral"}`;
  $("videoStateLabel").textContent=({idle:"尚未启动",running:app.status.video_mode==="live"?"实时视频检测":"视频回放",finished:"本次检测已结束",error:"检测异常"})[v.state]||v.state;
  $("videoFps").textContent=`推理 ${v.fps||0} FPS`;
  $("videoProbability").textContent=v.frames?`${(100*(v.prob||0)).toFixed(1)}%`:"--";
  $("videoFrames").textContent=v.frames||0;$("videoFalls").textContent=v.n_falls||0;
  $("videoVisibility").textContent=v.frames?(v.person_visible?"人体可见":"未检测到人体"):"未检测";
  $("startVideo").disabled=running||app.status.modules.find(m=>m.id==="video").state!=="ready";
  $("stopVideo").disabled=!running;
  $("videoModuleInfo").innerHTML=`<div><span>检测方案</span><strong>A4</strong></div><div><span>原始阈值</span><strong>0.60</strong></div><div><span>确认次数</span><strong>2</strong></div><div><span>目标采样帧率</span><strong>15 FPS</strong></div>${v.error?`<div class="error-inline">${esc(v.error)}</div>`:""}`;
  if(v.frames){const last=app.videoHistory.at(-1);if(!last||last.frame!==v.frames)app.videoHistory.push({frame:v.frames,value:v.prob});app.videoHistory=app.videoHistory.slice(-120);}
  drawLines($("videoChart"),[{values:app.videoHistory.map(x=>x.value),color:"#47836a"}],{min:0,max:1,threshold:.6});
  const fresh=sensor.state==="connected"&&Date.now()/1000-(sensor.last_received||0)<5;
  $("sensorStatus").textContent=sensor.state==="connected"?(fresh?`实时数据已连接 · ${sensor.windows||0} 个窗口`:"传感器数据中断"):"未连接";
  if(sensor.state==="connected"){
    $("signalState").textContent=fresh?"实时监测中":"数据中断";
    $("signalState").className=`badge ${fresh?"green":"amber"}`;
    const samples=sensor.samples||[],recent=sensor.recent||[];
    drawLines($("signalChart"),[1,2,3].map((axis,i)=>({values:samples.map(s=>s[axis]),color:["#547bb4","#5a9877","#bf9459"][i]})));
    drawLines($("signalProbabilityChart"),[{values:recent.map(s=>s.p),color:"#52886d"}],{min:0,max:1,threshold:.5});
    $("signalResult").textContent=`实时窗口跌倒概率：${((sensor.prob||0)*100).toFixed(1)}%`;
    $("signalResult").dataset.live="true";
  }else if($("signalResult").dataset.live){
    $("signalState").textContent="传感器已断开";$("signalState").className="badge neutral";
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
    const label=`疑似跌倒 ${((pose.prob||0)*100).toFixed(0)}%`;
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
    hint.textContent=shown===2?"全部显示":shown===1?(app.overlay.box?"仅检测框":"仅骨架关键点"):"已关闭";
  }
  if(!overlayVisible())app.pose=null;
  paintOverlay();
}

function renderEvents(){let list=app.events.filter(e=>(app.eventFilter==="all"||e.state===app.eventFilter)&&(app.eventMode==="all"||e.mode===app.eventMode));const q=$("eventSearch").value.trim().toLowerCase();if(q)list=list.filter(e=>`${e.id} ${e.room} ${e.resident}`.toLowerCase().includes(q));$("eventsTotal").textContent=`${list.length} 条记录`;$("eventList").innerHTML=list.length?list.map(e=>`<article class="event-record ${e.level==="EMERGENCY"?"emergency":""} ${e.state==="resolved"?"resolved":""}" data-event="${e.id}" role="button" tabindex="0"><div><h3>${esc(e.room)} · ${e.channel==="manual"?"主动求助":e.channel==="audio"?"语音求救":"疑似跌倒"}${levelBadge(e)}${modeBadge(e.mode)}${voiceChip(e.voice)}</h3><p>${channels[e.channel]}${e.channel==="manual"?"":"检测"} · ${esc(e.resident)}${e.outcome?` · ${outcomes[e.outcome]}`:""}</p><footer><span>${dateTime(e.occurred_at)}</span><span>事件 #${e.id}</span><span>${e.acknowledged_at?"照护者已接手":""}</span></footer></div><span class="badge neutral">${e.tag?({single_channel:"单通道",dual_channel:"双通道",triple_channel:"三通道"})[e.tag]:"主动求助"}</span>${icon("chevron-right")}</article>`).join(""):empty("暂无符合条件的事件","inbox");icons();}
async function openEvent(id){try{renderEventDetail(await api(`/api/events/${id}`));if(!$("eventDialog").open)$("eventDialog").showModal();}catch(e){toast(e.message,true);}}
function evidenceText(e){const v=e.evidence;if(e.channel==="vision")return `视频原算法检测到疑似跌倒。触发概率 ${((v.prob||0)*100).toFixed(1)}%，判定阈值 ${v.threshold??.6}${v.peak?`，峰值概率 ${(v.peak*100).toFixed(1)}%`:""}。`;if(e.channel==="accel")return `加速度原算法检测到疑似跌倒。触发概率 ${((v.score||0)*100).toFixed(1)}%，判定阈值 ${v.threshold??.5}${v.n_windows?`，命中 ${v.n_windows} 个窗口`:""}。`;if(e.channel==="audio")return "原语音求救监听器识别到明确求助。";return "长者通过求助按钮主动请求协助。";}
function renderEventDetail(e){app.selectedEvent=e;const settings=app.settings;const inquiry=e.inquiry;const inquiryText={not_started:"尚未发起",listening:"等待回复",audio_unavailable:"音频不可用",completed:"已完成",button_safe:"长者已表示安全"}[inquiry.status]||inquiry.status;$("eventDetail").innerHTML=`<div class="event-detail-body"><div class="event-detail-header"><h3>${esc(e.room)} · ${esc(e.resident)}</h3><div>${levelBadge(e)} ${modeBadge(e.mode)}</div></div><div class="event-detail-meta"><div><span>事件编号</span>#${e.id}</div><div><span>发生时间</span>${dateTime(e.occurred_at)}</div><div><span>检测通道</span>${channels[e.channel]}</div><div><span>语音询问</span>${e.voice?voiceBadge(e.voice):esc(inquiryText)}</div><div><span>检测时间</span>${dateTime(e.detected_at)}</div><div><span>输入来源</span>${esc(e.source)}</div></div>${e.withdrawn?`<div class="completion-banner">${icon("rotate-ccw")}<strong>\u957f\u8005\u5df2\u64a4\u56de\u6c42\u52a9\u3002\u4e8b\u4ef6\u4ecd\u7136\u4fdd\u7559\uff0c\u8bf7\u8054\u7cfb\u957f\u8005\u786e\u8ba4\u540e\u8bb0\u5f55\u5904\u7f6e\u7ed3\u8bba\u3002</strong></div>`:""}<div class="evidence-summary">${evidenceText(e)}${e.mode==="replay"?'<br><span class="badge blue">样例 / 回放记录</span>':""}</div>${e.evidence.clip?`<video class="event-video" controls preload="metadata" src="/api/video/clip/${e.id}"></video>`:""}${e.evidence.snippet?'<div class="section-heading"><h3>报警前后加速度</h3><span class="muted small">−2 秒至 +1 秒 · g</span></div><div class="chart-wrap"><canvas id="eventSignalChart"></canvas></div>':""}${e.state!=="resolved"?`<div class="event-actions"><button class="button primary" data-action="acknowledge" ${e.acknowledged_at?"disabled":""}>${icon("user-check")}${e.acknowledged_at?"已接手":"接手处理"}</button><button class="button danger-outline" data-action="escalate">${icon("arrow-up-right")}升级响应</button><a class="button secondary" href="/elder?event=${e.id}" target="_blank">${icon("audio-lines")}长者问询</a></div>`:""}<div>${(settings?.contacts||[]).map((c,i)=>`<div class="event-contact-row"><span class="badge neutral">${i+1}</span><strong>${esc(c.name)}</strong><span class="muted">${esc(c.relation)}</span><a class="button secondary" href="tel:${esc(c.phone.replace(/[^+0-9]/g,""))}" data-contact-call="true">${icon("phone")}${esc(c.phone)}</a></div>`).join("")||'<p class="muted small">暂无紧急联系人</p>'}</div><div class="section-heading spaced"><h3>语音问询结论</h3><span class="muted small">${e.inquiry?.asked_at?`发起于 ${dateTime(e.inquiry.asked_at)}`:"尚未发起问询"}</span></div><div class="inquiry-conclusion">${e.voice?voiceBadge(e.voice):'<span class="badge neutral">未问询</span>'}</div><div class="section-heading spaced"><h3>响应时间线</h3><a class="text-action" href="/reports/events/${e.id}" target="_blank">${icon("printer")}事件报告</a></div><ol class="timeline">${e.timeline.map(t=>`<li><time>${dateTime(t.at)}</time><div>${esc(t.note)}<small>${esc(t.actor)}</small></div></li>`).join("")}</ol>${e.state==="resolved"?`<div class="completion-banner">${icon("circle-check")}<strong>${outcomes[e.outcome]}</strong><span>${e.response_s!==null?`响应时长 ${duration(e.response_s)}`:""}</span></div>${e.notes?`<p class="muted small" style="margin-top:15px">${esc(e.notes)}</p>`:""}`:`<form id="resolveForm" class="resolution-form"><h3>处置结论</h3><label>最终结果<select id="resolveOutcome" required><option value="">选择处置结果</option><option value="confirmed_fall">确认跌倒</option><option value="false_alarm">误报</option><option value="safe">核实安全</option><option value="unresolved">结果未确定</option></select></label><label>处理备注<textarea id="resolveNote" rows="3" maxlength="2000" placeholder="联系情况、采取的措施等"></textarea></label><button class="button primary" type="submit">${icon("check")}记录结果并关闭事件</button></form>`}</div>`;icons();if(e.evidence.snippet){requestAnimationFrame(()=>{const s=e.evidence.snippet;drawLines($("eventSignalChart"),[{values:s.ax,color:"#547bb4"},{values:s.ay,color:"#5a9877"},{values:s.az,color:"#bf9459"}],{labels:s.t?.map(t=>`${t.toFixed(1)}s`)});});}$("resolveForm")?.addEventListener("submit",ev=>{ev.preventDefault();busy(ev.submitter,()=>eventAction("resolve",{outcome:$("resolveOutcome").value,note:$("resolveNote").value}));});}
async function eventAction(action,extra={}){const e=app.selectedEvent;const result=await api(`/api/events/${e.id}/action`,{method:"POST",body:{action,revision:e.revision,...extra}});renderEventDetail(result);await refresh();toast(action==="resolve"?"处置结果已记录":"操作已记录");}

function renderExerciseHistory(){const sessions=app.sessions;$("exerciseHistory").innerHTML=sessions.length?`<table><thead><tr><th>训练时间</th><th>动作</th><th>次数 / 目标</th><th>有效观察</th><th>来源</th><th>状态</th><th>报告</th></tr></thead><tbody>${sessions.slice(0,30).map(s=>`<tr><td>${dateTime(s.created_at)}</td><td>${names[s.exercise]}</td><td>${s.count} / ${s.target_reps}</td><td>${duration(s.valid_duration_s)}</td><td>${modeBadge(s.mode)}</td><td>${({running:"进行中",paused:"已暂停",ended:"已结束",interrupted:"已中断"})[s.status]||s.status}</td><td><a class="icon-button" title="下载训练报告" href="/api/sessions/${s.session_id}/export.json" aria-label="下载训练报告">${icon("download")}</a></td></tr>`).join("")}</tbody></table>`:empty("暂无训练记录","activity","",true);icons();}
function renderExerciseGuide(element,key){
  const guide=exerciseGuides[key];
  element.dataset.exerciseGuide=key;
  element.innerHTML=`<div class="exercise-guide-copy"><h3>${esc(guide.title)} · 动作与计数说明</h3><p class="exercise-count-rule">${esc(guide.count)}</p><ol>${guide.steps.map(step=>`<li>${esc(step)}</li>`).join("")}</ol><p class="exercise-motion-note">${esc(guide.note)}</p></div><figure><img src="/static/media/${key}-guide.png" alt="${esc(guide.title)}：起始、顶点与复位示意" width="780" height="252"><figcaption>${guide.endpoints.map(label=>`<span>${esc(label)}</span>`).join("")}</figcaption>${guide.animation?`<div class="exercise-guide-controls"><span>45° 动作骨架参考</span><button type="button" class="icon-button" title="播放动作示范" aria-label="播放动作示范" aria-pressed="false">${icon("play")}</button></div>`:""}</figure><p class="exercise-camera-guide">${icon("camera")}<span><strong>摄像头位置：</strong>正对镜头为 0°，纯侧身为 90°；建议身体斜侧约 45°，左右侧均可，不必精确量角。镜头约在髋部高度、水平拍摄，头顶到双脚完整入镜，四周留空，双腿不要互相遮挡。</span></p><p class="exercise-pace">建议节奏：抬起或起身约 1～2 秒，顶点稍停，返回约 1～2 秒；每次复位后站稳或坐稳约 1 秒，再开始下一次。按舒适幅度完成。</p>`;
  const playback=element.querySelector('.exercise-guide-controls button');
  if(playback)playback.onclick=()=>{
    const playing=playback.getAttribute('aria-pressed')!=='true';
    element.querySelector('figure img').src=`/static/media/${key}-guide.${playing?'gif':'png'}`;
    playback.setAttribute('aria-pressed',String(playing));
    playback.title=playback.ariaLabel=playing?"停止动作示范":"播放动作示范";
    playback.innerHTML=icon(playing?'square':'play');
    icons();
  };
}
function chooseExercise(key){app.selectedExercise=key;document.querySelectorAll(".exercise-option").forEach(x=>{const selected=x.dataset.exercise===key;x.classList.toggle("selected",selected);x.setAttribute("aria-pressed",selected);});renderExerciseGuide($("exerciseGuide"),key);icons();}
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
  if(s.status==="paused")cue="已暂停；继续后请从起始姿势重新开始。";
  else if(["ended","interrupted"].includes(s.status))cue=s.status==="ended"?"训练已结束。":"训练已中断。";
  else if(s.quality!=="good")cue="请让双肩、双髋、双膝和双踝完整入镜；恢复后从起始姿势重新开始。";
  else if(completed)cue="完整动作已计入，请稳住后开始下一次。";
  if($("exerciseCue").dataset.cue!==cue){$("exerciseCue").dataset.cue=cue;$("exerciseCue").textContent=cue;}
}
async function startExercise(ev){
  ev.preventDefault();
  await busy($("startExercise"),async()=>{
    const source=$("exerciseSource").value;
    let uploadId=null;
    if(source==="upload"){
      const file=$("exerciseUpload").files[0];if(!file)throw new Error("请选择运动视频");
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
    renderExerciseState(result);await refresh();
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
  app.exercise=s;$("activeExerciseName").textContent=names[s.exercise];$("exerciseMode").textContent=s.mode==="live"?"实时训练":"样例 / 回放";
  if($("activeExerciseGuide").dataset.exerciseGuide!==s.exercise)renderExerciseGuide($("activeExerciseGuide"),s.exercise);
  renderExerciseCycle(s);
  $("repCount").textContent=s.count;$("repTarget").textContent=s.target_reps;$("repProgress").value=s.progress;
  $("exercisePhase").textContent=phases[s.phase]||s.phase;$("exerciseDuration").textContent=duration(s.valid_duration_s);
  const q=["multiple_people","no_person"].includes(s.person_id)?s.person_id:s.quality;
  const diagnostic=s.capture_diagnostics;
  const jointNames={left_shoulder:"左肩",right_shoulder:"右肩",left_hip:"左髋",right_hip:"右髋",left_knee:"左膝",right_knee:"右膝",left_ankle:"左踝",right_ankle:"右踝"};
  let quality=qualityNames[q]||q;
  const unrecognized=q==="good"&&s.phase==="other";
  if(unrecognized)quality="骨架有效，尚未确认所选动作";
  if(diagnostic?.missing_joints?.length&&q==="insufficient_pose")quality+=`：${diagnostic.missing_joints.map(n=>jointNames[n]||n).join("、")}`;
  if(diagnostic?.gap_ms>250&&q==="good")quality="画面曾中断，当前动作重新观察";
  $("exerciseQuality").textContent=quality;
  $("exerciseQuality").title=diagnostic?`有效采样 ${diagnostic.observation_fps||0} FPS · 最近间隔 ${diagnostic.gap_ms||0} ms`:"";
  $("exerciseQuality").classList.toggle("bad",unrecognized||!["good","waiting_for_input"].includes(q)||diagnostic?.gap_ms>250);
  const counts=s.counting_diagnostics,details=$("exerciseDiagnostics");
  details.hidden=!counts;
  if(counts){
    const joints=Object.entries(diagnostic?.joint_confidences||{});
    const interruptions=Object.values(counts.interruption_counts).reduce((sum,n)=>sum+n,0);
    const last=counts.last_interruption;
    details.innerHTML=`<summary>关键点与输入状态</summary><div class="exercise-input-stats"><span>关键点完整 <strong>${Math.round(counts.visible_pose_ratio*100)}%</strong></span><span>连续可见 <strong>${counts.uninterrupted_s.toFixed(1)} 秒</strong></span><span>采样 <strong>${diagnostic?.observation_fps||0} FPS</strong></span><span>输入中断 <strong>${interruptions} 次</strong></span></div><div class="exercise-joints">${joints.map(([name,value])=>`<span class="${value<.3?"missing":"visible"}" title="${jointNames[name]||name} · ${(value*100).toFixed(1)}%"><i aria-hidden="true"></i>${jointNames[name]||name}<small>${Math.round(value*100)}%</small></span>`).join("")}</div>${last?`<p class="exercise-last-reset">最近中断：${esc(qualityNames[last.reason]||last.reason)}</p>`:""}`;
  }
  $("exerciseDiagnosticExport").href=`/api/sessions/${s.session_id}/diagnostics.json`;
  $("pauseExercise").innerHTML=icon(s.status==="paused"?"play":"pause")+`<span>${s.status==="paused"?"继续":"暂停"}</span>`;
  if(["ended","interrupted"].includes(s.status)){
    $("exerciseCompleted").hidden=false;$("exerciseCompleted").querySelector("strong").textContent=s.status==="ended"?`训练已结束 · ${s.count} 次`:"训练已中断";
    $("pauseExercise").disabled=true;$("endExercise").disabled=true;stopCamera();
  }
  drawPose($("poseCanvas"),s.keypoints,s.mode!=="live");icons();
}
async function pollExercise(){if(!app.sid||exercisePollBusy)return;const sid=app.sid;exercisePollBusy=true;try{const s=await api(`/api/sessions/${sid}`,{timeout:15000});if(app.sid===sid)renderExerciseState(s);}catch(e){}finally{exercisePollBusy=false;}}
function renderAudioHistory(){
  $("soundHistory").innerHTML=app.sounds.length?`<table><thead><tr><th>时间</th><th>任务</th><th>结果</th><th>来源</th></tr></thead><tbody>${app.sounds.map(s=>{
    const labels=(s.result.sounds?.events||s.result.events||[]).map(e=>soundNames[e.category]||e.category).join("、")||"无已确认声音事件";
    const help=(s.result.help_result||s.result).help?"检测到求救":"未检测到明确求救";
    return `<tr><td>${dateTime(s.created_at)}</td><td>${s.task==="monitor"?"联合分析":s.task==="help"?"求救监听":"声音标签"}</td><td>${esc(s.task==="monitor"?`${labels} · ${help}`:s.task==="help"?help:labels)}</td><td>${modeBadge(s.mode)}</td></tr>`;
  }).join("")}</tbody></table>`:empty("暂无声音记录","audio-lines","",true);
  const inquiries=app.events.filter(e=>e.mode==="live"&&e.state==="open").slice(0,8);
  $("inquiryQueue").innerHTML=inquiries.map(e=>`<div class="inquiry-row"><span>${esc(e.room)} · #${e.id.slice(0,6)}</span><span class="badge neutral">${({not_started:"待问询",listening:"等待语音回复",processing:"正在识别回复",completed:"问询已完成",audio_unavailable:"问询音频不可用",button_safe:"已按安全按钮"})[e.inquiry?.status]||"待问询"}</span><button class="text-action" data-event="${e.id}">查看事件 ${icon("arrow-right")}</button></div>`).join("")||empty("暂无待问询事件","circle-check","",true);
}
function renderJointAudioResult(result){
  renderAudioResult(result.sounds,"sounds");const sounds=$("audioResult").innerHTML;
  renderAudioResult(result.help_result,"help");
  $("audioResult").innerHTML=`<h3>环境声音</h3>${sounds}<div class="separator"></div><h3>语音求助</h3>${$("audioResult").innerHTML}`;
  $("audioResultDuration").textContent=`${Number(result.duration_s||0).toFixed(1)} 秒`;icons();
}
function renderLiveAudio(result){
  const degraded=result.pipeline?.status?.startsWith("degraded")||String(result.listener?.status||"").startsWith("degraded");
  $("audioLiveStatus").textContent=degraded?"监听覆盖受限，请检查运行状态":"麦克风已连接 · 声音与求助同时监听";
  for(const id of ["soundLiveState","helpLiveState"]){$(id).textContent=degraded?"覆盖受限":"监听中";$(id).className=`badge ${degraded?"amber":"green"}`;}
  $("liveSoundResult").innerHTML=(result.events||[]).slice().reverse().map(e=>`<div class="sound-event"><span class="sound-type">${icon("audio-lines")}</span><div><h3>${soundNames[e.category]||esc(e.category)}</h3><small>${dateTime(e.t_start_utc)} · ${e.n_windows||e.windows||0} 个窗口</small></div><span class="badge blue">${(e.confidence*100).toFixed(1)}%</span></div>`).join("")||empty("暂无已确认声音事件","volume-1","",true);
  app.liveCalls=[...(app.liveCalls||[]),...(result.calls||[])].slice(-20);
  $("liveHelpResult").innerHTML=app.liveCalls.slice().reverse().map(c=>`<div class="sound-event"><span class="sound-type">${icon("bell-ring")}</span><div><h3>检测到语音求助</h3><small>${dateTime(c.t_utc||new Date().toISOString())}</small></div><span class="badge red">已生成事件</span></div>`).join("")||empty("暂无语音求助","speech","",true);
  icons();
}
function renderAudioResult(result,task){$("audioResultDuration").textContent=result.duration_s?`${result.duration_s} 秒`:"";if(task==="sounds"){$("audioResult").innerHTML=(result.events||[]).map(e=>`<div class="sound-event"><span class="sound-type">${icon("audio-lines")}</span><div><h3>${soundNames[e.category]||esc(e.category)}</h3><small>${e.start_s??dateTime(e.t_start_utc)}${e.end_s!==undefined?` – ${e.end_s} 秒`:""} · ${e.windows||e.n_windows||0} 个窗口</small></div><span class="badge blue">${(e.confidence*100).toFixed(1)}%</span></div>`).join("")||empty("未确认声音事件","volume-1");}else{$("audioResult").innerHTML=`<div class="help-result">${icon(result.help?"bell-ring":"circle-check")}<h3>${result.help?"检测到明确语音求助":"未检测到明确求助"}</h3><p>${(result.segments||[]).length} 个语音片段</p>${(result.segments||[]).map(s=>`<div class="sound-event"><div><h3>${({help:"明确求助",not_help:"非求助意图",not_a_call:"非明确求助话语",cooldown:"处于冷却时间",too_short:"语音过短",too_long:"语音过长",low_confidence:"识别置信度不足",no_text:"未识别到文字",too_many_words:"语句过长"})[s.verdict]||esc(s.verdict)}</h3><small>${s.start_s} 秒 · ${s.duration_s?.toFixed(2)} 秒</small></div></div>`).join("")}</div>`;}icons();}
async function analyzeAudio(ev){ev.preventDefault();await busy($("analyzeAudio"),async()=>{if(!app.settings.audio_consent)throw new Error("请先取得音频处理授权");const task="monitor",source=$("audioSource").value;let response;if(source==="sample"){response=await api("/api/audio/sample",{method:"POST",body:{sample:$("audioSampleSelect").value,task}});}else{const file=source==="record"?app.recordedBlob:$("audioFile").files[0];if(!file)throw new Error(source==="record"?"请先完成录音":"请选择音频文件");const form=new FormData();form.append("file",file,"audio.webm");form.append("task",task);form.append("mode",source==="record"?"live":"replay");response=await api("/api/audio/analyze",{method:"POST",body:form});}renderJointAudioResult(response.result);await refresh();});}
async function recordAudio(){await busy($("recordAudio"),async()=>{
  if(!app.settings.audio_consent)throw new Error("请先取得音频处理授权");
  if(!navigator.mediaDevices?.getUserMedia)throw new Error("麦克风需要 localhost 或 HTTPS 访问");
  const stream=await navigator.mediaDevices.getUserMedia({audio:true});let interval;
  app.recordingStream=stream;
  try{
    const recorder=new MediaRecorder(stream),chunks=[];app.audioRecorder=recorder;
    recorder.ondataavailable=e=>{if(e.data.size)chunks.push(e.data);};
    await new Promise((resolve,reject)=>{
      recorder.onerror=()=>reject(new Error("录音失败"));
      recorder.onstop=()=>{
        clearInterval(interval);
        if(app.settings.audio_consent){app.recordedBlob=new Blob(chunks,{type:recorder.mimeType});$("audioPreview").src=URL.createObjectURL(app.recordedBlob);}
        else app.recordedBlob=null;
        resolve();
      };
      recorder.start();let remaining=10;$("recordStatus").textContent=`录音中 · ${remaining} 秒`;
      interval=setInterval(()=>{remaining--;$("recordStatus").textContent=`录音中 · ${remaining} 秒`;if(remaining<=0||!app.settings.audio_consent){clearInterval(interval);if(recorder.state!=="inactive")recorder.stop();}},1000);
    });
    $("recordStatus").textContent=app.settings.audio_consent?"录音已就绪":"音频授权已撤回";
  }finally{clearInterval(interval);stream.getTracks().forEach(t=>t.stop());app.audioRecorder=null;app.recordingStream=null;}
});}
async function startAudioLive(){
  await busy($("startAudioLive"),async()=>{
    if(!app.settings.audio_consent)throw new Error("请先取得音频处理授权");
    if(!navigator.mediaDevices?.getUserMedia)throw new Error("麦克风需要 localhost 或 HTTPS 访问");
    const stream=await navigator.mediaDevices.getUserMedia({audio:{channelCount:1,echoCancellation:false,noiseSuppression:false,autoGainControl:true}});
    const context=new AudioContext({sampleRate:16000});
    try{
      if(context.sampleRate!==16000)throw new Error("浏览器不支持 16 kHz 音频输入");
      await context.audioWorklet.addModule("/static/pcm-worklet.js");
      const socket=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/audio`);
      app.audioStream=stream;app.audioContext=context;app.audioSocket=socket;
      await new Promise((resolve,reject)=>{
        socket.onmessage=event=>{const r=JSON.parse(event.data);if(r.error)reject(new Error(r.error));else if(r.state==="listening")resolve();};
        socket.onerror=()=>reject(new Error("声音监听连接失败"));
        socket.onclose=()=>reject(new Error("声音监听未连接，请检查授权和模块状态"));
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
      socket.onclose=()=>{queue.close();stopAudioLive();};socket.onerror=()=>toast("麦克风连接中断",true);
      $("startAudioLive").disabled=true;$("stopAudioLive").disabled=false;renderLiveAudio({});await context.resume();
    }catch(e){stream.getTracks().forEach(t=>t.stop());await context.close();app.audioSocket?.close();app.audioSocket=null;app.audioStream=null;app.audioContext=null;throw e;}
  });
  if(app.audioStream)$("startAudioLive").disabled=true;
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
      $("audioLiveStatus").textContent="正在结束监听";
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
    $("audioLiveStatus").textContent="麦克风未连接";$("startAudioLive").disabled=false;
    for(const id of ["soundLiveState","helpLiveState"]){$(id).textContent="未监听";$(id).className="badge neutral";}
  })();
  try{await app.audioStopping;}finally{app.audioStopping=null;}
}

async function loadReport(overview=false){try{const report=await api(`/api/reports?days=${overview?7:app.reportDays}&mode=${overview?"live":app.reportMode}`);if(overview){app.report=report;if(app.page==="overview")drawBars($("overviewChart"),report.series,"reps","#579579");return;}app.report=report;renderReport();}catch(e){toast(e.message,true);}}
function renderReport(){const r=app.report;if(!r)return;const s=r.summary;$("reportRange").textContent=`${r.series[0].date} 至 ${r.series.at(-1).date} · ${r.mode==="live"?"实时记录":"样例 / 回放"}`;$("reportMetrics").innerHTML=metric("确认跌倒",s.confirmed_falls,"次","shield-alert",`${s.false_alarms} 次误报`)+metric("未处置事件",s.open_events,"条","bell-ring",`${s.unresolved_outcomes} 条结果未确定`)+metric("已完成训练",s.sessions,"次","activity",`${s.participation_days} 天参与`)+metric("有效运动次数",s.reps,"次","check-check","已结束训练累计")+metric("语音已应答",s.voice_answered,"条","speech","问询得到有效回答")+metric("高风险未应答",s.voice_high_risk,"条","shield-alert","问询未取得有效回答",s.voice_high_risk?"metric-risk":"")+metric("未问询",s.voice_not_asked,"条","circle-help","尚未发起或正在问询");renderReportRecords(r);drawBars($("safetyChart"),r.series,"events","#bc8d58");drawBars($("exerciseReportChart"),r.series,"reps","#5d83bb");$("coverageSummary").innerHTML=r.mode==="live"?Object.entries(s.coverage_unavailable_s).map(([ch,n])=>`<div><span class="muted small">${channels[ch]}受限 / 不可用时长</span><strong>${duration(n)}</strong></div>`).join(""):"";$("coverageHistory").innerHTML=r.coverage.length?`<table><thead><tr><th>通道</th><th>开始</th><th>结束</th><th>状态</th><th>原因</th></tr></thead><tbody>${r.coverage.slice(0,30).map(c=>`<tr><td>${channels[c.channel]}</td><td>${dateTime(c.started_at)}</td><td>${c.ended_at?dateTime(c.ended_at):"持续中"}</td><td><span class="badge ${c.state==="active"?"green":"amber"}">${c.state==="active"?"有效":c.state==="reduced"?"受限":"不可用"}</span></td><td>${esc(c.reason)}</td></tr>`).join("")}</tbody></table>`:empty(r.mode==="replay"?"回放不计入实时监测覆盖":"暂无覆盖记录","radio","",true);$("exerciseSummary").innerHTML=`<div class="exercise-summary-grid">${Object.entries(s.reps_by_exercise).map(([key,n])=>`<div>${names[key]}<strong>${n}<small>次</small></strong></div>`).join("")}</div>`;$("exportCsv").href=`/api/reports/export.csv?days=${app.reportDays}&mode=${app.reportMode}`;$("printReport").href=`/reports/print?days=${app.reportDays}&mode=${app.reportMode}`;icons();}
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
function recordSummary(result){return `安全事件 ${result.events||0} 条、运动训练 ${result.sessions||0} 条、通知 ${result.notifications||0} 条、录像片段 ${result.clips||0} 个`;}

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
  hint.textContent=picked.length?`已选择 ${picked.length} 条（安全事件 ${events} · 运动训练 ${sessions}）`:"未选择记录";
  remove.disabled=picked.length===0;
}
function renderReportRecords(report){
  const events=report.events||[],sessions=report.sessions||[];
  const eventRows=events.map(e=>`<tr><td class="pick"><input type="checkbox" class="row-pick" data-kind="event" value="${esc(e.id)}" aria-label="选择安全事件 ${esc(e.id)}"></td><td><span class="badge neutral">安全事件</span></td><td>#${esc(e.id)}</td><td>${dateTime(e.created_at)}</td><td>${modeBadge(e.mode)}</td><td>${channels[e.channel]||esc(e.channel)} · ${e.level==="EMERGENCY"?"紧急求助":"待核实"}</td><td>${voiceBadge(e.voice)}</td><td>${({open:"待处理",resolved:"已处置"})[e.state]||esc(e.state)}</td></tr>`);
  const sessionRows=sessions.map(s=>`<tr><td class="pick"><input type="checkbox" class="row-pick" data-kind="session" value="${esc(s.session_id)}" aria-label="选择运动训练"></td><td><span class="badge blue">运动训练</span></td><td>${esc(String(s.session_id).slice(0,8))}</td><td>${dateTime(s.created_at)}</td><td>${modeBadge(s.mode)}</td><td>${esc(names[s.exercise]||s.exercise)} · ${s.count} / ${s.target_reps} 次</td><td><span class="muted small">不适用</span></td><td>${({running:"进行中",paused:"已暂停",ended:"已结束",interrupted:"已中断"})[s.status]||esc(s.status)}</td></tr>`);
  const total=eventRows.length+sessionRows.length;
  $("reportRecords").innerHTML=total?`<table><thead><tr><th class="pick"><input type="checkbox" class="pick-all" data-scope="all" aria-label="全选报告记录"></th><th>类型</th><th>记录编号</th><th>时间</th><th>来源</th><th>级别 / 动作</th><th>语音问询</th><th>状态</th></tr></thead><tbody>${eventRows.join("")}${sessionRows.join("")}</tbody></table>`:empty("当前报告范围暂无记录","inbox","",true);
  syncReportPicker();
  icons();
}
async function deleteSelectedRecords(){
  const picked=reportPicked();
  if(!picked.length)return;
  const confirmed=await askConfirm("删除选中记录",`将永久删除选中的 ${picked.length} 条记录，以及随事件产生的通知与录像片段。删除后无法恢复，报告统计也会随之变化。`,"永久删除","取消");
  if(!confirmed)return;
  const button=$("deleteSelectedRecords");
  button.disabled=true;
  try{
    const result=await api("/api/records/delete",{method:"POST",body:{event_ids:picked.filter(row=>row.dataset.kind==="event").map(row=>row.value),session_ids:picked.filter(row=>row.dataset.kind==="session").map(row=>row.value)}});
    await askConfirm("删除完成",`已删除：${recordSummary(result)}。报告已按最新记录重新统计。`,"知道了",null);
    await loadReport();await refresh();
  }catch(e){
    button.disabled=false;
    await askConfirm("删除失败",e.message,"知道了",null);
  }
}
async function clearReportScope(){
  const confirmed=await askConfirm("清空报告范围",`将永久删除当前报告范围（${app.reportMode==="live"?"实时记录":"样例与回放"} · 近 ${app.reportDays} 天）内的全部安全事件与运动训练记录，含通知与录像片段。删除后无法恢复。`,"清空并删除","取消");
  if(!confirmed)return;
  const button=$("clearReportScope");
  button.disabled=true;
  try{
    const result=await api("/api/records/clear",{method:"POST",body:{mode:app.reportMode,days:app.reportDays,events:true,sessions:true}});
    await askConfirm("清空完成",`已清空：${recordSummary(result)}。`,"知道了",null);
    await loadReport();await refresh();
  }catch(e){
    button.disabled=false;
    await askConfirm("清空失败",e.message,"知道了",null);
  }
}

function contactEditor(c={name:"",relation:"家属",phone:""}){const div=document.createElement("div");div.className="contact-editor";div.innerHTML=`<label>姓名<input class="contact-name" value="${esc(c.name)}" maxlength="60" required></label><label>关系<input class="contact-relation" value="${esc(c.relation)}" maxlength="30"></label><label>联系电话<input class="contact-phone" type="tel" value="${esc(c.phone)}" maxlength="40" pattern="[+0-9 ()-]+" required></label><button class="icon-button remove-contact" type="button" title="移除联系人" aria-label="移除联系人">${icon("trash-2")}</button>`;$("contactsEditor").append(div);icons();}
function fillSettings(s){app.settings=s;$("residentName").value=s.resident.name;$("residentRoom").value=s.resident.room;$("residentNotes").value=s.resident.notes;$("audioConsent").checked=s.audio_consent;$("escalationSeconds").value=s.escalation_seconds;$("defaultTarget").value=s.exercise_target;$("targetReps").value=s.exercise_target;$("webhookUrl").value=s.webhook_url;$("contactsEditor").innerHTML="";s.contacts.forEach(contactEditor);$("consentStatusText").textContent=s.audio_consent?"已授权":"未授权";$("notificationTestStatus").textContent=s.notification_test_at?`最近测试 ${dateTime(s.notification_test_at)}`:"尚未测试";$("setupBanner").hidden=!!(s.resident.name&&s.contacts.length);$("overviewSubtitle").textContent=s.resident.name?`${s.resident.name} · 今天的安全与活动记录`:"今天的安全与活动记录";$("overviewRoom").textContent=s.resident.room.toUpperCase()+" / HOME CARE";$("cameraRoomLabel").textContent=s.resident.room;}
async function saveSettings(ev){ev.preventDefault();await busy($("saveSettings"),async()=>{const contacts=Array.from(document.querySelectorAll(".contact-editor")).map(el=>({name:el.querySelector(".contact-name").value.trim(),relation:el.querySelector(".contact-relation").value.trim(),phone:el.querySelector(".contact-phone").value.trim()}));const value={resident:{name:$("residentName").value.trim(),room:$("residentRoom").value.trim(),notes:$("residentNotes").value.trim()},contacts,audio_consent:$("audioConsent").checked,exercise_target:Number($("defaultTarget").value),escalation_seconds:Number($("escalationSeconds").value),webhook_url:$("webhookUrl").value.trim()};if(!value.audio_consent)await stopAudioLive();const result=await api("/api/settings",{method:"PUT",body:value});fillSettings(result);toast("设置已保存");await refresh();});}
function renderModules(){$("moduleList").innerHTML=app.status.modules.map(m=>`<div class="module-row"><div><h3>${moduleNames[m.id]}</h3><small>${modelNames[m.id]}</small>${m.error?`<small style="color:var(--red)">${esc(m.error)}</small>`:""}</div><span class="badge ${m.state==="ready"?"green":m.state==="unavailable"?"red":"amber"}">${stateNames[m.state]||m.state}</span><button class="icon-button" type="button" data-init="${m.id}" title="重新初始化模块" aria-label="重新初始化${moduleNames[m.id]}" ${m.state==="loading"?"disabled":""}>${icon("rotate-cw")}</button></div>`).join("");icons();}
async function loadDiagnostics(){try{const d=await api("/api/diagnostics");$("diagnosticsList").innerHTML=d.logs.length?d.logs.slice(0,30).map(x=>`<div class="diagnostic-item"><time>${dateTime(x.at)}</time><span>${esc(x.module)} · ${esc(x.action)}${x.payload.error?`<br>${esc(x.payload.error)}`:""}</span></div>`).join(""):empty("暂无运行记录","list", "",true);}catch(e){toast(e.message,true);}}

function canvasContext(canvas){if(!canvas)return null;const rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return null;const dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);const ctx=canvas.getContext("2d");ctx.scale(dpr,dpr);return {ctx,w:rect.width,h:rect.height};}
function chartBase(canvas,max,labels=[]){const setup=canvasContext(canvas);if(!setup)return null;const {ctx,w,h}=setup;const area={l:35,r:w-10,t:13,b:h-29};ctx.clearRect(0,0,w,h);ctx.font='10px "Segoe UI","Microsoft YaHei",sans-serif';ctx.textBaseline="middle";for(let i=0;i<4;i++){const y=area.t+(area.b-area.t)*i/3;ctx.strokeStyle="#e4ebe6";ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(area.l,y);ctx.lineTo(area.r,y);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle="#9ba99f";ctx.textAlign="right";const val=max*(1-i/3);ctx.fillText(max<=1?val.toFixed(1):String(Math.round(val)),area.l-9,y);}ctx.textAlign="center";if(labels.length){const step=Math.max(1,Math.ceil(labels.length/7));labels.forEach((label,i)=>{if(i%step===0||i===labels.length-1){const x=area.l+(area.r-area.l)*(i+.5)/labels.length;ctx.fillStyle="#98a89c";ctx.fillText(label,x,h-11);}});}return {...setup,area};}
function drawBars(canvas,series,key,color){const max=Math.max(3,...series.map(x=>x[key]||0));const setup=chartBase(canvas,max,series.map(x=>x.date.slice(5).replace("-","/")));if(!setup)return;const {ctx,area}=setup;const slot=(area.r-area.l)/series.length;series.forEach((s,i)=>{const n=s[key]||0;const bar=Math.max(2,n/max*(area.b-area.t));const width=Math.min(27,slot*.44);const x=area.l+slot*(i+.5)-width/2;ctx.fillStyle=n?color:"#e6ece8";ctx.beginPath();ctx.roundRect(x,area.b-bar,width,bar,[3,3,0,0]);ctx.fill();if(n){ctx.fillStyle=color;ctx.font="10px Segoe UI";ctx.fillText(String(n),x+width/2,area.b-bar-9);}});}
function drawLines(canvas,series,options={}){const all=series.flatMap(s=>s.values||[]).filter(Number.isFinite);let min=options.min??Math.min(0,...all),max=options.max??Math.max(1,...all);if(max===min)max=min+1;const setup=chartBase(canvas,max,options.labels||[]);if(!setup)return;const {ctx,area}=setup;const y=v=>area.b-(v-min)/(max-min)*(area.b-area.t);if(options.threshold!==undefined){ctx.strokeStyle="#c39c68";ctx.setLineDash([5,4]);ctx.beginPath();ctx.moveTo(area.l,y(options.threshold));ctx.lineTo(area.r,y(options.threshold));ctx.stroke();ctx.setLineDash([]);}for(const s of series){if(!s.values?.length)continue;ctx.strokeStyle=s.color;ctx.lineWidth=1.7;ctx.beginPath();s.values.forEach((v,i)=>{const x=area.l+(area.r-area.l)*i/Math.max(1,s.values.length-1);if(i===0)ctx.moveTo(x,y(v));else ctx.lineTo(x,y(v));});ctx.stroke();}if(!all.length){ctx.textAlign="center";ctx.fillStyle="#9aaba0";ctx.font='11px "Microsoft YaHei"';ctx.fillText(window.HomeEaseI18n?window.HomeEaseI18n.t("\u7b49\u5f85\u6570\u636e"):"等待数据",(area.l+area.r)/2,(area.t+area.b)/2);}}
function drawPose(canvas,points,background=true){const setup=canvasContext(canvas);if(!setup)return;const {ctx,w,h}=setup;ctx.clearRect(0,0,w,h);if(background){ctx.fillStyle="#eaf1ed";ctx.fillRect(0,0,w,h);ctx.strokeStyle="#dbe6de";ctx.lineWidth=1;for(let x=0;x<w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=0;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}}if(!points)return;const visible=points.filter(p=>p[2]>=.3);if(!visible.length)return;let minX,maxX,minY,maxY;if(background){minX=Math.min(...visible.map(p=>p[0]))-80;maxX=Math.max(...visible.map(p=>p[0]))+80;minY=Math.min(...visible.map(p=>p[1]))-60;maxY=Math.max(...visible.map(p=>p[1]))+60;}else{minX=0;maxX=app.exercise?.frame_shape?.[1]||640;minY=0;maxY=app.exercise?.frame_shape?.[0]||480;}const scale=Math.min(w/(maxX-minX),h/(maxY-minY));const dx=(w-(maxX-minX)*scale)/2-minX*scale,dy=(h-(maxY-minY)*scale)/2-minY*scale;const point=p=>[p[0]*scale+dx,p[1]*scale+dy];const edges=[[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],[11,13],[13,15],[12,14],[14,16]];ctx.lineWidth=background?5:3;ctx.lineCap="round";for(const [a,b] of edges){if(points[a]?.[2]<.3||points[b]?.[2]<.3)continue;ctx.strokeStyle=a%2?"#338567":"#5a84ba";ctx.beginPath();ctx.moveTo(...point(points[a]));ctx.lineTo(...point(points[b]));ctx.stroke();}for(const p of visible){ctx.fillStyle="#f9fcfa";ctx.strokeStyle="#4b876f";ctx.lineWidth=2;ctx.beginPath();ctx.arc(...point(p),background?5:3.5,0,Math.PI*2);ctx.fill();ctx.stroke();}}

document.addEventListener("DOMContentLoaded",async()=>{
  icons();$("headerDate").textContent=new Date().toLocaleDateString("zh-CN",{month:"long",day:"numeric",weekday:"long"});
  $("menuButton").onclick=()=>{$("sidebar").classList.toggle("open");$("navBackdrop").hidden=!$("sidebar").classList.contains("open");};$("navBackdrop").onclick=()=>{$("sidebar").classList.remove("open");$("navBackdrop").hidden=true;};
  window.addEventListener("hashchange",navigate);document.querySelectorAll("[data-close]").forEach(b=>b.onclick=()=>$(b.dataset.close).close());document.querySelectorAll("dialog").forEach(d=>d.addEventListener("click",e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}}));
  document.addEventListener("click",async ev=>{const eventEl=ev.target.closest("[data-event]");if(eventEl)openEvent(eventEl.dataset.event);const action=ev.target.closest("[data-action]");if(action)busy(action,()=>eventAction(action.dataset.action));const contact=ev.target.closest("[data-contact-call]");if(contact&&app.selectedEvent?.state!=="resolved")api(`/api/events/${app.selectedEvent.id}/action`,{method:"POST",body:{action:"contact",note:"已发起电话联系"}}).catch(e=>toast(e.message,true));const init=ev.target.closest("[data-init]");if(init)busy(init,async()=>{await api(`/api/modules/${init.dataset.init}/initialize`,{method:"POST"});toast("模块正在重新初始化");await refresh();});const remove=ev.target.closest(".remove-contact");if(remove)remove.closest(".contact-editor").remove();});document.addEventListener("keydown",ev=>{if(["Enter"," "].includes(ev.key)&&ev.target.matches("[data-event]")){ev.preventDefault();openEvent(ev.target.dataset.event);}});
  $("notificationButton").onclick=async()=>{await refresh();$("notificationList").innerHTML=app.notices.length?app.notices.slice(0,30).map(n=>`<div class="notification-row" data-notice="${n.id}" ${n.event_id?`data-event="${n.event_id}"`:""}><strong>${esc(n.title)}</strong><p>${esc(n.message)}</p><footer><span>${dateTime(n.created_at)}</span><span>${n.kind==="webhook"?"Webhook":"站内"} · ${({queued:"排队中",delivered:"已送达",seen:"已查看",retry:"重试中",failed:"失败"})[n.status]||n.status}</span></footer></div>`).join(""):empty("暂无通知","bell");$("notificationsDialog").showModal();$("notificationList").querySelectorAll("[data-notice]").forEach(el=>el.addEventListener("click",()=>{api(`/api/notifications/${el.dataset.notice}/seen`,{method:"POST"});$("notificationsDialog").close();}));icons();};
  $("videoSource").onchange=()=>{$("cameraIndexField").hidden=$("videoSource").value!=="camera";$("videoUploadField").hidden=$("videoSource").value!=="upload";};$("videoForm").onsubmit=ev=>{ev.preventDefault();busy($("startVideo"),async()=>{let upload_id=null;if($("videoSource").value==="upload"){const file=$("videoUpload").files[0];if(!file)throw new Error("请选择视频文件");const form=new FormData();form.append("file",file);$("videoUploadStatus").textContent="正在上传视频…";try{const result=await api("/api/uploads/video",{method:"POST",body:form});upload_id=result.upload_id;$("videoUploadStatus").textContent=result.filename;}catch(e){$("videoUploadStatus").textContent="上传失败";throw e;}}app.videoHistory=[];await api("/api/video/start",{method:"POST",body:{source:$("videoSource").value,camera_index:Number($("videoForm").elements.camera_index.value),upload_id}});await refresh();toast("检测已启动");});};$("stopVideo").onclick=()=>busy($("stopVideo"),async()=>{await api("/api/video/stop",{method:"POST"});await refresh();});
  restoreOverlay();$("toggleBoxes").onclick=()=>setOverlay("box",!app.overlay.box);$("toggleSkeleton").onclick=()=>setOverlay("skeleton",!app.overlay.skeleton);setOverlay("box",app.overlay.box);setOverlay("skeleton",app.overlay.skeleton);
  $("signalSource").onchange=()=>$("signalUploadField").hidden=$("signalSource").value!=="upload";$("analyzeSignal").onclick=()=>busy($("analyzeSignal"),async()=>{let result;if($("signalSource").value==="sample")result=await api("/api/signal/sample",{method:"POST"});else{const file=$("signalFile").files[0];if(!file)throw new Error("请选择 CSV 文件");const form=new FormData();form.append("file",file);result=await api("/api/signal/analyze",{method:"POST",body:form});}drawLines($("signalChart"),[{values:result.waveform.ax,color:"#547bb4"},{values:result.waveform.ay,color:"#5a9877"},{values:result.waveform.az,color:"#bf9459"}],{labels:result.waveform.t.map(t=>`${t.toFixed(1)}s`)});drawLines($("signalProbabilityChart"),[{values:result.info.probs,color:"#52886d"}],{min:0,max:1,threshold:.5,labels:result.info.t_end.map(t=>`${t.toFixed(1)}s`)});$("signalState").textContent="回放分析完成";$("signalResult").innerHTML=`${result.info.n_windows} 个窗口 · ${result.records.length} 个疑似跌倒事件 ${result.records.map(e=>`<button class="text-action" data-event="${e.id}">查看 #${e.id.slice(0,6)} ${icon("arrow-right")}</button>`).join(" ")}`;await refresh();icons();});
  $("exerciseOptions").innerHTML=Object.entries(names).map(([key,name],index)=>`<button type="button" class="exercise-option ${index===0?"selected":""}" data-exercise="${key}" aria-pressed="${index===0}"><img src="/static/media/${key}-pose.png" alt="${name}骨架动作"><span class="selection-mark">${icon("check")}</span><div><strong>${esc(exerciseGuides[key].title)}</strong><small>${["坐姿 / 站立完整周期","下蹲 / 起立完整周期","抬腿 / 回落完整周期","跨步 / 复位完整周期"][index]}</small></div></button>`).join("");document.querySelectorAll("[data-exercise]").forEach(b=>b.onclick=()=>chooseExercise(b.dataset.exercise));$("exerciseSource").onchange=()=>{$("exerciseUploadField").hidden=$("exerciseSource").value!=="upload";$("exerciseCameraField").hidden=$("exerciseSource").value!=="camera";};$("exerciseForm").onsubmit=startExercise;
  chooseExercise(app.selectedExercise);
  $("pauseExercise").onclick=async()=>{await busy($("pauseExercise"),()=>controlExercise(app.exercise?.status==="paused"?"resume":"pause"));syncExerciseControls();};$("endExercise").onclick=async()=>{await busy($("endExercise"),()=>controlExercise("end"));syncExerciseControls();};$("newExercise").onclick=()=>{app.sid=null;app.exercise=null;$("exerciseActive").hidden=true;$("exerciseChooser").hidden=false;};
  $("audioSource").onchange=()=>{const source=$("audioSource").value;$("audioSampleField").hidden=source!=="sample";$("audioFileField").hidden=source!=="upload";$("audioRecordField").hidden=source!=="record";$("audioPreview").removeAttribute("src");if(source==="sample")$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;};$("audioSampleSelect").onchange=()=>$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;$("audioFile").onchange=()=>{const file=$("audioFile").files[0];if(file)$("audioPreview").src=URL.createObjectURL(file);};$("audioForm").onsubmit=analyzeAudio;$("recordAudio").onclick=recordAudio;$("startAudioLive").onclick=startAudioLive;$("stopAudioLive").onclick=stopAudioLive;
  $("eventStateFilter").querySelectorAll("button").forEach(b=>b.onclick=()=>{app.eventFilter=b.dataset.value;$("eventStateFilter").querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));renderEvents();});$("eventModeFilter").onchange=()=>{app.eventMode=$("eventModeFilter").value;renderEvents();};$("eventSearch").oninput=renderEvents;$("reportDays").querySelectorAll("button").forEach(b=>b.onclick=()=>{app.reportDays=Number(b.dataset.value);$("reportDays").querySelectorAll("button").forEach(x=>x.classList.toggle("active",x===b));loadReport();});$("reportMode").onchange=()=>{app.reportMode=$("reportMode").value;loadReport();};
  $("selectAllRecords").onchange=()=>{const checked=$("selectAllRecords").checked;reportRows().forEach(row=>{row.checked=checked;});syncReportPicker();};$("reportRecords").addEventListener("change",event=>{const box=event.target.closest(".pick-all");if(box){const all=reportRows();(box.dataset.scope==="all"?all:all.filter(row=>row.dataset.kind===box.dataset.scope)).forEach(row=>{row.checked=box.checked;});}if(box||event.target.closest(".row-pick"))syncReportPicker();});$("deleteSelectedRecords").onclick=deleteSelectedRecords;$("clearReportScope").onclick=clearReportScope;
  $("settingsForm").onsubmit=saveSettings;$("addContact").onclick=()=>{if(document.querySelectorAll(".contact-editor").length>=10){toast("最多添加 10 位联系人",true);return;}contactEditor();};$("audioConsent").onchange=()=>$("consentStatusText").textContent=$("audioConsent").checked?"待保存授权":"待保存撤回";$("testNotification").onclick=()=>busy($("testNotification"),async()=>{await api("/api/notifications/test",{method:"POST"});if("Notification" in window&&Notification.permission==="default")await Notification.requestPermission();toast("测试通知已加入通知中心");app.settings=await api("/api/settings");$("notificationTestStatus").textContent=`最近测试 ${dateTime(app.settings.notification_test_at)}`;await refresh();});
  try{fillSettings(await api("/api/settings"));const samples=await api("/api/audio/samples");$("audioSampleSelect").innerHTML=[...(samples.scenes||[]),...(samples.replies||[])].map(s=>`<option value="${s.id}">${esc(s.title)}</option>`).join("");$("audioPreview").src=`/api/audio/samples/${$("audioSampleSelect").value}`;}catch(e){toast(e.message,true);}navigate();await refresh();icons();setInterval(refresh,2500);setInterval(pollExercise,600);setInterval(loadPose,500);window.addEventListener("resize",()=>{if(app.page==="overview"&&app.report)drawBars($("overviewChart"),app.report.series,"reps","#579579");if(app.page==="monitor")renderMonitor();if(app.page==="reports")renderReport();if(app.exercise)drawPose($("poseCanvas"),app.exercise.keypoints,app.exercise.mode!=="live");});
  window.addEventListener("pagehide",()=>{app.camera?.getTracks().forEach(t=>t.stop());app.audioStream?.getTracks().forEach(t=>t.stop());app.recordingStream?.getTracks().forEach(t=>t.stop());if(app.sid&&["running","paused"].includes(app.exercise?.status)&&app.exerciseSource==="camera")fetch(`/api/sessions/${app.sid}/control`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({action:"end"}),keepalive:true});});
});
