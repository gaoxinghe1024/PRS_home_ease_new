"use strict";
(() => {
const $=id=>document.getElementById(id);
const homeMode=document.body.dataset.household==="true";
const I=window.HomeEaseI18n;
function eventText(e){
  const urgent=e.level==="EMERGENCY",replay=e.mode!=="live";
  if(I&&I.lang()==="en")return `${e.room}: ${urgent?"an emergency call for help":"a suspected fall waiting for verification"} was recorded. ${e.withdrawn?"It was withdrawn; waiting for a caregiver to confirm.":e.acknowledged_at?"A caregiver has taken over.":"Waiting for a caregiver to verify."}${replay?" This is a sample / replay event, used to test the voice inquiry.":""}`;
  return `${e.room}\u6709\u4e00\u6761${urgent?"\u7d27\u6025\u6c42\u52a9":"\u5f85\u6838\u5b9e\u7684\u8dcc\u5012"}\u8bb0\u5f55\u3002${e.withdrawn?"\u5df2\u64a4\u56de\uff0c\u7b49\u5f85\u7167\u62a4\u8005\u786e\u8ba4\u3002":e.acknowledged_at?"\u7167\u62a4\u8005\u5df2\u63a5\u624b\u5904\u7406\u3002":"\u7b49\u5f85\u7167\u62a4\u8005\u6838\u5b9e\u3002"}${replay?"\u8fd9\u662f\u6837\u4f8b / \u56de\u653e\u4e8b\u4ef6\uff0c\u7528\u4e8e\u6d4b\u8bd5\u8bed\u97f3\u95ee\u8be2\u3002":""}`;
}
let withdrawUntil=0,withdrawTimer=null;
function showWithdraw(e){
  const button=$("elderWithdraw"),hint=$("elderWithdrawHint");
  clearInterval(withdrawTimer);
  if(!e||!e.withdrawable){button.hidden=true;hint.hidden=true;return;}
  withdrawUntil=Date.now()+e.withdraw_left_s*1000;
  const paint=()=>{const left=Math.max(0,Math.ceil((withdrawUntil-Date.now())/1000));if(!left){button.hidden=true;hint.hidden=true;clearInterval(withdrawTimer);return;}$("elderWithdrawText").textContent=`\u64a4\u56de\u6c42\u52a9\uff08${left} \u79d2\uff09`;};
  button.hidden=false;hint.hidden=false;paint();withdrawTimer=setInterval(paint,1000);
}
async function withdraw(){
  if(!currentEvent)return;
  const button=$("elderWithdraw");button.disabled=true;
  try{await api(`/api/events/${currentEvent.id}/action`,{method:"POST",body:{action:"withdraw"}});$("elderInquiryResult").textContent="\u5df2\u64a4\u56de\u6c42\u52a9\uff0c\u7b49\u5f85\u7167\u62a4\u8005\u786e\u8ba4\u5e76\u5173\u95ed\u3002";$("elderInquiryResult").hidden=false;await refresh();}
  catch(e){toast(e.message);}
  finally{button.disabled=false;}
}
const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
let currentEvent=null,settings=null,recording=false,stream=null,recorder=null,timer=null;
let voiceEnabled=false;
const promptedEvents=new Set();
document.addEventListener("DOMContentLoaded",()=>{
  if(homeMode){
    window.addEventListener("homeease:voice",event=>{voiceEnabled=event.detail.enabled;});
    return;
  }
  document.querySelectorAll('[data-lucide="house-heart"]').forEach(el=>el.dataset.lucide="house");
  const control=document.createElement("label");
  control.className="toggle-row elder-voice-control";
  control.innerHTML='<span><strong>接收语音问询</strong><small id="voiceReadyState">未开启</small></span><input id="enableVoiceInquiry" type="checkbox" class="switch" role="switch">';
  $("elderStatus").after(control);
  $("enableVoiceInquiry").onchange=async()=>{
    const toggle=$("enableVoiceInquiry");
    if(!toggle.checked){voiceEnabled=false;$("voiceReadyState").textContent="未开启";return;}
    try{
      if(!settings?.audio_consent)throw new Error("请先由照护者记录音频授权");
      if(!navigator.mediaDevices?.getUserMedia)throw new Error("麦克风需要 localhost 或 HTTPS 访问");
      const permissionStream=await navigator.mediaDevices.getUserMedia({audio:true});
      permissionStream.getTracks().forEach(t=>t.stop());
      const en=I&&I.lang()==="en",ready=new SpeechSynthesisUtterance(en?"Voice inquiry is on":"\u8bed\u97f3\u95ee\u8be2\u5df2\u5f00\u542f");ready.lang=en?"en-US":"zh-CN";speechSynthesis.speak(ready);
      voiceEnabled=true;$("voiceReadyState").textContent="已开启";
      await refresh();
    }catch(error){toggle.checked=false;voiceEnabled=false;toast(error.message);}
  };
});
async function api(url,options={}){if(options.body&&!(options.body instanceof FormData)){options.headers={"Content-Type":"application/json"};options.body=JSON.stringify(options.body);}const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw new Error(typeof(data.error||data.detail)==="string"?(data.error||data.detail):"操作失败，请重试");return data;}
function toast(message){$("toast").textContent=message;$("toast").hidden=false;setTimeout(()=>$("toast").hidden=true,6500);}
async function refresh(){
  try{
    const [status,profile,events,sessions]=await Promise.all([api("/api/status"),api("/api/settings"),api("/api/events?state=open"),api("/api/sessions?mode=live")]);
    settings=profile;$("elderGreeting").textContent=profile.resident.name?`${profile.resident.name}，您好`:"您好";
    if(!homeMode){
      const active=status.coverage_state==="active";
      $("elderStatus").classList.toggle("active",active);
      $("elderStatus").innerHTML=`<i data-lucide="${active?"shield-check":"shield-alert"}"></i><div><strong>${active?"居家监测已开启":status.coverage_state==="reduced"?"部分监测暂不可用":"居家监测尚未开启"}</strong><p>${active?`${esc(profile.resident.room)} · 输入有效`:status.coverage.filter(x=>x.channel!=="audio"&&x.state!=="active").map(x=>esc(x.reason)).join("；")}</p></div>`;
    }
    const query=new URLSearchParams(location.search).get("event");
    const live=events.filter(e=>e.mode==="live");
    if(homeMode)live.sort((a,b)=>Number(b.level==="EMERGENCY")-Number(a.level==="EMERGENCY"));
    const e=query?events.find(e=>e.id===query):(live[0]||(!homeMode&&events[0]));
    if(!recording){
      currentEvent=e||null;$("elderEvent").hidden=!e;
      if(e){$("elderEventMode").textContent=e.mode==="live"?"实时事件":"样例 / 回放";$("elderEventText").textContent=eventText(e);$("elderSafe").disabled=e.inquiry.status==="button_safe";$("elderInquiry").disabled=!profile.audio_consent||e.inquiry.status==="listening"||e.inquiry.status==="processing";}
      showWithdraw(e||null);
    }
    const today=new Date().toLocaleDateString("en-CA");
    $("elderExerciseCount").textContent=`${sessions.filter(s=>s.status==="ended"&&new Date(s.created_at).toLocaleDateString("en-CA")===today).length} 次训练`;
    $("elderContacts").innerHTML=profile.contacts.map(c=>`<div class="elder-contact"><strong>${esc(c.name)}</strong><small>${esc(c.relation)}</small><a class="button secondary" href="tel:${esc(c.phone.replace(/[^+0-9]/g,""))}"><i data-lucide="phone"></i>${esc(c.phone)}</a></div>`).join("");
    window.lucide?.createIcons();
  }catch(error){
    currentEvent=null;
    if(!homeMode)$("elderStatus").innerHTML='<i data-lucide="wifi-off"></i><div><strong>照护系统连接中断</strong><p>请直接联系家人</p></div>';
    window.lucide?.createIcons();
  }
}
const refreshStatus=refresh;
refresh=async function(){
  await refreshStatus();
  if(settings&&!settings.audio_consent){
    voiceEnabled=false;
    const toggle=$("enableVoiceInquiry");if(toggle)toggle.checked=false;
    if(recording){clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());if(recorder?.state==="recording")recorder.stop();}
  }
  if(voiceEnabled&&!recording&&currentEvent?.inquiry.status==="not_started"&&!promptedEvents.has(currentEvent.id)){
    promptedEvents.add(currentEvent.id);
    inquiry();
  }
};
function promptText(){return I&&I.lang()==="en"?"What's wrong? Are you OK? Do you need help?":"\u8bf7\u95ee\u4f60\u600e\u4e48\u5566\uff1f\u4f60\u9700\u8981\u5e2e\u52a9\u5417\uff1f";}
function speak(){return new Promise((resolve,reject)=>{
  if(!("speechSynthesis" in window))return reject(new Error("\u5f53\u524d\u6d4f\u89c8\u5668\u65e0\u6cd5\u64ad\u653e\u8be2\u95ee"));
  speechSynthesis.cancel();
  const en=I&&I.lang()==="en",phrase=new SpeechSynthesisUtterance(promptText());phrase.lang=en?"en-US":"zh-CN";
  const voice=speechSynthesis.getVoices().find(v=>v.lang&&v.lang.toLowerCase().startsWith(en?"en":"zh"));if(voice)phrase.voice=voice;
  const timer=setTimeout(()=>{speechSynthesis.cancel();reject(new Error("询问语音没有成功播放"));},30000);
  phrase.rate=.85;phrase.onend=()=>{clearTimeout(timer);resolve();};phrase.onerror=()=>{clearTimeout(timer);reject(new Error("询问语音没有成功播放"));};
  speechSynthesis.speak(phrase);
});}
async function unavailable(id,reason){try{await api(`/api/events/${id}/inquiry/unavailable`,{method:"POST",body:{reason}});}catch{}}
async function inquiry(){
  if(!currentEvent||recording)return;
  if(!settings.audio_consent){toast("尚未获得音频处理授权");return;}
  const id=currentEvent.id,mode=currentEvent.mode;
  recording=true;$("elderInquiry").disabled=true;$("inquiryProgress").hidden=false;$("elderInquiryResult").hidden=true;$("inquiryStatus").textContent="准备麦克风";
  let started=false;
  const promptState=speaking=>{if(homeMode)window.dispatchEvent(new CustomEvent("homeease:prompt",{detail:{speaking}}));};
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error("麦克风需要 localhost 或 HTTPS 访问");
    stream=homeMode&&window.HomeMonitoring?.stream()?window.HomeMonitoring.stream().clone():await navigator.mediaDevices.getUserMedia({audio:true});
    $("inquiryStatus").textContent="请听语音询问";$("inquiryQuestion").textContent=promptText();$("inquiryQuestion").hidden=false;
    promptState(true);
    await speak();
    if(!settings.audio_consent)throw new Error("音频授权已撤回");
    await api(`/api/events/${id}/inquiry`,{method:"POST",body:{prompt_played:true}});started=true;
    promptState(false);
    const parts=[];recorder=new MediaRecorder(stream);
    recorder.ondataavailable=e=>{if(e.data.size)parts.push(e.data);};
    const audio=await new Promise((resolve,reject)=>{
      recorder.onerror=()=>reject(new Error("麦克风录音失败"));recorder.onstop=()=>resolve(new Blob(parts,{type:recorder.mimeType}));recorder.start();
      const began=Date.now();
      $("inquiryStatus").textContent="正在听您的回答";$("inquiryCountdown").textContent="10 秒";$("inquiryBar").value=0;
      timer=setInterval(()=>{
        const seconds=(Date.now()-began)/1000;
        $("inquiryCountdown").textContent=`${Math.max(0,Math.ceil(10-seconds))} 秒`;$("inquiryBar").value=seconds;
        if(seconds>=10){clearInterval(timer);if(recorder.state==="recording")recorder.stop();}
      },250);
    });
    stream.getTracks().forEach(t=>t.stop());stream=null;
    $("inquiryStatus").textContent="正在识别回答";$("inquiryCountdown").textContent="";
    await api(`/api/events/${id}/inquiry/processing`,{method:"POST"});
    const form=new FormData();form.append("file",audio,"reply.webm");form.append("task","reply");
    if(!settings.audio_consent)throw new Error("音频授权已撤回");
    form.append("mode",mode);form.append("event_id",id);
    const result=await api("/api/audio/analyze",{method:"POST",body:form});const outcome=result.event.inquiry.result;
    $("elderInquiryResult").textContent=outcome.escalate?"求助已升级，等待照护者响应。":outcome.cancelled?"已记录您的回应。警报仍然有效，需要照护者核实并关闭。":outcome.state==="audio_unavailable"?"语音识别暂不可用，请使用求助按钮。":"已记录您的回应，等待照护者响应。";
    $("elderInquiryResult").hidden=false;
  }catch(error){
    toast(error.message);
    if(started)await unavailable(id,error.message);
    else{try{await api(`/api/events/${id}/inquiry`,{method:"POST",body:{prompt_played:false,reason:error.message}});}catch{}}
    $("elderInquiryResult").textContent="语音问询不可用，请使用下方求助按钮。";$("elderInquiryResult").hidden=false;
  }finally{
    promptState(false);clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());stream=null;recording=false;
    $("inquiryProgress").hidden=true;$("inquiryQuestion").hidden=true;$("elderInquiry").disabled=false;await refresh();
  }
}
document.addEventListener("DOMContentLoaded",()=>{
  $("elderDate").textContent=new Date().toLocaleDateString("zh-CN",{month:"long",day:"numeric",weekday:"long"});
  $("elderHelp").onclick=async()=>{
    $("elderHelp").disabled=true;
    try{await api("/api/help",{method:"POST"});$("helpConfirm").hidden=false;$("elderHelpSub").textContent="求助已发出";await refresh();}
    catch(error){toast(error.message);}finally{$("elderHelp").disabled=false;}
  };
  $("elderSafe").onclick=async()=>{
    if(!currentEvent)return;
    try{await api(`/api/events/${currentEvent.id}/action`,{method:"POST",body:{action:"safe"}});$("elderInquiryResult").textContent="已记录您表示安全。警报仍然有效，需要照护者核实并关闭。";$("elderInquiryResult").hidden=false;await refresh();}
    catch(error){toast(error.message);}
  };
  $("elderInquiry").onclick=inquiry;$("elderWithdraw").onclick=withdraw;
  refresh();setInterval(refresh,2500);window.lucide?.createIcons();
  window.addEventListener("pagehide",()=>{
    clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());
    if(recording&&currentEvent)fetch(`/api/events/${currentEvent.id}/inquiry/unavailable`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reason:"长者端已关闭"}),keepalive:true});
    window.speechSynthesis?.cancel();
  });
});
})();
