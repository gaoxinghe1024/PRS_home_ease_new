"use strict";
(() => {
const $=id=>document.getElementById(id);
const homeMode=document.body.dataset.household==="true";
function eventText(e){
  const urgent=e.level==="EMERGENCY",replay=e.mode!=="live";
  return `${e.room}: ${urgent?"an emergency call for help":"a suspected fall waiting for verification"} was recorded. ${e.withdrawn?"It was withdrawn; waiting for a caregiver to confirm.":e.acknowledged_at?"A caregiver has taken over.":"Waiting for a caregiver to verify."}${replay?" This is a sample / replay event, used to test the voice inquiry.":""}`;
}
let withdrawUntil=0,withdrawTimer=null;
function showWithdraw(e){
  const button=$("elderWithdraw"),hint=$("elderWithdrawHint");
  clearInterval(withdrawTimer);
  if(!e||!e.withdrawable){button.hidden=true;hint.hidden=true;return;}
  withdrawUntil=Date.now()+e.withdraw_left_s*1000;
  const paint=()=>{const left=Math.max(0,Math.ceil((withdrawUntil-Date.now())/1000));if(!left){button.hidden=true;hint.hidden=true;clearInterval(withdrawTimer);return;}$("elderWithdrawText").textContent=`Withdraw call for help (${left} s)`;};
  button.hidden=false;hint.hidden=false;paint();withdrawTimer=setInterval(paint,1000);
}
async function withdraw(){
  if(!currentEvent)return;
  const button=$("elderWithdraw");button.disabled=true;
  try{await api(`/api/events/${currentEvent.id}/action`,{method:"POST",body:{action:"withdraw"}});$("elderInquiryResult").textContent="Your call for help was withdrawn. A caregiver will confirm and close it.";$("elderInquiryResult").hidden=false;await refresh();}
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
  control.innerHTML='<span><strong>Receive voice inquiries</strong><small id="voiceReadyState">Off</small></span><input id="enableVoiceInquiry" type="checkbox" class="switch" role="switch">';
  $("elderStatus").after(control);
  $("enableVoiceInquiry").onchange=async()=>{
    const toggle=$("enableVoiceInquiry");
    if(!toggle.checked){voiceEnabled=false;$("voiceReadyState").textContent="Off";return;}
    try{
      if(!settings?.audio_consent)throw new Error("A caregiver must record the audio consent first");
      if(!navigator.mediaDevices?.getUserMedia)throw new Error("The microphone needs localhost or HTTPS access");
      const permissionStream=await navigator.mediaDevices.getUserMedia({audio:true});
      permissionStream.getTracks().forEach(t=>t.stop());
      const ready=new SpeechSynthesisUtterance("Voice inquiry is on");ready.lang="en-US";speechSynthesis.speak(ready);
      voiceEnabled=true;$("voiceReadyState").textContent="On";
      await refresh();
    }catch(error){toggle.checked=false;voiceEnabled=false;toast(error.message);}
  };
});
async function api(url,options={}){if(options.body&&!(options.body instanceof FormData)){options.headers={"Content-Type":"application/json"};options.body=JSON.stringify(options.body);}const response=await fetch(url,options);const data=await response.json();if(!response.ok)throw new Error(typeof(data.error||data.detail)==="string"?(data.error||data.detail):"The operation failed. Please try again.");return data;}
function toast(message){$("toast").textContent=message;$("toast").hidden=false;setTimeout(()=>$("toast").hidden=true,6500);}
async function refresh(){
  try{
    const [status,profile,events,sessions]=await Promise.all([api("/api/status"),api("/api/settings"),api("/api/events?state=open"),api("/api/sessions?mode=live")]);
    settings=profile;$("elderGreeting").textContent=profile.resident.name?`${profile.resident.name}, hello`:"Hello";
    if(!homeMode){
      const active=status.coverage_state==="active";
      $("elderStatus").classList.toggle("active",active);
      $("elderStatus").innerHTML=`<i data-lucide="${active?"shield-check":"shield-alert"}"></i><div><strong>${active?"Home monitoring is on":status.coverage_state==="reduced"?"Some monitoring is unavailable":"Home monitoring is not on yet"}</strong><p>${active?`${esc(profile.resident.room)} · Input valid`:status.coverage.filter(x=>x.channel!=="audio"&&x.state!=="active").map(x=>esc(x.reason)).join("; ")}</p></div>`;
    }
    const query=new URLSearchParams(location.search).get("event");
    const live=events.filter(e=>e.mode==="live");
    if(homeMode)live.sort((a,b)=>Number(b.level==="EMERGENCY")-Number(a.level==="EMERGENCY"));
    const e=query?events.find(e=>e.id===query):(live[0]||(!homeMode&&events[0]));
    if(!recording){
      currentEvent=e||null;$("elderEvent").hidden=!e;
      if(e){$("elderEventMode").textContent=e.mode==="live"?"Live event":"Sample / replay";$("elderEventText").textContent=eventText(e);$("elderSafe").disabled=e.inquiry.status==="button_safe";$("elderInquiry").disabled=!profile.audio_consent||e.inquiry.status==="listening"||e.inquiry.status==="processing";}
      showWithdraw(e||null);
    }
    const today=new Date().toLocaleDateString("en-CA");
    $("elderExerciseCount").textContent=`${sessions.filter(s=>s.status==="ended"&&new Date(s.created_at).toLocaleDateString("en-CA")===today).length} sessions`;
    $("elderContacts").innerHTML=profile.contacts.map(c=>`<div class="elder-contact"><strong>${esc(c.name)}</strong><small>${esc(c.relation)}</small><a class="button secondary" href="tel:${esc(c.phone.replace(/[^+0-9]/g,""))}"><i data-lucide="phone"></i>${esc(c.phone)}</a></div>`).join("");
    window.lucide?.createIcons();
  }catch(error){
    currentEvent=null;
    if(!homeMode)$("elderStatus").innerHTML='<i data-lucide="wifi-off"></i><div><strong>Connection to the care system lost</strong><p>Please contact your family directly</p></div>';
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
function promptText(){return "What's wrong? Are you OK? Do you need help?";}
function speak(){return new Promise((resolve,reject)=>{
  if(!("speechSynthesis" in window))return reject(new Error("This browser cannot play the question"));
  speechSynthesis.cancel();
  const phrase=new SpeechSynthesisUtterance(promptText());phrase.lang="en-US";
  const voice=speechSynthesis.getVoices().find(v=>v.lang&&v.lang.toLowerCase().startsWith("en"));if(voice)phrase.voice=voice;
  const timer=setTimeout(()=>{speechSynthesis.cancel();reject(new Error("The question audio did not play"));},30000);
  phrase.rate=.85;phrase.onend=()=>{clearTimeout(timer);resolve();};phrase.onerror=()=>{clearTimeout(timer);reject(new Error("The question audio did not play"));};
  speechSynthesis.speak(phrase);
});}
async function unavailable(id,reason){try{await api(`/api/events/${id}/inquiry/unavailable`,{method:"POST",body:{reason}});}catch{}}
async function inquiry(){
  if(!currentEvent||recording)return;
  if(!settings.audio_consent){toast("Audio-processing consent has not been given yet");return;}
  const id=currentEvent.id,mode=currentEvent.mode;
  recording=true;$("elderInquiry").disabled=true;$("inquiryProgress").hidden=false;$("elderInquiryResult").hidden=true;$("inquiryStatus").textContent="Preparing the microphone";
  let started=false;
  const promptState=speaking=>{if(homeMode)window.dispatchEvent(new CustomEvent("homeease:prompt",{detail:{speaking}}));};
  try{
    if(!navigator.mediaDevices?.getUserMedia)throw new Error("The microphone needs localhost or HTTPS access");
    stream=homeMode&&window.HomeMonitoring?.stream()?window.HomeMonitoring.stream().clone():await navigator.mediaDevices.getUserMedia({audio:true});
    $("inquiryStatus").textContent="Please listen to the voice question";$("inquiryQuestion").textContent=promptText();$("inquiryQuestion").hidden=false;
    promptState(true);
    await speak();
    if(!settings.audio_consent)throw new Error("Audio consent withdrawn");
    await api(`/api/events/${id}/inquiry`,{method:"POST",body:{prompt_played:true}});started=true;
    promptState(false);
    const parts=[];recorder=new MediaRecorder(stream);
    recorder.ondataavailable=e=>{if(e.data.size)parts.push(e.data);};
    const audio=await new Promise((resolve,reject)=>{
      recorder.onerror=()=>reject(new Error("Microphone recording failed"));recorder.onstop=()=>resolve(new Blob(parts,{type:recorder.mimeType}));recorder.start();
      const began=Date.now();
      $("inquiryStatus").textContent="Listening for your answer";$("inquiryCountdown").textContent="10 s";$("inquiryBar").value=0;
      timer=setInterval(()=>{
        const seconds=(Date.now()-began)/1000;
        $("inquiryCountdown").textContent=`${Math.max(0,Math.ceil(10-seconds))} s`;$("inquiryBar").value=seconds;
        if(seconds>=10){clearInterval(timer);if(recorder.state==="recording")recorder.stop();}
      },250);
    });
    stream.getTracks().forEach(t=>t.stop());stream=null;
    $("inquiryStatus").textContent="Recognizing your answer";$("inquiryCountdown").textContent="";
    await api(`/api/events/${id}/inquiry/processing`,{method:"POST"});
    const form=new FormData();form.append("file",audio,"reply.webm");form.append("task","reply");
    if(!settings.audio_consent)throw new Error("Audio consent withdrawn");
    form.append("mode",mode);form.append("event_id",id);
    const result=await api("/api/audio/analyze",{method:"POST",body:form});const outcome=result.event.inquiry.result;
    $("elderInquiryResult").textContent=outcome.escalate?"The call for help was escalated. Waiting for a caregiver to respond.":outcome.cancelled?"Your reply was recorded. The alarm stays active until a caregiver verifies and closes it.":outcome.state==="audio_unavailable"?"Speech recognition is unavailable. Please use the help button.":"Your reply was recorded. Waiting for a caregiver to respond.";
    $("elderInquiryResult").hidden=false;
  }catch(error){
    toast(error.message);
    if(started)await unavailable(id,error.message);
    else{try{await api(`/api/events/${id}/inquiry`,{method:"POST",body:{prompt_played:false,reason:error.message}});}catch{}}
    $("elderInquiryResult").textContent="The voice inquiry is unavailable. Please use the help button below.";$("elderInquiryResult").hidden=false;
  }finally{
    promptState(false);clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());stream=null;recording=false;
    $("inquiryProgress").hidden=true;$("inquiryQuestion").hidden=true;$("elderInquiry").disabled=false;await refresh();
  }
}
document.addEventListener("DOMContentLoaded",()=>{
  $("elderDate").textContent=new Date().toLocaleDateString("en-US",{month:"long",day:"numeric",weekday:"long"});
  $("elderHelp").onclick=async()=>{
    $("elderHelp").disabled=true;
    try{await api("/api/help",{method:"POST"});$("helpConfirm").hidden=false;$("elderHelpSub").textContent="Help request sent";await refresh();}
    catch(error){toast(error.message);}finally{$("elderHelp").disabled=false;}
  };
  $("elderSafe").onclick=async()=>{
    if(!currentEvent)return;
    try{await api(`/api/events/${currentEvent.id}/action`,{method:"POST",body:{action:"safe"}});$("elderInquiryResult").textContent="Your answer that you are safe was recorded. The alarm stays active until a caregiver verifies and closes it.";$("elderInquiryResult").hidden=false;await refresh();}
    catch(error){toast(error.message);}
  };
  $("elderInquiry").onclick=inquiry;$("elderWithdraw").onclick=withdraw;
  refresh();setInterval(refresh,2500);window.lucide?.createIcons();
  window.addEventListener("pagehide",()=>{
    clearInterval(timer);stream?.getTracks().forEach(t=>t.stop());
    if(recording&&currentEvent)fetch(`/api/events/${currentEvent.id}/inquiry/unavailable`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({reason:"The older adult page was closed"}),keepalive:true});
    window.speechSynthesis?.cancel();
  });
});
})();
