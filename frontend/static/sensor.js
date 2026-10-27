"use strict";
document.addEventListener("DOMContentLoaded",()=>document.querySelectorAll('[data-lucide="house-heart"]').forEach(el=>el.dataset.lucide="house"));
const $=id=>document.getElementById(id);let socket=null,pending=[],lastBatch=[],inflight=false,flushing=false;
function toast(message){$("toast").textContent=message;$("toast").hidden=false;setTimeout(()=>$("toast").hidden=true,6000);}
function resetConnection(){pending=[];flushing=false;socket=null;inflight=false;$("connectSensor").disabled=false;$("disconnectSensor").disabled=true;$("sendSamples").disabled=false;$("sensorConnection").textContent="Not connected";$("sensorDot").className="dot amber";}
function stop(){if(socket?.readyState===WebSocket.OPEN){flushing=true;$("disconnectSensor").disabled=true;$("connectSensor").disabled=true;$("sendSamples").disabled=true;$("sensorConnection").textContent="Saving the last batch of data";pump();}else resetConnection();}
function connect(){return new Promise((resolve,reject)=>{if(socket?.readyState===WebSocket.OPEN)return resolve();socket=new WebSocket(`${location.protocol==="https:"?"wss":"ws"}://${location.host}/ws/sensor`);socket.onmessage=event=>{const data=JSON.parse(event.data);if(data.error){toast(data.error);reject(new Error(data.error));stop();return;}if(data.state==="stopped"){socket.close();return;}if(data.state==="connected"){$("sensorConnection").textContent="Live data connected";$("sensorDot").className="dot green";$("disconnectSensor").disabled=false;resolve();return;}inflight=false;const recent=data.recent?.length?data.recent:lastBatch,last=recent.at(-1);if(last){["sensorX","sensorY","sensorZ"].forEach((id,i)=>$(id).textContent=last[i+1].toFixed(3));if(recent.length>1&&last[0]>recent[0][0])$("sensorRate").textContent=`${((recent.length-1)/(last[0]-recent[0][0])).toFixed(1)} Hz`;}if(data.prob!==null&&data.prob!==undefined)$("sensorProb").textContent=`${(data.prob*100).toFixed(1)}%`;if(data.alarms?.length)toast("A suspected fall was detected and an event was created");pump();};socket.onerror=()=>{reject(new Error("Sensor connection failed"));toast("Sensor connection failed");};socket.onclose=()=>{socket=null;stop();reject(new Error("Connection closed"));};});}
function pump(){if(!socket||socket.readyState!==WebSocket.OPEN||inflight)return;if(!pending.length){if(flushing){inflight=true;socket.send(JSON.stringify({type:"flush"}));}return;}const samples=pending.splice(0,100);lastBatch=samples;inflight=true;socket.send(JSON.stringify({samples}));}
document.addEventListener("DOMContentLoaded",()=>{
  window.lucide?.createIcons();
  $("connectSensor").onclick=async()=>{$("connectSensor").disabled=true;try{await connect();}catch(e){toast(e.message);$("connectSensor").disabled=false;}};
  $("disconnectSensor").onclick=stop;
  $("sendSamples").onclick=async()=>{try{
    const samples=JSON.parse($("sensorSamples").value);
    if(!Array.isArray(samples)||!samples.length||samples.length>20000)throw new Error("A batch needs 1 to 20000 samples");
    if(samples.some(row=>!Array.isArray(row)||row.length!==4||row.some(v=>typeof v!=="number"||!Number.isFinite(v))))throw new Error("Each sample needs 4 finite numbers");
    if(flushing)throw new Error("Saving the previous data connection");
    if(pending.length+samples.length>20000)throw new Error("Please wait until the previous batch has been processed");
    await connect();pending.push(...samples);pump();
  }catch(e){toast(e.message);}};
  window.addEventListener("pagehide",()=>socket?.close());
  setInterval(pump,150);
});
