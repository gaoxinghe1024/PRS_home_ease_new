"use strict";
(() => {
  if (!household) return;
  let desired = false, commandBusy = false, audioError = "", permissionBlocked = false, autoStartRequested = false;
  let audioRetryAt = 0, wakeLock = null, leaving = false, lastStatusAt = 0;
  const activeAudio = () => app.audioSocket?.readyState === WebSocket.OPEN &&
    app.audioStream?.getAudioTracks().some(track => track.readyState === "live") && app.audioContext?.state === "running";
  const voiceState = () => window.dispatchEvent(new CustomEvent("homeease:voice", {detail: {enabled: desired && !leaving && Date.now()-lastStatusAt < 12000 && !!activeAudio()}}));
  async function awake() {
    if (!desired || document.visibilityState !== "visible" || wakeLock || !navigator.wakeLock) return;
    try { wakeLock = await navigator.wakeLock.request("screen"); wakeLock.addEventListener("release", () => { wakeLock = null; }); } catch {}
  }
  async function setGuard(enabled, cameraIndex = app.status?.home_monitoring?.camera_index ?? 0) {
    if (commandBusy) return;
    commandBusy = true;
    try {
      const result = await api("/api/home/monitoring", {method: "PUT", body: {enabled, camera_index: cameraIndex}});
      app.status.home_monitoring = result;
      desired = enabled;
      if (enabled) { audioRetryAt = 0; await awake(); }
      else { await stopAudioLive(); await wakeLock?.release(); wakeLock = null; }
      render();
    } catch (error) { toast(error.message, true); }
    finally { commandBusy = false; }
  }
  async function connectAudio() {
    if (!desired || leaving || !app.settings?.audio_consent || app.audioStarting || app.audioStopping) return;
    if (app.audioStream && !app.audioStream.getAudioTracks().some(track => track.readyState === "live")) await stopAudioLive();
    if (!app.audioStream && !permissionBlocked && Date.now() >= audioRetryAt && app.status?.modules.find(m => m.id === "audio")?.state === "ready") {
      audioRetryAt = Date.now() + 15000;
      try { await startAudioLive(); } catch (error) { window.HomeMonitoring.audioFailed(error); }
    }
    voiceState();
  }
  function render() {
    if (!app.status) return;
    lastStatusAt = Date.now();
    desired = !!app.status.home_monitoring?.enabled;
    if (autoStartRequested && !commandBusy) { autoStartRequested = false; if (!desired) setGuard(true); }
    const video = desired && app.status.video_mode === "live" && app.status.video.state === "running";
    const visible = video && app.status.coverage.some(c => c.channel === "vision" && c.state === "active");
    const audio = !!activeAudio() && !!app.settings?.audio_consent && desired;
    const audioCoverage = app.status.coverage.find(c => c.channel === "audio");
    const audioHealthy = audio && audioCoverage?.state === "active";
    const healthy = visible && audioHealthy;
    $("homeMonitorDot").className = `dot ${healthy ? "green" : "amber"}`;
    $("homeMonitorSummary").textContent = healthy ? "居家守护中" : video ? "视频守护中" : desired ? "守护正在连接" : "守护已暂停";
    $("homeMicSummary").textContent = audioHealthy ? "声音与语音已连接" : audio ? "声音监听准备中" : "麦克风未连接";
    $("homeGuardBadge").textContent = healthy ? "守护中" : desired ? "等待输入" : "已暂停";
    $("homeGuardBadge").className = `badge ${healthy ? "green" : "amber"}`;
    $("homeSafetyStatus").textContent = !desired ? "守护已暂停" : app.status.home_monitoring?.error || (healthy ? "视频与声音输入有效" : "正在检查摄像头与麦克风");
    $("homeVideoState").textContent = !video ? "等待连接" : visible ? "监测中" : "已开启，未见人体";
    $("homeAudioState").textContent = audioHealthy ? "监听中" : audio ? "检查输入中" : !app.settings?.audio_consent ? "尚未授权" : audioError || "等待连接";
    $("homeInquiryState").textContent = audio ? "已就绪" : "等待麦克风";
    $("homeSensorState").textContent = app.status.coverage.some(c => c.channel === "accel" && c.state === "active") ? "已连接" : "未连接";
    for (const [id, ready] of [["homeVideoState", visible], ["homeAudioState", audioHealthy], ["homeInquiryState", audio]]) $(id).className = ready ? "ready" : "warning";
    $("homeAudioNotice").hidden = !desired || (audio && !audioError);
    $("homeAudioReason").textContent = audioError || (!app.settings?.audio_consent ? "麦克风尚未授权" : "麦克风等待连接");
    $("homeAllowAudio").querySelector("span").textContent = app.settings?.audio_consent ? "连接麦克风" : "允许麦克风";
    $("homePause").querySelector("span").textContent = desired ? "暂停守护" : "恢复守护";
    $("homeRoom").textContent = app.settings?.resident.room || "客厅";
    if (document.activeElement !== $("homeCameraIndex")) $("homeCameraIndex").value = app.status.home_monitoring?.camera_index ?? 0;
    $("exerciseCameraIndex").value = app.status.home_monitoring?.camera_index ?? 0;
    const image = $("homeFrame"), preview = video && app.page === "safety";
    if (preview && !image.getAttribute("src")) image.src = "/api/video/stream";
    if (!preview) image.removeAttribute("src");
    image.hidden = !preview;
    $("homeCameraEmpty").hidden = preview;
    document.querySelectorAll(".home-header nav a").forEach(a => {
      if (a.hash === `#${app.page}`) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (!desired && app.audioStream && !app.audioStopping) stopAudioLive();
    voiceState();
    connectAudio();
  }
  window.HomeMonitoring = {
    render,
    navigate() {
      autoStartRequested = app.page === "safety";
      render();
    },
    audioReady() { audioError = ""; permissionBlocked = false; render(); },
    audioFailed(error) { audioError = error.message; permissionBlocked = ["NotAllowedError", "NotFoundError", "SecurityError"].includes(error.name); audioRetryAt = Date.now() + 15000; render(); },
    stream() { return app.audioStream; }
  };
  document.addEventListener("DOMContentLoaded", () => {
    $("homePause").onclick = () => setGuard(!desired);
    $("homeRetry").onclick = async () => { audioError = ""; permissionBlocked = false; await setGuard(true); await connectAudio(); };
    $("homeCameraApply").onclick = () => {
      if (!$("homeCameraIndex").reportValidity()) return;
      return setGuard(true, Number($("homeCameraIndex").value));
    };
    $("homeAllowAudio").onclick = () => busy($("homeAllowAudio"), async () => {
      if (!app.settings.audio_consent) app.settings = await api("/api/home/audio-consent", {method: "POST"});
      audioError = ""; permissionBlocked = false; audioRetryAt = 0;
      if (app.audioContext?.state === "suspended") await app.audioContext.resume();
      await connectAudio(); await awake(); render();
    });
    $("homeFrame").onerror = () => { $("homeFrame").removeAttribute("src"); };
    window.addEventListener("homeease:prompt", event => app.audioStream?.getAudioTracks().forEach(track => { track.enabled = !event.detail.speaking; }));
    document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") { awake(); connectAudio(); } });
    document.addEventListener("click", () => { awake(); if (app.audioContext?.state === "suspended") app.audioContext.resume().then(voiceState); });
    window.addEventListener("pagehide", () => { leaving = true; voiceState(); });
    setInterval(() => {
      if (!lastStatusAt || Date.now()-lastStatusAt < 12000) return;
      $("homeMonitorSummary").textContent = "系统连接中断";
      $("homeMonitorDot").className = "dot amber";
      $("homeGuardBadge").textContent = "系统连接中断";
      $("homeGuardBadge").className = "badge amber";
      voiceState();
    }, 2000);
  });
})();
