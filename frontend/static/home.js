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
    $("homeMonitorSummary").textContent = healthy ? "Home monitoring is active" : video ? "Video monitoring is active" : desired ? "Connecting monitoring" : "Monitoring paused";
    $("homeMicSummary").textContent = audioHealthy ? "Sound and voice connected" : audio ? "Preparing sound monitoring" : "Microphone not connected";
    $("homeGuardBadge").textContent = healthy ? "Monitoring" : desired ? "Waiting for input" : "Paused";
    $("homeGuardBadge").className = `badge ${healthy ? "green" : "amber"}`;
    $("homeSafetyStatus").textContent = !desired ? "Monitoring paused" : app.status.home_monitoring?.error || (healthy ? "Video and sound inputs are available" : "Checking camera and microphone");
    $("homeVideoState").textContent = !video ? "Waiting for connection" : visible ? "Monitoring" : "On, no person visible";
    $("homeAudioState").textContent = audioHealthy ? "Listening" : audio ? "Checking input" : !app.settings?.audio_consent ? "Not authorized" : audioError || "Waiting for connection";
    $("homeInquiryState").textContent = audio ? "Ready" : "Waiting for microphone";
    $("homeSensorState").textContent = app.status.coverage.some(c => c.channel === "accel" && c.state === "active") ? "Connected" : "Not connected";
    for (const [id, ready] of [["homeVideoState", visible], ["homeAudioState", audioHealthy], ["homeInquiryState", audio]]) $(id).className = ready ? "ready" : "warning";
    $("homeAudioNotice").hidden = !desired || (audio && !audioError);
    $("homeAudioReason").textContent = audioError || (!app.settings?.audio_consent ? "Microphone permission needed" : "Microphone waiting to connect");
    $("homeAllowAudio").querySelector("span").textContent = app.settings?.audio_consent ? "Connect microphone" : "Allow microphone";
    $("homeRoom").textContent = app.settings?.resident.room || "Living room";
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
    renderPlan();
  }
  /* 照护者安排的运动计划：长者端首页展示，点一项直接带着这个动作开始训练。
     完成与否只看当天有该动作的已结束训练，不做额外判定。 */
  let planBusy = false;
  function planSignature(plan) {
    return JSON.stringify([plan?.enabled, plan?.title, plan?.note, plan?.updated_at, (plan?.items || []).map(i => [i.exercise, i.target_reps, i.enabled]),
      (plan?.progress?.items || []).map(i => [i.exercise, i.reps, i.done]), plan?.progress?.skipped]);
  }
  let planPaint = "";
  function renderPlan() {
    const plan = app.plan;
    if (!plan || !$("homePlan")) return;
    const signature = planSignature(plan);
    if (signature === planPaint) return;
    planPaint = signature;
    const progress = plan.progress || {items: [], completed: 0, total: 0};
    const rows = (progress.items || []).filter(item => item.enabled);
    // 长者点过「今天先不做」，当天就不再显示这张卡片，明天照常出现。
    $("homePlan").hidden = !plan.enabled || !rows.length || !!progress.skipped;
    if ($("homePlan").hidden) return;
    $("homePlanTitle").textContent = plan.title || "Today's movement plan";
    $("homePlanNote").textContent = plan.note || "";
    $("homePlanNote").hidden = !plan.note;
    const done = rows.filter(item => item.done).length;
    $("homePlanCount").textContent = `${done} / ${rows.length} completed`;
    $("homePlanCount").className = `badge ${done === rows.length ? "green" : "amber"}`;
    $("homePlanItems").innerHTML = rows.map((item, index) => {
      const guide = exerciseGuides[item.exercise];
      const percent = Math.min(100, Math.round(100 * (item.reps || 0) / Math.max(1, item.target_reps)));
      const state = item.done ? "Completed" : item.reps ? `Done ${item.reps} reps` : "Not started";
      return `<li class="home-plan-item ${item.done ? "is-done" : ""}">
        <button type="button" class="home-plan-button" data-plan-start="${item.exercise}" ${item.disabled ? "disabled" : ""} aria-label="${item.done ? "Again" : "Start"}${names[item.exercise]}">
          <span class="home-plan-order">${item.done ? icon("circle-check") : index + 1}</span>
          <img src="/static/media/${item.exercise}-guide.png" alt="${names[item.exercise]} movement guide">
          <span class="home-plan-copy"><strong>${names[item.exercise]}</strong><small>${item.disabled ? "Detection for this movement is off; please contact your family" : `${state} · Target ${item.target_reps} reps`}</small>
            <span class="home-plan-bar"><span style="width:${percent}%"></span></span></span>
          <span class="home-plan-go">${item.done ? "Again" : item.reps ? "Resume" : "Start"}${icon("arrow-right")}</span>
        </button></li>`;
    }).join("");
    $("homePlanHint").textContent = done === rows.length
      ? "Today's plan is complete, well done"
      : done ? `Still ${rows.length - done} to go, take your time` : "Start in order; each item is ticked off automatically";
    $("homePlanStart").hidden = done === rows.length;
    icons();
    document.querySelectorAll("[data-plan-start]").forEach(button => {
      button.onclick = () => startFromPlan(button.dataset.planStart);
    });
  }
  function startFromPlan(key) {
    // 先把动作选好再切页：hashchange 之后 app.js 才渲染运动页，
    // 那时再塞标记就晚了，长者会看到默认的第一个动作。
    sessionStorage.setItem("homeease.plan", key);
    if (typeof chooseExercise === "function") chooseExercise(key);
    location.hash = "#exercise";
  }
  document.addEventListener("DOMContentLoaded", () => {
    $("homePlanDefer").onclick = async () => {
      if (planBusy) return;
      // 顺延几天交给长者自己定，别替他猜「明天一定有空」。
      const days = Number($("homePlanDeferDays")?.value || 1);
      if (days > 1 && !confirm("No training for these days. Are you sure?")) return;
      planBusy = true;
      try {
        await api("/api/plan/defer", {method: "POST", body: {days}});
        app.plan = await api("/api/plan");
        planPaint = "";
        render();
        toast(days > 1 ? `All right, resting for ${days} day(s), then reminders resume as usual` : "All right, rest today; the plan moves to tomorrow");
      } catch (error) { toast(error.message, true); }
      finally { planBusy = false; }
    };
  });
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
      $("homeMonitorSummary").textContent = "System connection lost";
      $("homeMonitorDot").className = "dot amber";
      $("homeGuardBadge").textContent = "System connection lost";
      $("homeGuardBadge").className = "badge amber";
      voiceState();
    }, 2000);
  });
})();
