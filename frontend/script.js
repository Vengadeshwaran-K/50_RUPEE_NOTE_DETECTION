/* RupeeVision AI — frontend logic (vanilla JS, no frameworks).
 *
 * The browser only captures webcam frames and images; all inference happens
 * on the FastAPI backend. Server-provided strings are written with
 * textContent, never innerHTML.
 */
(function () {
  "use strict";

  const metaBase = document.querySelector('meta[name="rv-api-base"]');
  const CONFIG = Object.freeze({
    apiBase: ((metaBase && metaBase.content) || "http://127.0.0.1:8000").replace(/\/+$/, ""),
    frameIntervalMs: 200,      // up to ~5 requests per second
    maxFrameWidth: 512,        // refined from /api/model-info once the model is loaded
    frameJpegQuality: 0.75,
    maxLiveFailures: 3,
    healthPollMs: 15000,
    defaultMaxUploadBytes: 10 * 1024 * 1024,
  });
  const ACCEPTED_TYPES = ["image/jpeg", "image/png"];
  const ACCEPTED_EXT = /\.(jpe?g|png)$/i;

  const $ = (id) => document.getElementById(id);
  const el = {
    backendStatus: $("backend-status"),
    modelStatus: $("model-status"),
    confInput: $("conf-input"),
    confValue: $("conf-value"),
    modelMeta: $("model-meta"),
    docsLink: $("docs-link"),
    heroTarget: $("hero-target"),
    heroEngine: $("hero-engine"),
    // webcam
    video: $("webcam-video"),
    liveImg: $("webcam-result"),
    placeholder: $("webcam-placeholder"),
    liveBadge: $("live-badge"),
    canvas: $("capture-canvas"),
    btnCamStart: $("btn-camera-start"),
    btnCamStop: $("btn-camera-stop"),
    btnDetStart: $("btn-detect-start"),
    btnDetStop: $("btn-detect-stop"),
    liveCount: $("live-count"),
    liveMs: $("live-ms"),
    liveRate: $("live-rate"),
    webcamMsg: $("webcam-message"),
    // upload
    dropzone: $("dropzone"),
    fileInput: $("file-input"),
    preview: $("upload-preview"),
    dropText: $("dropzone-text"),
    btnChoose: $("btn-choose"),
    btnUpload: $("btn-upload-detect"),
    spinner: $("upload-spinner"),
    fileName: $("file-name"),
    resultWrap: $("upload-result-wrap"),
    resultImg: $("upload-result"),
    uploadCount: $("upload-count"),
    uploadConf: $("upload-conf"),
    uploadMs: $("upload-ms"),
    uploadMsg: $("upload-message"),
    // stats + table
    statCount: $("stat-count"),
    statConf: $("stat-conf"),
    statMs: $("stat-ms"),
    statStatus: $("stat-status"),
    statsSource: $("stats-source"),
    resultsBody: $("results-body"),
    resultsMeta: $("results-meta"),
  };

  const state = {
    backendOnline: null,
    modelLoaded: false,
    maxUploadBytes: CONFIG.defaultMaxUploadBytes,
    maxFrameWidth: CONFIG.maxFrameWidth,
    stream: null,
    live: { running: false, pending: false, timer: null, abort: null, failures: 0, lastStart: 0, rate: 0 },
    upload: { file: null, previewUrl: null, pending: false },
  };

  // ---------------------------------------------------------------- helpers

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.status = status;
    }
  }

  function apiUrl(path) {
    return CONFIG.apiBase + path;
  }

  function detailToMessage(detail, fallback) {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      // FastAPI validation errors: [{loc, msg, ...}]
      return detail.map((d) => {
        const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "";
        return field ? `${field}: ${d.msg}` : String(d.msg);
      }).join("; ");
    }
    return fallback;
  }

  async function fetchJson(path, options) {
    let response;
    try {
      response = await fetch(apiUrl(path), options);
    } catch (err) {
      if (err.name === "AbortError") throw err;
      throw new ApiError(`Cannot reach the backend at ${CONFIG.apiBase}. Is the server running?`, 0);
    }
    let body = null;
    try {
      body = await response.json();
    } catch (_) {
      /* non-JSON body */
    }
    if (!response.ok) {
      const fallback = `Request failed (HTTP ${response.status})`;
      throw new ApiError(detailToMessage(body && body.detail, fallback), response.status);
    }
    return body;
  }

  function setMessage(target, text, kind) {
    target.textContent = text || "";
    target.dataset.kind = kind || "info";
  }

  function setPill(pill, text, pillState) {
    pill.dataset.state = pillState;
    pill.querySelector(".pill-text").textContent = text;
  }

  function formatConf(value) {
    return typeof value === "number" ? value.toFixed(2) : "–";
  }

  function formatMs(value) {
    return typeof value === "number" ? `${value.toFixed(1)} ms` : "–";
  }

  function detectPath(endpoint) {
    const conf = Number(el.confInput.value);
    return `/api/detect/${endpoint}?conf=${encodeURIComponent(conf.toFixed(2))}`;
  }

  // --------------------------------------------------------- backend status

  async function checkHealth() {
    try {
      const health = await fetchJson("/api/health");
      const wasOffline = state.backendOnline !== true;
      state.backendOnline = true;
      state.modelLoaded = Boolean(health.model_loaded);
      if (wasOffline) await loadModelInfo();
      if (!state.modelLoaded && health.model_error) {
        el.modelStatus.title = health.model_error;
      }
    } catch (err) {
      state.backendOnline = false;
      state.modelLoaded = false;
    }
    renderStatus();
  }

  async function loadModelInfo() {
    try {
      const info = await fetchJson("/api/model-info");
      state.modelLoaded = Boolean(info.model_loaded);
      if (typeof info.max_upload_bytes === "number") state.maxUploadBytes = info.max_upload_bytes;
      if (typeof info.input_width === "number" && info.input_width > 0) {
        state.maxFrameWidth = Math.max(320, Math.min(1280, info.input_width));
      }
      if (typeof info.conf_threshold === "number" && !el.confInput.dataset.touched) {
        el.confInput.value = String(info.conf_threshold);
        el.confValue.textContent = Number(info.conf_threshold).toFixed(2);
      }
      if (info.target_class) el.heroTarget.textContent = info.target_class;
      if (info.inference_backend) el.heroEngine.textContent = info.inference_backend;
      el.modelMeta.textContent = info.model_loaded
        ? `Target ${info.target_class} · input ${info.input_width}×${info.input_height} · ${info.inference_backend}`
        : `Model not loaded: ${info.model_error || "unknown error"}`;
    } catch (err) {
      el.modelMeta.textContent = "";
    }
  }

  function renderStatus() {
    if (state.backendOnline === true) {
      setPill(el.backendStatus, "Backend: online", "ok");
      setPill(el.modelStatus, state.modelLoaded ? "Model: loaded" : "Model: not loaded",
        state.modelLoaded ? "ok" : "error");
    } else if (state.backendOnline === false) {
      setPill(el.backendStatus, "Backend: offline", "error");
      setPill(el.modelStatus, "Model: unknown", "warn");
    }
    el.statStatus.textContent = state.backendOnline
      ? (state.modelLoaded ? "Online · model loaded" : "Online · model missing")
      : (state.backendOnline === false ? "Offline" : "Checking…");
    el.btnUpload.disabled = !state.upload.file || state.upload.pending || !state.modelLoaded;
    el.btnDetStart.disabled = !state.stream || state.live.running || !state.modelLoaded;
  }

  // ----------------------------------------------------------------- webcam

  function describeCameraError(err) {
    switch (err && err.name) {
      case "NotAllowedError":
      case "SecurityError":
        return "Camera permission was denied. Allow camera access in the browser's site settings and try again.";
      case "NotFoundError":
      case "OverconstrainedError":
        return "No camera was found. Connect a webcam and try again.";
      case "NotReadableError":
      case "AbortError":
        return "The camera is in use by another application or could not be started.";
      default:
        return `Could not start the camera: ${(err && err.message) || "unknown error"}`;
    }
  }

  async function startCamera() {
    if (state.stream) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setMessage(el.webcamMsg,
        "This browser does not support camera access here. Use a modern browser on http://127.0.0.1 or localhost.",
        "error");
      return;
    }
    el.btnCamStart.disabled = true;
    setMessage(el.webcamMsg, "Requesting camera permission…", "info");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      state.stream = stream;
      el.video.srcObject = stream;
      await el.video.play();
      el.placeholder.hidden = true;
      el.btnCamStop.disabled = false;
      setMessage(el.webcamMsg, "Camera on. Press Start Detection to analyse frames.", "info");
      // Handle the camera being unplugged or revoked while running.
      stream.getVideoTracks().forEach((track) => {
        track.addEventListener("ended", () => {
          stopCamera();
          setMessage(el.webcamMsg, "The camera stopped unexpectedly.", "warn");
        }, { once: true });
      });
    } catch (err) {
      stopCamera();
      setMessage(el.webcamMsg, describeCameraError(err), "error");
    }
    renderStatus();
  }

  function stopCamera() {
    stopLiveDetection();
    if (state.stream) {
      state.stream.getTracks().forEach((track) => track.stop());
      state.stream = null;
    }
    el.video.pause();
    el.video.srcObject = null;
    el.placeholder.hidden = false;
    el.btnCamStart.disabled = false;
    el.btnCamStop.disabled = true;
    renderStatus();
  }

  // --------------------------------------------------------- live detection

  function startLiveDetection() {
    if (!state.stream || state.live.running) return;
    const live = state.live;
    live.running = true;
    live.failures = 0;
    live.rate = 0;
    live.lastStart = 0;
    el.btnDetStart.disabled = true;
    el.btnDetStop.disabled = false;
    el.liveBadge.hidden = false;
    setMessage(el.webcamMsg, "Detecting…", "info");
    liveLoop();
  }

  function stopLiveDetection() {
    const live = state.live;
    live.running = false;
    clearTimeout(live.timer);
    live.timer = null;
    if (live.abort) live.abort.abort();
    live.abort = null;
    el.liveImg.hidden = true;
    el.liveImg.removeAttribute("src");
    el.liveBadge.hidden = true;
    el.btnDetStop.disabled = true;
    el.liveRate.textContent = "–";
    renderStatus();
  }

  async function liveLoop() {
    const live = state.live;
    if (!live.running || live.pending) return;
    const started = performance.now();
    if (live.lastStart) {
      const instant = 1000 / (started - live.lastStart);
      live.rate = live.rate ? 0.8 * live.rate + 0.2 * instant : instant;
      el.liveRate.textContent = `${live.rate.toFixed(1)} req/s`;
    }
    live.lastStart = started;

    live.pending = true;
    try {
      const blob = await captureFrame();
      if (blob && live.running) await submitFrame(blob);
    } finally {
      live.pending = false;
    }
    if (live.running) {
      const wait = Math.max(0, CONFIG.frameIntervalMs - (performance.now() - started));
      live.timer = setTimeout(liveLoop, wait);
    }
  }

  function captureFrame() {
    const video = el.video;
    if (!video.videoWidth || !video.videoHeight) return Promise.resolve(null);
    const scale = Math.min(1, state.maxFrameWidth / video.videoWidth);
    const canvas = el.canvas;
    canvas.width = Math.round(video.videoWidth * scale);
    canvas.height = Math.round(video.videoHeight * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", CONFIG.frameJpegQuality));
  }

  async function submitFrame(blob) {
    const live = state.live;
    const form = new FormData();
    form.append("file", blob, "frame.jpg");
    live.abort = new AbortController();
    try {
      const result = await fetchJson(detectPath("frame"), { method: "POST", body: form, signal: live.abort.signal });
      if (!live.running) return;
      live.failures = 0;
      showAnnotated(el.liveImg, result);
      el.liveCount.textContent = String(result.count);
      el.liveMs.textContent = formatMs(result.inference_ms);
      setMessage(el.webcamMsg,
        result.count ? `${result.count} ₹50 note(s) detected.` : "No ₹50 note detected in the current frame.",
        result.count ? "ok" : "info");
      updateResults(result, "Live webcam");
    } catch (err) {
      if (err.name === "AbortError") return;
      live.failures += 1;
      setMessage(el.webcamMsg, err.message, "error");
      if (err.status === 0 || err.status === 503 || live.failures >= CONFIG.maxLiveFailures) {
        stopLiveDetection();
        setMessage(el.webcamMsg, `Live detection stopped: ${err.message}`, "error");
        checkHealth();
      }
    } finally {
      live.abort = null;
    }
  }

  // ----------------------------------------------------------- image upload

  function validateFile(file) {
    if (!file) return "No file selected.";
    const typeOk = ACCEPTED_TYPES.includes(file.type) || (!file.type && ACCEPTED_EXT.test(file.name));
    if (!typeOk) return `"${file.name}" is not a JPG or PNG image.`;
    if (file.size === 0) return "The selected file is empty.";
    if (file.size > state.maxUploadBytes) {
      return `The file is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${(state.maxUploadBytes / 1048576).toFixed(0)} MB.`;
    }
    return null;
  }

  function handleFileSelected(file) {
    const error = validateFile(file);
    if (error) {
      setMessage(el.uploadMsg, error, "error");
      return;
    }
    if (state.upload.previewUrl) URL.revokeObjectURL(state.upload.previewUrl);
    state.upload.file = file;
    state.upload.previewUrl = URL.createObjectURL(file);
    el.preview.src = state.upload.previewUrl;
    el.preview.hidden = false;
    el.dropText.hidden = true;
    el.fileName.textContent = `${file.name} (${(file.size / 1024).toFixed(0)} KB)`;
    el.resultWrap.hidden = true;
    setMessage(el.uploadMsg, "Ready. Press Detect to analyse the image.", "info");
    renderStatus();
  }

  async function uploadImage() {
    const upload = state.upload;
    if (!upload.file || upload.pending) return;
    upload.pending = true;
    el.spinner.hidden = false;
    setMessage(el.uploadMsg, "Detecting…", "info");
    renderStatus();

    const form = new FormData();
    form.append("file", upload.file, upload.file.name);
    try {
      const result = await fetchJson(detectPath("image"), { method: "POST", body: form });
      showAnnotated(el.resultImg, result);
      el.resultWrap.hidden = false;
      el.uploadCount.textContent = String(result.count);
      el.uploadConf.textContent = result.count
        ? result.detections.map((d) => formatConf(d.confidence)).join(", ")
        : "–";
      el.uploadMs.textContent = formatMs(result.inference_ms);
      setMessage(el.uploadMsg,
        result.count ? `${result.count} ₹50 note(s) detected.` : "No ₹50 note detected in this image.",
        result.count ? "ok" : "info");
      updateResults(result, "Image upload");
    } catch (err) {
      setMessage(el.uploadMsg, err.message, "error");
      if (err.status === 0 || err.status === 503) checkHealth();
    } finally {
      upload.pending = false;
      el.spinner.hidden = true;
      renderStatus();
    }
  }

  // ---------------------------------------------------------------- results

  function showAnnotated(img, result) {
    const b64 = typeof result.annotated_image === "string" ? result.annotated_image : "";
    if (!/^[A-Za-z0-9+/]*={0,2}$/.test(b64)) return;
    img.src = `data:image/jpeg;base64,${b64}`;
    img.hidden = false;
  }

  function updateResults(result, source) {
    updateStats(result, source);
    renderTable(result);
  }

  function updateStats(result, source) {
    const confidences = result.detections.map((d) => d.confidence);
    el.statCount.textContent = String(result.count);
    el.statConf.textContent = confidences.length ? formatConf(Math.max(...confidences)) : "–";
    el.statMs.textContent = formatMs(result.inference_ms);
    el.statsSource.textContent = `(latest: ${source})`;
  }

  function renderTable(result) {
    const body = el.resultsBody;
    body.replaceChildren();
    el.resultsMeta.textContent =
      `Coordinates in pixels of the processed ${result.image_width}×${result.image_height} image · threshold ${formatConf(result.conf_threshold)}`;

    if (!result.detections.length) {
      const row = document.createElement("tr");
      row.className = "empty-row";
      const cell = document.createElement("td");
      cell.colSpan = 5;
      cell.textContent = "No ₹50 note detected in the latest result.";
      row.appendChild(cell);
      body.appendChild(row);
      return;
    }

    result.detections.forEach((d, i) => {
      const b = d.bbox;
      const cells = [
        String(i + 1),
        d.label,
        formatConf(d.confidence),
        `(${b.x1}, ${b.y1}, ${b.x2}, ${b.y2})`,
        `${b.width} × ${b.height}`,
      ];
      const row = document.createElement("tr");
      cells.forEach((text) => {
        const cell = document.createElement("td");
        cell.textContent = text;
        row.appendChild(cell);
      });
      body.appendChild(row);
    });
  }

  // ------------------------------------------------------------------- init

  function bindEvents() {
    el.btnCamStart.addEventListener("click", startCamera);
    el.btnCamStop.addEventListener("click", () => {
      stopCamera();
      setMessage(el.webcamMsg, "Camera stopped.", "info");
    });
    el.btnDetStart.addEventListener("click", startLiveDetection);
    el.btnDetStop.addEventListener("click", () => {
      stopLiveDetection();
      setMessage(el.webcamMsg, "Detection stopped. Camera is still on.", "info");
    });

    el.confInput.addEventListener("input", () => {
      el.confInput.dataset.touched = "1";
      el.confValue.textContent = Number(el.confInput.value).toFixed(2);
    });

    el.btnChoose.addEventListener("click", () => el.fileInput.click());
    el.fileInput.addEventListener("change", () => {
      if (el.fileInput.files && el.fileInput.files[0]) handleFileSelected(el.fileInput.files[0]);
      el.fileInput.value = ""; // allow re-selecting the same file
    });
    el.btnUpload.addEventListener("click", uploadImage);

    ["dragenter", "dragover"].forEach((type) => el.dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      el.dropzone.classList.add("is-dragover");
    }));
    ["dragleave", "drop"].forEach((type) => el.dropzone.addEventListener(type, (e) => {
      e.preventDefault();
      el.dropzone.classList.remove("is-dragover");
    }));
    el.dropzone.addEventListener("drop", (e) => {
      const files = e.dataTransfer && e.dataTransfer.files;
      if (files && files.length) handleFileSelected(files[0]);
    });
    // Stop a stray drop outside the zone from navigating away from the page.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());

    window.addEventListener("pagehide", () => {
      stopCamera();
      if (state.upload.previewUrl) URL.revokeObjectURL(state.upload.previewUrl);
    });
  }

  function init() {
    el.docsLink.href = apiUrl("/docs");
    bindEvents();
    renderStatus();
    checkHealth();
    setInterval(checkHealth, CONFIG.healthPollMs);
  }

  init();
})();
