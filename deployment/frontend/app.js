const API_BASE = window.__API_BASE__ || "/api";

const form = document.getElementById("risk-form");
const result = document.getElementById("result");
const error = document.getElementById("error");
const liveStatus = document.getElementById("live-status");
const streamToggle = document.getElementById("stream-toggle");
const streamState = document.getElementById("stream-state");

let debounceTimer = null;
let activeRequest = 0;
let streamSocket = null;
let streamTimer = null;
let streamRunning = false;

function riskColor(band) {
  if (band === "high") return "#b42318";
  if (band === "moderate") return "#b54708";
  return "#027a48";
}

function setStatus(text, mode = "idle") {
  liveStatus.textContent = text;
  liveStatus.classList.remove("status-idle", "status-running", "status-error");
  liveStatus.classList.add(`status-${mode}`);
}

function setStreamState(isRunning, detail = "") {
  streamRunning = isRunning;
  streamToggle.textContent = isRunning ? "Stop stream" : "Start stream";
  streamState.textContent = isRunning ? `running${detail ? ` (${detail})` : ""}` : "stopped";
}

function buildPayload() {
  return {
    condition: document.getElementById("condition").value,
    heart_rate: Number(document.getElementById("heart_rate").value),
    sbp: Number(document.getElementById("sbp").value),
    map: Number(document.getElementById("map").value),
    resp_rate: Number(document.getElementById("resp_rate").value),
    spo2: Number(document.getElementById("spo2").value),
    temp_f: Number(document.getElementById("temp_f").value),
  };
}

function hasInvalidValue(payload) {
  return Object.entries(payload).some(([key, value]) => key !== "condition" && Number.isNaN(value));
}

function updateResult(data, timestamp = null) {
  const color = riskColor(data.risk_band);
  const stamp = timestamp ? new Date(timestamp) : new Date();
  const lead = data.lead_hours ? `${data.lead_hours}h` : "1h";
  result.innerHTML = `
    <h2>Prediction</h2>
    <p><strong>Condition:</strong> ${data.condition}</p>
    <p><strong>Prediction horizon:</strong> ${lead}</p>
    <p><strong>Risk probability:</strong> ${(data.risk_probability * 100).toFixed(2)}%</p>
    <p><strong>High-risk threshold:</strong> ${(data.high_risk_threshold * 100).toFixed(2)}%</p>
    <p><strong>Risk band:</strong> <span style="color:${color}; font-weight: 700; text-transform: uppercase;">${data.risk_band}</span></p>
    <p><strong>Last updated:</strong> ${stamp.toLocaleTimeString()}</p>
  `;
  result.classList.remove("hidden");
}

async function predictNow() {
  const requestId = ++activeRequest;
  const payload = buildPayload();

  if (hasInvalidValue(payload)) {
    setStatus("Waiting for complete numeric values", "idle");
    return;
  }

  error.textContent = "";
  setStatus("Updating risk prediction...", "running");

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction request failed");
    }

    if (requestId !== activeRequest) {
      return;
    }

    updateResult(data);
    setStatus("Live mode active", "idle");
  } catch (err) {
    if (requestId !== activeRequest) {
      return;
    }
    error.textContent = err.message;
    setStatus("Prediction update failed", "error");
  }
}

function jitter(current, min, max, step) {
  const delta = (Math.random() * 2 - 1) * step;
  return Math.max(min, Math.min(max, current + delta));
}

function setInputValue(id, value) {
  document.getElementById(id).value = value.toFixed(1);
}

function applySimulationStep() {
  const heartRate = jitter(Number(document.getElementById("heart_rate").value), 45, 180, 4.0);
  const sbp = jitter(Number(document.getElementById("sbp").value), 70, 190, 3.5);
  const mapVal = jitter(Number(document.getElementById("map").value), 45, 130, 2.5);
  const respRate = jitter(Number(document.getElementById("resp_rate").value), 8, 45, 1.8);
  const spo2 = jitter(Number(document.getElementById("spo2").value), 82, 100, 0.8);
  const tempF = jitter(Number(document.getElementById("temp_f").value), 95, 105, 0.3);

  setInputValue("heart_rate", heartRate);
  setInputValue("sbp", sbp);
  setInputValue("map", mapVal);
  setInputValue("resp_rate", respRate);
  setInputValue("spo2", spo2);
  setInputValue("temp_f", tempF);
}

function websocketUrl() {
  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  return `${protocol}://${window.location.host}/api/ws/live`;
}

function sendCurrentPayloadToStream() {
  if (!streamSocket || streamSocket.readyState !== WebSocket.OPEN) {
    return;
  }
  const payload = buildPayload();
  if (hasInvalidValue(payload)) {
    return;
  }
  streamSocket.send(JSON.stringify(payload));
}

function startStreamLoop() {
  if (streamTimer) {
    clearInterval(streamTimer);
  }
  streamTimer = setInterval(() => {
    applySimulationStep();
    sendCurrentPayloadToStream();
  }, 1200);
}

function stopStreamLoop() {
  if (streamTimer) {
    clearInterval(streamTimer);
    streamTimer = null;
  }
}

function stopStreaming() {
  stopStreamLoop();
  if (streamSocket) {
    streamSocket.close();
    streamSocket = null;
  }
  setStreamState(false);
  setStatus("Live mode active", "idle");
}

function startStreaming() {
  if (streamRunning) {
    return;
  }
  error.textContent = "";
  setStatus("Connecting stream...", "running");
  setStreamState(true, "connecting");

  streamSocket = new WebSocket(websocketUrl());

  streamSocket.onopen = () => {
    setStreamState(true, "connected");
    setStatus("Streaming from simulated monitor", "running");
    sendCurrentPayloadToStream();
    startStreamLoop();
  };

  streamSocket.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "error") {
      error.textContent = msg.detail || "Stream error";
      setStatus("Stream error", "error");
      return;
    }
    if (msg.type === "prediction") {
      updateResult(msg, msg.timestamp);
      setStatus("Streaming from simulated monitor", "running");
    }
  };

  streamSocket.onclose = () => {
    if (streamRunning) {
      stopStreaming();
    }
  };

  streamSocket.onerror = () => {
    error.textContent = "Unable to connect streaming endpoint";
    setStatus("Stream connection failed", "error");
    stopStreaming();
  };
}

function schedulePredict() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => {
    predictNow();
  }, 350);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!streamRunning) {
    predictNow();
  }
});

form.querySelectorAll("input, select").forEach((element) => {
  element.addEventListener("input", () => {
    if (streamRunning) {
      sendCurrentPayloadToStream();
    } else {
      schedulePredict();
    }
  });
  element.addEventListener("change", () => {
    if (streamRunning) {
      sendCurrentPayloadToStream();
    } else {
      schedulePredict();
    }
  });
});

streamToggle.addEventListener("click", () => {
  if (streamRunning) {
    stopStreaming();
  } else {
    startStreaming();
  }
});

setStreamState(false);
setStatus("Live mode active", "idle");
predictNow();
