const API_BASE = window.__API_BASE__ || "/api";

const form = document.getElementById("risk-form");
const result = document.getElementById("result");
const error = document.getElementById("error");

function riskColor(band) {
  if (band === "high") return "#b42318";
  if (band === "moderate") return "#b54708";
  return "#027a48";
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  error.textContent = "";
  result.classList.add("hidden");

  const payload = {
    condition: document.getElementById("condition").value,
    heart_rate: Number(document.getElementById("heart_rate").value),
    sbp: Number(document.getElementById("sbp").value),
    map: Number(document.getElementById("map").value),
    resp_rate: Number(document.getElementById("resp_rate").value),
    spo2: Number(document.getElementById("spo2").value),
    temp_f: Number(document.getElementById("temp_f").value),
  };

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

    const color = riskColor(data.risk_band);
    result.innerHTML = `
      <h2>Prediction</h2>
      <p><strong>Condition:</strong> ${data.condition}</p>
      <p><strong>Risk probability:</strong> ${(data.risk_probability * 100).toFixed(2)}%</p>
      <p><strong>High-risk threshold:</strong> ${(data.high_risk_threshold * 100).toFixed(2)}%</p>
      <p><strong>Risk band:</strong> <span style="color:${color}; font-weight: 700; text-transform: uppercase;">${data.risk_band}</span></p>
    `;
    result.classList.remove("hidden");
  } catch (err) {
    error.textContent = err.message;
  }
});
