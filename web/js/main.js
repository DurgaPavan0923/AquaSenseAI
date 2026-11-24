/**
 * AquaSense AI Frontend JavaScript
 * Handles single-sample prediction, batch CSV upload, local occlusion attribution,
 * and client-side history storage (localStorage).
 */

document.addEventListener("DOMContentLoaded", () => {
  setupSinglePredictionForm();
  setupBatchUploadForm();
  loadSessionHistory();
});

const API_BASE = "";

// Format numbers cleanly
function fmt(val, digits = 4) {
  if (typeof val !== "number") return val;
  return val.toFixed(digits);
}

// 1. Single Sample Prediction Form
function setupSinglePredictionForm() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  const resultContainer = document.getElementById("prediction-result");
  const attributionContainer = document.getElementById("attribution-result");
  const anomalyContainer = document.getElementById("anomaly-result");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const sample = {
      pH: parseFloat(document.getElementById("input-ph").value),
      Turbidity: parseFloat(document.getElementById("input-turbidity").value),
      Temperature: parseFloat(document.getElementById("input-temp").value),
      "Dissolved Oxygen": parseFloat(document.getElementById("input-do").value),
      "Biochemical Oxygen Demand": parseFloat(document.getElementById("input-bod").value),
      Lead: parseFloat(document.getElementById("input-lead").value),
      Mercury: parseFloat(document.getElementById("input-mercury").value),
      Arsenic: parseFloat(document.getElementById("input-arsenic").value),
    };

    resultContainer.innerHTML = "<p>Processing screening inference...</p>";
    if (attributionContainer) attributionContainer.innerHTML = "";
    if (anomalyContainer) anomalyContainer.innerHTML = "";

    try {
      // 1. Prediction with Conformal Uncertainty
      const predRes = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(sample),
      });

      if (!predRes.ok) {
        throw new Error(`Prediction failed with status: ${predRes.status}`);
      }

      const predData = await predRes.json();
      displayPredictionResult(predData, sample);

      // 2. Local Occlusion Attribution
      if (attributionContainer) {
        const expRes = await fetch(`${API_BASE}/explain`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(sample),
        });
        if (expRes.ok) {
          const expData = await expRes.json();
          displayAttributionResult(expData);
        }
      }

      // 3. Sensor & Statutory Anomaly Audit
      if (anomalyContainer) {
        const anomRes = await fetch(`${API_BASE}/anomaly`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(sample),
        });
        if (anomRes.ok) {
          const anomData = await anomRes.json();
          displayAnomalyResult(anomData);
        }
      }

      // Save to session history
      saveToSessionHistory(predData, sample);

    } catch (err) {
      resultContainer.innerHTML = `<div class="badge badge-alert">Error: ${err.message}</div>`;
    }
  });
}

function displayPredictionResult(data, sample) {
  const container = document.getElementById("prediction-result");
  if (!container) return;

  const badgeClass =
    data.risk_tier === "Good"
      ? "badge-good"
      : data.risk_tier === "Moderate"
      ? "badge-moderate"
      : "badge-polluted";

  const labNotice = data.needs_laboratory_test
    ? `<div style="background-color: #fee2e2; border: 1px solid #fca5a5; color: #991b1b; padding: 12px; border-radius: 4px; margin-top: 14px; font-weight: 600;">
        Flagged: ${data.confidence_assessment}
       </div>`
    : "";

  container.innerHTML = `
    <div style="border: 1px solid var(--border-color); border-radius: 4px; padding: 20px; background-color: var(--bg-surface);">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <span class="badge ${badgeClass}" style="font-size: 1.1rem; padding: 6px 14px;">${data.risk_tier} Risk</span>
        <span style="font-size: 0.9rem; color: var(--text-muted);">Conformal Guarantee: ${data.conformal_coverage_guarantee}</span>
      </div>
      
      <p style="margin-bottom: 12px;">
        <strong>Conformal Prediction Set:</strong> [ ${data.conformal_prediction_set.join(", ")} ]
      </p>

      <div style="margin-bottom: 14px;">
        <strong>Calibrated Class Probabilities:</strong>
        <div style="margin-top: 6px; font-size: 0.9rem;">
          <div>Good: ${(data.probabilities.Good * 100).toFixed(1)}%</div>
          <div>Moderate: ${(data.probabilities.Moderate * 100).toFixed(1)}%</div>
          <div>Polluted: ${(data.probabilities.Polluted * 100).toFixed(1)}%</div>
        </div>
      </div>

      ${labNotice}
    </div>
  `;
}

function displayAttributionResult(data) {
  const container = document.getElementById("attribution-result");
  if (!container) return;

  let tableRows = "";
  for (const [feat, val] of Object.entries(data.feature_attributions)) {
    const sign = val >= 0 ? "+" : "";
    const color = val > 0 ? "#991b1b" : "#166534";
    tableRows += `
      <tr>
        <td>${feat}</td>
        <td style="color: ${color}; font-weight: 600;">${sign}${val.toFixed(4)}</td>
      </tr>
    `;
  }

  container.innerHTML = `
    <div class="card" style="margin-top: 20px;">
      <h3>Local Feature Attribution (Approximate Occlusion)</h3>
      <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 12px;">
        ${data.notice}
      </p>
      <table>
        <thead>
          <tr>
            <th>Parameter</th>
            <th>Probability Delta (&Delta;P)</th>
          </tr>
        </thead>
        <tbody>
          ${tableRows}
        </tbody>
      </table>
    </div>
  `;
}

function displayAnomalyResult(data) {
  const container = document.getElementById("anomaly-result");
  if (!container) return;

  const severityBadge =
    data.severity === "NORMAL"
      ? '<span class="badge badge-good">Normal</span>'
      : data.severity === "WARNING"
      ? '<span class="badge badge-moderate">Warning</span>'
      : '<span class="badge badge-alert">Critical</span>';

  let violationsHtml = "";
  if (data.physical_violations.length > 0 || data.statutory_violations.length > 0) {
    const items = [...data.physical_violations, ...data.statutory_violations]
      .map((v) => `<li>${v}</li>`)
      .join("");
    violationsHtml = `<ul style="margin-left: 20px; font-size: 0.88rem; margin-top: 8px;">${items}</ul>`;
  }

  container.innerHTML = `
    <div class="card" style="margin-top: 20px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <h3>Sensor & Limit Audit</h3>
        ${severityBadge}
      </div>
      <p style="font-size: 0.9rem;">${data.status_message}</p>
      ${violationsHtml}
    </div>
  `;
}

// 2. Batch CSV Upload
function setupBatchUploadForm() {
  const form = document.getElementById("batch-upload-form");
  if (!form) return;

  const fileInput = document.getElementById("batch-file-input");
  const resultContainer = document.getElementById("batch-result");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files || fileInput.files.length === 0) {
      alert("Please select a CSV file to upload.");
      return;
    }

    const file = fileInput.files[0];
    if (file.size > 2 * 1024 * 1024) {
      alert("File size exceeds 2 MB limit.");
      return;
    }

    resultContainer.innerHTML = "<p>Processing batch records...</p>";

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/predict/batch`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      displayBatchResult(data);
    } catch (err) {
      resultContainer.innerHTML = `<div class="badge badge-alert">Error: ${err.message}</div>`;
    }
  });
}

function displayBatchResult(data) {
  const container = document.getElementById("batch-result");
  if (!container) return;

  let rowsHtml = "";
  data.results.slice(0, 50).forEach((r) => {
    if (r.error) {
      rowsHtml += `<tr><td>${r.row_index}</td><td colspan="4" style="color: #991b1b;">${r.error}</td></tr>`;
    } else {
      const flagText = r.needs_laboratory_test ? '<span style="color: #991b1b; font-weight: 600;">Needs Lab Test</span>' : "Pass";
      rowsHtml += `
        <tr>
          <td>${r.row_index}</td>
          <td><strong>${r.predicted_class}</strong></td>
          <td>${(r.confidence * 100).toFixed(1)}%</td>
          <td>[ ${r.conformal_set.join(", ")} ]</td>
          <td>${flagText}</td>
        </tr>
      `;
    }
  });

  container.innerHTML = `
    <div style="margin-top: 24px;">
      <p style="font-size: 0.9rem; margin-bottom: 12px;">
        Processed <strong>${data.total_processed}</strong> records. (Displaying top 50 rows).
      </p>
      <table>
        <thead>
          <tr>
            <th>Row</th>
            <th>Screening Tier</th>
            <th>Confidence</th>
            <th>Conformal Set</th>
            <th>Lab Status</th>
          </tr>
        </thead>
        <tbody>
          ${rowsHtml}
        </tbody>
      </table>
    </div>
  `;
}

// 3. Session History (localStorage)
function saveToSessionHistory(pred, sample) {
  const history = JSON.parse(localStorage.getItem("aquasense_history") || "[]");
  history.unshift({
    timestamp: new Date().toISOString(),
    tier: pred.risk_tier,
    confidence: pred.probabilities[pred.risk_tier],
    conformal: pred.conformal_prediction_set,
    needsLab: pred.needs_laboratory_test,
    sample: sample,
  });

  // Keep last 15 items
  if (history.length > 15) history.pop();
  localStorage.setItem("aquasense_history", JSON.stringify(history));
  loadSessionHistory();
}

function loadSessionHistory() {
  const container = document.getElementById("session-history");
  if (!container) return;

  const history = JSON.parse(localStorage.getItem("aquasense_history") || "[]");
  if (history.length === 0) {
    container.innerHTML = "<p style='font-size: 0.88rem; color: var(--text-muted);'>No past screenings in this browser session.</p>";
    return;
  }

  let rows = history
    .map(
      (h) => `
    <tr>
      <td>${new Date(h.timestamp).toLocaleTimeString()}</td>
      <td><strong>${h.tier}</strong></td>
      <td>${(h.confidence * 100).toFixed(1)}%</td>
      <td>${h.needsLab ? "Needs Lab Test" : "Standard"}</td>
    </tr>
  `
    )
    .join("");

  container.innerHTML = `
    <div style="margin-top: 16px;">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Risk Tier</th>
            <th>Confidence</th>
            <th>Audit</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>
      <button id="clear-history-btn" class="btn btn-secondary" style="margin-top: 10px; font-size: 0.8rem; padding: 6px 12px;">
        Clear Session History
      </button>
    </div>
  `;

  document.getElementById("clear-history-btn")?.addEventListener("click", () => {
    localStorage.removeItem("aquasense_history");
    loadSessionHistory();
  });
}
