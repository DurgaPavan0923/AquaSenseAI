/**
 * AquaSense AI Production Client Logic
 * Handles single-sample screening, batch CSV processing, local occlusion attribution,
 * geographic monitoring maps, session history management, accessible lightboxes, accordions,
 * and WCAG AA theme switching.
 * Strict Design: Zero emojis, zero em dashes, zero gradients, border-radius <= 4px.
 */

// 0. Immediate Theme Initialization to prevent theme flicker
(function initTheme() {
  const saved = localStorage.getItem("aquasense_theme");
  const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  const theme = saved || (prefersDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
})();

document.addEventListener("DOMContentLoaded", () => {
  setupThemeToggle();
  setupNavigation();
  setupHomepagePreview();
  setupSinglePredictionForm();
  setupBatchUploadForm();
  setupGeographicMap();
  setupFigureLightbox();
  setupAccordions();
  setupResultsTabs();
  loadSessionHistory();
});

const API_BASE = "";

// Safe HTML Escaping Helper to prevent XSS
function escapeHtml(str) {
  if (str === null || str === undefined) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Resilient API Call Helper with /api fallback
async function callApi(endpoint, options) {
  let res = await fetch(`${API_BASE}${endpoint}`, options);
  if (res.status === 404) {
    const fallbackEndpoint = endpoint.startsWith("/api")
      ? endpoint.replace(/^\/api/, "")
      : `/api${endpoint}`;
    try {
      const fallbackRes = await fetch(`${API_BASE}${fallbackEndpoint}`, options);
      if (fallbackRes.ok || fallbackRes.status !== 404) {
        return fallbackRes;
      }
    } catch {
      // Fallback network error; return original response
    }
  }
  return res;
}

// 1. Theme Toggle
function setupThemeToggle() {
  const toggleBtns = document.querySelectorAll(".theme-toggle-btn");
  if (toggleBtns.length === 0) return;

  const updateIcons = (theme) => {
    toggleBtns.forEach((btn) => {
      btn.setAttribute("aria-label", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
      btn.setAttribute("title", theme === "dark" ? "Switch to light theme" : "Switch to dark theme");
      const sun = btn.querySelector(".sun-icon");
      const moon = btn.querySelector(".moon-icon");
      if (sun && moon) {
        if (theme === "dark") {
          sun.style.display = "inline";
          moon.style.display = "none";
        } else {
          sun.style.display = "none";
          moon.style.display = "inline";
        }
      }
    });
  };

  const currentTheme = document.documentElement.getAttribute("data-theme") || "light";
  updateIcons(currentTheme);

  toggleBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const active = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
      const next = active === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("aquasense_theme", next);
      updateIcons(next);
    });
  });

  if (window.matchMedia) {
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", (e) => {
      if (!localStorage.getItem("aquasense_theme")) {
        const next = e.matches ? "dark" : "light";
        document.documentElement.setAttribute("data-theme", next);
        updateIcons(next);
      }
    });
  }
}

// 2. Mobile Navigation
function setupNavigation() {
  const toggleBtn = document.getElementById("mobile-nav-btn");
  const navMenu = document.getElementById("nav-menu");
  if (toggleBtn && navMenu) {
    toggleBtn.addEventListener("click", () => {
      navMenu.classList.toggle("open");
    });
  }
}

// 3. Homepage Interactive Dashboard Preview
function setupHomepagePreview() {
  const btnGood = document.getElementById("preview-btn-good");
  const btnMod = document.getElementById("preview-btn-mod");
  const btnPoll = document.getElementById("preview-btn-poll");

  if (!btnGood) return;

  const sampleGood = {
    ph: "7.20", turb: "2.50 NTU", temp: "22.0 deg C", do: "6.80 mg/L", bod: "1.80 mg/L",
    lead: "0.0050 mg/L", merc: "0.0004 mg/L", ars: "0.0030 mg/L",
    tier: "Good Risk Tier", badge: "badge-good",
    good: "98.5%", mod: "1.2%", poll: "0.3%",
    cSet: "[ Good ]",
    driver: "Organic Stress Ratio (BOD/DO = 0.26, well below 0.60 threshold)"
  };

  const sampleMod = {
    ph: "6.40", turb: "6.50 NTU", temp: "24.5 deg C", do: "4.90 mg/L", bod: "3.80 mg/L",
    lead: "0.0120 mg/L", merc: "0.0011 mg/L", ars: "0.0120 mg/L",
    tier: "Moderate Risk Tier", badge: "badge-moderate",
    good: "3.4%", mod: "92.1%", poll: "4.5%",
    cSet: "[ Moderate ]",
    driver: "BOD Elevation (3.80 mg/L, exceeding potable baseline +0.34 impact)"
  };

  const samplePoll = {
    ph: "4.50", turb: "38.00 NTU", temp: "28.0 deg C", do: "2.10 mg/L", bod: "18.50 mg/L",
    lead: "0.0450 mg/L", merc: "0.0042 mg/L", ars: "0.0350 mg/L",
    tier: "Polluted Risk Tier", badge: "badge-polluted",
    good: "0.1%", mod: "1.8%", poll: "98.1%",
    cSet: "[ Polluted ]",
    driver: "Heavy Metal Toxicity (Lead & Mercury exceed statutory threshold by 4.5x)"
  };

  function updatePreview(s, activeBtn) {
    document.getElementById("prev-ph").textContent = s.ph;
    document.getElementById("prev-turb").textContent = s.turb;
    document.getElementById("prev-temp").textContent = s.temp;
    document.getElementById("prev-do").textContent = s.do;
    document.getElementById("prev-bod").textContent = s.bod;
    document.getElementById("prev-lead").textContent = s.lead;
    document.getElementById("prev-merc").textContent = s.merc;
    document.getElementById("prev-ars").textContent = s.ars;

    const badge = document.getElementById("prev-tier-badge");
    badge.textContent = s.tier;
    badge.className = `badge ${s.badge}`;

    document.getElementById("prev-prob-good").textContent = s.good;
    document.getElementById("prev-prob-mod").textContent = s.mod;
    document.getElementById("prev-prob-poll").textContent = s.poll;

    document.getElementById("prev-bar-good").style.width = s.good;
    document.getElementById("prev-bar-mod").style.width = s.mod;
    document.getElementById("prev-bar-poll").style.width = s.poll;

    document.getElementById("prev-conformal-set").textContent = s.cSet;

    const driverElem = document.getElementById("prev-attribution");
    if (driverElem) {
      driverElem.textContent = s.driver;
    }

    [btnGood, btnMod, btnPoll].forEach(b => {
      if (b) b.classList.remove("active");
    });
    if (activeBtn) activeBtn.classList.add("active");
  }

  btnGood.addEventListener("click", () => updatePreview(sampleGood, btnGood));
  btnMod.addEventListener("click", () => updatePreview(sampleMod, btnMod));
  btnPoll.addEventListener("click", () => updatePreview(samplePoll, btnPoll));
}

// 4. Single Sample Prediction Form with Live Inline Range Validation
function setupSinglePredictionForm() {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  const btnReset = document.getElementById("btn-reset-form");
  const btnLoadPotable = document.getElementById("btn-load-potable");
  const btnLoadRunoff = document.getElementById("btn-load-runoff");
  const btnLoadEffluent = document.getElementById("btn-load-effluent");
  const btnQuickRun = document.getElementById("btn-quick-run");
  const errorBox = document.getElementById("form-error-box");
  const btnPrintReport = document.getElementById("btn-print-report");
  const submitBtn = document.getElementById("btn-submit-screening");

  // Input references and validation rules
  const validationRules = [
    { id: "input-ph", errId: "err-ph", min: 0.0, max: 14.0, name: "pH" },
    { id: "input-turbidity", errId: "err-turbidity", min: 0.0, max: 500.0, name: "Turbidity" },
    { id: "input-temp", errId: "err-temp", min: -5.0, max: 60.0, name: "Temperature" },
    { id: "input-do", errId: "err-do", min: 0.0, max: 25.0, name: "Dissolved Oxygen" },
    { id: "input-bod", errId: "err-bod", min: 0.0, max: 100.0, name: "BOD" },
    { id: "input-lead", errId: "err-lead", min: 0.0, max: 5.0, name: "Lead" },
    { id: "input-mercury", errId: "err-mercury", min: 0.0, max: 1.0, name: "Mercury" },
    { id: "input-arsenic", errId: "err-arsenic", min: 0.0, max: 5.0, name: "Arsenic" },
  ];

  function validateFormFields() {
    let allValid = true;
    validationRules.forEach((rule) => {
      const input = document.getElementById(rule.id);
      const err = document.getElementById(rule.errId);
      if (!input) return;

      const val = parseFloat(input.value);
      const isInvalid = isNaN(val) || val < rule.min || val > rule.max;

      if (isInvalid) {
        allValid = false;
        input.classList.add("is-invalid");
        input.classList.remove("is-valid");
        if (err) {
          err.classList.add("visible");
          err.textContent = `${rule.name} must be a number between ${rule.min} and ${rule.max}.`;
        }
      } else {
        input.classList.remove("is-invalid");
        input.classList.add("is-valid");
        if (err) {
          err.classList.remove("visible");
        }
      }
    });

    if (submitBtn) {
      submitBtn.disabled = !allValid;
    }
    return allValid;
  }

  validationRules.forEach((rule) => {
    const input = document.getElementById(rule.id);
    if (input) {
      input.addEventListener("input", validateFormFields);
      input.addEventListener("change", validateFormFields);
    }
  });

  // Sample templates
  const templates = {
    potable: {
      id: "SAMP-POT-001", loc: "Northern Reservoir, Station A",
      ph: 7.20, turb: 2.50, temp: 22.0, do: 6.80, bod: 1.80,
      lead: 0.0050, merc: 0.00040, ars: 0.0030
    },
    runoff: {
      id: "SAMP-AGR-002", loc: "Agricultural Tributary Basin",
      ph: 6.40, turb: 6.50, temp: 24.5, do: 4.90, bod: 3.80,
      lead: 0.0120, merc: 0.00110, ars: 0.0120
    },
    effluent: {
      id: "SAMP-IND-003", loc: "Industrial Canal Outfall #4",
      ph: 4.50, turb: 38.00, temp: 28.0, do: 2.10, bod: 18.50,
      lead: 0.0450, merc: 0.00420, ars: 0.0350
    }
  };

  function setFormValues(data) {
    document.getElementById("input-sample-id").value = data.id;
    document.getElementById("input-location").value = data.loc;
    document.getElementById("input-ph").value = data.ph;
    document.getElementById("input-turbidity").value = data.turb;
    document.getElementById("input-temp").value = data.temp;
    document.getElementById("input-do").value = data.do;
    document.getElementById("input-bod").value = data.bod;
    document.getElementById("input-lead").value = data.lead;
    document.getElementById("input-mercury").value = data.merc;
    document.getElementById("input-arsenic").value = data.ars;

    const dtInput = document.getElementById("input-datetime");
    if (dtInput) {
      const now = new Date();
      now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
      dtInput.value = now.toISOString().slice(0, 16);
    }
    validateFormFields();
  }

  // Set initial datetime and run validation check
  const dtInput = document.getElementById("input-datetime");
  if (dtInput) {
    const now = new Date();
    now.setMinutes(now.getMinutes() - now.getTimezoneOffset());
    dtInput.value = now.toISOString().slice(0, 16);
  }
  validateFormFields();

  if (btnLoadPotable) btnLoadPotable.addEventListener("click", () => setFormValues(templates.potable));
  if (btnLoadRunoff) btnLoadRunoff.addEventListener("click", () => setFormValues(templates.runoff));
  if (btnLoadEffluent) btnLoadEffluent.addEventListener("click", () => setFormValues(templates.effluent));
  if (btnQuickRun) {
    btnQuickRun.addEventListener("click", () => {
      setFormValues(templates.potable);
      form.dispatchEvent(new Event("submit"));
    });
  }

  if (btnReset) {
    btnReset.addEventListener("click", () => {
      form.reset();
      setFormValues(templates.potable);
      if (errorBox) errorBox.style.display = "none";
    });
  }

  // Form Submission
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (errorBox) errorBox.style.display = "none";

    if (!validateFormFields()) {
      if (errorBox) {
        errorBox.textContent = "Please correct all invalid parameter values before running screening.";
        errorBox.style.display = "block";
      }
      return;
    }

    const sampleId = document.getElementById("input-sample-id").value.trim() || "SAMP-UNSPECIFIED";
    const location = document.getElementById("input-location").value.trim() || "Field Station";
    const collectionTime = document.getElementById("input-datetime").value || new Date().toISOString();

    const ph = parseFloat(document.getElementById("input-ph").value);
    const turb = parseFloat(document.getElementById("input-turbidity").value);
    const temp = parseFloat(document.getElementById("input-temp").value);
    const dOxygen = parseFloat(document.getElementById("input-do").value);
    const bod = parseFloat(document.getElementById("input-bod").value);
    const lead = parseFloat(document.getElementById("input-lead").value);
    const merc = parseFloat(document.getElementById("input-mercury").value);
    const ars = parseFloat(document.getElementById("input-arsenic").value);

    const payload = {
      sample_id: sampleId,
      location: location,
      collection_time: collectionTime,
      pH: ph,
      Turbidity: turb,
      Temperature: temp,
      "Dissolved Oxygen": dOxygen,
      "Biochemical Oxygen Demand": bod,
      Lead: lead,
      Mercury: merc,
      Arsenic: ars,
    };

    if (submitBtn) {
      submitBtn.disabled = true;
      submitBtn.textContent = "Processing Calibrated Screening Inference...";
    }

    try {
      // 1. Prediction Call with automatic API path resolution
      const predRes = await callApi("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!predRes.ok) {
        const errJson = await predRes.json().catch(() => ({}));
        throw new Error(errJson.detail || `Prediction failed with HTTP ${predRes.status}`);
      }
      const predData = await predRes.json();

      // 2. Local Occlusion Attribution Call with automatic API path resolution
      let expData = null;
      try {
        const expRes = await callApi("/explain", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (expRes.ok) {
          expData = await expRes.json();
        }
      } catch (e) {
        console.warn("Attribution endpoint failed:", e);
      }

      // Render Complete Results
      renderAssessmentResult(predData, payload, expData);

      // Save to Session History
      saveToSessionHistory(predData, payload);

    } catch (err) {
      if (errorBox) {
        errorBox.textContent = `Server screening error: ${err.message}`;
        errorBox.style.display = "block";
      }
    } finally {
      if (submitBtn) {
        submitBtn.disabled = false;
        submitBtn.textContent = "Run Environmental Risk Screening";
      }
    }
  });

  if (btnPrintReport) {
    btnPrintReport.addEventListener("click", () => {
      window.print();
    });
  }
}

function renderAssessmentResult(data, sample, expData) {
  const container = document.getElementById("assessment-container");
  const attributionCard = document.getElementById("attribution-card");
  const observationsCard = document.getElementById("observations-card");
  const btnPrintReport = document.getElementById("btn-print-report");

  if (!container) return;

  const tier = data.risk_tier;
  const badgeClass =
    tier === "Good" ? "badge-good" : tier === "Moderate" ? "badge-moderate" : "badge-polluted";

  const pGood = (data.probabilities.Good * 100).toFixed(1);
  const pMod = (data.probabilities.Moderate * 100).toFixed(1);
  const pPoll = (data.probabilities.Polluted * 100).toFixed(1);

  const labFlagHtml = data.needs_laboratory_test
    ? `<div style="background-color: var(--status-polluted-bg); border: 1px solid var(--status-polluted-border); color: var(--status-polluted); padding: 12px; border-radius: 4px; margin-top: 14px; font-weight: 600; font-size: 0.9rem;">
        Action Required: Certified Laboratory Testing Recommended (${escapeHtml(data.confidence_assessment)})
       </div>`
    : `<div style="background-color: var(--status-good-bg); border: 1px solid var(--status-good-border); color: var(--status-good); padding: 10px; border-radius: 4px; margin-top: 14px; font-size: 0.88rem; font-weight: 600;">
        High Model Confidence: Screening prediction set contains single risk tier.
       </div>`;

  const meta = data.sample_metadata || {};
  const metaSampleId = escapeHtml(meta.sample_id || sample.sample_id || "SAMP-001");
  const metaLocation = escapeHtml(meta.location || sample.location || "Station");
  const metaTime = escapeHtml(meta.analysis_timestamp || new Date().toISOString());

  container.innerHTML = `
    <!-- Assessment Report Summary Header -->
    <div style="margin-bottom: 16px;">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; margin-bottom: 8px;">
        <span class="badge ${badgeClass}" style="font-size: 1.15rem; padding: 6px 16px;">
          ${escapeHtml(tier)} Water Quality Risk
        </span>
        <span style="font-size: 0.82rem; color: var(--text-muted);">
          Coverage Guarantee: ${escapeHtml(data.conformal_coverage_guarantee || "90% Marginal")}
        </span>
      </div>

      <div style="font-size: 0.82rem; color: var(--text-light); margin-bottom: 12px;">
        <strong>Sample ID:</strong> ${metaSampleId} | <strong>Location:</strong> ${metaLocation} | <strong>Timestamp:</strong> ${metaTime}
      </div>
    </div>

    <!-- Probability Distribution Bars -->
    <div style="background-color: var(--bg-primary); border: 1px solid var(--border-subtle); border-radius: 4px; padding: 16px; margin-bottom: 16px;">
      <div style="font-size: 0.85rem; font-weight: 700; color: var(--text-main); margin-bottom: 10px;">
        Calibrated Class Probability Distribution:
      </div>
      <div class="prob-item">
        <div class="prob-header"><span>Good Quality (Potable / Aquatic Baseline)</span><span>${pGood}%</span></div>
        <div class="prob-track"><div class="prob-fill-good" style="width: ${pGood}%;"></div></div>
      </div>
      <div class="prob-item">
        <div class="prob-header"><span>Moderate Risk (Elevated Runoff / Advisory)</span><span>${pMod}%</span></div>
        <div class="prob-track"><div class="prob-fill-moderate" style="width: ${pMod}%;"></div></div>
      </div>
      <div class="prob-item">
        <div class="prob-header"><span>Polluted Risk (Industrial / Sewage Discharge)</span><span>${pPoll}%</span></div>
        <div class="prob-track"><div class="prob-fill-polluted" style="width: ${pPoll}%;"></div></div>
      </div>
    </div>

    <!-- Conformal Prediction Set Details -->
    <div style="border: 1px solid var(--border-subtle); border-radius: 4px; padding: 14px; margin-bottom: 16px;">
      <div style="font-size: 0.88rem; margin-bottom: 4px;">
        <strong>Conformal Prediction Set C(x):</strong> [ ${data.conformal_prediction_set.map(c => `<strong>${escapeHtml(c)}</strong>`).join(", ")} ]
      </div>
      <p style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0;">
        Under exchangeability, the true environmental risk category belongs to this set with at least 90% finite-sample probability.
      </p>
    </div>

    <!-- Statutory Laboratory Recommendation -->
    <div style="border-left: 4px solid var(--accent); background-color: var(--bg-primary); padding: 12px 16px; border-radius: 0 4px 4px 0; font-size: 0.88rem; margin-bottom: 14px;">
      <strong>Actionable Recommendation:</strong> ${escapeHtml(data.laboratory_recommendation || data.confidence_assessment)}
    </div>

    ${labFlagHtml}
  `;

  if (btnPrintReport) btnPrintReport.style.display = "inline-block";

  // Render Local Feature Attribution Table with Horizontal Growing Bars (occlusion-based, approximate)
  if (attributionCard && expData && expData.feature_attributions) {
    const tbody = document.getElementById("attribution-tbody");
    if (tbody) {
      let rowsHtml = "";
      const baselineMedians = {
        pH: 7.15, Turbidity: 2.90, Temperature: 22.1,
        "Dissolved Oxygen": 5.62, "Biochemical Oxygen Demand": 2.35,
        Lead: 0.0058, Mercury: 0.00037, Arsenic: 0.0056
      };

      const maxAttr = Math.max(...Object.values(expData.feature_attributions).map(v => Math.abs(v)), 0.001);

      for (const [feat, val] of Object.entries(expData.feature_attributions)) {
        const sign = val >= 0 ? "+" : "";
        const color = val > 0 ? "var(--status-polluted)" : "var(--status-good)";
        const sampleVal = sample[feat] !== undefined ? sample[feat] : "-";
        const medVal = baselineMedians[feat] !== undefined ? baselineMedians[feat] : "-";
        const barWidthPct = Math.min(100, Math.round((Math.abs(val) / maxAttr) * 100));

        rowsHtml += `
          <tr>
            <td>
              <strong>${escapeHtml(feat)}</strong>
              <div class="attribution-bar-track">
                <div class="attribution-bar-fill" style="width: 0%; background-color: ${color};" data-target-width="${barWidthPct}%"></div>
              </div>
            </td>
            <td>${sampleVal}</td>
            <td>${medVal}</td>
            <td style="color: ${color}; font-weight: 700;">${sign}${val.toFixed(4)}</td>
          </tr>
        `;
      }
      tbody.innerHTML = rowsHtml;
      attributionCard.style.display = "block";

      // Animate bar growth once horizontally with 300ms transition
      setTimeout(() => {
        tbody.querySelectorAll(".attribution-bar-fill").forEach(bar => {
          bar.style.width = bar.dataset.targetWidth;
        });
      }, 50);
    }
  }

  // Render Granular Parameter Observations Table
  if (observationsCard && data.parameter_observations && data.parameter_observations.length > 0) {
    const obsBody = document.getElementById("observations-tbody");
    if (obsBody) {
      let obsHtml = "";
      data.parameter_observations.forEach((o) => {
        const statusBadge =
          o.status === "Compliant" || o.status === "Ambient"
            ? '<span class="badge badge-good">Compliant</span>'
            : '<span class="badge badge-polluted">Breached</span>';

        obsHtml += `
          <tr>
            <td><strong>${escapeHtml(o.parameter)}</strong></td>
            <td>${o.value} ${escapeHtml(o.unit)}</td>
            <td>${statusBadge}</td>
            <td style="font-size: 0.8rem; color: var(--text-muted);">${escapeHtml(o.authority || "WHO / BIS")}</td>
            <td style="font-size: 0.85rem;">${escapeHtml(o.observation)}</td>
          </tr>
        `;
      });
      obsBody.innerHTML = obsHtml;
      observationsCard.style.display = "block";
    }
  }
}

// 5. Batch CSV Processing with Drag-and-Drop, File Checks, Formula Injection Sanitization & Table Sorting
let currentBatchData = null;
let currentSortColumn = "row_index";
let currentSortAsc = true;

// CSV Formula Injection Sanitizer: prepends single quote if cell starts with =, +, -, or @
function sanitizeCsvCell(val) {
  if (val === null || val === undefined) return '""';
  let str = String(val).replace(/"/g, '""');
  if (str.length > 0 && ["=", "+", "-", "@"].includes(str.charAt(0))) {
    str = "'" + str;
  }
  return `"${str}"`;
}

function setupBatchUploadForm() {
  const form = document.getElementById("batch-upload-form");
  const fileInput = document.getElementById("batch-file-input");
  const dropzone = document.getElementById("batch-dropzone");
  const dropzoneText = document.getElementById("dropzone-text");
  const btnDemoBatch = document.getElementById("btn-load-sample-batch");
  const errorBox = document.getElementById("batch-upload-error");
  const btnDownloadCsv = document.getElementById("btn-download-batch-csv");
  const btnPrintBatch = document.getElementById("btn-print-batch");
  const filterInput = document.getElementById("batch-filter-input");
  const tierSelect = document.getElementById("batch-tier-select");

  if (!form) return;

  // Drag and Drop Zone Listeners
  if (dropzone && fileInput) {
    dropzone.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput.click();
      }
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.add("dragover");
      });
    });

    ["dragleave", "drop"].forEach((eventName) => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropzone.classList.remove("dragover");
      });
    });

    dropzone.addEventListener("drop", async (e) => {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const file = e.dataTransfer.files[0];
        if (dropzoneText) dropzoneText.textContent = `Selected: ${file.name}`;
        await processBatchFile(file);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files.length > 0 && dropzoneText) {
        dropzoneText.textContent = `Selected: ${fileInput.files[0].name}`;
      }
    });
  }

  // Demo Batch CSV Loader
  if (btnDemoBatch) {
    btnDemoBatch.addEventListener("click", async () => {
      const demoCsv =
        "Sample_ID,pH,Turbidity,Temperature,Dissolved Oxygen,Biochemical Oxygen Demand,Lead,Mercury,Arsenic\n" +
        "STN-001,7.20,2.50,22.0,6.80,1.80,0.0050,0.00040,0.0030\n" +
        "STN-002,7.40,1.80,21.5,7.10,1.40,0.0030,0.00020,0.0020\n" +
        "STN-003,6.40,6.50,24.5,4.90,3.80,0.0120,0.00110,0.0120\n" +
        "STN-004,6.20,7.20,25.0,4.60,4.20,0.0140,0.00130,0.0140\n" +
        "STN-005,4.50,38.00,28.0,2.10,18.50,0.0450,0.00420,0.0350\n" +
        "STN-006,3.80,55.00,29.5,1.50,24.00,0.0650,0.00650,0.0550\n" +
        "STN-007,7.10,3.10,22.2,6.50,2.10,0.0060,0.00050,0.0040\n" +
        "STN-008,6.80,4.20,23.0,5.80,2.80,0.0090,0.00080,0.0080\n" +
        "STN-009,4.90,32.00,27.0,2.50,15.20,0.0380,0.00380,0.0320\n" +
        "STN-010,7.30,2.20,21.8,7.00,1.60,0.0040,0.00030,0.0025\n";

      const blob = new Blob([demoCsv], { type: "text/csv" });
      const file = new File([blob], "demo_batch.csv", { type: "text/csv" });
      if (dropzoneText) dropzoneText.textContent = "Selected: demo_batch.csv (10 Samples)";
      await processBatchFile(file);
    });
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!fileInput.files || fileInput.files.length === 0) {
      if (errorBox) {
        errorBox.textContent = "Please select or drop a valid CSV file to upload.";
        errorBox.style.display = "block";
      }
      return;
    }
    const file = fileInput.files[0];
    await processBatchFile(file);
  });

  async function processBatchFile(file) {
    if (errorBox) errorBox.style.display = "none";
    
    // Client-side file size check (5 MB limit)
    if (file.size > 5 * 1024 * 1024) {
      if (errorBox) {
        errorBox.textContent = "File size exceeds 5 MB maximum allowable limit.";
        errorBox.style.display = "block";
      }
      return;
    }

    const resultContainer = document.getElementById("batch-result");
    if (resultContainer) {
      resultContainer.innerHTML = "<p style='text-align: center; padding: 40px;'>Processing multi-sample batch screening...</p>";
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await callApi("/predict/batch", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Upload failed with status ${res.status}`);
      }

      const data = await res.json();
      currentBatchData = data;
      renderBatchResults(data);

    } catch (err) {
      if (errorBox) {
        errorBox.textContent = `Batch processing error: ${err.message}`;
        errorBox.style.display = "block";
      }
      if (resultContainer) {
        resultContainer.innerHTML = `<p style="color: var(--status-polluted); text-align: center; padding: 30px;">Failed to evaluate batch: ${escapeHtml(err.message)}</p>`;
      }
    }
  }

  // Filter input listeners
  if (filterInput && tierSelect) {
    const handleFilter = () => {
      if (!currentBatchData) return;
      const q = filterInput.value.toLowerCase().trim();
      const tier = tierSelect.value;

      const filtered = currentBatchData.results.filter((r) => {
        const matchesQuery =
          !q ||
          (r.sample_id && r.sample_id.toLowerCase().includes(q)) ||
          (r.predicted_class && r.predicted_class.toLowerCase().includes(q));
        const matchesTier = tier === "all" || r.predicted_class === tier;
        return matchesQuery && matchesTier;
      });

      renderBatchTableRows(filtered);
    };

    filterInput.addEventListener("input", handleFilter);
    tierSelect.addEventListener("change", handleFilter);
  }

  // Export Results CSV with Formula Injection Sanitization
  if (btnDownloadCsv) {
    btnDownloadCsv.addEventListener("click", () => {
      if (!currentBatchData || !currentBatchData.results) return;
      let csvContent = "Row,Sample_ID,Risk_Tier,Confidence,Prob_Good,Prob_Moderate,Prob_Polluted,Conformal_Set,Lab_Verification_Required\n";

      currentBatchData.results.forEach((r) => {
        if (!r.error) {
          const cSet = sanitizeCsvCell(r.conformal_set.join(";"));
          const pGood = r.probabilities ? r.probabilities.Good : "";
          const pMod = r.probabilities ? r.probabilities.Moderate : "";
          const pPoll = r.probabilities ? r.probabilities.Polluted : "";
          csvContent += `${r.row_index},${sanitizeCsvCell(r.sample_id || "")},${sanitizeCsvCell(r.predicted_class)},${r.confidence},${pGood},${pMod},${pPoll},${cSet},${r.needs_laboratory_test ? "YES" : "NO"}\n`;
        }
      });

      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `aquasense_batch_results_${Date.now()}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    });
  }

  if (btnPrintBatch) {
    btnPrintBatch.addEventListener("click", () => {
      window.print();
    });
  }
}

function renderBatchResults(data) {
  const summarySec = document.getElementById("batch-summary-section");
  if (summarySec) summarySec.style.display = "block";

  const sum = data.batch_summary || {};
  const total = data.total_processed || 0;
  const good = sum.good_samples_count || 0;
  const mod = sum.moderate_samples_count || 0;
  const poll = sum.polluted_samples_count || 0;

  document.getElementById("batch-stat-total").textContent = total;
  document.getElementById("batch-stat-good").textContent = good;
  document.getElementById("batch-stat-mod").textContent = mod;
  document.getElementById("batch-stat-poll").textContent = poll;
  document.getElementById("batch-id-label").textContent = `ID: ${data.batch_id || "BATCH"}`;

  const pctGood = total > 0 ? ((good / total) * 100).toFixed(1) : 0;
  const pctMod = total > 0 ? ((mod / total) * 100).toFixed(1) : 0;
  const pctPoll = total > 0 ? ((poll / total) * 100).toFixed(1) : 0;

  document.getElementById("batch-pct-good").textContent = `${pctGood}% (${good})`;
  document.getElementById("batch-pct-mod").textContent = `${pctMod}% (${mod})`;
  document.getElementById("batch-pct-poll").textContent = `${pctPoll}% (${poll})`;

  document.getElementById("batch-bar-good").style.width = `${pctGood}%`;
  document.getElementById("batch-bar-mod").style.width = `${pctMod}%`;
  document.getElementById("batch-bar-poll").style.width = `${pctPoll}%`;

  document.getElementById("batch-stat-needs-lab").textContent = sum.needs_laboratory_test_count || 0;
  document.getElementById("batch-stat-anomalies").textContent = sum.anomalous_samples_count || 0;

  renderBatchTableRows(data.results);
}

function renderBatchTableRows(rows) {
  const container = document.getElementById("batch-result");
  if (!container) return;

  if (!rows || rows.length === 0) {
    container.innerHTML = "<p style='text-align: center; padding: 30px; color: var(--text-muted);'>No matching sample records found.</p>";
    return;
  }

  let tbodyHtml = "";
  rows.forEach((r) => {
    if (r.error) {
      tbodyHtml += `
        <tr>
          <td>${r.row_index}</td>
          <td>${escapeHtml(r.sample_id || "-")}</td>
          <td colspan="5" style="color: var(--status-polluted); font-weight: 600;">${escapeHtml(r.error)}</td>
        </tr>
      `;
    } else {
      const tierBadge =
        r.predicted_class === "Good"
          ? '<span class="badge badge-good">Good</span>'
          : r.predicted_class === "Moderate"
          ? '<span class="badge badge-moderate">Moderate</span>'
          : '<span class="badge badge-polluted">Polluted</span>';

      const labFlag = r.needs_laboratory_test
        ? '<span style="color: var(--status-polluted); font-weight: 700;">Needs Lab Test</span>'
        : '<span style="color: var(--status-good); font-weight: 600;">Pass</span>';

      tbodyHtml += `
        <tr>
          <td>${r.row_index}</td>
          <td><strong>${escapeHtml(r.sample_id || `ROW-${r.row_index}`)}</strong></td>
          <td>${tierBadge}</td>
          <td>${(r.confidence * 100).toFixed(1)}%</td>
          <td>[ ${r.conformal_set.map(c => escapeHtml(c)).join(", ")} ]</td>
          <td>${labFlag}</td>
        </tr>
      `;
    }
  });

  container.innerHTML = `
    <div class="table-wrapper">
      <table id="batch-results-table">
        <thead>
          <tr>
            <th class="sortable ${currentSortColumn === 'row_index' ? (currentSortAsc ? 'sorted-asc' : 'sorted-desc') : ''}" data-col="row_index">Row</th>
            <th class="sortable ${currentSortColumn === 'sample_id' ? (currentSortAsc ? 'sorted-asc' : 'sorted-desc') : ''}" data-col="sample_id">Sample Identifier</th>
            <th class="sortable ${currentSortColumn === 'predicted_class' ? (currentSortAsc ? 'sorted-asc' : 'sorted-desc') : ''}" data-col="predicted_class">Screening Tier</th>
            <th class="sortable ${currentSortColumn === 'confidence' ? (currentSortAsc ? 'sorted-asc' : 'sorted-desc') : ''}" data-col="confidence">Model Confidence</th>
            <th>Conformal Set C(x)</th>
            <th class="sortable ${currentSortColumn === 'needs_laboratory_test' ? (currentSortAsc ? 'sorted-asc' : 'sorted-desc') : ''}" data-col="needs_laboratory_test">Laboratory Triage</th>
          </tr>
        </thead>
        <tbody>
          ${tbodyHtml}
        </tbody>
      </table>
    </div>
  `;

  // Attach sort listeners to table headers
  const ths = container.querySelectorAll("th.sortable");
  ths.forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (currentSortColumn === col) {
        currentSortAsc = !currentSortAsc;
      } else {
        currentSortColumn = col;
        currentSortAsc = true;
      }

      rows.sort((a, b) => {
        let valA = a[col];
        let valB = b[col];
        if (typeof valA === "string") valA = valA.toLowerCase();
        if (typeof valB === "string") valB = valB.toLowerCase();
        if (valA < valB) return currentSortAsc ? -1 : 1;
        if (valA > valB) return currentSortAsc ? 1 : -1;
        return 0;
      });

      renderBatchTableRows(rows);
    });
  });
}

// 6. Accessible Lightbox Modal for Diagnostic Figures
function setupFigureLightbox() {
  const modal = document.getElementById("lightbox-modal");
  if (!modal) return;
  const modalImg = document.getElementById("lightbox-img");
  const modalTitle = document.getElementById("lightbox-title");
  const modalCaption = document.getElementById("lightbox-caption");
  const closeBtn = document.getElementById("lightbox-close-btn");
  let lastFocusedElement = null;

  function openLightbox(imgSrc, title, caption, triggerElem) {
    lastFocusedElement = triggerElem;
    modalImg.src = imgSrc;
    modalImg.alt = title;
    modalTitle.textContent = title;
    modalCaption.textContent = caption;
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
    document.body.style.overflow = "hidden";
    if (closeBtn) closeBtn.focus();
  }

  function closeLightbox() {
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
    document.body.style.overflow = "";
    if (lastFocusedElement) lastFocusedElement.focus();
  }

  document.querySelectorAll(".figure-container").forEach((fig) => {
    const img = fig.querySelector(".figure-img");
    const title = fig.querySelector("h4") ? fig.querySelector("h4").textContent : "Diagnostic Figure";
    const caption = fig.querySelector(".figure-caption") ? fig.querySelector(".figure-caption").textContent : "";
    if (img) {
      fig.style.cursor = "pointer";
      fig.setAttribute("tabindex", "0");
      fig.setAttribute("role", "button");
      fig.setAttribute("aria-label", `View full resolution: ${title}`);
      const handler = () => openLightbox(img.src, title, caption, fig);
      fig.addEventListener("click", handler);
      fig.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          handler();
        }
      });
    }
  });

  if (closeBtn) closeBtn.addEventListener("click", closeLightbox);
  modal.addEventListener("click", (e) => {
    if (e.target === modal) closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("open")) {
      closeLightbox();
    }
  });
}

// 7. Accessible Accordions
function setupAccordions() {
  const headers = document.querySelectorAll(".accordion-header");
  headers.forEach((header) => {
    header.addEventListener("click", () => {
      const isExpanded = header.getAttribute("aria-expanded") === "true";
      const targetId = header.getAttribute("aria-controls");
      const body = document.getElementById(targetId);
      if (!body) return;

      if (isExpanded) {
        header.setAttribute("aria-expanded", "false");
        body.hidden = true;
      } else {
        header.setAttribute("aria-expanded", "true");
        body.hidden = false;
      }
    });
  });
}

// 8. Results Page Tabbed Interface
function setupResultsTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  if (tabBtns.length === 0) return;

  tabBtns.forEach((btn, idx) => {
    btn.addEventListener("click", () => {
      const targetPanelId = btn.getAttribute("aria-controls");
      tabBtns.forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      document.querySelectorAll(".tab-panel").forEach(panel => {
        panel.classList.remove("active");
        panel.hidden = true;
      });

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");
      const targetPanel = document.getElementById(targetPanelId);
      if (targetPanel) {
        targetPanel.classList.add("active");
        targetPanel.hidden = false;
      }
    });

    btn.addEventListener("keydown", (e) => {
      let nextIdx = null;
      if (e.key === "ArrowRight") nextIdx = (idx + 1) % tabBtns.length;
      else if (e.key === "ArrowLeft") nextIdx = (idx - 1 + tabBtns.length) % tabBtns.length;
      if (nextIdx !== null) {
        e.preventDefault();
        tabBtns[nextIdx].focus();
        tabBtns[nextIdx].click();
      }
    });
  });
}

// 9. Geographic Map Setup (Leaflet)
function setupGeographicMap() {
  const mapElement = document.getElementById("monitoring-map");
  if (!mapElement) return;

  const stations = [
    { id: "STN-BBS-01", name: "Kuakhai River Barrage (North Intake)", lat: 20.3540, lng: 85.8650, tier: "Good", ph: 7.3, turb: 2.2, do: 6.9, bod: 1.6, lead: 0.004 },
    { id: "STN-BBS-02", name: "Daya River Confluence (South Catchment)", lat: 20.2180, lng: 85.8520, tier: "Moderate", ph: 6.5, turb: 6.8, do: 4.8, bod: 3.9, lead: 0.012 },
    { id: "STN-BBS-03", name: "Bindu Sagar Heritage Lake (Old Town)", lat: 20.2415, lng: 85.8340, tier: "Moderate", ph: 6.7, turb: 7.2, do: 4.5, bod: 4.1, lead: 0.009 },
    { id: "STN-BBS-04", name: "Gangua Nullah Outfall Canal", lat: 20.2520, lng: 85.8610, tier: "Polluted", ph: 4.5, turb: 42.0, do: 1.8, bod: 22.0, lead: 0.048 },
    { id: "STN-BBS-05", name: "Mancheswar Industrial Runoff Sluice", lat: 20.3150, lng: 85.8600, tier: "Polluted", ph: 4.2, turb: 52.0, do: 1.4, bod: 26.5, lead: 0.052 },
    { id: "STN-BBS-06", name: "Chandaka Wetland & Reservoir Intake", lat: 20.3700, lng: 85.7750, tier: "Good", ph: 7.4, turb: 1.9, do: 7.3, bod: 1.3, lead: 0.003 },
    { id: "STN-BBS-07", name: "Vani Vihar Lake Ecological Point", lat: 20.3015, lng: 85.8450, tier: "Good", ph: 7.1, turb: 2.8, do: 6.6, bod: 1.8, lead: 0.005 },
    { id: "STN-BBS-08", name: "Daya West Canal Agricultural Inflow", lat: 20.2750, lng: 85.8050, tier: "Good", ph: 7.2, turb: 2.5, do: 6.8, bod: 1.7, lead: 0.004 },
  ];

  const detailBody = document.getElementById("station-detail-body");
  const filterSelect = document.getElementById("map-filter-tier");

  function showStationDetail(stn) {
    if (!detailBody) return;
    const badgeClass =
      stn.tier === "Good" ? "badge-good" : stn.tier === "Moderate" ? "badge-moderate" : "badge-polluted";

    detailBody.innerHTML = `
      <div style="margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center;">
        <span class="badge ${badgeClass}" style="font-size: 0.95rem; padding: 4px 10px;">${escapeHtml(stn.tier)} Risk</span>
        <span style="font-size: 0.8rem; color: var(--text-light);">${escapeHtml(stn.id)}</span>
      </div>
      <h4>${escapeHtml(stn.name)}</h4>
      <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 12px;">
        Coordinates: ${stn.lat.toFixed(4)}, ${stn.lng.toFixed(4)} (Bhubaneswar Hydrological Monitoring Point)
      </p>

      <table style="font-size: 0.82rem; margin-bottom: 14px;">
        <tbody>
          <tr><td>pH</td><td style="font-weight: 600;">${stn.ph}</td><td>Turbidity</td><td style="font-weight: 600;">${stn.turb} NTU</td></tr>
          <tr><td>Dissolved Oxygen</td><td style="font-weight: 600;">${stn.do} mg/L</td><td>BOD (5-Day)</td><td style="font-weight: 600;">${stn.bod} mg/L</td></tr>
          <tr><td>Lead (Pb)</td><td style="font-weight: 600;">${stn.lead} mg/L</td><td>Conformal Set</td><td style="font-weight: 600;">[ ${stn.tier} ]</td></tr>
        </tbody>
      </table>

      <a href="/analyze.html" class="btn btn-primary btn-sm" style="width: 100%;">
        Open Station Parameters in Analysis Workspace
      </a>
    `;
  }

  if (typeof L !== "undefined") {
    try {
      const map = L.map("monitoring-map").setView([20.2961, 85.8245], 12);
      L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 18,
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      }).addTo(map);

      const markers = [];

      function getMarkerColor(tier) {
        if (tier === "Good") return "#166534";
        if (tier === "Moderate") return "#9a3412";
        return "#991b1b";
      }

      function addStationMarkers(filterTier) {
        markers.forEach(m => map.removeLayer(m));
        markers.length = 0;

        stations.forEach((stn) => {
          if (filterTier !== "all" && stn.tier !== filterTier) return;

          const color = getMarkerColor(stn.tier);
          const circle = L.circleMarker([stn.lat, stn.lng], {
            radius: 8,
            fillColor: color,
            color: "#ffffff",
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9,
          }).addTo(map);

          circle.bindPopup(`<strong>${escapeHtml(stn.name)}</strong><br>Tier: ${escapeHtml(stn.tier)}<br>pH: ${stn.ph}, DO: ${stn.do} mg/L`);
          circle.on("click", () => showStationDetail(stn));
          markers.push(circle);
        });
      }

      addStationMarkers("all");

      if (filterSelect) {
        filterSelect.addEventListener("change", () => {
          addStationMarkers(filterSelect.value);
        });
      }

      const coordForm = document.getElementById("custom-coord-form");
      if (coordForm) {
        coordForm.addEventListener("submit", (e) => {
          e.preventDefault();
          const lat = parseFloat(document.getElementById("input-custom-lat").value);
          const lng = parseFloat(document.getElementById("input-custom-lng").value);
          const name = document.getElementById("input-custom-name").value.trim() || "Field Station";

          if (isNaN(lat) || isNaN(lng)) return;

          const newStn = {
            id: `CUSTOM-${Date.now().toString().slice(-4)}`,
            name: name,
            lat: lat,
            lng: lng,
            tier: "Good",
            ph: 7.2,
            turb: 2.5,
            do: 6.8,
            bod: 1.8,
            lead: 0.005,
          };

          stations.push(newStn);
          map.setView([lat, lng], 12);

          const circle = L.circleMarker([lat, lng], {
            radius: 9,
            fillColor: "#0284c7",
            color: "#ffffff",
            weight: 2,
            opacity: 1,
            fillOpacity: 0.9,
          }).addTo(map);

          circle.bindPopup(`<strong>${escapeHtml(name)}</strong><br>Custom User Coordinate`).openPopup();
          circle.on("click", () => showStationDetail(newStn));
          markers.push(circle);

          showStationDetail(newStn);
        });
      }

    } catch (err) {
      console.warn("Leaflet map initialization failed:", err);
      renderFallbackMapSvg(mapElement, stations, showStationDetail);
    }
  } else {
    renderFallbackMapSvg(mapElement, stations, showStationDetail);
  }
}

function renderFallbackMapSvg(container, stations, onSelect) {
  container.innerHTML = `
    <div style="padding: 24px; text-align: center; color: var(--text-main);">
      <h4>Regional Station Network (Vector View)</h4>
      <p style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 14px;">Select a simulated station below to inspect measurements:</p>
      <div style="display: flex; flex-direction: column; gap: 8px; max-height: 380px; overflow-y: auto;">
        ${stations.map(s => `
          <button type="button" class="btn btn-secondary btn-sm" style="text-align: left; justify-content: space-between; display: flex;" onclick='window.__selectStation(${JSON.stringify(s)})'>
            <span><strong>${escapeHtml(s.name)}</strong> (${s.lat.toFixed(2)}, ${s.lng.toFixed(2)})</span>
            <span class="badge ${s.tier === 'Good' ? 'badge-good' : s.tier === 'Moderate' ? 'badge-moderate' : 'badge-polluted'}">${s.tier}</span>
          </button>
        `).join("")}
      </div>
    </div>
  `;
  window.__selectStation = onSelect;
}

// 10. Session History (localStorage)
function saveToSessionHistory(pred, sample) {
  const history = JSON.parse(localStorage.getItem("aquasense_history") || "[]");
  const meta = pred.sample_metadata || {};

  history.unshift({
    timestamp: meta.analysis_timestamp || new Date().toISOString(),
    sampleId: meta.sample_id || sample.sample_id || `SAMP-${Date.now().toString().slice(-4)}`,
    location: meta.location || sample.location || "Station",
    tier: pred.risk_tier,
    confidence: pred.probabilities ? pred.probabilities[pred.risk_tier] : 0.95,
    conformal: pred.conformal_prediction_set || [pred.risk_tier],
    needsLab: pred.needs_laboratory_test || false,
    sample: sample,
  });

  if (history.length > 25) history.pop();
  localStorage.setItem("aquasense_history", JSON.stringify(history));
  loadSessionHistory();
}

function loadSessionHistory() {
  const container = document.getElementById("session-history-container");
  if (!container) return;

  const history = JSON.parse(localStorage.getItem("aquasense_history") || "[]");
  if (history.length === 0) {
    container.innerHTML = "<p style='font-size: 0.88rem; color: var(--text-light);'>No past evaluations recorded in this browser session.</p>";
    return;
  }

  let rowsHtml = "";
  history.forEach((h) => {
    const tierBadge =
      h.tier === "Good"
        ? '<span class="badge badge-good">Good</span>'
        : h.tier === "Moderate"
        ? '<span class="badge badge-moderate">Moderate</span>'
        : '<span class="badge badge-polluted">Polluted</span>';

    const labStatus = h.needsLab
      ? '<span style="color: var(--status-polluted); font-weight: 700;">Needs Lab Test</span>'
      : '<span style="color: var(--status-good); font-weight: 600;">Pass</span>';

    const dateStr = new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    rowsHtml += `
      <tr>
        <td>${dateStr}</td>
        <td><strong>${escapeHtml(h.sampleId)}</strong></td>
        <td>${escapeHtml(h.location)}</td>
        <td>${tierBadge}</td>
        <td>${(h.confidence * 100).toFixed(1)}%</td>
        <td>[ ${h.conformal.map(c => escapeHtml(c)).join(", ")} ]</td>
        <td>${labStatus}</td>
      </tr>
    `;
  });

  container.innerHTML = `
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Time</th>
            <th>Sample ID</th>
            <th>Station / Location</th>
            <th>Screening Tier</th>
            <th>Confidence</th>
            <th>Conformal Set C(x)</th>
            <th>Laboratory Triage</th>
          </tr>
        </thead>
        <tbody>
          ${rowsHtml}
        </tbody>
      </table>
    </div>
  `;

  const btnClear = document.getElementById("btn-clear-history");
  if (btnClear) {
    btnClear.onclick = () => {
      if (confirm("Are you sure you want to clear your local session history?")) {
        localStorage.removeItem("aquasense_history");
        loadSessionHistory();
      }
    };
  }

  const btnExportJson = document.getElementById("btn-export-history-json");
  if (btnExportJson) {
    btnExportJson.onclick = () => {
      const blob = new Blob([JSON.stringify(history, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `aquasense_history_${Date.now()}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };
  }

  const btnExportCsv = document.getElementById("btn-export-history-csv");
  if (btnExportCsv) {
    btnExportCsv.onclick = () => {
      let csv = "Timestamp,Sample_ID,Location,Risk_Tier,Confidence,Conformal_Set,Needs_Lab_Test\n";
      history.forEach(h => {
        csv += `${sanitizeCsvCell(h.timestamp)},${sanitizeCsvCell(h.sampleId)},${sanitizeCsvCell(h.location)},${sanitizeCsvCell(h.tier)},${h.confidence},${sanitizeCsvCell(h.conformal.join(';'))},${h.needsLab ? 'YES' : 'NO'}\n`;
      });
      const blob = new Blob([csv], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `aquasense_history_${Date.now()}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    };
  }
}
