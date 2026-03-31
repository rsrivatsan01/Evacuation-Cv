/* ─────────────────────────────────────────────
   dashboard.js — Live Dashboard Logic
   Polls /status every 500ms and updates DOM
───────────────────────────────────────────── */

const STATUS_COLORS = {
  SAFE     : 'safe',
  MODERATE : 'moderate',
  CRITICAL : 'critical',
};

const BADGE_CLASS = {
  SAFE     : 'badge-safe',
  MODERATE : 'badge-moderate',
  CRITICAL : 'badge-critical',
};

// ── DOM helpers ───────────────────────────────
const $ = id => document.getElementById(id);

function setText(id, val) {
  const el = $(id);
  if (el) el.textContent = val;
}

// ── Status polling ────────────────────────────
async function fetchStatus() {
  try {
    const res  = await fetch('/status');
    const data = await res.json();
    updateDashboard(data);
  } catch (err) {
    console.warn('Status fetch error:', err);
  }
}

function updateDashboard(data) {
  // ── Navbar stats ──────────────────────────
  setText('sys-fps',     `FPS: ${data.fps || '--'}`);
  setText('sys-persons', `Persons: ${data.active_persons || '--'}`);

  const lstmEl = $('lstm-status');
  if (lstmEl) {
    if (data.lstm_ready) {
      lstmEl.textContent  = 'LSTM: Active';
      lstmEl.className    = 'badge badge-safe';
    } else {
      lstmEl.textContent  = 'LSTM: Warming';
      lstmEl.className    = 'badge badge-warming';
    }
  }

  // ── Alert banner ──────────────────────────
  const banner = $('alert-banner');
  if (banner) {
    if (data.any_critical) {
      const criticals = (data.zones || [])
        .filter(z => z.status === 'CRITICAL')
        .map(z => z.name)
        .join(', ');
      $('alert-text').textContent =
        `⚠️  EVACUATION ALERT — Critical congestion at: ${criticals}`;
      banner.classList.remove('hidden');
    } else {
      banner.classList.add('hidden');
    }
  }

  // ── Exit status cards ─────────────────────
  const cardsEl = $('exit-cards');
  if (cardsEl && data.zones) {
    cardsEl.innerHTML = data.zones.map(z => {
      const cls   = STATUS_COLORS[z.status] || 'safe';
      const badge = BADGE_CLASS[z.status]   || 'badge-safe';
      const conf  = z.confidence != null
        ? `<span style="font-size:10px;color:var(--text-muted)">
             ${(z.confidence * 100).toFixed(0)}% conf
           </span>` : '';
      return `
        <div class="exit-card ${cls}">
          <div>
            <div class="exit-name">${z.name}</div>
            <div class="exit-count">${z.count} / ${z.capacity} persons</div>
          </div>
          <div style="display:flex;flex-direction:column;align-items:flex-end;gap:3px">
            <span class="exit-badge ${badge}">${z.status}</span>
            ${conf}
          </div>
        </div>`;
    }).join('');
  }

  // ── Zone features ─────────────────────────
  const featEl = $('zone-features');
  if (featEl && data.zone_features) {
    featEl.innerHTML = data.zone_features.map(zf => `
      <div class="zone-feat-card">
        <div class="zone-feat-name">${zf.zone_name}</div>
        <div class="feat-row">
          <span>Count</span>
          <span class="feat-val">${zf.count}</span>
        </div>
        <div class="feat-row">
          <span>Avg Speed</span>
          <span class="feat-val">${(zf.avg_speed || 0).toFixed(2)} px/fr</span>
        </div>
        <div class="feat-row">
          <span>Density</span>
          <span class="feat-val">${(zf.density || 0).toFixed(4)}</span>
        </div>
        <div class="feat-row">
          <span>Status</span>
          <span class="feat-val" style="color:var(--${(STATUS_COLORS[zf.status] || 'safe')})">
            ${zf.status || 'SAFE'}
          </span>
        </div>
      </div>`
    ).join('');
  }

  // ── Global stats ──────────────────────────
  setText('g-count',   data.active_persons ?? '--');
  setText('g-speed',   data.global_speed   != null ? data.global_speed.toFixed(2)   : '--');
  setText('g-density', data.global_density != null ? data.global_density.toFixed(4) : '--');
  setText('g-fps',     data.fps            != null ? data.fps.toFixed(1)            : '--');
}

// ── Settings panel ────────────────────────────
function toggleSettings() {
  const panel = $('settings-panel');
  if (!panel) return;
  panel.classList.toggle('hidden');
  if (!panel.classList.contains('hidden')) {
    loadZonesList();
  }
}

async function loadZonesList() {
  try {
    const res   = await fetch('/api/zones');
    const data  = await res.json();
    const zones = data.zones || [];
    const list  = $('zones-list');
    if (!list) return;

    list.innerHTML = zones.map(z => `
      <div class="zone-edit-row" id="zone-row-${z.id}">
        <input class="name-input" id="zone-name-${z.id}"
               type="text" value="${z.name}" placeholder="Zone name"/>
        <span style="color:var(--text-muted);font-size:11px">Cap:</span>
        <input class="cap-input" id="zone-cap-${z.id}"
               type="number" value="${z.capacity}" min="1" max="999"/>
        <button class="btn btn-sm btn-save" onclick="saveZone(${z.id})">Save</button>
        <button class="btn btn-sm btn-danger" onclick="deleteZone(${z.id})">✕</button>
      </div>`
    ).join('');

    if (zones.length === 0) {
      list.innerHTML = '<p style="color:var(--text-muted);font-size:12px">No zones defined. Run define_zones.py first.</p>';
    }
  } catch (err) {
    console.error('Load zones error:', err);
  }
}

async function saveZone(zoneId) {
  const name     = $(`zone-name-${zoneId}`)?.value?.trim();
  const capacity = parseInt($(`zone-cap-${zoneId}`)?.value);

  if (!name || isNaN(capacity)) {
    alert('Please enter a valid name and capacity.');
    return;
  }

  try {
    const res = await fetch('/api/zones/update', {
      method  : 'POST',
      headers : { 'Content-Type': 'application/json' },
      body    : JSON.stringify({ id: zoneId, name, capacity }),
    });
    const data = await res.json();
    if (data.success) {
      showMsg('zones-msg', '✅ Zone saved', 'var(--safe)');
    } else {
      showMsg('zones-msg', `❌ ${data.error}`, 'var(--critical)');
    }
  } catch (err) {
    showMsg('zones-msg', '❌ Save failed', 'var(--critical)');
  }
}

async function deleteZone(zoneId) {
  if (!confirm(`Delete zone ${zoneId}?`)) return;
  try {
    const res  = await fetch('/api/zones/delete', {
      method  : 'POST',
      headers : { 'Content-Type': 'application/json' },
      body    : JSON.stringify({ id: zoneId }),
    });
    const data = await res.json();
    if (data.success) {
      $(`zone-row-${zoneId}`)?.remove();
      showMsg('zones-msg', '✅ Zone deleted', 'var(--safe)');
    }
  } catch (err) {
    showMsg('zones-msg', '❌ Delete failed', 'var(--critical)');
  }
}

// ── Floor map upload ──────────────────────────
async function uploadFloormap() {
  const fileInput = $('floormap-file');
  const msgEl     = $('upload-msg');
  if (!fileInput?.files?.length) {
    if (msgEl) msgEl.textContent = '⚠️ Please select a file first.';
    return;
  }

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const res  = await fetch('/api/upload_floormap', {
      method : 'POST',
      body   : formData,
    });
    const data = await res.json();
    if (msgEl) {
      msgEl.textContent = data.success
        ? '✅ Floor map updated!'
        : `❌ ${data.error}`;
      msgEl.style.color = data.success ? 'var(--safe)' : 'var(--critical)';
    }
  } catch (err) {
    if (msgEl) {
      msgEl.textContent = '❌ Upload failed';
      msgEl.style.color = 'var(--critical)';
    }
  }
}

function showMsg(containerId, msg, color) {
  let el = $(containerId);
  if (!el) {
    el = document.createElement('p');
    el.id = containerId;
    el.style.fontSize = '11px';
    el.style.marginTop = '4px';
    $('zones-list')?.after(el);
  }
  el.textContent = msg;
  el.style.color = color;
  setTimeout(() => { el.textContent = ''; }, 3000);
}

// ── Fullscreen Support ────────────────────────
function toggleFullScreen(elem) {
  elem.classList.toggle('video-expanded');
}

// ── Start polling ─────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  fetchStatus();
  setInterval(fetchStatus, 500);
});