/**
 * FixProtoGPT — FIX Version Selector Module
 * Handles version listing, selection, and status display.
 * @module versions
 */

/** Currently active FIX version (cached from server). */
let _currentFixVersion = null;

/**
 * Populate the version selector dropdown from the API.
 */
async function loadVersions() {
    const sel = document.getElementById('fixVersionSelect');
    try {
        const resp = await authFetch(`${API_BASE}/versions`);
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();

        _currentFixVersion = data.current;

        if (!sel) return;

        sel.innerHTML = '';
        (data.versions || []).forEach(function (v) {
            const opt = document.createElement('option');
            opt.value = v.version;
            opt.textContent = v.label;
            if (!v.has_data && !v.has_model) {
                opt.textContent += ' (no data)';
            } else if (!v.has_model) {
                opt.textContent += ' (untrained)';
            }
            if (v.selected) opt.selected = true;
            sel.appendChild(opt);
        });

        updateVersionBadge(data.current, data.versions);
    } catch (e) {
        console.error('Failed to load versions:', e);
        if (sel) {
            sel.innerHTML = '<option value="5.0SP2">FIX 5.0 SP2</option>';
        }
    }
}

/**
 * Called when user changes the version dropdown.
 */
async function onVersionChange() {
    const sel = document.getElementById('fixVersionSelect');
    if (!sel) return;
    const ver = sel.value;

    try {
        const resp = await authFetch(`${API_BASE}/version`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ version: ver }),
        });
        const data = await resp.json();

        if (data.success) {
            _currentFixVersion = ver;
            showToast('Switched to ' + (data.label || ver), 'success');

            // Warn if the selected model is not available
            if (data.model_unavailable && data.available_models) {
                var names = data.available_models.map(function(m) { return m.label; }).join(', ');
                var msg = 'No trained model for ' + (data.label || ver) + '. Running in demo mode.';
                if (names) {
                    msg += ' Available models: ' + names + '.';
                }
                showToast(msg, 'warning', 8000);
            }

            // Refresh status to reflect new version's model state
            checkStatus();
        } else {
            showToast('Error: ' + (data.error || 'Unknown'), 'danger');
        }
    } catch (e) {
        console.error('Version switch failed:', e);
        showToast('Failed to switch version', 'danger');
    }
}

/**
 * Update the version badge next to status.
 * @param {string} current - Active version key
 * @param {Array} versions - Full version list from API (optional)
 */
function updateVersionBadge(current, versions) {
    const badge = document.getElementById('versionBadge');
    if (!badge) return;

    let label = 'FIX ' + current;
    let hasModel = false;

    if (versions) {
        const v = versions.find(function (x) { return x.version === current; });
        if (v) {
            label = v.label;
            hasModel = v.has_model;
        }
    }

    badge.style.display = 'inline-block';
    badge.innerHTML = '<i class="fas fa-code-branch"></i> ' + label;
    badge.style.background = hasModel
        ? 'rgba(16,185,129,0.15)'
        : 'rgba(245,158,11,0.15)';
    badge.style.color = hasModel ? '#10b981' : '#f59e0b';
}

/**
 * Display the brain/checkpoint badge showing which version brain is loaded
 * and what FIX versions it was trained on.
 * @param {object} checkpointInfo - From /api/status checkpoint_info
 * @param {string} versionLabel - Human-readable active version label
 */
function updateBrainBadge(checkpointInfo, versionLabel) {
    var badge = document.getElementById('brainBadge');
    if (!badge) return;

    if (!checkpointInfo || !checkpointInfo.step) {
        badge.style.display = 'none';
        return;
    }

    var versions = checkpointInfo.fix_versions_trained || [];
    var versionText = versions.length ? versions.join(', ') : (versionLabel || '?');

    var html = '<i class="fas fa-brain me-1"></i> Brain: FIX ' + escapeHtml(versionText);

    badge.style.display = 'inline-block';
    badge.innerHTML = html;
}
