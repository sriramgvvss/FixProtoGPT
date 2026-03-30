/**
 * FixProtoGPT — API Module
 * Handles all server communication, status checking, and auth.
 * @module api
 */

const API_BASE = '/api';

/**
 * Wrapper around fetch that auto-redirects to login on 401.
 */
async function authFetch(url, options = {}) {
    const resp = await fetch(url, { ...options, credentials: 'include' });
    if (resp.status === 401) {
        window.location.href = '/auth/login';
        throw new Error('Session expired');
    }
    return resp;
}

/**
 * Sign out the current user and redirect to login page.
 */
async function handleLogout() {
    try {
        await fetch('/auth/logout', { method: 'POST', credentials: 'include' });
    } catch (_) { /* ignore */ }
    window.location.href = '/auth/login';
}

/**
 * Check system status and update the status badge in the DOM.
 */
async function checkStatus() {
    try {
        const response = await authFetch(`${API_BASE}/status`);
        const data = await response.json();

        const statusBadge = document.getElementById('statusBadge');
        if (data.model_loaded) {
            statusBadge.className = 'status-badge status-active';
            statusBadge.innerHTML = '<i class="fas fa-check-circle"></i> Model Active';
        } else if (data.demo_mode) {
            statusBadge.className = 'status-badge status-demo';
            statusBadge.innerHTML = '<i class="fas fa-exclamation-triangle"></i> Demo Only';
        } else {
            statusBadge.className = 'status-badge status-demo';
            statusBadge.innerHTML = '<i class="fas fa-circle"></i> Loading...';
        }

        // Show learning stats if interactions exist
        const learnBadge = document.getElementById('learningBadge');
        if (learnBadge && data.interactions && data.interactions.total_interactions > 0) {
            const s = data.interactions;
            learnBadge.style.display = 'inline-block';
            learnBadge.innerHTML =
                '<i class="fas fa-graduation-cap"></i> ' + s.total_interactions + ' interactions'
                + (s.positive ? ' · ' + s.positive + ' <i class="fas fa-thumbs-up"></i>' : '');
        }

        // Show token usage badge (always show, even with 0 tokens)
        updateTokenBadge(data.token_usage || {});

        // Update version badge from status response
        if (data.fix_version) {
            updateVersionBadge(data.fix_version);
        }

        // Show checkpoint / brain info
        updateBrainBadge(data.checkpoint_info || {}, data.fix_version_label || '');
    } catch (error) {
        console.error('Status check failed:', error);
        const statusBadge = document.getElementById('statusBadge');
        statusBadge.className = 'status-badge status-demo';
        statusBadge.innerHTML = '<i class="fas fa-times-circle"></i> Offline';
    }
}

/**
 * Submit user feedback (thumbs up/down) for an interaction.
 * @param {string} interactionId - The interaction UUID
 * @param {string} rating - "positive" or "negative"
 * @param {HTMLElement} btnEl - The clicked button element
 */
async function sendFeedback(interactionId, rating, btnEl) {
    try {
        const bar = btnEl.closest('.feedback-bar');
        const statusEl = bar.querySelector('.feedback-status');

        const response = await authFetch(API_BASE + '/feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ interaction_id: interactionId, rating: rating }),
        });
        const data = await response.json();

        if (data.success) {
            bar.querySelectorAll('.feedback-btn').forEach(b => {
                b.disabled = true;
                b.style.opacity = '0.5';
            });
            btnEl.style.opacity = '1';
            btnEl.classList.remove('btn-outline-success', 'btn-outline-danger');
            btnEl.classList.add(rating === 'positive' ? 'btn-success' : 'btn-danger');
            statusEl.innerHTML = '<i class="fas fa-check ms-1"></i> Thanks! This helps the model learn.';
            statusEl.style.color = '#10b981';
        } else {
            statusEl.textContent = 'Error: ' + (data.error || 'unknown');
            statusEl.style.color = '#ef4444';
        }
    } catch (e) {
        console.error('Feedback error:', e);
    }
}
