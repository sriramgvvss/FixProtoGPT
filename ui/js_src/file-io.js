/**
 * FixProtoGPT — File Import / Export Module
 * FIX message import (JSON/XML), export, and format conversion display.
 * Depends on: api.js, ui.js
 * @module file-io
 */

/**
 * Import a FIX message from a JSON or XML file into a tab's input.
 * @param {string} tab  - Tab identifier (validate, explain, complete)
 * @param {string} format - File format: 'json' | 'xml'
 */
async function importFile(tab, format) {
    const fileInput = document.getElementById(tab + 'FileInput');
    const file = fileInput.files[0];

    if (!file) {
        showToast('⚠️ Please select a file first', 'error');
        return;
    }

    if (!file.name.toLowerCase().endsWith('.' + format)) {
        showToast('⚠️ Please select a ' + format.toUpperCase() + ' file', 'error');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        showToast('📂 Importing file...', 'success');

        const response = await authFetch(API_BASE + '/import/' + format, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            showToast('❌ Import failed: ' + data.error, 'error');
            if (data.details) console.error('Import errors:', data.details);
        } else {
            document.getElementById(tab + 'Input').value = data.fix_message;
            showToast('✅ ' + format.toUpperCase() + ' imported successfully!', 'success');
            if (data.warnings && data.warnings.length > 0) console.warn('Import warnings:', data.warnings);
            fileInput.value = '';
        }
    } catch (error) {
        showToast('❌ Import error: ' + error.message, 'error');
        console.error('Import error:', error);
    }
}

/**
 * Export a FIX message from a tab's input as a downloadable file.
 * @param {string} tab  - Tab identifier (validate, explain, complete)
 * @param {string} format - Export format: 'json' | 'xml'
 */
async function exportFile(tab, format) {
    const input = document.getElementById(tab + 'Input').value.trim();
    if (!input) {
        showToast('⚠️ Please enter a FIX message first', 'error');
        return;
    }

    try {
        showToast('📤 Exporting to ' + format.toUpperCase() + '...', 'success');

        const response = await authFetch(API_BASE + '/export/' + format, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: input })
        });

        if (!response.ok) {
            const data = await response.json();
            showToast('❌ Export failed: ' + (data.error || 'Unknown error'), 'error');
            return;
        }

        _downloadBlob(response, 'fix_message.' + format, format);
    } catch (error) {
        showToast('❌ Export error: ' + error.message, 'error');
        console.error('Export error:', error);
    }
}

/**
 * Export the generated FIX message (from the Generate tab cache).
 * @param {string} format - 'json' | 'xml'
 */
async function exportGeneratedFixMessage(format) {
    const fixMessage = document.getElementById('generatedFixCache').value.trim();
    if (!fixMessage) {
        showToast('⚠️ No generated message to export', 'error');
        return;
    }

    try {
        showToast('📤 Exporting to ' + format.toUpperCase() + '...', 'success');

        const response = await authFetch(API_BASE + '/export/' + format, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: fixMessage })
        });

        if (!response.ok) {
            const data = await response.json();
            showToast('❌ Export failed: ' + (data.error || 'Unknown error'), 'error');
            return;
        }

        _downloadBlob(response, 'generated_fix_message.' + format, format);
    } catch (error) {
        showToast('❌ Export error: ' + error.message, 'error');
        console.error('Export error:', error);
    }
}

// ── Private helpers ─────────────────────────────────────────────

/**
 * Download a blob from a fetch response.
 * @param {Response} response - Fetch response
 * @param {string} defaultName - Fallback filename
 * @param {string} format - File format label for the toast
 */
async function _downloadBlob(response, defaultName, format) {
    const contentDisposition = response.headers.get('Content-Disposition');
    let filename = defaultName;
    if (contentDisposition) {
        const match = contentDisposition.match(/filename="(.+)"/);
        if (match) filename = match[1];
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    showToast('✅ Exported to ' + filename + '!', 'success');
}
