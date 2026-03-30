/**
 * FixProtoGPT — Application Orchestrator
 * Bootstraps status check, sidebar, and example loader on DOMContentLoaded.
 * Load order: api.js → ui.js → generate.js → conversations.js → file-io.js → app.js
 * @module app
 */

document.addEventListener('DOMContentLoaded', function () {
    checkStatus();
    loadVersions();
    loadConversations();
});

/**
 * Populate the Generate FIX input with a sample prompt.
 * @param {string} text - Example prompt text
 */
function setGenerateFixExample(text) {
    document.getElementById('generateFixInput').value = text;
    document.getElementById('generateFixInput').focus();
    showToast('📝 Example loaded!', 'success');
}
