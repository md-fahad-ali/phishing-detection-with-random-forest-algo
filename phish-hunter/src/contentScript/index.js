console.info('contentScript is running')

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request) => {
  if (request.type === 'SUSPICIOUS_URL') {
    showWarning(request.url);
  }
});

// Function to show warning to user
function showWarning(url) {
  // Create warning element
  const warningDiv = document.createElement('div');
  warningDiv.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    background-color: #ff4444;
    color: white;
    padding: 10px;
    text-align: center;
    z-index: 999999;
    font-family: Arial, sans-serif;
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  `;

  warningDiv.innerHTML = `
    ⚠️ Warning: This URL (${url}) may be suspicious! Be careful about entering any personal information.
    <button onclick="this.parentElement.remove()" style="
      margin-left: 10px;
      padding: 5px 10px;
      background: white;
      border: none;
      border-radius: 3px;
      cursor: pointer;
    ">Dismiss</button>
  `;

  document.body.appendChild(warningDiv);
}
