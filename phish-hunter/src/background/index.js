console.log('background is running')

async function fetchUrl(url) {
  // Simulate fetching URL and checking for phishing
  console.log('Fetching URL:', url)
  try {
    // Normalize URL to remove trailing slash and ensure consistent format
    const normalizedUrl = url.replace(/\/$/, '').replace(/^https?:\/\/www\./, 'https://')
    console.log('Normalized URL:', normalizedUrl)
    
    const response = await fetch('http://localhost:5000/api/check', {
      method: 'POST',
      body: JSON.stringify({
        url: normalizedUrl,
      }),
      headers: {
        'Content-Type': 'application/json',
      },
    })
    const data = await response.json()
    console.log('Response from URL check:', data)
    return data
  } catch (error) {
    console.error('Error checking URL:', error)
    return { error: true }
  }
  
  // Here you would implement your URL checking logic
}
// Listen for tab updates
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  console.log('background has received a message from tab, and changeInfo is ', changeInfo?.url)

  if (changeInfo.url && changeInfo?.url !== "chrome://newtab/"  ) {
    const data = await fetchUrl(changeInfo.url)
    if (data.is_phishing) {
      // Show warning notification for phishing
      chrome.notifications.create({
        type: 'basic',
        iconUrl: '/img/logo-128.png',
        title: 'Warning: Suspicious Website',
        message: 'This website might be a phishing attempt. Please be careful!',
        priority: 2
      });
      
      // You can also inject a warning into the page
      chrome.scripting.executeScript({
        target: { tabId: tabId },
        func: () => {
          const warning = document.createElement('div');
          warning.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: #ff4444;
            color: white;
            padding: 15px;
            text-align: center;
            z-index: 999999;
            font-weight: bold;
          `;
          warning.textContent = '⚠️ Warning: This might be a phishing website!';
          document.body.appendChild(warning);
        }
      });
    } else {
      console.log('URL is safe:', changeInfo.url)
      // Show success notification for safe URL
      chrome.notifications.create({
        type: 'basic',
        iconUrl: '/img/logo-128.png',
        title: 'Safe Website',
        message: 'This website appears to be safe!',
        priority: 1
      });
    }
  }
})

// Listen for messages from popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('background has received a message from popup:', request)
  
  if (request.type === 'CHECK_URL') {
    // Call fetchUrl and send response back to popup
    fetchUrl(request.url)
      .then(data => sendResponse(data))
      .catch(error => sendResponse({ error: true, message: error.message }));
    return true; // Required to use sendResponse asynchronously
  }
  
  if (request.type === 'COUNT') {
    console.log('background has received a message from popup, and count is ', request?.count)
  }
})
