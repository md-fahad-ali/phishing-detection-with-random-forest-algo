import { useState, useEffect } from 'react'
import './Popup.css'

export const Popup = () => {
  const [phishingScore, setPhishingScore] = useState(0)
  const [currentUrl, setCurrentUrl] = useState('')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    // Get current tab URL and analyze it
    chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
      if (tabs[0]?.url) {
        const url = tabs[0].url;
        setCurrentUrl(url);
        try {
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
          if (data.error) {
            setPhishingScore(0)
          } else {
            console.log(data.phishing_probability)
            setPhishingScore(data.phishing_probability * 100)
          }
        } catch (error) {
          console.error('Error checking URL:', error)
          setPhishingScore(0)
        } finally {
          setLoading(false)
        }
      } else {
        setLoading(false)
      }
    })
  }, [])

  const getRiskLevel = (score) => {
    if (score < 30) return { text: 'Low Risk', color: '#4CAF50' }
    if (score < 70) return { text: 'Medium Risk', color: '#FF9800' }
    return { text: 'High Risk', color: '#F44336' }
  }

  const risk = getRiskLevel(phishingScore)

  return (
    <main className="phishing-popup">
      <h3>Phishing Detection</h3>
    {console.log('loading', loading)}
      {loading ? (
        <div className="loading">Analyzing website...</div>
      ) : (
        <>
          <div className="url-display">
            <span className="label">Current URL:</span>
            <span className="url">{currentUrl}</span>
          </div>

          <div className="score-container">
            <div className="score-label">
              Risk Level: <span style={{ color: risk.color }}>{risk.text}</span>
            </div>

            <div className="slider-container">
              <input
                type="range"
                min="0"
                max="100"
                value={phishingScore}
                className="risk-slider"
                readOnly
              />
              <div className="score-value" style={{ color: risk.color }}>
                {Math.round(phishingScore)}%
              </div>
            </div>
          </div>

          <div className="details">
            <h4>Safety Tips:</h4>
            <ul>
              <li>Always verify the website's URL</li>
              <li>Check for secure HTTPS connection</li>
              <li>Be cautious of requests for sensitive information</li>
            </ul>
          </div>
          <div className="footer">
            <p>Powered by Phishing Detection API</p>
            <p>Version 1.0</p>
            <p>Find us:</p>
            <a href="https://github.com/md-fahad-ali" target="_blank" rel="noopener noreferrer">
              <img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub" />
            </a>
          </div>
        </>
      )}
    </main>
  )
}

export default Popup
