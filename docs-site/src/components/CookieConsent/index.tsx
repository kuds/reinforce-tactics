import React, { useState, useEffect } from 'react';
import styles from './styles.module.css';

const COOKIE_CONSENT_KEY = 'cookie-consent';

export default function CookieConsent(): JSX.Element | null {
  const [showBanner, setShowBanner] = useState(false);

  useEffect(() => {
    // Check if user has already made a choice
    const consent = localStorage.getItem(COOKIE_CONSENT_KEY);
    if (!consent) {
      setShowBanner(true);
    } else if (consent === 'accepted') {
      // If consent was previously given, enable analytics
      enableAnalytics();
    }
  }, []);

  const enableAnalytics = () => {
    // Enable Google Analytics by setting the consent
    if (window.gtag) {
      window.gtag('consent', 'update', {
        'analytics_storage': 'granted'
      });
    }
  };

  const handleAccept = () => {
    localStorage.setItem(COOKIE_CONSENT_KEY, 'accepted');
    enableAnalytics();
    setShowBanner(false);
  };

  const handleDecline = () => {
    localStorage.setItem(COOKIE_CONSENT_KEY, 'declined');
    // Analytics remains disabled (default state)
    if (window.gtag) {
      window.gtag('consent', 'update', {
        'analytics_storage': 'denied'
      });
    }
    setShowBanner(false);
  };

  if (!showBanner) {
    return null;
  }

  return (
    <div className={styles.cookieConsent}>
      <div className={styles.cookieConsentContent}>
        <div className={styles.cookieConsentText}>
          <strong>🍪 Cookie Notice</strong>
          <p>
            We use cookies and analytics to improve your experience on our site. 
            This includes Google Analytics to understand how visitors interact with our documentation.
            {' '}
            <a 
              href="https://policies.google.com/technologies/cookies" 
              target="_blank" 
              rel="noopener noreferrer"
              className={styles.cookieConsentLink}
            >
              Learn more about cookies
            </a>
          </p>
        </div>
        <div className={styles.cookieConsentButtons}>
          <button 
            onClick={handleDecline}
            className={styles.cookieConsentButtonDecline}
            aria-label="Decline cookies"
          >
            Decline
          </button>
          <button 
            onClick={handleAccept}
            className={styles.cookieConsentButtonAccept}
            aria-label="Accept cookies"
          >
            Accept
          </button>
        </div>
      </div>
    </div>
  );
}

// Extend Window interface for TypeScript
declare global {
  interface Window {
    gtag?: (...args: any[]) => void;
  }
}
