import React from 'react';
import CookieConsent from '@site/src/components/CookieConsent';

// Root component wrapper to add global components
export default function Root({children}: {children: React.ReactNode}): JSX.Element {
  return (
    <>
      {children}
      <CookieConsent />
    </>
  );
}
