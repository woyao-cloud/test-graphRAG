import React from 'react';

const NotFound: React.FC = () => {
  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: '60vh',
        textAlign: 'center',
      }}
    >
      <h1 style={{ fontSize: 72, fontWeight: 800, color: 'var(--primary)', marginBottom: 8 }}>
        404
      </h1>
      <p style={{ fontSize: 18, color: 'var(--text-secondary)', marginBottom: 24 }}>
        Page not found
      </p>
      <a href="#/" className="btn btn-primary">
        Back to Dashboard
      </a>
    </div>
  );
};

export default NotFound;
