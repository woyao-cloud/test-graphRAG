import React from 'react';

const links = [
  { href: '#/', label: 'Dashboard' },
  { href: '#/users', label: 'Users' },
  { href: '#/settings', label: 'Settings' },
];

const Sidebar: React.FC = () => {
  const currentRoute = window.location.hash.replace('#', '') || '/';

  return (
    <aside className="sidebar">
      {links.map((link) => (
        <a
          key={link.href}
          href={link.href}
          className={`sidebar-link ${currentRoute === (link.href.replace('#', '') || '/') ? 'active' : ''}`}
        >
          {link.label}
        </a>
      ))}
    </aside>
  );
};

export default React.memo(Sidebar);
