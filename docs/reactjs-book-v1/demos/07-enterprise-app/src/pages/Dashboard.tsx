import React, { useEffect, useState, useMemo } from 'react';
import { fetchDashboard } from '../utils/api';
import { useToast } from '../hooks/useToast';

interface DashboardData {
  stats: { totalUsers: number; revenue: number; orders: number; growth: number };
  recentActivity: { id: number; text: string; time: string; type: string }[];
}

const activityColors: Record<string, string> = {
  user: '#4f46e5',
  order: '#16a34a',
  deploy: '#f59e0b',
  payment: '#06b6d4',
  system: '#8b5cf6',
  support: '#ef4444',
};

const Dashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const { addToast } = useToast();

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    fetchDashboard().then((result) => {
      if (!cancelled) {
        setData(result);
        setLoading(false);
      }
    });
    return () => { cancelled = true; };
  }, []);

  const stats = useMemo(
    () => [
      { label: 'Total Users', value: data?.stats.totalUsers.toLocaleString() ?? '...', change: '+12%', positive: true },
      { label: 'Revenue', value: `$${data?.stats.revenue.toLocaleString() ?? '...'}`, change: '+8.3%', positive: true },
      { label: 'Orders', value: data?.stats.orders.toLocaleString() ?? '...', change: '+3.2%', positive: true },
      { label: 'Growth', value: `${data?.stats.growth ?? '...'}%`, change: '+2.1%', positive: true },
    ],
    [data]
  );

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  return (
    <div>
      <h2 style={{ marginBottom: 20, fontSize: 22 }}>Dashboard</h2>

      <div className="card-grid">
        {stats.map((stat) => (
          <div key={stat.label} className="card stat-card">
            <div className="stat-value">{stat.value}</div>
            <div className="stat-label">{stat.label}</div>
            <div className={`stat-change ${stat.positive ? 'positive' : 'negative'}`}>
              {stat.change} vs last month
            </div>
          </div>
        ))}
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 12, fontSize: 16 }}>Recent Activity</h3>
        <ul className="activity-list">
          {data?.recentActivity.map((item) => (
            <li key={item.id} className="activity-item">
              <span
                className="activity-dot"
                style={{ background: activityColors[item.type] ?? '#9ca3af' }}
              />
              <span className="activity-text">{item.text}</span>
              <span className="activity-time">{item.time}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default Dashboard;
