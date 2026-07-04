import React, { useEffect, useState, useCallback } from 'react';
import { fetchSettings, saveSettings, Settings as SettingsType } from '../utils/api';
import { useToast } from '../hooks/useToast';

const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SettingsType | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [errors, setErrors] = useState<Partial<Record<keyof SettingsType, string>>>({});
  const { addToast } = useToast();

  useEffect(() => {
    let cancelled = false;
    fetchSettings().then((result) => {
      if (!cancelled) {
        setSettings(result);
        setLoading(false);
      }
    });
    return () => { cancelled = true; };
  }, []);

  const validate = useCallback((): boolean => {
    const newErrors: Partial<Record<keyof SettingsType, string>> = {};
    if (!settings?.name?.trim()) newErrors.name = 'Name is required';
    if (!settings?.email?.trim()) newErrors.email = 'Email is required';
    else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(settings.email))
      newErrors.email = 'Invalid email format';
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [settings]);

  const handleSave = async () => {
    if (!settings || !validate()) return;
    setSaving(true);
    try {
      await saveSettings(settings);
      addToast('Settings saved successfully', 'success');
    } catch {
      addToast('Failed to save settings', 'error');
    } finally {
      setSaving(false);
    }
  };

  const update = (key: keyof SettingsType, value: string) => {
    setSettings((prev) => (prev ? { ...prev, [key]: value } : prev));
    setErrors((prev) => ({ ...prev, [key]: undefined }));
  };

  if (loading) {
    return <div className="loading">Loading settings...</div>;
  }

  if (!settings) {
    return <div className="loading">Failed to load settings</div>;
  }

  return (
    <div>
      <h2 style={{ fontSize: 22, marginBottom: 20 }}>Settings</h2>

      <div className="card" style={{ maxWidth: 500 }}>
        <div className="form-group">
          <label>Name</label>
          <input
            className={`form-input ${errors.name ? 'error' : ''}`}
            value={settings.name}
            onChange={(e) => update('name', e.target.value)}
            placeholder="Your name"
          />
          {errors.name && <p className="form-error">{errors.name}</p>}
        </div>

        <div className="form-group">
          <label>Email</label>
          <input
            className={`form-input ${errors.email ? 'error' : ''}`}
            value={settings.email}
            onChange={(e) => update('email', e.target.value)}
            placeholder="Your email"
          />
          {errors.email && <p className="form-error">{errors.email}</p>}
        </div>

        <div className="form-group">
          <label>Language</label>
          <select
            className="form-input"
            value={settings.language}
            onChange={(e) => update('language', e.target.value)}
          >
            <option value="en">English</option>
            <option value="zh">Chinese</option>
            <option value="ja">Japanese</option>
            <option value="ko">Korean</option>
            <option value="es">Spanish</option>
          </select>
        </div>

        <div className="form-group">
          <label>Theme</label>
          <select
            className="form-input"
            value={settings.theme}
            onChange={(e) => update('theme', e.target.value as 'light' | 'dark')}
          >
            <option value="light">Light</option>
            <option value="dark">Dark</option>
          </select>
        </div>

        <button
          className="btn btn-primary"
          onClick={handleSave}
          disabled={saving}
          style={{ marginTop: 8 }}
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
};

export default SettingsPage;
