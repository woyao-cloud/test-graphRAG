/* ═══════════════════════════════════════════
   Mock API — returns Promises with delays
   ═══════════════════════════════════════════ */

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/* ── Auth ── */
export async function login(
  username: string,
  password: string
): Promise<{ id: number; name: string; email: string; role: string }> {
  await delay(600);
  if (!username || !password) {
    throw new Error('Invalid credentials');
  }
  return {
    id: 1,
    name: username,
    email: `${username}@example.com`,
    role: 'Admin',
  };
}

/* ── Dashboard ── */
export async function fetchDashboard(): Promise<{
  stats: { totalUsers: number; revenue: number; orders: number; growth: number };
  recentActivity: { id: number; text: string; time: string; type: string }[];
}> {
  await delay(400);
  return {
    stats: { totalUsers: 12483, revenue: 84720, orders: 3621, growth: 12.5 },
    recentActivity: [
      { id: 1, text: 'New user registered: alice@example.com', time: '2 min ago', type: 'user' },
      { id: 2, text: 'Order #3821 completed', time: '15 min ago', type: 'order' },
      { id: 3, text: 'Server deployment v2.4.1 finished', time: '1 hr ago', type: 'deploy' },
      { id: 4, text: 'Payment of $4,200 received', time: '2 hr ago', type: 'payment' },
      { id: 5, text: 'Database backup completed', time: '3 hr ago', type: 'system' },
      { id: 6, text: 'New support ticket opened', time: '4 hr ago', type: 'support' },
      { id: 7, text: 'SSL certificate renewed', time: '6 hr ago', type: 'system' },
    ],
  };
}

/* ── Users ── */
export interface User {
  id: number;
  name: string;
  email: string;
  role: string;
  status: 'active' | 'inactive';
  joined: string;
}

const mockUsers: User[] = [
  { id: 1, name: 'Alice Johnson', email: 'alice@example.com', role: 'Admin', status: 'active', joined: '2024-01-15' },
  { id: 2, name: 'Bob Smith', email: 'bob@example.com', role: 'Editor', status: 'active', joined: '2024-03-22' },
  { id: 3, name: 'Carol White', email: 'carol@example.com', role: 'Viewer', status: 'inactive', joined: '2024-05-10' },
  { id: 4, name: 'David Lee', email: 'david@example.com', role: 'Editor', status: 'active', joined: '2024-06-01' },
  { id: 5, name: 'Eve Martinez', email: 'eve@example.com', role: 'Admin', status: 'active', joined: '2024-07-18' },
  { id: 6, name: 'Frank Brown', email: 'frank@example.com', role: 'Viewer', status: 'active', joined: '2024-08-05' },
  { id: 7, name: 'Grace Kim', email: 'grace@example.com', role: 'Editor', status: 'inactive', joined: '2024-09-12' },
  { id: 8, name: 'Henry Davis', email: 'henry@example.com', role: 'Viewer', status: 'active', joined: '2024-10-30' },
  { id: 9, name: 'Iris Chen', email: 'iris@example.com', role: 'Editor', status: 'active', joined: '2024-11-15' },
  { id: 10, name: 'Jack Wilson', email: 'jack@example.com', role: 'Admin', status: 'active', joined: '2024-12-01' },
  { id: 11, name: 'Karen Taylor', email: 'karen@example.com', role: 'Viewer', status: 'inactive', joined: '2025-01-20' },
  { id: 12, name: 'Leo Anderson', email: 'leo@example.com', role: 'Editor', status: 'active', joined: '2025-02-14' },
];

let nextUserId = 13;

export async function fetchUsers(): Promise<User[]> {
  await delay(300);
  return [...mockUsers];
}

export async function addUser(user: Omit<User, 'id'>): Promise<User> {
  await delay(400);
  const newUser = { ...user, id: nextUserId++ };
  mockUsers.push(newUser);
  return newUser;
}

export async function deleteUser(id: number): Promise<void> {
  await delay(300);
  const idx = mockUsers.findIndex((u) => u.id === id);
  if (idx !== -1) mockUsers.splice(idx, 1);
}

/* ── Settings ── */
export interface Settings {
  name: string;
  email: string;
  language: string;
  theme: 'light' | 'dark';
}

export async function fetchSettings(): Promise<Settings> {
  await delay(300);
  return {
    name: 'Admin User',
    email: 'admin@example.com',
    language: 'en',
    theme: 'light',
  };
}

export async function saveSettings(settings: Settings): Promise<void> {
  await delay(500);
  console.log('Settings saved:', settings);
}
