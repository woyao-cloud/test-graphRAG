import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { fetchUsers, addUser, deleteUser, User } from '../utils/api';
import DataTable, { Column } from '../components/DataTable';
import { useToast } from '../hooks/useToast';

const UserManagement: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [showAdd, setShowAdd] = useState(false);
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const { addToast } = useToast();

  const loadUsers = useCallback(async () => {
    setLoading(true);
    const result = await fetchUsers();
    setUsers(result);
    setLoading(false);
  }, []);

  useEffect(() => {
    loadUsers();
  }, [loadUsers]);

  const columns: Column<User>[] = useMemo(
    () => [
      { key: 'id', label: 'ID', sortable: true },
      { key: 'name', label: 'Name', sortable: true },
      { key: 'email', label: 'Email', sortable: true },
      { key: 'role', label: 'Role', sortable: true },
      {
        key: 'status',
        label: 'Status',
        sortable: true,
        render: (value) => (
          <span
            style={{
              padding: '2px 8px',
              borderRadius: 4,
              fontSize: 12,
              fontWeight: 600,
              background: value === 'active' ? '#f0fdf4' : '#fef2f2',
              color: value === 'active' ? '#166534' : '#991b1b',
            }}
          >
            {value as string}
          </span>
        ),
      },
      {
        key: 'joined',
        label: 'Joined',
        sortable: true,
      },
      {
        key: 'id',
        label: 'Actions',
        sortable: false,
        render: (id) => (
          <button
            className="btn btn-danger btn-sm"
            onClick={() => setDeleteId(id as number)}
          >
            Delete
          </button>
        ),
      },
    ],
    []
  );

  const handleAddUser = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const form = e.currentTarget;
    const formData = new FormData(form);
    const name = formData.get('name') as string;
    const email = formData.get('email') as string;
    const role = formData.get('role') as string;

    if (!name || !email) {
      addToast('Name and email are required', 'error');
      return;
    }

    try {
      await addUser({
        name,
        email,
        role: role || 'Viewer',
        status: 'active',
        joined: new Date().toISOString().slice(0, 10),
      });
      addToast('User added successfully', 'success');
      setShowAdd(false);
      await loadUsers();
    } catch {
      addToast('Failed to add user', 'error');
    }
  };

  const handleDelete = async () => {
    if (deleteId === null) return;
    try {
      await deleteUser(deleteId);
      addToast('User deleted', 'success');
      setDeleteId(null);
      await loadUsers();
    } catch {
      addToast('Failed to delete user', 'error');
    }
  };

  if (loading) {
    return <div className="loading">Loading users...</div>;
  }

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
        <h2 style={{ fontSize: 22 }}>User Management</h2>
        <button className="btn btn-primary" onClick={() => setShowAdd(true)}>
          + Add User
        </button>
      </div>

      <DataTable
        columns={columns}
        data={users}
        pageSize={8}
        searchable
        searchKeys={['name', 'email', 'role']}
      />

      {/* Add User Modal */}
      {showAdd &&
        createPortal(
          <div className="modal-overlay" onClick={() => setShowAdd(false)}>
            <div className="modal" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>Add User</h3>
                <button className="modal-close" onClick={() => setShowAdd(false)}>
                  &times;
                </button>
              </div>
              <form onSubmit={handleAddUser}>
                <div className="form-group">
                  <label>Name *</label>
                  <input className="form-input" name="name" placeholder="Full name" required />
                </div>
                <div className="form-group">
                  <label>Email *</label>
                  <input className="form-input" name="email" type="email" placeholder="Email address" required />
                </div>
                <div className="form-group">
                  <label>Role</label>
                  <select className="form-input" name="role" defaultValue="Viewer">
                    <option>Admin</option>
                    <option>Editor</option>
                    <option>Viewer</option>
                  </select>
                </div>
                <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end', marginTop: 16 }}>
                  <button type="button" className="btn btn-ghost" onClick={() => setShowAdd(false)}>
                    Cancel
                  </button>
                  <button type="submit" className="btn btn-primary">
                    Add User
                  </button>
                </div>
              </form>
            </div>
          </div>,
          document.body
        )}

      {/* Delete Confirmation Modal */}
      {deleteId !== null &&
        createPortal(
          <div className="modal-overlay" onClick={() => setDeleteId(null)}>
            <div className="modal" onClick={(e) => e.stopPropagation()} style={{ minWidth: 320 }}>
              <div className="modal-header">
                <h3>Confirm Delete</h3>
                <button className="modal-close" onClick={() => setDeleteId(null)}>
                  &times;
                </button>
              </div>
              <p style={{ fontSize: 14, color: 'var(--text-secondary)', marginBottom: 16 }}>
                Are you sure you want to delete this user? This action cannot be undone.
              </p>
              <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
                <button className="btn btn-ghost" onClick={() => setDeleteId(null)}>
                  Cancel
                </button>
                <button className="btn btn-danger" onClick={handleDelete}>
                  Delete
                </button>
              </div>
            </div>
          </div>,
          document.body
        )}
    </div>
  );
};

export default UserManagement;
