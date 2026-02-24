import { useState } from 'react';
import { Upload, Loader2, LogOut, User } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ConfirmDialog } from '@/components/ui/confirm-dialog';
import { useAuth } from '@/context/AuthContext';
import { Logo } from '@/components/Logo';

interface HeaderProps {
  onUpload: () => void;
  uploading: boolean;
  onLogoClick?: () => void;
}

export function Header({ onUpload, uploading, onLogoClick }: HeaderProps) {
  const { user, logout } = useAuth();
  const [showLogoutDialog, setShowLogoutDialog] = useState(false);

  return (
    <header className="relative h-20 border-b bg-background flex items-center px-6">
      {/* Left: Animated Logo */}
      <Logo onClick={onLogoClick} />

      {/* Right: Upload + User Info + Logout */}
      <div className="ml-auto flex items-center gap-3">
        <Button onClick={onUpload} disabled={uploading} size="sm">
          {uploading ? (
            <Loader2 className="h-4 w-4 animate-spin mr-2" />
          ) : (
            <Upload className="h-4 w-4 mr-2" />
          )}
          {uploading ? 'Uploading...' : 'Upload PDF'}
        </Button>

        {user && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground border-l pl-3">
            <User className="h-4 w-4" />
            <span>{user.username}</span>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowLogoutDialog(true)}
              className="text-muted-foreground hover:text-destructive"
            >
              <LogOut className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>

      {/* Logout Confirmation */}
      <ConfirmDialog
        open={showLogoutDialog}
        onOpenChange={setShowLogoutDialog}
        title="Log out"
        description="Are you sure you want to log out?"
        confirmLabel="Logout"
        cancelLabel="Cancel"
        variant="destructive"
        onConfirm={logout}
      />
    </header>
  );
}
