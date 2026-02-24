/**
 * Main Application Component
 *
 * Simple layout:
 * - Header with upload button
 * - Main content: Table view or Run detail view
 */

import { useState, useRef } from 'react';
import { ArrowLeft, Loader2 } from 'lucide-react';

import { Header } from '@/components/Header';
import { RunsTable } from '@/components/RunsTable';
import { RunDetail } from '@/components/RunDetail';
import { LoginPage } from '@/components/LoginPage';
import { Button } from '@/components/ui/button';
import { uploadPDF, analyzePDF } from '@/api/client';
import { useAuth } from '@/context/AuthContext';

export default function App() {
  const { isAuthenticated, isLoading } = useAuth();

  if (isLoading) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <LoginPage />;
  }

  return <AuthenticatedApp />;
}

function AuthenticatedApp() {
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const fileInputRef = useRef<HTMLInputElement>(null);

  function handleUploadClick() {
    fileInputRef.current?.click();
  }

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    try {
      const uploadResponse = await uploadPDF(file);
      const analyzeResponse = await analyzePDF(uploadResponse.file_id);

      // Refresh table and select the new run
      setRefreshTrigger(prev => prev + 1);
      setSelectedRunId(analyzeResponse.run_id);
    } catch (err) {
      console.error('Upload failed:', err);
      alert('Upload failed: ' + (err instanceof Error ? err.message : 'Unknown error'));
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  }

  function handleBack() {
    setSelectedRunId(null);
    setRefreshTrigger(prev => prev + 1);
  }

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={handleFileChange}
        disabled={uploading}
      />

      {/* Header */}
      <Header onUpload={handleUploadClick} uploading={uploading} onLogoClick={handleBack} />

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {selectedRunId ? (
          <div className="h-full flex flex-col">
            {/* Back button */}
            <div className="px-6 py-3 border-b bg-muted/30">
              <Button variant="ghost" size="sm" onClick={handleBack}>
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back to all runs
              </Button>
            </div>
            {/* Run detail */}
            <div className="flex-1 overflow-hidden">
              <RunDetail runId={selectedRunId} />
            </div>
          </div>
        ) : (
          <div className="p-6 max-w-6xl mx-auto">
            <RunsTable
              onSelectRun={setSelectedRunId}
              refreshTrigger={refreshTrigger}
            />
          </div>
        )}
      </main>
    </div>
  );
}
