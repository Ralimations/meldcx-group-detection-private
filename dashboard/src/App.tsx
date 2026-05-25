import { ConfigProvider as AntProvider, Empty, Layout } from 'antd';
import React, { Suspense, lazy } from 'react';
import AppStateView from './components/shell/AppStateView';
import { ConfigProvider } from './contexts/ConfigContext';
import { useConfigContext } from './contexts/useConfigContext';
import { useDashboardController } from './hooks/useDashboardController';
import './App.css';

const SensorList = lazy(() => import('./views/SensorList'));
const SensorConfig = lazy(() => import('./views/SensorConfig'));

const phaseFallback = (
  <AppStateView
    tone="loading"
    title="Loading Dashboard View"
    subtitle="Preparing the next dashboard module."
  />
);

const DashboardContent: React.FC = () => {
  const { sensors } = useConfigContext();
  const { phase, bootError, workspaceView, setWorkspaceView, retryBootstrap } = useDashboardController();

  if (phase === 'booting') {
    return (
      <AppStateView
        tone="loading"
        title="Booting Dashboard"
        subtitle="Starting the backend API, loading sensor metadata, and preparing the workspace."
      />
    );
  }

  if (phase === 'error') {
    return (
      <AppStateView
        tone="error"
        title="Dashboard Unavailable"
        subtitle="The UI is running, but the dashboard services did not finish booting cleanly."
        details={bootError ?? undefined}
        actionLabel="Retry Connection"
        onAction={retryBootstrap}
      />
    );
  }

  if (phase === 'sensor-list') {
    return (
      <Suspense fallback={phaseFallback}>
        <Layout style={{ minHeight: '100vh', background: '#f0f2f5' }}>
          <SensorList onViewChange={(view) => setWorkspaceView(view === 'config' ? 'zones' : 'live')} />
          {sensors.length === 0 && (
            <div style={{ position: 'fixed', right: 24, bottom: 24, background: '#fff', border: '1px solid #e2e8f0', padding: 16 }}>
              <Empty
                description="The dashboard is ready. Add a sensor source to enter the workspace."
                image={Empty.PRESENTED_IMAGE_SIMPLE}
              />
            </div>
          )}
        </Layout>
      </Suspense>
    );
  }

  return (
    <Suspense fallback={phaseFallback}>
      <SensorConfig
        initialView={workspaceView}
        onViewChange={(view) => setWorkspaceView(view === 'config' ? 'zones' : (view as typeof workspaceView))}
      />
    </Suspense>
  );
};

function App() {
  return (
    <AntProvider
      theme={{
        token: {
          colorPrimary: '#1677ff',
          borderRadius: 0,
          fontFamily: "'Inter', sans-serif",
        },
      }}
    >
      <ConfigProvider>
        <DashboardContent />
      </ConfigProvider>
    </AntProvider>
  );
}

export default App;
