import { useMemo, useState } from 'react';
import { useConfigContext } from '../contexts/useConfigContext';

export type DashboardPhase = 'booting' | 'error' | 'sensor-list' | 'workspace';
export type WorkspaceView = 'live' | 'zones' | 'undistort' | 'summary';

export function useDashboardController() {
    const { loading, bootError, activeSensorId, bootstrap } = useConfigContext();
    const [workspaceViewOverride, setWorkspaceView] = useState<WorkspaceView>('live');
    const workspaceView = activeSensorId === null ? 'live' : workspaceViewOverride;

    const phase = useMemo<DashboardPhase>(() => {
        if (loading) return 'booting';
        if (bootError) return 'error';
        if (activeSensorId === null) return 'sensor-list';
        return 'workspace';
    }, [activeSensorId, bootError, loading]);

    return {
        phase,
        bootError,
        workspaceView,
        setWorkspaceView,
        retryBootstrap: bootstrap,
    };
}
