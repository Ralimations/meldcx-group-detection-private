import { useState, useEffect, useCallback } from 'react';
import { api } from '../services/api';
import type { ConfigResponse, ConfigValue, EngineStatus, LogEntry, PlaybackStatus, Sensor } from '../services/api';

export type DashboardAction =
    | 'startingEngine'
    | 'stoppingEngine'
    | 'resettingEngine'
    | 'pausingEngine'
    | 'creatingSensor'
    | 'deletingSensor'
    | 'renamingSensor'
    | 'loadingSensor'
    | 'deactivatingSensor';

const isStreamSettingKey = (key: string) => (
    key === 'source' ||
    key === 'camera_index' ||
    key === 'input_mode' ||
    key === 'preview_width' ||
    key === 'preview_height'
);

const isFrameAffectingKey = (key: string) => (
    isStreamSettingKey(key) ||
    key === 'roi_polygon' ||
    key === 'height_roi_polygon' ||
    key === 'use_roi' ||
    key === 'show_roi_mask' ||
    key === 'show_height_roi_mask' ||
    key.startsWith('undistort_')
);

export function useConfig() {
    const [config, setConfig] = useState<ConfigResponse | null>(null);
    const [status, setStatus] = useState<EngineStatus>({ running: false, paused: false });
    const [playback, setPlayback] = useState<PlaybackStatus>({
        is_file: false,
        playing: true,
        speed: 1,
        loop: false,
        current_frame: 0,
        total_frames: 0,
        fps: 0,
        position_seconds: 0,
        duration_seconds: 0,
    });
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [videos, setVideos] = useState<{ videos: string[], current: string }>({ videos: [], current: '' });
    const [sensors, setSensors] = useState<Sensor[]>([]);
    const [activeSensorId, setActiveSensorId] = useState<number | null>(null);
    const [refreshKey, setRefreshKey] = useState(Date.now());
    const [loading, setLoading] = useState(true);
    const [bootError, setBootError] = useState<string | null>(null);
    const [actionState, setActionState] = useState<Record<DashboardAction, boolean>>({
        startingEngine: false,
        stoppingEngine: false,
        resettingEngine: false,
        pausingEngine: false,
        creatingSensor: false,
        deletingSensor: false,
        renamingSensor: false,
        loadingSensor: false,
        deactivatingSensor: false,
    });

    const refreshConfig = useCallback(async () => {
        try {
            const data = await api.getConfig();
            setConfig(data);
            return data;
        } catch (err) {
            console.error('Failed to fetch config', err);
            return null;
        }
    }, []);

    const refreshStatus = useCallback(async () => {
        try {
            const data = await api.getStatus();
            setStatus(data);
            return data;
        } catch (err) {
            console.error('Failed to fetch status', err);
            return null;
        }
    }, []);

    const refreshLogs = useCallback(async () => {
        try {
            const data = await api.getLogs();
            setLogs(data.logs);
            return data;
        } catch (err) {
            console.error('Failed to fetch logs', err);
            return null;
        }
    }, []);

    const refreshVideos = useCallback(async () => {
        try {
            const data = await api.getVideos();
            setVideos(data);
            return data;
        } catch (err) {
            console.error('Failed to fetch videos', err);
            return null;
        }
    }, []);

    const refreshSensors = useCallback(async () => {
        try {
            const data = await api.getSensors();
            setSensors(data.sensors);
            setActiveSensorId(data.active_id);
            return data;
        } catch (err) {
            console.error('Failed to fetch sensors', err);
            return null;
        }
    }, []);

    const refreshPlayback = useCallback(async () => {
        try {
            const data = await api.getPlayback();
            setPlayback(data);
            return data;
        } catch (err) {
            console.error('Failed to fetch playback', err);
            return null;
        }
    }, []);

    const bootstrap = useCallback(async () => {
        setLoading(true);
        setBootError(null);
        const [cfg, statusData, videosData, sensorData, playbackData] = await Promise.all([
            refreshConfig(),
            refreshStatus(),
            refreshVideos(),
            refreshSensors(),
            refreshPlayback(),
        ]);
        if (!cfg || !statusData || !videosData || !sensorData || !playbackData) {
            setBootError('Dashboard services are not reachable yet. Check the backend and frontend startup logs.');
        }
        setLoading(false);
    }, [refreshConfig, refreshPlayback, refreshSensors, refreshStatus, refreshVideos]);

    const runAction = useCallback(async <T,>(action: DashboardAction, task: () => Promise<T>) => {
        setActionState((prev) => ({ ...prev, [action]: true }));
        try {
            return await task();
        } finally {
            setActionState((prev) => ({ ...prev, [action]: false }));
        }
    }, []);

    useEffect(() => {
        bootstrap();

        const interval = setInterval(() => {
            if (typeof document !== 'undefined' && document.visibilityState !== 'visible') {
                return;
            }
            refreshStatus();
            refreshPlayback();
        }, 1000);

        return () => clearInterval(interval);
    }, [bootstrap, refreshPlayback, refreshStatus]);

    const updateParams = useCallback(async (updates: Record<string, ConfigValue>) => {
        setConfig((currentConfig) => {
            if (!currentConfig) return currentConfig;
            return {
                ...currentConfig,
                values: {
                    ...currentConfig.values,
                    ...updates,
                },
            };
        });

        try {
            await api.setConfig(updates);
            if (Object.keys(updates).some(isFrameAffectingKey)) {
                setRefreshKey(Date.now());
            }
            if (Object.keys(updates).some(isStreamSettingKey)) {
                refreshPlayback();
            }
        } catch (err) {
            console.error('Failed to update params', err);
            refreshConfig();
            throw err;
        }
    }, [refreshConfig, refreshPlayback]);

    const updateParam = useCallback(async (key: string, value: ConfigValue) => {
        await updateParams({ [key]: value });
    }, [updateParams]);

    return {
        config,
        status,
        playback,
        logs,
        videos,
        sensors,
        activeSensorId,
        refreshKey,
        loading,
        bootError,
        bootstrap,
        updateParam,
        updateParams,
        refreshConfig,
        refreshStatus,
        refreshLogs,
        refreshSensors,
        refreshPlayback,
        startEngine: async () => {
            await runAction('startingEngine', async () => {
                await api.startEngine();
                await Promise.all([refreshStatus(), refreshPlayback()]);
            });
        },
        stopEngine: async () => {
            await runAction('stoppingEngine', async () => {
                await api.stopEngine();
                await Promise.all([refreshStatus(), refreshPlayback()]);
                setRefreshKey(Date.now());
            });
        },
        pauseEngine: async () => {
            return runAction('pausingEngine', async () => {
                const paused = await api.pauseEngine();
                setStatus((prev) => ({ ...prev, paused }));
                return paused;
            });
        },
        captureWorkspaceSnapshot: api.captureWorkspaceSnapshot,
        resetEngine: async () => {
            await runAction('resettingEngine', async () => {
                await api.resetEngine();
                setRefreshKey(Date.now());
                await Promise.all([refreshStatus(), refreshPlayback()]);
            });
        },
        resetDefaults: async () => {
            const values = await api.resetDefaults();
            if (config) setConfig({ ...config, values });
        },
        getPresets: api.getPresets,
        savePreset: api.savePreset,
        loadPreset: async (name: string) => {
            const values = await api.loadPreset(name);
            if (config) setConfig({ ...config, values });
        },
        // Sensor API pass-throughs
        createSensor: async (name: string, overrides: Record<string, ConfigValue>) => {
            await runAction('creatingSensor', async () => {
                await api.createSensor(name, overrides);
                await refreshSensors();
            });
        },
        deleteSensor: async (id: number) => {
            await runAction('deletingSensor', async () => {
                await api.deleteSensor(id);
                await refreshSensors();
            });
        },
        renameSensor: async (id: number, name: string) => {
            await runAction('renamingSensor', async () => {
                await api.updateSensorName(id, name);
                await refreshSensors();
            });
        },
        loadSensor: async (id: number) => {
            await runAction('loadingSensor', async () => {
                const loaded = await api.loadSensor(id);
                setActiveSensorId(id);
                setConfig((currentConfig) => {
                    if (!currentConfig) return currentConfig;
                    return {
                        ...currentConfig,
                        values: loaded.values,
                    };
                });
                setRefreshKey(Date.now());
                void Promise.allSettled([refreshConfig(), refreshSensors(), refreshVideos(), refreshPlayback()]);
            });
        },
        deactivateSensor: async () => {
            await runAction('deactivatingSensor', async () => {
                await api.deactivateSensor();
                await api.clearLogs();
                setLogs([]);
                setRefreshKey(Date.now());
                await Promise.all([refreshSensors(), refreshStatus(), refreshPlayback()]);
            });
        },
        updatePlayback: async (payload: Partial<Pick<PlaybackStatus, 'playing' | 'speed' | 'loop'>> & { seek_frame?: number; restart?: boolean }) => {
            const data = await api.updatePlayback(payload);
            setPlayback(data);
            return data;
        },
        clearLogs: async () => {
            await api.clearLogs();
            setLogs([]);
        },
        actionState,
        isActionPending: (action: DashboardAction) => actionState[action],
    };
}
