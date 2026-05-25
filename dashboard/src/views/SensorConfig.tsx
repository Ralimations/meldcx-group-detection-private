import React, { Suspense, lazy, useEffect, useMemo, useState } from 'react';
import { 
    Layout, 
    Spin,
    Flex,
    Typography, 
    Space, 
    message,
} from 'antd';
import WorkspaceHeader from '../components/workspace/WorkspaceHeader';
import WorkspaceNav from '../components/workspace/WorkspaceNav';
import WorkspaceSidebar from '../components/workspace/WorkspaceSidebar';
import WorkspaceStatusBanner from '../components/workspace/WorkspaceStatusBanner';
import ZoneCanvas from '../components/workspace/ZoneCanvas';
import UndistortCanvas from '../components/workspace/UndistortCanvas';
import { useConfigContext } from '../contexts/useConfigContext';
import type { WorkspaceView } from '../hooks/useDashboardController';

const LiveFeed = lazy(() => import('./LiveFeed'));
const TestSummary = lazy(() => import('./TestSummary'));

const { Header, Content, Sider } = Layout;
const { Title, Text } = Typography;
const MAX_ZONE_POINTS = 4;
const SIDEBAR_MIN_WIDTH = 300;
const SIDEBAR_MAX_WIDTH = 420;
const SIDEBAR_DEFAULT_WIDTH = 300;
const DEFAULT_ZONE_BOX: [number, number][] = [
    [0.2, 0.2],
    [0.8, 0.2],
    [0.8, 0.8],
    [0.2, 0.8],
];
type ZoneType = 'detection' | 'height' | 'doorway';

function clampPoint([x, y]: [number, number]): [number, number] {
    return [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))];
}

function toBoxPoints(points: [number, number][]): [number, number][] {
    if (points.length === 4) {
        return points.map((point) => clampPoint(point as [number, number]));
    }
    if (points.length >= 3) {
        const xs = points.map((point) => point[0]);
        const ys = points.map((point) => point[1]);
        const minX = Math.min(...xs);
        const maxX = Math.max(...xs);
        const minY = Math.min(...ys);
        const maxY = Math.max(...ys);
        return [
            [minX, minY],
            [maxX, minY],
            [maxX, maxY],
            [minX, maxY],
        ].map((point) => clampPoint(point as [number, number]));
    }
    return [];
}

interface SensorConfigProps {
    initialView?: string;
    onViewChange?: (view: string) => void;
}

interface RoiDraftState {
    sensorId: number | null;
    detectionPoints: [number, number][];
    heightPoints: [number, number][];
    doorwayPoints: [number, number][];
}

const viewFallback = (
    <Flex align="center" justify="center" style={{ flex: 1, minHeight: 320 }}>
        <Spin size="large" />
    </Flex>
);

const SensorConfig: React.FC<SensorConfigProps> = ({ initialView = 'live', onViewChange }) => {
    const {
        activeSensorId,
        sensors,
        config,
        status,
        playback,
        deactivateSensor,
        updateParam,
        updateParams,
        refreshKey,
        startEngine,
        stopEngine,
        resetEngine,
        pauseEngine,
        captureWorkspaceSnapshot,
        actionState,
        renameSensor,
    } = useConfigContext();
    const [currentSubView, setCurrentSubView] = useState<WorkspaceView>(
        initialView === 'undistort' || initialView === 'summary'
            ? initialView
            : initialView === 'config'
                ? 'zones'
                : 'live'
    );
    const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT_WIDTH);
    const [isResizingSidebar, setIsResizingSidebar] = useState(false);
    const [activeZoneType, setActiveZoneType] = useState<ZoneType>('detection');
    const [roiDraft, setRoiDraft] = useState<RoiDraftState | null>(null);
    const savedDetectionRoiPoints = useMemo(() => {
        const roiPolygon = config?.values?.roi_polygon;
        return Array.isArray(roiPolygon) ? toBoxPoints(roiPolygon as [number, number][]) : [];
    }, [config?.values?.roi_polygon]);
    const savedHeightRoiPoints = useMemo(() => {
        const roiPolygon = config?.values?.height_roi_polygon;
        return Array.isArray(roiPolygon) ? toBoxPoints(roiPolygon as [number, number][]) : [];
    }, [config?.values?.height_roi_polygon]);
    const savedDoorwayRoiPoints = useMemo(() => {
        const roiPolygon = config?.values?.doorway_roi_polygon;
        return Array.isArray(roiPolygon) ? toBoxPoints(roiPolygon as [number, number][]) : [];
    }, [config?.values?.doorway_roi_polygon]);
    const detectionRoiPoints = roiDraft?.sensorId === activeSensorId ? roiDraft.detectionPoints : savedDetectionRoiPoints;
    const heightRoiPoints = roiDraft?.sensorId === activeSensorId ? roiDraft.heightPoints : savedHeightRoiPoints;
    const doorwayRoiPoints = roiDraft?.sensorId === activeSensorId ? roiDraft.doorwayPoints : savedDoorwayRoiPoints;

    const setZonePoints = (zone: ZoneType, points: [number, number][]) => {
        setRoiDraft({
            sensorId: activeSensorId,
            detectionPoints: zone === 'detection' ? points : detectionRoiPoints,
            heightPoints: zone === 'height' ? points : heightRoiPoints,
            doorwayPoints: zone === 'doorway' ? points : doorwayRoiPoints,
        });
    };

    const clearZone = async (zone: ZoneType) => {
        setZonePoints(zone, []);
        message.loading({ content: `Removing ${zone === 'detection' ? 'detection' : 'height'} area...`, key: 'save_roi' });
        try {
            if (zone === 'detection') {
                await updateParams({
                    roi_polygon: [],
                    use_roi: false,
                    show_roi_mask: false,
                });
            } else if (zone === 'height') {
                await updateParams({
                    height_roi_polygon: [],
                    show_height_roi_mask: false,
                });
            } else {
                await updateParams({
                    doorway_roi_polygon: [],
                    use_doorway_monitor: false,
                    show_doorway_status: false,
                });
            }
            setRoiDraft(null);
            const zoneLabel = zone === 'detection' ? 'Detection' : zone === 'height' ? 'Height' : 'Doorway';
            message.success({ content: `${zoneLabel} area removed.`, key: 'save_roi' });
        } catch {
            const zoneLabel = zone === 'detection' ? 'detection' : zone === 'height' ? 'height' : 'doorway';
            message.error({ content: `Failed to remove ${zoneLabel} area`, key: 'save_roi' });
        }
    };

    const createZoneBox = (zone: ZoneType) => {
        setZonePoints(zone, DEFAULT_ZONE_BOX);
    };

    const saveRoi = async () => {
        try {
            if (
                (detectionRoiPoints.length !== 0 && detectionRoiPoints.length !== MAX_ZONE_POINTS)
                || (heightRoiPoints.length !== 0 && heightRoiPoints.length !== MAX_ZONE_POINTS)
                || (doorwayRoiPoints.length !== 0 && doorwayRoiPoints.length !== MAX_ZONE_POINTS)
            ) {
                message.warning(`Each ROI must use exactly ${MAX_ZONE_POINTS} points.`);
                return;
            }

            message.loading({ content: 'Saving configuration...', key: 'save_roi' });
            const hasValidDetectionZone = detectionRoiPoints.length === MAX_ZONE_POINTS;
            const hasValidHeightZone = heightRoiPoints.length === MAX_ZONE_POINTS;
            const hasValidDoorwayZone = doorwayRoiPoints.length === MAX_ZONE_POINTS;
            await updateParams({
                roi_polygon: detectionRoiPoints,
                use_roi: hasValidDetectionZone,
                show_roi_mask: hasValidDetectionZone,
                height_roi_polygon: heightRoiPoints,
                show_height_roi_mask: hasValidHeightZone,
                doorway_roi_polygon: doorwayRoiPoints,
                use_doorway_monitor: hasValidDoorwayZone ? Boolean(config?.values?.use_doorway_monitor ?? true) : false,
                show_doorway_status: hasValidDoorwayZone ? Boolean(config?.values?.show_doorway_status ?? true) : false,
            });
            setRoiDraft(null);
            message.success({
                content: 'Detection, height, and doorway areas saved.',
                key: 'save_roi'
            });
        } catch {
            message.error({ content: 'Failed to save configuration', key: 'save_roi' });
        }
    };

    const exitSensor = async () => {
        await deactivateSensor();
    };

    const setSubView = async (view: WorkspaceView) => {
        if (currentSubView === 'live' && view !== 'live' && status.running) {
            try {
                await captureWorkspaceSnapshot();
            } catch {
                // Continue switching views even if snapshot capture fails.
            }
            if (!status.paused && !actionState.pausingEngine) {
                if (playback.is_file) {
                    void stopEngine();
                } else {
                    void pauseEngine();
                }
            }
        }
        setCurrentSubView(view);
        onViewChange?.(view === 'zones' ? 'config' : view);
    };

    const activeSensor = sensors.find((sensor) => sensor.id === activeSensorId);
    const sceneName = activeSensor?.name ?? (typeof config?.values?.name === 'string' ? config.values.name : undefined);

    const engineBusyAction = actionState.startingEngine
        ? 'starting'
        : actionState.stoppingEngine
            ? 'stopping'
            : actionState.resettingEngine
                ? 'resetting'
                : actionState.pausingEngine
                    ? 'pausing'
                    : null;

    const workspaceBanner = actionState.deactivatingSensor
        ? 'Saving scene and exiting workspace...'
        : actionState.loadingSensor
            ? 'Loading sensor workspace...'
            : actionState.startingEngine
                ? 'Starting engine...'
                : actionState.stoppingEngine
                    ? 'Stopping engine...'
                    : actionState.resettingEngine
                        ? 'Resetting engine...'
                        : actionState.pausingEngine
                            ? (status.paused ? 'Resuming engine...' : 'Pausing engine...')
                            : null;

    useEffect(() => {
        if (!isResizingSidebar) {
            return undefined;
        }

        const handlePointerMove = (event: PointerEvent) => {
            const nextWidth = window.innerWidth - event.clientX;
            setSidebarWidth(Math.round(Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, nextWidth))));
        };

        const handlePointerUp = () => {
            setIsResizingSidebar(false);
        };

        document.body.style.cursor = 'col-resize';
        document.body.style.userSelect = 'none';
        window.addEventListener('pointermove', handlePointerMove);
        window.addEventListener('pointerup', handlePointerUp);

        return () => {
            document.body.style.cursor = '';
            document.body.style.userSelect = '';
            window.removeEventListener('pointermove', handlePointerMove);
            window.removeEventListener('pointerup', handlePointerUp);
        };
    }, [isResizingSidebar]);

    return (
        <Layout style={{ height: '100vh', background: '#f8fafc' }}>
            <Header style={{ padding: 0, height: 64, background: 'transparent' }}>
                <WorkspaceHeader
                    sceneName={sceneName}
                    exiting={actionState.deactivatingSensor}
                    renaming={actionState.renamingSensor}
                    onRename={activeSensorId !== null ? (name) => renameSensor(activeSensorId, name) : undefined}
                    onExit={exitSensor}
                />
            </Header>

            <Layout>
                <Sider width={72} style={{ background: '#fff' }}>
                    <WorkspaceNav currentView={currentSubView} onChange={setSubView} />
                </Sider>

                <Content style={{ position: 'relative', display: 'flex', flexDirection: 'column', padding: 0, background: '#f1f5f9' }}>
                    {workspaceBanner && (
                        <WorkspaceStatusBanner
                            tone={actionState.stoppingEngine || actionState.deactivatingSensor ? 'warning' : 'info'}
                            message={workspaceBanner}
                        />
                    )}

                    {/* Main Content Area */}
                    <div style={{ 
                        flex: 1, 
                        display: 'flex', 
                        flexDirection: 'column',
                        overflow: 'hidden',
                        position: 'relative'
                    }}>
                        {currentSubView === 'live' ? (
                            <Suspense fallback={viewFallback}>
                                <div style={{ flex: 1, padding: 20, overflowY: 'auto' }}>
                                    <LiveFeed
                                        showEngineControls
                                        onStart={startEngine}
                                        onStop={stopEngine}
                                        onReset={resetEngine}
                                        onPauseToggle={pauseEngine}
                                        busyAction={engineBusyAction}
                                    />
                                </div>
                            </Suspense>
                        ) : currentSubView === 'undistort' ? (
                            <UndistortCanvas refreshKey={refreshKey} />
                        ) : currentSubView === 'summary' ? (
                            <Suspense fallback={viewFallback}>
                                <div style={{ flex: 1, padding: 20, overflowY: 'auto' }}>
                                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                                        <div>
                                            <Title level={3} style={{ margin: 0 }}>Run Summaries</Title>
                                            <Text type="secondary">Historical session metrics captured when the engine stops.</Text>
                                        </div>
                                        <TestSummary />
                                    </Space>
                                </div>
                            </Suspense>
                        ) : (
                            <ZoneCanvas
                                refreshKey={refreshKey}
                                detectionRoiPoints={detectionRoiPoints}
                                heightRoiPoints={heightRoiPoints}
                                doorwayRoiPoints={doorwayRoiPoints}
                                activeZoneType={activeZoneType}
                                onChangeZonePoints={setZonePoints}
                                onCreateZoneBox={createZoneBox}
                            />
                        )}
                    </div>
                </Content>

                <div style={{ position: 'relative', display: 'flex', background: '#fff' }}>
                    <div
                        aria-label="Resize sidebar"
                        onPointerDown={() => setIsResizingSidebar(true)}
                        style={{
                            position: 'absolute',
                            left: -4,
                            top: 0,
                            bottom: 0,
                            width: 8,
                            cursor: 'col-resize',
                            zIndex: 5,
                        }}
                    />
                    <Sider
                        width={sidebarWidth}
                        style={{
                            background: '#fff',
                            borderLeft: '1px solid #e2e8f0',
                            flex: `0 0 ${sidebarWidth}px`,
                            overflow: 'hidden',
                        }}
                    >
                        <WorkspaceSidebar
                            currentView={currentSubView}
                            activeZoneType={activeZoneType}
                            detectionRoiPoints={detectionRoiPoints}
                            heightRoiPoints={heightRoiPoints}
                            doorwayRoiPoints={doorwayRoiPoints}
                            showRoiMask={Boolean(config?.values?.show_roi_mask)}
                            showHeightRoiMask={Boolean(config?.values?.show_height_roi_mask)}
                            doorwayMonitorEnabled={Boolean(config?.values?.use_doorway_monitor)}
                            showDoorwayStatus={Boolean(config?.values?.show_doorway_status)}
                            onActiveZoneChange={setActiveZoneType}
                            onCreateZoneBox={createZoneBox}
                            onClearZone={clearZone}
                            onToggleRoiMask={() => updateParam('show_roi_mask', !config?.values?.show_roi_mask)}
                            onToggleHeightRoiMask={() => updateParam('show_height_roi_mask', !config?.values?.show_height_roi_mask)}
                            onToggleDoorwayMonitor={() => updateParams({
                                use_doorway_monitor: !config?.values?.use_doorway_monitor,
                                show_doorway_status: !config?.values?.use_doorway_monitor ? true : Boolean(config?.values?.show_doorway_status),
                            })}
                            onSaveConfiguration={saveRoi}
                        />
                    </Sider>
                </div>
            </Layout>

            {/* Global styles for the custom UI feel */}
            <style dangerouslySetInnerHTML={{ __html: `
                .custom-tabs .ant-tabs-nav {
                    margin-bottom: 0;
                }
                .custom-tabs .ant-tabs-tab {
                    font-weight: 500 !important;
                    color: #64748b !important;
                }
                .custom-tabs .ant-tabs-tab-active .ant-tabs-tab-btn {
                    color: #1677ff !important;
                }
                .custom-tabs .ant-tabs-ink-bar {
                    background: #1677ff !important;
                    height: 3px !important;
                }
                .ant-collapse-header {
                    padding: 16px 20px !important;
                    align-items: center !important;
                }
                .ant-collapse-content-box {
                    padding: 0 !important;
                }
            `}} />
        </Layout>
    );
};

export default SensorConfig;
