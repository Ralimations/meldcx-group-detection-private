import React from 'react';
import { DeleteOutlined, PlusOutlined, SaveOutlined } from '@ant-design/icons';
import { Button, Card, Segmented, Space, Switch, Tabs, Typography } from 'antd';
import Tuning from '../../views/Tuning';
import type { WorkspaceView } from '../../hooks/useDashboardController';

const { Text } = Typography;

type ZoneType = 'detection' | 'height' | 'doorway';

interface WorkspaceSidebarProps {
    currentView: WorkspaceView;
    activeZoneType: ZoneType;
    detectionRoiPoints: [number, number][];
    heightRoiPoints: [number, number][];
    doorwayRoiPoints: [number, number][];
    showRoiMask: boolean;
    showHeightRoiMask: boolean;
    doorwayMonitorEnabled: boolean;
    showDoorwayStatus: boolean;
    onActiveZoneChange: (zone: ZoneType) => void;
    onCreateZoneBox: (zone: ZoneType) => void;
    onClearZone: (zone: ZoneType) => void;
    onToggleRoiMask: () => void | Promise<void>;
    onToggleHeightRoiMask: () => void | Promise<void>;
    onToggleDoorwayMonitor: () => void | Promise<void>;
    onSaveConfiguration: () => void | Promise<void>;
}

const ZoneCard: React.FC<{
    title: string;
    points: [number, number][];
    active: boolean;
    showOverlay: boolean;
    onSelect: () => void;
    onCreate: () => void;
    onClear: () => void;
    overlayLabel?: string;
    onToggleOverlay?: () => void | Promise<void>;
}> = ({ title, points, active, showOverlay, onSelect, onCreate, onClear, overlayLabel = 'Show Overlay', onToggleOverlay }) => (
    <Card
        size="small"
        style={{
            background: active ? '#eff6ff' : '#f8fafc',
            border: active ? '1px solid #93c5fd' : '1px solid #e2e8f0',
        }}
    >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
            <div>
                <Text strong>{title}</Text>
                <div>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                        {points.length === 4 ? '4 points configured' : 'Not configured'}
                    </Text>
                </div>
            </div>
            <Button type={active ? 'primary' : 'default'} size="small" onClick={onSelect}>
                {active ? 'Editing' : 'Edit'}
            </Button>
        </div>

        {onToggleOverlay && (
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
                <Text style={{ fontSize: 13 }}>{overlayLabel}</Text>
                <Switch size="small" checked={showOverlay} onChange={() => { void onToggleOverlay(); }} />
            </div>
        )}

        <Space.Compact block>
            <Button size="small" icon={<PlusOutlined />} onClick={onCreate} style={{ flex: 1 }}>
                {points.length === 4 ? 'Reset' : 'Add'}
            </Button>
            <Button size="small" danger icon={<DeleteOutlined />} onClick={onClear} style={{ flex: 1 }}>
                Clear
            </Button>
        </Space.Compact>
    </Card>
);

const WorkspaceSidebar: React.FC<WorkspaceSidebarProps> = ({
    currentView,
    activeZoneType,
    detectionRoiPoints,
    heightRoiPoints,
    doorwayRoiPoints,
    showRoiMask,
    showHeightRoiMask,
    doorwayMonitorEnabled,
    showDoorwayStatus,
    onActiveZoneChange,
    onCreateZoneBox,
    onClearZone,
    onToggleRoiMask,
    onToggleHeightRoiMask,
    onToggleDoorwayMonitor,
    onSaveConfiguration,
}) => {
    if (currentView === 'undistort') {
        return (
            <div style={{ background: '#fff', overflow: 'hidden', display: 'flex', flexDirection: 'column', width: '100%', minWidth: 0, height: '100%' }}>
                <div style={{ padding: '10px 12px 12px 12px', height: '100%', overflowY: 'auto', overflowX: 'hidden' }}>
                    <Tabs
                        defaultActiveKey="preview"
                        className="custom-tabs"
                        items={[
                            {
                                key: 'preview',
                                label: 'Preview',
                                children: (
                                    <div style={{ padding: '16px 4px' }}>
                                        <Card size="small" style={{ background: '#f8fafc', border: '1px solid #e2e8f0' }}>
                                            <Space direction="vertical" size="small" style={{ width: '100%' }}>
                                                <Text strong>Camera Undistort</Text>
                                                <Text type="secondary" style={{ fontSize: 13 }}>
                                                    This workspace uses a static side-by-side frame so lens tuning does not keep a second live preview loop active.
                                                </Text>
                                            </Space>
                                        </Card>
                                    </div>
                                ),
                            },
                            {
                                key: 'config',
                                label: 'Config',
                                children: (
                                    <div style={{ maxHeight: 'calc(100vh - 260px)', overflowY: 'auto', padding: '18px 6px 0 6px' }}>
                                        <Tuning isSidebar forceCategory="Lens" />
                                    </div>
                                ),
                            },
                        ]}
                    />
                </div>
            </div>
        );
    }

    return (
        <div style={{ background: '#fff', overflow: 'hidden', display: 'flex', flexDirection: 'column', width: '100%', minWidth: 0, height: '100%' }}>
            <div style={{ padding: '10px 12px 12px 12px', height: '100%', overflowY: 'auto', overflowX: 'hidden' }}>
                <Tabs
                    defaultActiveKey="area"
                    className="custom-tabs"
                    items={[
                        {
                            key: 'area',
                            label: 'Area',
                            children: (
                                <div style={{ padding: '16px 4px' }}>
                                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                                        <Segmented
                                            block
                                            value={activeZoneType}
                                            onChange={(value) => onActiveZoneChange(value as ZoneType)}
                                            options={[
                                                { label: 'Detection ROI', value: 'detection' },
                                                { label: 'Height ROI', value: 'height' },
                                                { label: 'Doorway ROI', value: 'doorway' },
                                            ]}
                                        />
                                        <ZoneCard
                                            title="Detection ROI"
                                            points={detectionRoiPoints}
                                            active={activeZoneType === 'detection'}
                                            showOverlay={showRoiMask}
                                            onSelect={() => onActiveZoneChange('detection')}
                                            onCreate={() => onCreateZoneBox('detection')}
                                            onClear={() => onClearZone('detection')}
                                            onToggleOverlay={onToggleRoiMask}
                                        />
                                        <ZoneCard
                                            title="Height ROI"
                                            points={heightRoiPoints}
                                            active={activeZoneType === 'height'}
                                            showOverlay={showHeightRoiMask}
                                            onSelect={() => onActiveZoneChange('height')}
                                            onCreate={() => onCreateZoneBox('height')}
                                            onClear={() => onClearZone('height')}
                                            onToggleOverlay={onToggleHeightRoiMask}
                                        />
                                        <ZoneCard
                                            title="Doorway ROI"
                                            points={doorwayRoiPoints}
                                            active={activeZoneType === 'doorway'}
                                            showOverlay={doorwayMonitorEnabled && showDoorwayStatus}
                                            onSelect={() => onActiveZoneChange('doorway')}
                                            onCreate={() => onCreateZoneBox('doorway')}
                                            onClear={() => onClearZone('doorway')}
                                            overlayLabel="Enable Monitor"
                                            onToggleOverlay={onToggleDoorwayMonitor}
                                        />
                                    </Space>
                                </div>
                            ),
                        },
                        {
                            key: 'config',
                            label: currentView === 'zones' ? 'ROI Config' : 'Config',
                            children: (
                                <div style={{ maxHeight: 'calc(100vh - 260px)', overflowY: 'auto', padding: '18px 6px 0 6px' }}>
                                    {currentView === 'zones' ? (
                                        <Space direction="vertical" style={{ width: '100%' }} size="middle">
                                            <Segmented
                                                block
                                                value={activeZoneType}
                                                onChange={(value) => onActiveZoneChange(value as ZoneType)}
                                            options={[
                                                { label: 'Detection ROI', value: 'detection' },
                                                { label: 'Height ROI', value: 'height' },
                                                { label: 'Doorway ROI', value: 'doorway' },
                                            ]}
                                            />
                                            <Tuning
                                                isSidebar
                                                forceCategory={
                                                    activeZoneType === 'detection'
                                                        ? 'Detection ROI'
                                                        : activeZoneType === 'height'
                                                            ? 'Height ROI'
                                                            : 'Doorway ROI'
                                                }
                                            />
                                        </Space>
                                    ) : (
                                        <Tuning isSidebar initialCategory="Display" />
                                    )}
                                </div>
                            ),
                        },
                    ]}
                />

                {currentView === 'zones' && (
                    <Button
                        type="primary"
                        block
                        size="large"
                        icon={<SaveOutlined />}
                        style={{
                            marginTop: 20,
                            borderRadius: 8,
                            background: '#3b82f6',
                            height: 46,
                            fontWeight: 600,
                            boxShadow: '0 4px 6px -1px rgba(59, 130, 246, 0.5)',
                        }}
                        onClick={onSaveConfiguration}
                    >
                        Save Configuration
                    </Button>
                )}
            </div>
        </div>
    );
};

export default WorkspaceSidebar;
