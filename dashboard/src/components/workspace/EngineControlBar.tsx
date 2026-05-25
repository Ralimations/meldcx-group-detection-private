import React from 'react';
import { PauseCircleOutlined, PlayCircleOutlined, ReloadOutlined, StopOutlined } from '@ant-design/icons';
import { Button, Space, Typography } from 'antd';
import type { EngineStatus } from '../../services/api';

const { Text } = Typography;

interface EngineControlBarProps {
    status: EngineStatus;
    busyAction?: 'starting' | 'stopping' | 'resetting' | 'pausing' | null;
    onStart: () => void | Promise<void>;
    onStop: () => void | Promise<void>;
    onReset: () => void | Promise<void>;
    onPauseToggle: () => void | Promise<void>;
}

const EngineControlBar: React.FC<EngineControlBarProps> = ({
    status,
    busyAction = null,
    onStart,
    onStop,
    onReset,
    onPauseToggle,
}) => {
    const engineStateLabel = status.running ? (status.paused ? 'Paused' : 'Running') : 'Stopped';
    const feedStateLabel = status.running ? 'Feed Active' : 'Feed Offline';
    const engineStateColor = status.running ? (status.paused ? '#faad14' : '#52c41a') : '#94a3b8';
    const feedStateColor = status.running ? '#1677ff' : '#94a3b8';
    const engineBusy = busyAction !== null;

    return (
        <div style={{ position: 'absolute', top: 20, left: 20, zIndex: 5 }}>
            <Space wrap size="small">
                <Button
                    type="primary"
                    icon={<PlayCircleOutlined />}
                    onClick={onStart}
                    disabled={status.running || engineBusy}
                    loading={busyAction === 'starting'}
                    style={{ background: '#52c41a', borderColor: '#52c41a' }}
                >
                    Start Engine
                </Button>
                <Button
                    danger
                    icon={<StopOutlined />}
                    onClick={onStop}
                    disabled={!status.running || engineBusy}
                    loading={busyAction === 'stopping'}
                >
                    Stop Engine
                </Button>
                <Button
                    icon={<ReloadOutlined />}
                    onClick={onReset}
                    disabled={!status.running || engineBusy}
                    loading={busyAction === 'resetting'}
                >
                    Reset
                </Button>
                <Button
                    icon={<PauseCircleOutlined />}
                    onClick={onPauseToggle}
                    disabled={!status.running || engineBusy}
                    loading={busyAction === 'pausing'}
                >
                    {status.paused ? 'Resume' : 'Pause'}
                </Button>
                <div style={{ padding: '0 8px', display: 'flex', alignItems: 'center' }}>
                    <Space size="small">
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: engineStateColor }} />
                        <Text>{engineStateLabel}</Text>
                    </Space>
                </div>
                <div style={{ padding: '0 8px', display: 'flex', alignItems: 'center' }}>
                    <Space size="small">
                        <div style={{ width: 8, height: 8, borderRadius: '50%', background: feedStateColor }} />
                        <Text>{feedStateLabel}</Text>
                    </Space>
                </div>
            </Space>
        </div>
    );
};

export default EngineControlBar;
