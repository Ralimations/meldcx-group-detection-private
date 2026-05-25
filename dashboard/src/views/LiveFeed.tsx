import React, { useEffect, useState } from 'react';
import { Button, Card, Col, Row, Select, Slider, Space, Spin, Statistic, Switch, Typography, message } from 'antd';
import { 
  PauseCircleOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  UserOutlined,
  TeamOutlined
} from '@ant-design/icons';
import { useConfigContext } from '../contexts/useConfigContext';
import LogFeed from '../components/logs/LogFeed';
import EngineControlBar from '../components/workspace/EngineControlBar';

const { Title, Text } = Typography;

interface LiveFeedProps {
    showEngineControls?: boolean;
    onStart?: () => void | Promise<void>;
    onStop?: () => void | Promise<void>;
    onReset?: () => void | Promise<void>;
    onPauseToggle?: () => Promise<boolean> | boolean | void;
    busyAction?: 'starting' | 'stopping' | 'resetting' | 'pausing' | null;
}

const PLAYBACK_SPEEDS = [0.5, 1, 1.5, 2];

const LiveFeed: React.FC<LiveFeedProps> = ({
    showEngineControls = false,
    onStart,
    onStop,
    onReset,
    onPauseToggle,
    busyAction = null,
}) => {
    const { status, playback, logs, loading, config, refreshKey, refreshLogs, updatePlayback } = useConfigContext();
    const [streamError, setStreamError] = useState(false);
    const [mountNonce] = useState(() => Date.now());
    const isFileSource = playback.is_file;
    const isEngineBusy = busyAction !== null;

    useEffect(() => {
        refreshLogs();
        const interval = setInterval(() => {
            refreshLogs();
        }, 3000);

        return () => clearInterval(interval);
    }, [refreshLogs]);

    if (loading) {
        return (
            <div style={{ height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '16px' }}>
               <Spin size="large" />
               <Text type="secondary">Initializing stream...</Text>
            </div>
        );
    }

    // Compose image key from both refreshKey and the per-mount timestamp so every
    // navigation back forces React to create a new <img> element and open a fresh
    // HTTP connection to the MJPEG endpoint.
    const imgKey = `${refreshKey}-${mountNonce}`;
    const feedSrc = `http://localhost:8765/video_feed?t=${imgKey}&w=${config?.values.preview_width || 640}&h=${config?.values.preview_height || 360}`;

    const handleEnginePause = async () => {
        if (!onPauseToggle) return;
        try {
            const paused = await onPauseToggle();
            if (typeof paused === 'boolean') {
                message.success(paused ? 'Engine paused' : 'Engine resumed');
            }
        } catch {
            message.error('Failed to toggle engine pause');
        }
    };

    const handlePlaybackSeek = async (value: number) => {
        try {
            await updatePlayback({ seek_frame: value });
        } catch {
            message.error('Failed to seek video playback');
        }
    };

    const handlePlaybackToggle = async () => {
        try {
            await updatePlayback({ playing: !playback.playing });
        } catch {
            message.error('Failed to toggle video playback');
        }
    };

    const handlePlaybackRestart = async () => {
        try {
            await updatePlayback({ restart: true });
        } catch {
            message.error('Failed to restart video playback');
        }
    };

    return (
        <Space direction="vertical" style={{ width: '100%', height: '100%' }} size="middle">
            {/* Stats */}
            <Row gutter={16}>
                <Col span={6}>
                    <Card bordered={false} style={{ background: '#f8fafc' }}>
                        <Statistic 
                            title="Real-time People" 
                            value={status.people ?? 0} 
                            prefix={<UserOutlined style={{ color: '#1677ff' }} />} 
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card bordered={false} style={{ background: '#f8fafc' }}>
                        <Statistic 
                            title="Real-time Groups" 
                            value={status.groups ?? 0} 
                            prefix={<TeamOutlined style={{ color: '#52c41a' }} />} 
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card bordered={false} style={{ background: '#f8fafc' }}>
                        <Statistic
                            title="Detected Groups"
                            value={status.total_groups ?? 0}
                            prefix={<TeamOutlined style={{ color: '#16a34a' }} />}
                        />
                    </Card>
                </Col>
                <Col span={6}>
                    <Card bordered={false} style={{ background: '#f8fafc' }}>
                        <Statistic title="Engine FPS" value={status.fps ?? 0} precision={1} suffix="fps" />
                    </Card>
                </Col>
            </Row>

            {/* Video Feed */}
            <div className="video-container" style={{ position: 'relative', minHeight: streamError ? 360 : undefined }}>
                {showEngineControls && onStart && onStop && onReset && (
                    <EngineControlBar
                        status={status}
                        busyAction={busyAction}
                        onStart={onStart}
                        onStop={onStop}
                        onReset={onReset}
                        onPauseToggle={handleEnginePause}
                    />
                )}
                {/* Always keep the img mounted so it auto-reconnects when engine starts */}
                <img
                    key={imgKey}
                    src={feedSrc}
                    style={{
                        width: '100%',
                        height: '100%',
                        objectFit: 'contain',
                        // Hide the img only when the engine is not running AND we have no stream error overlay;
                        // when streamError is true the overlay covers it but the img keeps retrying in the background.
                        display: (status.running && !streamError) ? 'block' : 'none',
                    }}
                    alt="Live Stream"
                    onLoad={() => setStreamError(false)}
                    onError={() => {
                        // Only surface the offline overlay if the engine is genuinely not running.
                        // While the engine IS running, the MJPEG server never terminates the stream,
                        // so a transient onError (e.g. during the reconnect after navigation) is safe
                        // to ignore — the browser will retry the connection automatically.
                        if (!status.running) {
                            setStreamError(true);
                        }
                    }}
                />
                {/* Offline overlay — shown when streamError and engine not running */}
                {streamError && !status.running && (
                    <div
                        className="placeholder-overlay"
                        style={{
                            position: 'absolute',
                            inset: 0,
                            minHeight: 360,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: '#ddd',
                            fontSize: 20,
                            background: '#222',
                        }}
                    >
                        Stream Offline – press Start Engine
                    </div>
                )}
            </div>

            {isFileSource && (
                <Card bordered={false} style={{ background: '#f8fafc' }}>
                    <Space direction="vertical" style={{ width: '100%' }} size="middle">
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                            <div>
                                <Title level={5} style={{ margin: 0 }}>Video Playback</Title>
                                <Text type="secondary">Playback controls for file sources. AI processing continues on the active stream.</Text>
                            </div>
                            <Space>
                                <Button
                                    icon={playback.playing ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
                                    onClick={handlePlaybackToggle}
                                    disabled={!status.running || isEngineBusy}
                                >
                                    {playback.playing ? 'Pause Video' : 'Play Video'}
                                </Button>
                                <Button
                                    icon={<ReloadOutlined />}
                                    onClick={handlePlaybackRestart}
                                    disabled={!status.running || isEngineBusy}
                                >
                                    Restart
                                </Button>
                            </Space>
                        </div>
                        <Slider
                            min={0}
                            max={Math.max(playback.total_frames - 1, 0)}
                            value={Math.min(playback.current_frame, Math.max(playback.total_frames - 1, 0))}
                            onChangeComplete={handlePlaybackSeek}
                            disabled={!status.running || isEngineBusy || playback.total_frames <= 1}
                            tooltip={{ formatter: () => `${playback.position_seconds.toFixed(1)}s / ${playback.duration_seconds.toFixed(1)}s` }}
                        />
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
                            <Text type="secondary">
                                Frame {playback.current_frame + 1} / {Math.max(playback.total_frames, 1)} at {playback.position_seconds.toFixed(1)}s of {playback.duration_seconds.toFixed(1)}s
                            </Text>
                            <Space>
                                <Text type="secondary">Loop</Text>
                                <Switch
                                    checked={playback.loop}
                                    onChange={(checked) => void updatePlayback({ loop: checked })}
                                    disabled={!status.running || isEngineBusy}
                                />
                                <Select
                                    style={{ width: 120 }}
                                    value={playback.speed}
                                    options={PLAYBACK_SPEEDS.map((speed) => ({ label: `${speed}x`, value: speed }))}
                                    onChange={(speed) => void updatePlayback({ speed })}
                                    disabled={!status.running || isEngineBusy}
                                />
                            </Space>
                        </div>
                    </Space>
                </Card>
            )}
                

            {/* Logs */}
            <div>
                <Title level={5}>System Logs</Title>
                <LogFeed logs={logs} />
            </div>
        </Space>
    );
};

export default LiveFeed;
