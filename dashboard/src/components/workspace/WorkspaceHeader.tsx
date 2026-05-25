import React, { useRef, useState } from 'react';
import { ArrowLeftOutlined } from '@ant-design/icons';
import { Breadcrumb, Button, Input, Space, Typography, message } from 'antd';

const { Title, Text } = Typography;

interface WorkspaceHeaderProps {
    sceneName?: string;
    exiting?: boolean;
    renaming?: boolean;
    onRename?: (name: string) => Promise<void> | void;
    onExit: () => void;
}

const WorkspaceHeader: React.FC<WorkspaceHeaderProps> = ({
    sceneName,
    exiting = false,
    renaming = false,
    onRename,
    onExit,
}) => {
    const [isEditing, setIsEditing] = useState(false);
    const [draftName, setDraftName] = useState('');
    const commitInFlightRef = useRef(false);

    const handleCommit = async () => {
        if (commitInFlightRef.current) {
            return;
        }

        const nextName = draftName.trim();
        const currentName = sceneName || 'New Scene';
        commitInFlightRef.current = true;

        if (!nextName || nextName === currentName) {
            setDraftName('');
            setIsEditing(false);
            commitInFlightRef.current = false;
            return;
        }

        setIsEditing(false);
        setDraftName('');
        try {
            await onRename?.(nextName);
            message.success(`Scene renamed to "${nextName}"`);
        } catch {
            message.error('Failed to rename scene. If the dashboard backend was already running, restart it and try again.');
        } finally {
            commitInFlightRef.current = false;
        }
    };

    return (
        <div
            style={{
                background: '#fff',
                height: 64,
                padding: '0 24px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                borderBottom: '1px solid #e2e8f0',
            }}
        >
            <Space size="large">
                <Button type="text" icon={<ArrowLeftOutlined />} onClick={onExit} disabled={exiting}>
                    Back to Sensors
                </Button>
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                    {isEditing ? (
                        <Input
                            autoFocus
                            value={draftName}
                            disabled={renaming}
                            onChange={(event) => setDraftName(event.target.value)}
                            onPressEnter={() => void handleCommit()}
                            onBlur={() => {
                                if (!renaming) {
                                    void handleCommit();
                                }
                            }}
                            style={{ width: 320 }}
                        />
                    ) : (
                        <Title
                            level={4}
                            style={{ margin: 0, fontSize: 18, cursor: onRename ? 'text' : 'default' }}
                            onDoubleClick={() => {
                                if (onRename && !renaming) {
                                    setDraftName(sceneName || 'New Scene');
                                    setIsEditing(true);
                                }
                            }}
                        >
                            Scene {sceneName || 'New Scene'}
                        </Title>
                    )}
                    <Breadcrumb separator="/" style={{ fontSize: 12, color: '#94a3b8' }}>
                        <Breadcrumb.Item>
                            <Space size={4}>
                                <Text type="secondary">📁</Text>
                                Sensors
                            </Space>
                        </Breadcrumb.Item>
                        <Breadcrumb.Item>
                            <Space size={4}>
                                <Text type="secondary">🎬</Text>
                                {sceneName || 'Scene'}
                            </Space>
                        </Breadcrumb.Item>
                    </Breadcrumb>
                </div>
            </Space>
        </div>
    );
};

export default WorkspaceHeader;
