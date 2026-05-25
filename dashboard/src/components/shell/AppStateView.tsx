import React from 'react';
import { Alert, Button, Result, Space, Spin, Typography } from 'antd';

const { Paragraph, Text, Title } = Typography;

interface AppStateViewProps {
    title: string;
    subtitle: string;
    tone?: 'loading' | 'empty' | 'error';
    actionLabel?: string;
    onAction?: () => void;
    details?: string;
}

const AppStateView: React.FC<AppStateViewProps> = ({
    title,
    subtitle,
    tone = 'empty',
    actionLabel,
    onAction,
    details,
}) => {
    const icon = tone === 'loading'
        ? <Spin size="large" />
        : undefined;

    return (
        <div
            style={{
                minHeight: '100vh',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: 'linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%)',
                padding: 32,
            }}
        >
            <div
                style={{
                    width: '100%',
                    maxWidth: 720,
                    background: '#fff',
                    border: '1px solid #e2e8f0',
                    boxShadow: '0 24px 80px rgba(15, 23, 42, 0.08)',
                    padding: 32,
                }}
            >
                <Space direction="vertical" size="large" style={{ width: '100%' }}>
                    <div>
                        <Text type="secondary" style={{ letterSpacing: '0.12em', textTransform: 'uppercase' }}>
                            Group Detection Dashboard
                        </Text>
                        <Title level={2} style={{ margin: '8px 0 12px 0' }}>
                            {title}
                        </Title>
                        <Paragraph type="secondary" style={{ margin: 0 }}>
                            {subtitle}
                        </Paragraph>
                    </div>

                    {tone === 'error' ? (
                        <Alert
                            type="error"
                            message="Dashboard startup failed"
                            description={details ?? 'An unknown error occurred.'}
                            showIcon
                        />
                    ) : (
                        <Result
                            status={tone === 'loading' ? 'info' : '404'}
                            icon={icon}
                            title={tone === 'loading' ? 'Starting services' : 'No active workspace'}
                            subTitle={tone === 'loading'
                                ? 'Waiting for the dashboard runtime to become ready.'
                                : 'Select or create a sensor to enter the workspace.'}
                            style={{ padding: 0 }}
                        />
                    )}

                    {actionLabel && onAction && (
                        <Button type="primary" size="large" onClick={onAction}>
                            {actionLabel}
                        </Button>
                    )}
                </Space>
            </div>
        </div>
    );
};

export default AppStateView;
