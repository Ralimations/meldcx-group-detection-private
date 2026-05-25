import React, { useEffect, useMemo, useState } from 'react';
import { Button, Space, Typography } from 'antd';
import { ReloadOutlined } from '@ant-design/icons';

const { Text } = Typography;

interface SnapshotImageProps {
    endpoint: string;
    alt: string;
    refreshKey?: number;
    pollMs?: number;
    style?: React.CSSProperties;
    imgStyle?: React.CSSProperties;
    footer?: React.ReactNode;
}

const SnapshotImage: React.FC<SnapshotImageProps> = ({
    endpoint,
    alt,
    refreshKey = 0,
    pollMs = 1500,
    style,
    imgStyle,
    footer,
}) => {
    const [tick, setTick] = useState(0);

    useEffect(() => {
        const timer = window.setInterval(() => {
            setTick((prev) => prev + 1);
        }, pollMs);
        return () => window.clearInterval(timer);
    }, [pollMs]);

    const src = useMemo(
        () => `${endpoint}${endpoint.includes('?') ? '&' : '?'}t=${refreshKey}-${tick}`,
        [endpoint, refreshKey, tick],
    );

    return (
        <div style={style}>
            <img
                src={src}
                alt={alt}
                style={{
                    width: '100%',
                    display: 'block',
                    ...imgStyle,
                }}
            />
            <Space style={{ width: '100%', justifyContent: 'space-between', marginTop: 12 }}>
                <Text type="secondary">Snapshot view refreshing every {Math.round(pollMs / 1000)}s</Text>
                <Button size="small" icon={<ReloadOutlined />} onClick={() => setTick((prev) => prev + 1)}>
                    Refresh
                </Button>
            </Space>
            {footer}
        </div>
    );
};

export default SnapshotImage;
