import React, { useState } from 'react';
import { ReloadOutlined } from '@ant-design/icons';
import { Alert, Button, Tag, Typography } from 'antd';

const { Text } = Typography;

interface UndistortCanvasProps {
    refreshKey: number;
}

const IMAGE_FRAME_STYLE: React.CSSProperties = {
    flex: 1,
    minWidth: 0,
    background: '#000',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    overflow: 'hidden',
};

const IMAGE_STYLE: React.CSSProperties = {
    display: 'block',
    width: '100%',
    height: 'auto',
    maxHeight: '70vh',
    objectFit: 'contain',
};

const UndistortCanvas: React.FC<UndistortCanvasProps> = ({ refreshKey }) => {
    const [snapshotVersion, setSnapshotVersion] = useState(0);
    const cacheKey = `${refreshKey}-${snapshotVersion}`;

    return (
        <div
            style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                padding: 16,
                gap: 10,
            }}
        >
            <Alert
                type="info"
                showIcon
                message="Camera undistort uses a static preview"
                description="Both previews come from the current frame buffer and refresh only when requested, so this tab does not keep an extra live snapshot loop running."
            />

            <div
                style={{
                    width: '100%',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                    gap: 0,
                    alignItems: 'center',
                }}
            >
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <Tag color="gold">Original Lens View</Tag>
                </div>
                <div style={{ display: 'flex', justifyContent: 'center' }}>
                    <Tag color="green">Corrected Preview</Tag>
                </div>
            </div>

            <div
                style={{
                    width: '100%',
                    maxHeight: '100%',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                    background: '#000',
                    borderRadius: 4,
                    overflow: 'hidden',
                    position: 'relative',
                }}
            >
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))' }}>
                    <div style={{ ...IMAGE_FRAME_STYLE, borderRight: '1px solid rgba(255,255,255,0.12)' }}>
                        <img
                            src={`http://localhost:8765/api/undistort/comparison?side=original&t=${cacheKey}`}
                            alt="Original Lens View"
                            style={IMAGE_STYLE}
                        />
                    </div>
                    <div style={IMAGE_FRAME_STYLE}>
                        <img
                            src={`http://localhost:8765/api/undistort/comparison?side=corrected&t=${cacheKey}`}
                            alt="Corrected Preview"
                            style={IMAGE_STYLE}
                        />
                    </div>
                </div>
            </div>

            <div>
                <Button icon={<ReloadOutlined />} onClick={() => setSnapshotVersion((prev) => prev + 1)}>
                    Refresh Preview
                </Button>
            </div>
            <div>
                <Text type="secondary">
                    Adjust the lens settings in the sidebar, then refresh to compare the original and corrected frame.
                </Text>
            </div>
        </div>
    );
};

export default UndistortCanvas;
