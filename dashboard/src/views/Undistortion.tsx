import React from 'react';
import { Alert, Card, Space, Row, Col } from 'antd';
import { useConfigContext } from '../contexts/useConfigContext';
import SnapshotImage from '../components/workspace/SnapshotImage';

const Undistortion: React.FC = () => {
    const { refreshKey } = useConfigContext();

    return (
        <Space direction="vertical" style={{ width: '100%' }} size="large">
            <Alert
                type="info"
                showIcon
                message="Camera undistort uses snapshot previews"
                description="This tab mirrors the active live source with periodic snapshots instead of a second continuous stream."
            />
            <Row gutter={16}>
                <Col span={12}>
                    <Card title="Original" style={{ borderRadius: 0, overflow: 'hidden' }} bodyStyle={{ padding: 0 }}>
                        <div style={{ background: '#000', height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <SnapshotImage
                                endpoint="http://localhost:8765/api/undistort/frame?side=original"
                                alt="Original"
                                refreshKey={refreshKey}
                                pollMs={1500}
                                style={{ width: '100%', padding: 12 }}
                                imgStyle={{ maxWidth: '100%', maxHeight: 320, objectFit: 'contain' }}
                            />
                        </div>
                    </Card>
                </Col>
                <Col span={12}>
                    <Card title="Corrected" style={{ borderRadius: 0, overflow: 'hidden' }} bodyStyle={{ padding: 0 }}>
                        <div style={{ background: '#000', height: '400px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                            <SnapshotImage
                                endpoint="http://localhost:8765/api/undistort/frame?side=corrected"
                                alt="Corrected"
                                refreshKey={refreshKey}
                                pollMs={1500}
                                style={{ width: '100%', padding: 12 }}
                                imgStyle={{ maxWidth: '100%', maxHeight: 320, objectFit: 'contain' }}
                            />
                        </div>
                    </Card>
                </Col>
            </Row>
        </Space>
    );
};

export default Undistortion;
