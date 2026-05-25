import React from 'react';
import { Alert } from 'antd';

interface WorkspaceStatusBannerProps {
    tone?: 'info' | 'warning';
    message: string;
}

const WorkspaceStatusBanner: React.FC<WorkspaceStatusBannerProps> = ({ tone = 'info', message }) => {
    return (
        <div style={{ padding: '16px 24px 0 24px' }}>
            <Alert
                type={tone}
                message={message}
                showIcon
                style={{ borderRadius: 0 }}
            />
        </div>
    );
};

export default WorkspaceStatusBanner;
