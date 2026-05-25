import React from 'react';
import { AimOutlined, ColumnWidthOutlined, DashboardOutlined, HistoryOutlined } from '@ant-design/icons';
import { Button, Tooltip } from 'antd';
import type { WorkspaceView } from '../../hooks/useDashboardController';

interface WorkspaceNavProps {
    currentView: WorkspaceView;
    onChange: (view: WorkspaceView) => void;
}

interface NavButtonProps {
    icon: React.ReactNode;
    title: string;
    active?: boolean;
    onClick?: () => void;
}

const NavButton: React.FC<NavButtonProps> = ({ icon, title, active, onClick }) => (
    <Tooltip title={title} placement="right">
        <Button
            type={active ? 'primary' : 'text'}
            icon={icon}
            style={{
                width: 44,
                height: 44,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                background: active ? '#1677ff' : 'transparent',
                borderRadius: 4,
                marginBottom: 12,
                color: active ? '#fff' : '#555',
            }}
            onClick={onClick}
        />
    </Tooltip>
);

const WorkspaceNav: React.FC<WorkspaceNavProps> = ({ currentView, onChange }) => {
    return (
        <div style={{ background: '#fff', borderRight: '1px solid #e2e8f0', padding: '18px 14px', width: 72 }}>
            <NavButton
                icon={<DashboardOutlined style={{ fontSize: 20 }} />}
                title="Live Monitoring"
                active={currentView === 'live'}
                onClick={() => onChange('live')}
            />
            <NavButton
                icon={<ColumnWidthOutlined style={{ fontSize: 20 }} />}
                title="ROI Zones"
                active={currentView === 'zones'}
                onClick={() => onChange('zones')}
            />
            <NavButton
                icon={<AimOutlined style={{ fontSize: 20 }} />}
                title="Camera Undistort"
                active={currentView === 'undistort'}
                onClick={() => onChange('undistort')}
            />
            <NavButton
                icon={<HistoryOutlined style={{ fontSize: 20 }} />}
                title="Run Summaries"
                active={currentView === 'summary'}
                onClick={() => onChange('summary')}
            />
        </div>
    );
};

export default WorkspaceNav;
