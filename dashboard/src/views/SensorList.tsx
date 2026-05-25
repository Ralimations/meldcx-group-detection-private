import React, { useState } from 'react';
import { Typography, Space, Button, Modal, Input, Select, InputNumber, Table, Dropdown, Alert } from 'antd';
import type { MenuProps, TableColumnsType } from 'antd';
import { 
    VideoCameraOutlined, 
    PlusOutlined,
    GlobalOutlined,
    FileImageOutlined,
    MoreOutlined,
    DeleteOutlined,
    DatabaseOutlined
} from '@ant-design/icons';
import { useConfigContext } from '../contexts/useConfigContext';
import type { ConfigValues, Sensor } from '../services/api';

const { Title, Text } = Typography;

const RTSP_SOURCE = 'rtsp://root:Skunkw0rks@192.168.100.17/axis-media/media.amp';

interface SensorListProps {
    onViewChange: (view: string) => void;
}

const SensorList: React.FC<SensorListProps> = ({ onViewChange }) => {
    const { sensors, videos, createSensor, deleteSensor, loadSensor, activeSensorId, status, actionState } = useConfigContext();
    const [isAddModalOpen, setIsAddModalOpen] = useState(false);
    
    // Modal State
    const [newName, setNewName] = useState('');
    const [inputMode, setInputMode] = useState<'camera' | 'network' | 'file'>('file');
    const [selectedVideo, setSelectedVideo] = useState<string>('');
    const [cameraIndex, setCameraIndex] = useState<number>(0);
    const [networkUrl, setNetworkUrl] = useState<string>(RTSP_SOURCE);

    const handleCreate = async () => {
        if (!newName.trim()) return;

        let sourceOverrides: ConfigValues;

        if (inputMode === 'camera') {
            sourceOverrides = {
                input_mode: 'camera',
                camera_index: cameraIndex,
                source: cameraIndex.toString(),
            };
        } else if (inputMode === 'network') {
            sourceOverrides = {
                input_mode: 'source',
                source: networkUrl.trim(),
            };
        } else {
            const selected = selectedVideo || videos.videos[0];
            sourceOverrides = {
                input_mode: 'source',
                source: selected ? `media/${selected}` : '',
            };
        }

        await createSensor(newName.trim(), sourceOverrides);
        
        setIsAddModalOpen(false);
        setNewName('');
        setInputMode('file');
        setSelectedVideo('');
        setNetworkUrl(RTSP_SOURCE);
    };

    const getSensorKind = (config: ConfigValues) => {
        if (config.input_mode === 'camera') return 'camera';
        const source = String(config.source || '');
        if (source.startsWith('rtsp://') || source.startsWith('http://') || source.startsWith('https://')) {
            return 'network';
        }
        return 'file';
    };

    const getIconForSource = (kind: string) => {
        if (kind === 'camera') return <VideoCameraOutlined style={{ color: '#1677ff' }} />;
        if (kind === 'network') return <GlobalOutlined style={{ color: '#52c41a' }} />;
        return <FileImageOutlined style={{ color: '#eb2f96' }} />;
    };

    const getSourceLabel = (config: ConfigValues) => {
        const kind = getSensorKind(config);
        if (kind === 'camera') return `Camera ${config.camera_index ?? 0}`;
        if (kind === 'network') return `Network Stream`;
        return config.source ? String(config.source).split('/').pop() : 'Unknown File';
    };

    const columns: TableColumnsType<Sensor> = [
        {
            title: 'Name',
            dataIndex: 'name',
            key: 'name',
            render: (text: string) => (
                <div style={{ fontWeight: 500, color: '#1677ff', display: 'flex', alignItems: 'center', gap: '8px' }}>
                   <PlusOutlined style={{ fontSize: '10px', color: '#1677ff', border: '1px solid #1677ff', padding: '1px', borderRadius: '2px' }} /> 
                   {text}
                </div>
            ),
        },
        {
            title: 'Type',
            key: 'type',
            render: (_value, record) => (
                <Space>
                    {getIconForSource(getSensorKind(record.config))}
                    <span style={{ textTransform: 'capitalize' }}>{getSensorKind(record.config)}</span>
                </Space>
            ),
        },
        {
            title: 'Source',
            key: 'source',
            render: (_value, record) => (
                <Text type="secondary">{getSourceLabel(record.config)}</Text>
            ),
        },
        {
            title: 'Status',
            key: 'status',
            render: (_value, record) => {
                const isActive = record.id === activeSensorId;
                const label = isActive
                    ? (status.running ? (status.paused ? 'Paused' : 'Running') : 'Selected')
                    : 'Configured';
                const color = isActive
                    ? (status.running ? (status.paused ? '#faad14' : '#52c41a') : '#1677ff')
                    : '#94a3b8';

                return (
                <Space size="small">
                   <div style={{ width: 8, height: 8, borderRadius: '50%', background: color }} />
                   <span>{label}</span>
                </Space>
                );
            },
        },
        {
            title: 'Actions',
            key: 'actions',
            width: 100,
            align: 'center' as const,
            render: (_value, record) => {
                const items: MenuProps['items'] = [
                    {
                        key: 'delete',
                        label: 'Delete Sensor',
                        icon: <DeleteOutlined />,
                        danger: true,
                    }
                ];
                
                return (
                    <div onClick={(e) => e.stopPropagation()}>
                        <Dropdown 
                            menu={{ 
                                items, 
                                onClick: ({ key }) => {
                                    if (key === 'delete') {
                                        Modal.confirm({
                                            title: 'Delete this sensor?',
                                            content: 'Are you sure you want to delete this sensor config?',
                                            okText: 'Yes',
                                            okType: 'danger',
                                            cancelText: 'No',
                                            okButtonProps: { loading: actionState.deletingSensor },
                                            onOk: () => deleteSensor(record.id)
                                        });
                                    }
                                }
                            }} 
                            trigger={['click']}
                            disabled={actionState.loadingSensor || actionState.deletingSensor}
                        >
                            <Button
                                type="text"
                                icon={<MoreOutlined style={{ fontSize: '16px', color: '#888' }} />}
                                disabled={actionState.loadingSensor || actionState.deletingSensor}
                                loading={actionState.deletingSensor}
                            />
                        </Dropdown>
                    </div>
                );
            },
        },
    ];

    return (
        <div style={{ padding: '24px 40px', background: '#fff', minHeight: '100vh' }}>
            {(actionState.loadingSensor || actionState.creatingSensor || actionState.deletingSensor) && (
                <div style={{ marginBottom: 16 }}>
                    <Alert
                        type="info"
                        showIcon
                        message={
                            actionState.loadingSensor
                                ? 'Loading sensor workspace...'
                                : actionState.creatingSensor
                                    ? 'Creating sensor...'
                                    : 'Deleting sensor...'
                        }
                    />
                </div>
            )}
            {/* Breadcrumb / Title Area */}
            <div style={{ marginBottom: '24px' }}>
                <Space style={{ marginBottom: '8px', color: '#666' }}>
                    <DatabaseOutlined /> 
                    <span>Sensors</span>
                </Space>
                
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div>
                        <Title level={2} style={{ color: '#1677ff', margin: 0, fontWeight: 600 }}>Manage Sensors</Title>
                        <Text type="secondary" style={{ fontSize: '14px' }}>Manage all your sensors in one place.</Text>
                    </div>
                    <Button 
                        type="primary" 
                        icon={<PlusOutlined />} 
                        size="large"
                        onClick={() => setIsAddModalOpen(true)}
                        loading={actionState.creatingSensor}
                        disabled={actionState.loadingSensor || actionState.creatingSensor}
                        style={{ borderRadius: '4px', background: '#1677ff', fontWeight: 500 }}
                    >
                        Add Sensor
                    </Button>
                </div>
            </div>

            {/* Table Area */}
            <div style={{ display: 'flex', alignItems: 'center', padding: '16px 0', borderBottom: '1px solid #f0f0f0' }}>
                <Text strong style={{ marginRight: '16px' }}>All Sensors</Text>
                <Text type="secondary">{sensors.length} results</Text>
            </div>

            <Table
                dataSource={sensors}
                columns={columns}
                rowKey="id"
                pagination={{ pageSize: 10, position: ['topRight', 'bottomRight'] }}
                style={{ border: '1px solid #f0f0f0', borderBottom: 'none', borderTop: 'none' }}
                onRow={(record) => ({
                    onClick: async () => {
                        if (actionState.loadingSensor || actionState.creatingSensor || actionState.deletingSensor) return;
                        await loadSensor(record.id);
                        onViewChange('live');
                    },
                    style: { cursor: actionState.loadingSensor ? 'progress' : 'pointer', opacity: actionState.loadingSensor ? 0.7 : 1 }
                })}
                rowClassName={() => 'sensor-row-hover'}
                loading={actionState.loadingSensor}
            />

            <Modal
                title="Add New Sensor"
                open={isAddModalOpen}
                onOk={handleCreate}
                onCancel={() => setIsAddModalOpen(false)}
                okText="Create Sensor"
                width={500}
                confirmLoading={actionState.creatingSensor}
                okButtonProps={{ disabled: actionState.loadingSensor }}
                cancelButtonProps={{ disabled: actionState.creatingSensor }}
            >
                <Space direction="vertical" style={{ width: '100%', marginTop: '16px' }} size="large">
                    <div>
                        <Text strong>Sensor Name</Text>
                        <Input 
                            placeholder="e.g. Front Door Camera" 
                            size="large"
                            value={newName}
                            onChange={e => setNewName(e.target.value)}
                            style={{ marginTop: '8px' }}
                        />
                    </div>

                    <div>
                        <Text strong>Input Type</Text>
                        <Select 
                            value={inputMode} 
                            onChange={setInputMode}
                            size="large"
                            style={{ width: '100%', marginTop: '8px' }}
                            options={[
                                { label: 'Video File (Local)', value: 'file' },
                                { label: 'Network Stream (RTSP)', value: 'network' },
                                { label: 'Local Camera (USB)', value: 'camera' },
                            ]}
                        />
                    </div>

                    {inputMode === 'camera' && (
                        <div>
                            <Text strong>Camera Index</Text><br/>
                            <InputNumber 
                                min={0} 
                                max={10} 
                                value={cameraIndex} 
                                onChange={(val) => setCameraIndex(val || 0)}
                                size="large"
                                style={{ width: '100%' }}
                            />
                        </div>
                    )}

                    {inputMode === 'network' && (
                        <div>
                            <Text strong>RTSP URL</Text>
                            <Input
                                value={networkUrl}
                                onChange={(e) => setNetworkUrl(e.target.value)}
                                size="large"
                                placeholder="rtsp://camera-host/stream"
                            />
                        </div>
                    )}

                    {inputMode === 'file' && (
                        <div>
                            <Text strong>Select Video</Text>
                            <Select
                                value={selectedVideo || undefined}
                                placeholder="Select a test video..."
                                size="large"
                                style={{ width: '100%' }}
                                onChange={setSelectedVideo}
                                options={videos.videos.map(v => ({ label: v, value: v }))}
                            />
                        </div>
                    )}
                </Space>
            </Modal>
        </div>
    );
};

export default SensorList;
