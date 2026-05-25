import React, { useState, useEffect } from 'react';
import { Button, Table, Typography, Space, Tag } from 'antd';
import { FileTextOutlined, VideoCameraOutlined, UserOutlined, TeamOutlined } from '@ant-design/icons';
import { api } from '../services/api';
import type { TestSummary as ITestSummary } from '../services/api';

const { Text } = Typography;

const TestSummary: React.FC = () => {
    const [summaries, setSummaries] = useState<ITestSummary[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        try {
            const data = await api.fetchSummaries();
            setSummaries(data);
        } catch (error) {
            console.error('Failed to fetch summaries:', error);
        } finally {
            setLoading(false);
        }
    };

    const columns = [
        {
            title: 'Timestamp',
            dataIndex: 'timestamp',
            key: 'timestamp',
            width: 150,
            ellipsis: true,
            render: (text: string) => <Text copyable ellipsis={{ tooltip: text }}>{text}</Text>,
        },
        {
            title: 'Source',
            dataIndex: 'source',
            key: 'source',
            width: 150,
            ellipsis: true,
            render: (text: string) => (
                <Space style={{ minWidth: 0 }}>
                    <VideoCameraOutlined />
                    <Text ellipsis={{ tooltip: text }} style={{ minWidth: 0 }}>
                        {text}
                    </Text>
                </Space>
            ),
        },
        {
            title: 'Peak People',
            dataIndex: 'max_people',
            key: 'max_people',
            width: 88,
            render: (val: number) => (
                <Space>
                    <UserOutlined style={{ color: '#1677ff' }} />
                    <Text strong>{val}</Text>
                </Space>
            ),
        },
        {
            title: 'Peak Groups',
            dataIndex: 'max_groups',
            key: 'max_groups',
            width: 88,
            render: (val: number) => (
                <Space>
                    <TeamOutlined style={{ color: '#52c41a' }} />
                    <Text strong>{val}</Text>
                </Space>
            ),
        },
        {
            title: 'Avg FPS',
            dataIndex: 'avg_fps',
            key: 'avg_fps',
            width: 92,
            render: (val: number) => (
                <Tag color={val > 15 ? 'green' : val > 5 ? 'blue' : 'orange'}>
                    {val.toFixed(1)} fps
                </Tag>
            ),
        },
        {
            title: 'Duration',
            dataIndex: 'duration',
            key: 'duration',
            width: 90,
            render: (val: number) => {
                const mins = Math.floor(val / 60);
                const secs = Math.floor(val % 60);
                return `${mins}m ${secs}s`;
            },
        },
        {
            title: 'Artifacts',
            key: 'artifacts',
            width: 140,
            render: (_: unknown, record: ITestSummary) => (
                <Space size="small" wrap={false}>
                    <Button
                        size="small"
                        type="text"
                        icon={<VideoCameraOutlined />}
                        disabled={!record.annotated_video_path}
                        onClick={() => window.open(`http://localhost:8765/api/summaries/${record.id}/artifact/video`, '_blank')}
                    >
                        Video
                    </Button>
                    <Button
                        size="small"
                        type="text"
                        icon={<FileTextOutlined />}
                        disabled={!record.detection_json_path}
                        onClick={() => window.open(`http://localhost:8765/api/summaries/${record.id}/artifact/json`, '_blank')}
                    >
                        JSON
                    </Button>
                </Space>
            ),
        },
    ];

    return (
        <Table 
            dataSource={summaries} 
            columns={columns} 
            rowKey="id" 
            loading={loading}
            pagination={{ pageSize: 5, showSizeChanger: false }}
            size="small"
            tableLayout="fixed"
            style={{ borderRadius: 0 }}
        />
    );
};

export default TestSummary;
