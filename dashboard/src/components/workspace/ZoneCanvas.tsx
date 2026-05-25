import React, { useEffect, useRef, useState } from 'react';
import { PlusOutlined, ReloadOutlined } from '@ant-design/icons';
import { Alert, Button, Space, Tag, Typography } from 'antd';

const { Text } = Typography;

type ZoneType = 'detection' | 'height' | 'doorway';
type DragState =
    | { type: 'corner'; index: number }
    | { type: 'box'; startMouse: [number, number]; startPoints: [number, number][] }
    | null;

interface ZoneCanvasProps {
    refreshKey: number;
    detectionRoiPoints: [number, number][];
    heightRoiPoints: [number, number][];
    doorwayRoiPoints: [number, number][];
    activeZoneType: ZoneType;
    onChangeZonePoints: (zone: ZoneType, points: [number, number][]) => void;
    onCreateZoneBox: (zone: ZoneType) => void;
}

function clamp(value: number) {
    return Math.min(1, Math.max(0, value));
}

function clampPoint([x, y]: [number, number]): [number, number] {
    return [clamp(x), clamp(y)];
}

function pointInPolygon(point: [number, number], polygon: [number, number][]) {
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0];
        const yi = polygon[i][1];
        const xj = polygon[j][0];
        const yj = polygon[j][1];
        const intersects = ((yi > point[1]) !== (yj > point[1]))
            && (point[0] < ((xj - xi) * (point[1] - yi)) / ((yj - yi) || 1e-6) + xi);
        if (intersects) inside = !inside;
    }
    return inside;
}

const ZONE_THEME: Record<ZoneType, { stroke: string; fill: string; label: string }> = {
    detection: {
        stroke: '#2563eb',
        fill: 'rgba(37, 99, 235, 0.18)',
        label: 'Detection ROI',
    },
    height: {
        stroke: '#d97706',
        fill: 'rgba(217, 119, 6, 0.18)',
        label: 'Height ROI',
    },
    doorway: {
        stroke: '#14b8a6',
        fill: 'rgba(20, 184, 166, 0.18)',
        label: 'Doorway ROI',
    },
};

const ZoneCanvas: React.FC<ZoneCanvasProps> = ({
    refreshKey,
    detectionRoiPoints,
    heightRoiPoints,
    doorwayRoiPoints,
    activeZoneType,
    onChangeZonePoints,
    onCreateZoneBox,
}) => {
    const imageRef = useRef<HTMLImageElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [snapshotVersion, setSnapshotVersion] = useState(0);
    const [imageVersion, setImageVersion] = useState(0);
    const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
    const [dragState, setDragState] = useState<DragState>(null);

    const activePoints = activeZoneType === 'detection'
        ? detectionRoiPoints
        : activeZoneType === 'height'
            ? heightRoiPoints
            : doorwayRoiPoints;

    useEffect(() => {
        const drawPolygon = (
            ctx: CanvasRenderingContext2D,
            points: [number, number][],
            zone: ZoneType,
            editable: boolean,
        ) => {
            if (points.length !== 4) return;
            const theme = ZONE_THEME[zone];
            ctx.beginPath();
            ctx.lineWidth = editable ? 2.5 : 2;
            ctx.strokeStyle = theme.stroke;
            ctx.fillStyle = theme.fill;

            points.forEach((pt, index) => {
                const x = pt[0] * ctx.canvas.width;
                const y = pt[1] * ctx.canvas.height;
                if (index === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });

            ctx.closePath();
            ctx.fill();
            ctx.stroke();

            if (!editable) return;

            points.forEach((pt, index) => {
                const x = pt[0] * ctx.canvas.width;
                const y = pt[1] * ctx.canvas.height;
                const isActive = dragState?.type === 'corner' && dragState.index === index;
                const isHovered = hoveredIndex === index;
                ctx.beginPath();
                ctx.arc(x, y, isActive || isHovered ? 8 : 6, 0, Math.PI * 2);
                ctx.fillStyle = '#fff';
                ctx.fill();
                ctx.strokeStyle = isActive || isHovered ? '#ef4444' : theme.stroke;
                ctx.lineWidth = 2;
                ctx.stroke();
            });
        };

        const draw = () => {
            const canvas = canvasRef.current;
            const img = imageRef.current;
            if (!canvas || !img) return;

            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            canvas.width = img.clientWidth;
            canvas.height = img.clientHeight;
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            drawPolygon(ctx, detectionRoiPoints, 'detection', activeZoneType === 'detection');
            drawPolygon(ctx, heightRoiPoints, 'height', activeZoneType === 'height');
            drawPolygon(ctx, doorwayRoiPoints, 'doorway', activeZoneType === 'doorway');
        };

        const timer = setTimeout(draw, 60);
        window.addEventListener('resize', draw);
        return () => {
            clearTimeout(timer);
            window.removeEventListener('resize', draw);
        };
    }, [activeZoneType, detectionRoiPoints, doorwayRoiPoints, dragState, heightRoiPoints, hoveredIndex, imageVersion]);

    const getNormalizedPoint = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const canvas = canvasRef.current;
        if (!canvas) return null;
        return [
            clamp(e.nativeEvent.offsetX / canvas.width),
            clamp(e.nativeEvent.offsetY / canvas.height),
        ] as [number, number];
    };

    const findClosestPointIndex = (point: [number, number], threshold = 0.03) => {
        let closestIndex: number | null = null;
        let closestDistance = Number.POSITIVE_INFINITY;

        activePoints.forEach((candidate, index) => {
            const distance = Math.hypot(point[0] - candidate[0], point[1] - candidate[1]);
            if (distance < threshold && distance < closestDistance) {
                closestDistance = distance;
                closestIndex = index;
            }
        });

        return closestIndex;
    };

    const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const point = getNormalizedPoint(e);
        if (!point || activePoints.length !== 4) return;

        const pointIndex = findClosestPointIndex(point);
        if (pointIndex !== null) {
            setDragState({ type: 'corner', index: pointIndex });
            return;
        }

        if (pointInPolygon(point, activePoints)) {
            setDragState({ type: 'box', startMouse: point, startPoints: activePoints });
        }
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
        const point = getNormalizedPoint(e);
        if (!point) return;

        if (activePoints.length === 4) {
            setHoveredIndex(findClosestPointIndex(point));
        } else {
            setHoveredIndex(null);
        }

        if (!dragState) return;

        if (dragState.type === 'corner') {
            onChangeZonePoints(
                activeZoneType,
                activePoints.map((roiPoint, index) => (index === dragState.index ? clampPoint(point) : roiPoint)),
            );
            return;
        }

        const dx = point[0] - dragState.startMouse[0];
        const dy = point[1] - dragState.startMouse[1];
        onChangeZonePoints(
            activeZoneType,
            dragState.startPoints.map(([x, y]) => clampPoint([x + dx, y + dy])),
        );
    };

    const clearDragState = () => {
        setDragState(null);
        setHoveredIndex(null);
    };

    const cursor = dragState?.type === 'box'
        ? 'grabbing'
        : hoveredIndex !== null
            ? 'grab'
            : activePoints.length === 4
                ? 'move'
                : 'default';

    return (
        <div
            style={{
                flex: 1,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                overflow: 'hidden',
                position: 'relative',
                padding: 40,
            }}
        >
            <div style={{ position: 'absolute', top: 20, left: 20, right: 20, zIndex: 5 }}>
                <Alert
                    type="info"
                    showIcon
                    message="ROI setup uses a static snapshot"
                    description="Detection ROI limits where detection runs. Height ROI limits pose-based age samples. Doorway ROI defines the room-entry zone used by the monitor alerts."
                />
            </div>

            <div style={{ position: 'absolute', top: 92, left: 32, zIndex: 6 }}>
                <Space>
                    <Tag color={activeZoneType === 'detection' ? 'blue' : 'default'}>Detection ROI</Tag>
                    <Tag color={activeZoneType === 'height' ? 'orange' : 'default'}>Height ROI</Tag>
                    <Tag color={activeZoneType === 'doorway' ? 'cyan' : 'default'}>Doorway ROI</Tag>
                </Space>
            </div>

            <div
                style={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                    background: '#000',
                    borderRadius: 4,
                    overflow: 'hidden',
                    position: 'relative',
                }}
            >
                <img
                    ref={imageRef}
                    src={`http://localhost:8765/api/workspace/frame?t=${refreshKey}-${snapshotVersion}`}
                    style={{ display: 'block', maxWidth: '100%', height: 'auto', marginTop: 72 }}
                    alt="Scene Snapshot"
                    onLoad={() => {
                        const canvas = canvasRef.current;
                        const img = imageRef.current;
                        if (!canvas || !img) return;
                        canvas.width = img.clientWidth;
                        canvas.height = img.clientHeight;
                        setImageVersion((prev) => prev + 1);
                    }}
                />
                <canvas
                    ref={canvasRef}
                    style={{
                        position: 'absolute',
                        top: 72,
                        left: 0,
                        width: '100%',
                        height: 'calc(100% - 72px)',
                        cursor,
                    }}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={clearDragState}
                    onMouseLeave={clearDragState}
                />
            </div>

            <div style={{ position: 'absolute', bottom: 60, left: '50%', transform: 'translateX(-50%)' }}>
                <Space>
                    <Button type="primary" icon={<PlusOutlined />} onClick={() => onCreateZoneBox(activeZoneType)}>
                        {activePoints.length === 4 ? `Reset ${ZONE_THEME[activeZoneType].label}` : `Add ${ZONE_THEME[activeZoneType].label}`}
                    </Button>
                    <Button icon={<ReloadOutlined />} onClick={() => setSnapshotVersion((prev) => prev + 1)}>
                        Refresh Snapshot
                    </Button>
                </Space>
            </div>
            <div style={{ position: 'absolute', bottom: 20, left: '50%', transform: 'translateX(-50%)' }}>
                <Text type="secondary">
                    {activePoints.length === 4
                        ? `Drag a corner to reshape the ${ZONE_THEME[activeZoneType].label.toLowerCase()} or drag inside it to move it.`
                        : `Add a ${ZONE_THEME[activeZoneType].label.toLowerCase()} to start editing.`}
                </Text>
            </div>
        </div>
    );
};

export default ZoneCanvas;
