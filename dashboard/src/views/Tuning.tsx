import React, { useMemo, useState } from 'react';
import { Card, Typography, Space, Row, Col, Button, Modal, message, Input, Alert, Popover, Spin, Switch } from 'antd';
import { useConfigContext } from '../contexts/useConfigContext';
import ConfigPanel from '../components/controls/ConfigPanel';
import type { ConfigValue } from '../services/api';
import { ANALYTICS_PRESETS, PERFORMANCE_PRESETS } from '../presets/performanceProfiles';
import { ALL_CATEGORY_BUNDLE_IDS, CATEGORY_BUNDLES, buildCategoryBundleUpdates, bundleIdsToCategories } from '../presets/categoryBundles';
import { 
    ControlOutlined, 
    ExportOutlined, 
    UndoOutlined,
    SaveOutlined,
    FolderOpenOutlined,
    UnorderedListOutlined,
    EyeOutlined,
    AimOutlined,
    UserOutlined,
    DeploymentUnitOutlined,
    RocketOutlined
} from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

const CATEGORY_LABELS: Record<string, string> = {
    ROI: 'Area',
    'Detection ROI': 'Detection ROI',
    'Height ROI': 'Height ROI',
    'Doorway ROI': 'Doorway ROI',
    Alerts: 'Alerts',
    Archive: 'Archive',
    Advanced: 'Advanced',
};

const CATEGORY_ORDER = [
    'Detection ROI',
    'Height ROI',
    'Doorway ROI',
    'Input',
    'Detection',
    'Demographics',
    'Alerts',
    'Archive',
    'Groups',
    'Tracking',
    'Performance',
    'Display',
    'Lens',
    'Advanced',
    'ROI',
];

interface TuningProps {
    isSidebar?: boolean;
    forceCategory?: string;
    initialCategory?: string;
}

const Tuning: React.FC<TuningProps> = ({ isSidebar, forceCategory, initialCategory }) => {
    const {
        config, updateParam, updateParams, loading,
        resetDefaults, savePreset, loadPreset, getPresets, resetEngine,
    } = useConfigContext();
    
    const [pendingRestart, setPendingRestart] = useState(false);
    const [isExportVisible, setIsExportVisible] = useState(false);
    const [isSaveVisible, setIsSaveVisible] = useState(false);
    const [isLoadVisible, setIsLoadVisible] = useState(false);
    const [presetName, setPresetName] = useState('');
    const [availablePresets, setAvailablePresets] = useState<string[]>([]);
    const [activeCategoryOverride, setActiveCategory] = useState<string>(initialCategory ?? '');
    const [selectedBundles, setSelectedBundles] = useState<string[]>(ALL_CATEGORY_BUNDLE_IDS);
    const [appliedBundles, setAppliedBundles] = useState<string[]>(ALL_CATEGORY_BUNDLE_IDS);
    const [bundlePopoverOpen, setBundlePopoverOpen] = useState(false);
    const [busyState, setBusyState] = useState<string | null>(null);
    const [showAdvancedOptions, setShowAdvancedOptions] = useState(false);

    const categories = useMemo(() => {
        if (!config) return [];
        const cats = new Set<string>();
        const HIDDEN_KEYS = ['image', 'batch_sources'];
        Object.keys(config.meta).forEach(key => {
            if (!HIDDEN_KEYS.includes(key) && !config.meta[key].advanced) {
                const cat = config.meta[key].cat;
                cats.add(cat);
            }
        });
        return Array.from(cats).sort((a, b) => {
            const aIndex = CATEGORY_ORDER.indexOf(a);
            const bIndex = CATEGORY_ORDER.indexOf(b);
            const safeA = aIndex === -1 ? Number.MAX_SAFE_INTEGER : aIndex;
            const safeB = bIndex === -1 ? Number.MAX_SAFE_INTEGER : bIndex;
            return safeA - safeB || a.localeCompare(b);
        });
    }, [config]);

    const activeBundleCategories = useMemo(() => bundleIdsToCategories(appliedBundles), [appliedBundles]);
    const visibleCategories = useMemo(() => {
        if (forceCategory) {
            return categories;
        }
        if (activeBundleCategories.length === 0) {
            return showAdvancedOptions ? ['Advanced'] : [];
        }
        const allowed = new Set(activeBundleCategories);
        const baseCategories = categories.filter((cat) => allowed.has(cat));
        return showAdvancedOptions ? [...baseCategories, 'Advanced'] : baseCategories;
    }, [activeBundleCategories, categories, forceCategory, showAdvancedOptions]);

    if (loading || !config) return null;

    const activeCategory = forceCategory
        ?? (visibleCategories.includes(activeCategoryOverride) ? activeCategoryOverride : visibleCategories[0] ?? '');

    const handleUpdate = async (key: string, value: ConfigValue) => {
        if (busyState) {
            return;
        }
        await updateParam(key, value);
        if (config.meta[key]?.needs_restart) {
            setPendingRestart(true);
        }
    };

    const handleApplyPerformancePreset = async (presetKey: keyof typeof PERFORMANCE_PRESETS) => {
        const preset = PERFORMANCE_PRESETS[presetKey];
        try {
            setBusyState(`Applying ${preset.label}...`);
            await updateParams(preset.updates);
            const needsRestart = Object.keys(preset.updates).some((key) => config.meta[key]?.needs_restart);
            if (needsRestart) {
                setPendingRestart(true);
            }
            message.success(`${preset.label} preset applied`);
        } catch {
            message.error(`Failed to apply ${preset.label.toLowerCase()} preset`);
        } finally {
            setBusyState(null);
        }
    };

    const handleApplyAnalyticsPreset = async (presetKey: keyof typeof ANALYTICS_PRESETS) => {
        const preset = ANALYTICS_PRESETS[presetKey];
        try {
            setBusyState(`Applying ${preset.label}...`);
            await updateParams(preset.updates);
            const needsRestart = Object.keys(preset.updates).some((key) => config.meta[key]?.needs_restart);
            if (needsRestart) {
                setPendingRestart(true);
            }
            message.success(`${preset.label} preset applied`);
        } catch {
            message.error(`Failed to apply ${preset.label.toLowerCase()} preset`);
        } finally {
            setBusyState(null);
        }
    };

    const handleApplyCategoryBundles = async () => {
        const updates = buildCategoryBundleUpdates(selectedBundles);
        try {
            setBusyState('Applying category bundles...');
            await updateParams(updates);
            const needsRestart = Object.keys(updates).some((key) => config.meta[key]?.needs_restart);
            if (needsRestart) {
                setPendingRestart(true);
            }
            setAppliedBundles(selectedBundles);
            setBundlePopoverOpen(false);
            message.success(
                selectedBundles.length === 0
                    ? 'Raw video bundle applied'
                    : `Applied ${selectedBundles.length} category bundle${selectedBundles.length === 1 ? '' : 's'}`,
            );
        } catch {
            message.error('Failed to apply category bundles');
        } finally {
            setBusyState(null);
        }
    };

    const handleApplyRestart = async () => {
        try {
            setBusyState('Restarting engine...');
            message.loading({ content: 'Restarting engine...', key: 'restart' });
            await resetEngine();
            setPendingRestart(false);
            message.success({ content: 'Engine restarted and settings applied', key: 'restart' });
        } catch {
            message.error({ content: 'Failed to restart engine', key: 'restart' });
        } finally {
            setBusyState(null);
        }
    };

    const categoryIcons: Record<string, React.ReactNode> = {
        Detection: <EyeOutlined />,
        Tracking: <AimOutlined />,
        Groups: <UserOutlined />,
        Semantics: <DeploymentUnitOutlined />,
        Demographics: <DeploymentUnitOutlined />,
        Alerts: <DeploymentUnitOutlined />,
        Archive: <FolderOpenOutlined />,
        Performance: <RocketOutlined />,
        Display: <UnorderedListOutlined />,
        ROI: <ControlOutlined />,
        'Detection ROI': <ControlOutlined />,
        'Height ROI': <ControlOutlined />,
        'Doorway ROI': <ControlOutlined />,
        Lens: <AimOutlined />,
        Advanced: <ControlOutlined />,
    };

    const handleResetDefaults = async () => {
        Modal.confirm({
            title: 'Reset to Defaults?',
            content: 'This will reset all parameters to their engine defaults.',
            onOk: async () => {
                try {
                    setBusyState('Resetting to defaults...');
                    await resetDefaults();
                    message.success('Parameters reset to defaults');
                } catch {
                    message.error('Failed to reset parameters');
                } finally {
                    setBusyState(null);
                }
            }
        });
    };

    const handleSavePreset = async () => {
        if (!presetName.trim()) {
            message.warning('Please enter a preset name');
            return;
        }
        try {
            setBusyState('Saving preset...');
            await savePreset(presetName.trim());
            message.success(`Preset "${presetName}" saved`);
            setIsSaveVisible(false);
            setPresetName('');
        } catch {
            message.error('Failed to save preset');
        } finally {
            setBusyState(null);
        }
    };

    const handleOpenLoad = async () => {
        try {
            setBusyState('Loading preset list...');
            const data = await getPresets();
            setAvailablePresets(data.presets);
            setIsLoadVisible(true);
        } catch {
            message.error('Failed to fetch presets');
        } finally {
            setBusyState(null);
        }
    };

    const handleLoadPreset = async (name: string) => {
        try {
            setBusyState(`Loading ${name}...`);
            await loadPreset(name);
            message.success(`Preset "${name}" loaded`);
            setIsLoadVisible(false);
        } catch {
            message.error('Failed to load preset');
        } finally {
            setBusyState(null);
        }
    };

    const exportModal = (
        <Modal
            title="Configuration Export"
            open={isExportVisible}
            onCancel={() => setIsExportVisible(false)}
            footer={[
                <Button key="close" onClick={() => setIsExportVisible(false)}>Close</Button>,
                <Button key="copy" type="primary" onClick={() => {
                    navigator.clipboard.writeText(JSON.stringify(config.values, null, 2));
                    message.success('Copied to clipboard!');
                }}>Copy</Button>
            ]}
        >
            <Paragraph>
                <pre style={{ background: '#f5f5f5', padding: '12px', borderRadius: '4px', fontSize: '12px', maxHeight: '400px', overflow: 'auto' }}>
                    {JSON.stringify(config.values, null, 2)}
                </pre>
            </Paragraph>
        </Modal>
    );

    const bundleSelectorCard = (
        <Card size="small" style={{ borderRadius: '12px', width: isSidebar ? 360 : 520 }}>
            <Space direction="vertical" style={{ width: '100%' }} size="small">
                <div>
                    <Title level={5} style={{ margin: 0 }}>Category Bundles</Title>
                    <Text type="secondary">
                        Combine category-focused preset files. Unselect all gives raw video only. Select all enables the full feature stack.
                    </Text>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: isSidebar ? 'repeat(2, minmax(0, 1fr))' : 'repeat(3, minmax(0, 1fr))', gap: '8px' }}>
                    {CATEGORY_BUNDLES.map((bundle) => {
                        const selected = selectedBundles.includes(bundle.id);
                        return (
                            <Popover
                                key={bundle.id}
                                trigger="hover"
                                placement="top"
                                content={
                                    <div style={{ maxWidth: 260 }}>
                                        <Text strong>{bundle.label}</Text>
                                        <div>
                                            <Text type="secondary">{bundle.description}</Text>
                                        </div>
                                    </div>
                                }
                            >
                                <Button
                                    type={selected ? 'primary' : 'default'}
                                    disabled={Boolean(busyState)}
                                    onClick={() => {
                                        setSelectedBundles((current) => (
                                            current.includes(bundle.id)
                                                ? current.filter((id) => id !== bundle.id)
                                                : [...current, bundle.id]
                                        ));
                                    }}
                                    style={{ width: '100%', justifyContent: 'center' }}
                                >
                                    {bundle.label}
                                </Button>
                            </Popover>
                        );
                    })}
                </div>
                <Space wrap>
                    <Button disabled={Boolean(busyState)} onClick={() => setSelectedBundles(ALL_CATEGORY_BUNDLE_IDS)}>Select All</Button>
                    <Button disabled={Boolean(busyState)} onClick={() => setSelectedBundles([])}>Unselect All</Button>
                    <Button disabled={Boolean(busyState)} type="primary" onClick={() => void handleApplyCategoryBundles()}>
                        Apply Selected
                    </Button>
                </Space>
            </Space>
        </Card>
    );

    const bundleSelector = (
        <div style={{ marginBottom: 12 }}>
            <Text
                strong
                style={{
                    display: 'block',
                    marginBottom: 6,
                    fontSize: '11px',
                    color: '#999',
                    textTransform: 'uppercase',
                    letterSpacing: '1px',
                    textAlign: 'left',
                }}
            >
                Category Bundles
            </Text>
            <Popover
                trigger="click"
                placement="bottomLeft"
                open={bundlePopoverOpen}
                onOpenChange={(open) => {
                    if (!busyState) {
                        setBundlePopoverOpen(open);
                    }
                }}
                content={bundleSelectorCard}
            >
                <Button disabled={Boolean(busyState)} style={{ width: '100%' }}>
                    Open Category Bundles ({selectedBundles.length}/{CATEGORY_BUNDLES.length})
                </Button>
            </Popover>
        </div>
    );

    const saveModal = (
        <Modal
            title="Save Preset"
            open={isSaveVisible}
            onCancel={() => setIsSaveVisible(false)}
            onOk={handleSavePreset}
        >
            <div style={{ padding: '16px 0' }}>
                <Input 
                    placeholder="Preset name (e.g. Night Mode)" 
                    value={presetName}
                    onChange={e => setPresetName(e.target.value)}
                    onPressEnter={handleSavePreset}
                />
            </div>
        </Modal>
    );

    const loadModal = (
        <Modal
            title="Load Preset"
            open={isLoadVisible}
            onCancel={() => setIsLoadVisible(false)}
            footer={null}
        >
            <Space direction="vertical" style={{ width: '100%' }}>
                {availablePresets.length === 0 ? <Text type="secondary">No presets found</Text> : 
                    availablePresets.map(name => (
                        <Button key={name} block onClick={() => handleLoadPreset(name)}>{name}</Button>
                    ))
                }
            </Space>
        </Modal>
    );

    const systemChangeCoaster = pendingRestart ? (
        <div
            style={{
                position: 'sticky',
                top: 12,
                zIndex: 30,
                margin: '0 12px 12px 12px',
            }}
        >
            <Alert
                message="System Change"
                description="Restart required to apply new settings."
                type="warning"
                showIcon
                action={(
                    <Button
                        size="small"
                        type="primary"
                        disabled={Boolean(busyState)}
                        onClick={handleApplyRestart}
                    >
                        Apply & Restart
                    </Button>
                )}
                style={{
                    borderRadius: 12,
                    boxShadow: '0 8px 24px rgba(0,0,0,0.12)',
                }}
            />
        </div>
    ) : null;

    if (isSidebar) {
        const showGlobalControls = !forceCategory;
        const responsiveTwoColGrid = 'repeat(auto-fit, minmax(120px, 1fr))';
        const responsiveCategoryGrid = 'repeat(auto-fit, minmax(88px, 1fr))';
        return (
            <div style={{ display: 'flex', flexDirection: 'column', position: 'relative' }}>
                {busyState && (
                    <div style={{ position: 'absolute', inset: 0, zIndex: 100, background: 'rgba(255,255,255,0.72)', display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'auto' }}>
                        <Spin size="large" tip={busyState} />
                    </div>
                )}
                {systemChangeCoaster}
                {/* Locked Header */}
                <div style={{ 
                    padding: '16px 14px 10px 14px', 
                    background: '#fff', 
                    borderBottom: '1px solid #f0f0f0',
                    zIndex: 10,
                    boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                }}>
                    <Title level={4} style={{ margin: '0 0 14px 0', textAlign: 'center', fontSize: 24 }}>
                        {forceCategory ? `${CATEGORY_LABELS[forceCategory] ?? forceCategory} Tuning` : 'Configuration'}
                    </Title>

                    {showGlobalControls && (
                        <div style={{ marginBottom: 12 }}>
                            <Text
                                strong
                                style={{
                                    display: 'block',
                                    marginBottom: 6,
                                    fontSize: '11px',
                                    color: '#999',
                                    textTransform: 'uppercase',
                                    letterSpacing: '1px',
                                    textAlign: 'left',
                                }}
                            >
                                Performance Presets
                            </Text>
                            <div style={{ display: 'grid', gridTemplateColumns: responsiveTwoColGrid, gap: '8px', alignItems: 'stretch' }}>
                                {Object.entries(PERFORMANCE_PRESETS).map(([key, preset]) => (
                                    <Button
                                        key={key}
                                        size="middle"
                                        style={{ width: '100%', justifyContent: 'center', minHeight: 36, height: 'auto', fontSize: 13, whiteSpace: 'normal' }}
                                        disabled={Boolean(busyState)}
                                        onClick={() => void handleApplyPerformancePreset(key as keyof typeof PERFORMANCE_PRESETS)}
                                    >
                                        {preset.label}
                                    </Button>
                                ))}
                            </div>
                        </div>
                    )}

                    {showGlobalControls && (
                        <div style={{ marginBottom: 12 }}>
                            <Text
                                strong
                                style={{
                                    display: 'block',
                                    marginBottom: 6,
                                    fontSize: '11px',
                                    color: '#999',
                                    textTransform: 'uppercase',
                                    letterSpacing: '1px',
                                    textAlign: 'left',
                                }}
                            >
                                PAR Validation Presets
                            </Text>
                            <div style={{ display: 'grid', gridTemplateColumns: responsiveTwoColGrid, gap: '8px', alignItems: 'stretch' }}>
                                {Object.entries(ANALYTICS_PRESETS).map(([key, preset]) => (
                                    <Button
                                        key={key}
                                        size="middle"
                                        style={{ width: '100%', justifyContent: 'center', minHeight: 36, height: 'auto', fontSize: 13, whiteSpace: 'normal' }}
                                        disabled={Boolean(busyState)}
                                        onClick={() => void handleApplyAnalyticsPreset(key as keyof typeof ANALYTICS_PRESETS)}
                                    >
                                        {preset.label}
                                    </Button>
                                ))}
                            </div>
                        </div>
                    )}

                    {showGlobalControls && bundleSelector}

                    {showGlobalControls && (
                        <div style={{ marginBottom: 12 }}>
                            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12 }}>
                                <div>
                                    <Text strong style={{ display: 'block', fontSize: '12px' }}>Show Advanced Options</Text>
                                    <Text type="secondary" style={{ fontSize: '11px' }}>
                                        Reveal group scoring, ReID, and debug controls.
                                    </Text>
                                </div>
                                <Switch checked={showAdvancedOptions} onChange={setShowAdvancedOptions} disabled={Boolean(busyState)} />
                            </div>
                        </div>
                    )}

                    {showGlobalControls && (
                        <div style={{ margin: '0 0 12px 0' }}>
                            <Text
                                strong
                                style={{
                                    display: 'block',
                                    marginBottom: 6,
                                    fontSize: '11px',
                                    color: '#999',
                                    textTransform: 'uppercase',
                                    letterSpacing: '1px',
                                    textAlign: 'left',
                                }}
                            >
                                Config Controls
                            </Text>
                            <div style={{ display: 'grid', gridTemplateColumns: responsiveTwoColGrid, gap: '8px', alignItems: 'stretch' }}>
                                <Button disabled={Boolean(busyState)} size="middle" style={{ width: '100%', justifyContent: 'center', minHeight: 38, height: 'auto', fontSize: 13, whiteSpace: 'normal' }} icon={<UndoOutlined />} onClick={handleResetDefaults}>Reset</Button>
                                <Button disabled={Boolean(busyState)} size="middle" style={{ width: '100%', justifyContent: 'center', minHeight: 38, height: 'auto', fontSize: 13, whiteSpace: 'normal' }} icon={<ExportOutlined />} onClick={() => setIsExportVisible(true)}>Export</Button>
                                <Button disabled={Boolean(busyState)} size="middle" style={{ width: '100%', justifyContent: 'center', minHeight: 38, height: 'auto', fontSize: 13, whiteSpace: 'normal' }} icon={<FolderOpenOutlined />} onClick={handleOpenLoad}>Load Preset</Button>
                                <Button disabled={Boolean(busyState)} size="middle" style={{ width: '100%', justifyContent: 'center', minHeight: 38, height: 'auto', fontSize: 13, whiteSpace: 'normal' }} type="primary" icon={<SaveOutlined />} onClick={() => setIsSaveVisible(true)}>Save Preset</Button>
                            </div>
                        </div>
                    )}
                    
                    {showGlobalControls && (
                        <div style={{ 
                            display: 'grid', 
                            gridTemplateColumns: responsiveCategoryGrid, 
                            gap: '8px',
                            marginBottom: '10px',
                            alignItems: 'stretch',
                        }}>
                            {visibleCategories.map(cat => (
                                <Button 
                                    key={cat}
                                    type={activeCategory === cat ? 'primary' : 'default'}
                                    disabled={Boolean(busyState)}
                                    onClick={() => setActiveCategory(cat)}
                                    style={{ 
                                        width: '100%', 
                                        minHeight: '64px', 
                                        height: 'auto',
                                        padding: '6px',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        alignItems: 'center',
                                        justifyContent: 'center',
                                        borderRadius: '10px',
                                        fontSize: '11px'
                                    }}
                                >
                                    <div style={{ fontSize: '20px', marginBottom: '4px' }}>
                                        {categoryIcons[cat] || <ControlOutlined />}
                                    </div>
                                    <div style={{ 
                                        overflow: 'hidden', 
                                        textOverflow: 'ellipsis', 
                                        whiteSpace: 'nowrap',
                                        width: '100%',
                                        textAlign: 'center',
                                        lineHeight: '1.2'
                                    }}>
                                        {CATEGORY_LABELS[cat] ?? cat}
                                    </div>
                                </Button>
                            ))}
                        </div>
                    )}

                </div>

                {/* Scrollable Body */}
                <div style={{ padding: '18px' }}>
                    <ConfigPanel 
                        config={config} 
                        onUpdate={handleUpdate} 
                        activeCategory={activeCategory} 
                        activeCategories={forceCategory ? undefined : activeBundleCategories}
                        disabled={Boolean(busyState)}
                        showAdvanced={showAdvancedOptions}
                    />
                </div>

                {exportModal}
                {saveModal}
                {loadModal}
            </div>
        );
    }

    // Default view stays clean
    return (
        <div style={{ position: 'relative' }}>
            {busyState && (
                <div style={{ position: 'absolute', inset: 0, zIndex: 100, background: 'rgba(255,255,255,0.72)', display: 'flex', alignItems: 'center', justifyContent: 'center', pointerEvents: 'auto' }}>
                    <Spin size="large" tip={busyState} />
                </div>
            )}
        {systemChangeCoaster}
        <Space direction="vertical" style={{ width: '100%' }} size="large">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', marginBottom: '8px' }}>
                <div>
                    <Title level={3} style={{ margin: 0 }}>Configuration & Tuning</Title>
                    <Text type="secondary">Fine-tune detection parameters.</Text>
                </div>
                <Space>
                    <Button disabled={Boolean(busyState)} icon={<UndoOutlined />} onClick={handleResetDefaults}>Reset Defaults</Button>
                    <Button disabled={Boolean(busyState)} icon={<ExportOutlined />} onClick={() => setIsExportVisible(true)}>Export</Button>
                    <Button disabled={Boolean(busyState)} icon={<FolderOpenOutlined />} onClick={handleOpenLoad}>Load Preset</Button>
                    <Button disabled={Boolean(busyState)} type="primary" icon={<SaveOutlined />} onClick={() => setIsSaveVisible(true)}>Save Preset</Button>
                </Space>
            </div>

            <Card size="small" style={{ borderRadius: '12px' }}>
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                    <Title level={5} style={{ margin: 0 }}>Performance Presets</Title>
                    <Text type="secondary">Apply a ready-made latency profile, then restart only if prompted.</Text>
                    <Space wrap>
                        {Object.entries(PERFORMANCE_PRESETS).map(([key, preset]) => (
                            <Button disabled={Boolean(busyState)} key={key} onClick={() => void handleApplyPerformancePreset(key as keyof typeof PERFORMANCE_PRESETS)}>
                                {preset.label}
                            </Button>
                        ))}
                    </Space>
                </Space>
            </Card>

            <Card size="small" style={{ borderRadius: '12px' }}>
                <Space direction="vertical" style={{ width: '100%' }} size="small">
                    <Title level={5} style={{ margin: 0 }}>PAR Validation Presets</Title>
                    <Text type="secondary">Switch quickly between body PAR, height-age, face override, and full analytics validation modes.</Text>
                    <Space wrap>
                        {Object.entries(ANALYTICS_PRESETS).map(([key, preset]) => (
                            <Button disabled={Boolean(busyState)} key={key} onClick={() => void handleApplyAnalyticsPreset(key as keyof typeof ANALYTICS_PRESETS)}>
                                {preset.label}
                            </Button>
                        ))}
                    </Space>
                </Space>
            </Card>

            {bundleSelector}

            <Card size="small" style={{ borderRadius: '12px' }}>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 16 }}>
                    <div>
                        <Title level={5} style={{ margin: 0 }}>Show Advanced Options</Title>
                        <Text type="secondary">Reveal group scoring, ReID, and debug controls.</Text>
                    </div>
                    <Switch checked={showAdvancedOptions} onChange={setShowAdvancedOptions} disabled={Boolean(busyState)} />
                </div>
            </Card>
            
            <Row gutter={[24, 24]}>
                <Col span={6}>
                    <Card size="small" style={{ borderRadius: '12px', position: 'sticky', top: '24px' }}>
                        <Space direction="vertical" style={{ width: '100%' }}>
                            {visibleCategories.map(cat => (
                                <Button 
                                    key={cat}
                                    block
                                    disabled={Boolean(busyState)}
                                    type={activeCategory === cat ? 'primary' : 'text'}
                                    icon={categoryIcons[cat] || <ControlOutlined />}
                                    onClick={() => setActiveCategory(cat)}
                                    style={{ textAlign: 'left', borderRadius: '8px' }}
                                >
                                    {CATEGORY_LABELS[cat] ?? cat}
                                </Button>
                            ))}
                        </Space>
                    </Card>
                </Col>
                <Col span={18}>
                    <ConfigPanel
                        config={config}
                        onUpdate={handleUpdate}
                        activeCategory={activeCategory}
                        activeCategories={forceCategory ? undefined : activeBundleCategories}
                        disabled={Boolean(busyState)}
                        showAdvanced={showAdvancedOptions}
                    />
                </Col>
            </Row>
            
            {exportModal}
            {saveModal}
            {loadModal}
        </Space>
        </div>
    );
};

export default Tuning;
