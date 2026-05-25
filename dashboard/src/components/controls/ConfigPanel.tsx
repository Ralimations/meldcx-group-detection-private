import React from 'react';
import { Slider, Switch, Typography, Space, Select, Tooltip, Row, Col, Card, InputNumber, Input } from 'antd';
import { InfoCircleOutlined } from '@ant-design/icons';
import type { ConfigResponse, ConfigValue } from '../../services/api';

const { Text } = Typography;

interface ConfigPanelProps {
    config: ConfigResponse | null;
    onUpdate: (key: string, value: ConfigValue) => void;
    activeCategory?: string;
    activeCategories?: string[];
    disabled?: boolean;
    showAdvanced?: boolean;
}

const PARAM_IMPACT: Record<string, string> = {
    "stable": "Ensures IDs are consistent across frames. Turning this off will reset IDs every frame.",
    "group_detect": "Main toggle for the social distancing/grouping engine. Critical for analytics.",
    "person_detection_conf": "Lower values catch more people but increase false positives (ghosts).",
    "person_overlap_threshold": "Controls how overlapping person boxes are merged. High values lead to duplicate detections surviving longer.",
    "head_detection_conf": "Head-only confidence threshold. Higher values reduce false head detections but may miss small heads.",
    "head_overlap_threshold": "Controls how overlapping head detections are merged.",
    "show_merged_warnings": "Shows warnings when the runtime suspects multiple people are merged into one detection region.",
    "skip_frames": "Increases performance by running AI less often. Higher = Faster but choppier tracking.",
    "track_overlap_threshold": "Threshold for matching a box to a previous track. Affects ID stability.",
    "track_min_conf": "Drops weak tracks earlier. Higher values reduce noisy IDs but can lose distant people.",
    "track_max_draw_misses": "How long a lost track is still drawn before disappearing from the overlay.",
    "track_reid_window": "How long a disappeared person can still be matched back to an old ID.",
    "track_reid_similarity_thresh": "Higher values make ID recovery stricter and less likely to mismatch.",
    "track_reid_update_interval": "How often ReID features are refreshed for active tracks.",
    "track_reid_backend": "Chooses whether identity recovery is off, histogram-based, OpenVINO-based, or hybrid.",
    "track_reid_model": "Path to the OpenVINO person ReID model used when the backend needs embeddings.",
    "track_reid_embedding_similarity_thresh": "Higher values make embedding-based identity recovery stricter.",
    "track_reid_require_not_touching_frame_edge": "Rejects low-quality ReID identity evidence when the crop touches the image edge.",
    "track_reid_min_quality_frames": "Quality frames required before OpenVINO ReID starts trusting an identity candidate.",
    "track_reid_identity_confirm_frames": "Frames required before a ReID identity candidate is promoted to a recovered ID.",
    "track_reid_use_spatial_gate": "Restricts identity recovery to a local spatial neighborhood around a lost track.",
    "track_reid_spatial_gate_scale": "Controls how large the ReID spatial recovery area is.",
    "smooth": "Reduces jitter in the boxes. High values make boxes feel 'floaty' but stable.",
    "group_history_len": "How much recent motion history the group scorer keeps per pair.",
    "group_idle_speed_thresh": "People moving slower than this count as idle for grouping logic.",
    "group_lock_threshold": "Higher values require people to stand together longer before forming a group.",
    "group_max_score": "Limits how 'solid' a group bond can become. Affects separation speed.",
    "group_pair_angle_history_len": "Controls how much pair-angle history contributes to stable grouping.",
    "group_merged_warn_interval": "How often the runtime warns about merged or ambiguous groups.",
    "group_origin_gap_threshold": "Breaks weak group continuity when origins drift too far apart.",
    "use_doorway_monitor": "Tracks minor-adult groups through the doorway ROI and starts an in-room timer after entry.",
    "show_doorway_status": "Shows the doorway room-timer statuses directly in the overlay.",
    "room_presence_alert_seconds": "How long a group may remain in-room before the doorway alert escalates.",
    "doorway_commit_frames": "Frames required before the doorway monitor commits an entry event.",
    "doorway_missing_frames": "Missing frames tolerated by the doorway monitor before exit logic advances.",
    "show_carry_status": "Shows carry-alert overlays for minor-on-adult interaction detections.",
    "show_carry_overlap_debug": "Exposes carry overlap diagnostics that are mainly useful when validating alert behavior.",
    "carry_overlap_score_threshold": "Required carry-overlap score before the alert path starts to lock.",
    "carry_overlap_lock_frames": "Frames required before a carry alert is considered locked.",
    "carry_overlap_release_frames": "Frames required before a carry alert clears.",
    "out": "Legacy/manual output path used for non-dashboard runs and file outputs.",
    "auto_output_name": "Automatically names output files from the active source instead of using the manual path.",
    "testing_output_root": "Root folder used for testing outputs and saved results.",
    "nvr_record_enable": "Enables rolling archive video recording.",
    "nvr_segment_minutes": "Length of each rolling archive segment.",
    "nvr_retention_hours": "How long archive video segments are kept before cleanup.",
    "nvr_output_root": "Root folder where rolling archive segments and related outputs are stored.",
    "group_log_enable": "Writes group-event logs alongside archive output.",
    "label_segment_colors": "Prefix-based color rules for labels. Use comma-separated entries like Male=#4DA6FF, Adult=#7DDA58.",
    "detect_face": "Enables the face detector used for overlays and face-based demographics overrides.",
    "gender_use_face_override": "Lets a confident face result replace the body PAR gender result when available.",
    "gender_face_override_min_conf": "Higher values make face-based gender overrides rarer but safer.",
    "enable_face_analysis": "Runs the face age/gender models. This adds cost but enables face-based overrides.",
    "face_age_override": "Lets a face age result replace the height-based age label when available.",
    "face_detection_conf": "Minimum face detector score before a face crop is considered.",
    "face_overlap_threshold": "Controls how overlapping face detections are merged.",
    "use_pose_visibility_gate": "Blocks height-based age updates when pose visibility is too weak.",
    "show_pose_indices": "Draws pose keypoint index numbers for debugging the pose model output.",
    "pose_force_run": "Forces pose inference on tracks that would normally be skipped by the height-age pipeline.",
    "pose_full_frame": "Runs pose on the full frame and matches observations back to tracks.",
    "pose_conf": "Minimum score for pose detections.",
    "pose_overlap_threshold": "Controls how overlapping pose boxes are merged.",
    "pose_crop_padding": "Extra border around the person crop sent into the pose model.",
    "pose_keypoint_conf": "Minimum score for keypoints used in visibility and height checks.",
    "pose_min_visible_keypoints": "Required visible keypoints before a pose sample is accepted.",
    "pose_min_torso_keypoints": "Required torso keypoints before a pose sample is accepted.",
    "pose_min_lower_body_keypoints": "Required lower-body keypoints before a pose sample is accepted.",
    "pose_min_lowest_keypoint_ratio": "Requires the lowest visible keypoint to be near the bottom of the person box.",
    "height_average_stability_px": "How much the measured height may drift before the tracker treats it as unstable.",
    "height_display_freeze_samples": "How many valid ROI samples are needed before freezing a trusted baseline.",
    "height_lock_min_samples": "Minimum ROI samples before the height-based age label may lock.",
    "show_perf": "Shows extra performance counters directly in the runtime overlay.",
    "verbose_logging": "Prints extra runtime diagnostics that are useful when debugging bad behavior.",
    "save_output_video": "Writes the runtime preview stream to disk during processing.",
    "output_video_fps": "Overrides the saved output video FPS. Leave at 0 to use the source FPS.",
    "output_processed_only": "Skips output on frames where inference was skipped, improving throughput.",
    "preview_window_mode": "Controls the native OpenCV preview window mode when preview is enabled.",
};

const MULTILINE_KEYS = new Set([
    'label_segment_colors',
]);

function isEnabled(value: ConfigValue | undefined): boolean {
    return value === true;
}

function shouldShowParam(key: string, values: ConfigResponse['values']): boolean {
    const ageEnabled = isEnabled(values.detect_age);
    const genderEnabled = isEnabled(values.detect_gender);
    const attributesEnabled = isEnabled(values.show_attributes);
    const faceDetectionEnabled = isEnabled(values.detect_face);
    const faceAnalysisEnabled = faceDetectionEnabled && isEnabled(values.enable_face_analysis);
    const poseGateEnabled = ageEnabled && isEnabled(values.use_pose_visibility_gate);
    const doorwayMonitorEnabled = isEnabled(values.use_doorway_monitor);
    const carryStatusEnabled = isEnabled(values.show_carry_status);
    const reidBackend = String(values.track_reid_backend ?? 'openvino');
    const usesEmbeddingReid = reidBackend === 'openvino' || reidBackend === 'hybrid';

    if ([
        'head_detection_conf',
        'head_overlap_threshold',
    ].includes(key)) {
        return isEnabled(values.detect_head);
    }

    if ([
        'gender_use_face_override',
        'gender_face_override_min_conf',
        'show_gender_confidence',
    ].includes(key)) {
        return genderEnabled;
    }

    if (['show_body_par_raw_labels'].includes(key)) {
        return attributesEnabled;
    }

    if ([
        'show_age_confidence',
        'show_age_height_stats',
        'show_age_height_unlocked',
        'show_height_measurement_line',
        'use_pose_visibility_gate',
        'show_pose_keypoints',
        'show_pose_indices',
        'pose_interval',
        'pose_max_tracks_per_frame',
        'pose_force_run',
        'pose_full_frame',
        'pose_conf',
        'pose_overlap_threshold',
        'pose_crop_padding',
    ].includes(key)) {
        return ageEnabled;
    }

    if ([
        'height_average_stability_px',
        'height_display_freeze_samples',
        'height_peak_stability_frames',
        'height_lock_min_samples',
        'height_lock_stability_frames',
        'height_lock_requires_trusted_baseline',
    ].includes(key)) {
        return ageEnabled;
    }

    if ([
        'pose_keypoint_conf',
        'pose_min_visible_keypoints',
        'pose_min_torso_keypoints',
        'pose_min_lower_body_keypoints',
        'pose_min_lowest_keypoint_ratio',
    ].includes(key)) {
        return poseGateEnabled;
    }

    if ([
        'gender_face_override_min_conf',
    ].includes(key)) {
        return genderEnabled && faceDetectionEnabled && isEnabled(values.gender_use_face_override);
    }

    if ([
        'show_face',
        'show_face_confidence',
        'enable_face_analysis',
        'face_detection_conf',
        'face_overlap_threshold',
    ].includes(key)) {
        return faceDetectionEnabled;
    }

    if ([
        'face_age_override',
        'face_age_senior_threshold',
        'show_face_analysis_unlocked',
    ].includes(key)) {
        return ageEnabled && faceAnalysisEnabled;
    }

    if ([
        'track_reid_model',
        'track_reid_embedding_similarity_thresh',
        'track_reid_require_not_touching_frame_edge',
        'track_reid_min_quality_frames',
        'track_reid_identity_confirm_frames',
    ].includes(key)) {
        return usesEmbeddingReid;
    }

    if (['track_reid_spatial_gate_scale'].includes(key)) {
        return isEnabled(values.track_reid_use_spatial_gate);
    }

    if ([
        'show_doorway_status',
        'room_presence_alert_seconds',
        'doorway_commit_frames',
        'doorway_missing_frames',
    ].includes(key)) {
        return doorwayMonitorEnabled;
    }

    if ([
        'show_carry_overlap_debug',
        'carry_overlap_score_threshold',
        'carry_overlap_lock_frames',
        'carry_overlap_release_frames',
    ].includes(key)) {
        return carryStatusEnabled;
    }

    if (['output_video_fps'].includes(key)) {
        return isEnabled(values.save_output_video);
    }

    if ([
        'nvr_segment_minutes',
        'nvr_retention_hours',
        'nvr_output_root',
    ].includes(key)) {
        return isEnabled(values.nvr_record_enable);
    }

    return true;
}

const ConfigPanel: React.FC<ConfigPanelProps> = ({ config, onUpdate, activeCategory, activeCategories, disabled = false, showAdvanced = false }) => {
    if (!config) return <Text type="secondary">Loading parameters...</Text>;

    const categories: Record<string, string[]> = {};
    const HIDDEN_KEYS = ['input_mode', 'source', 'camera_index', 'image', 'batch_sources'];
    const categoryFilter = activeCategories && activeCategories.length > 0 ? new Set(activeCategories) : null;
    const showingAdvancedCategory = activeCategory === 'Advanced';

    Object.keys(config.meta).forEach(key => {
        if (HIDDEN_KEYS.includes(key)) return;
        const isAdvanced = Boolean(config.meta[key].advanced);
        if (isAdvanced && !showAdvanced) return;
        if (!shouldShowParam(key, config.values)) return;
        const cat = config.meta[key].cat;
        if (showingAdvancedCategory) {
            if (!isAdvanced) return;
        } else {
            if (isAdvanced) return;
            if (activeCategory && cat !== activeCategory) return;
        }
        if (!showingAdvancedCategory && categoryFilter && !categoryFilter.has(cat)) return;
        const targetCategory = showingAdvancedCategory ? 'Advanced' : cat;
        if (!categories[targetCategory]) categories[targetCategory] = [];
        categories[targetCategory].push(key);
    });

    if (Object.keys(categories).length === 0) {
        return <Text type="secondary">No config controls match the current category selection.</Text>;
    }

    return (
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
            {Object.entries(categories).map(([cat, keys]) => (
                <Card 
                    key={cat}
                    size="small" 
                    bordered={true}
                    style={{ marginBottom: '12px', borderRadius: '8px' }}
                    bodyStyle={{ padding: '6px 12px' }}
                >
                    {keys.map(key => {
                        const meta = config.meta[key];
                        const val = config.values[key];
                        const impact = PARAM_IMPACT[key] || meta.desc;

                        return (
                            <div key={key} style={{ padding: '10px 0', borderBottom: '1px solid #f0f0f0' }}>
                                <Row align="middle" gutter={12}>
                                    {/* Column 1: Label & Info */}
                                    <Col span={8}>
                                        <Space size={4}>
                                            <Text strong style={{ fontSize: '11px', color: '#595959', lineHeight: 1.25 }}>
                                                {key.replace(/_/g, ' ')}
                                            </Text>
                                            <Tooltip title={impact} placement="right">
                                                <InfoCircleOutlined style={{ fontSize: '10px', color: '#bfbfbf', cursor: 'help' }} />
                                            </Tooltip>
                                        </Space>
                                    </Col>
 
                                    {/* Column 2: Control */}
                                    <Col span={13}>
                                        {meta.type === 'bool' ? (
                                            <div style={{ textAlign: 'right', paddingRight: '20%' }}>
                                                <Switch 
                                                    size="small"
                                                    checked={val as boolean} 
                                                    disabled={disabled}
                                                    onChange={(v) => onUpdate(key, v)} 
                                                />
                                            </div>
                                        ) : meta.type === 'select' ? (
                                            <Select 
                                                size="middle"
                                                style={{ width: '100%' }}
                                                value={val}
                                                disabled={disabled}
                                                onChange={(v) => onUpdate(key, v)}
                                                options={
                                                    typeof meta.options === 'string' 
                                                        ? meta.options.split(',').map((o: string) => ({ label: o.trim(), value: o.trim() }))
                                                        : Array.isArray(meta.options) 
                                                            ? (meta.options as unknown[]).map((option) => ({ label: String(option), value: String(option) }))
                                                            : []
                                                }
                                            />
                                        ) : meta.type === 'str' ? (
                                            MULTILINE_KEYS.has(key) ? (
                                                <Input.TextArea
                                                    value={
                                                        typeof val === 'string'
                                                            ? val
                                                            : Array.isArray(val) || (val && typeof val === 'object')
                                                                ? JSON.stringify(val)
                                                                : String(val ?? '')
                                                    }
                                                    disabled={disabled}
                                                    onChange={(e) => onUpdate(key, e.target.value)}
                                                    autoSize={{ minRows: 2, maxRows: 5 }}
                                                    style={{ width: '100%' }}
                                                />
                                            ) : (
                                            <Input 
                                                size="middle"
                                                value={
                                                    typeof val === 'string'
                                                        ? val
                                                        : Array.isArray(val) || (val && typeof val === 'object')
                                                            ? JSON.stringify(val)
                                                            : String(val ?? '')
                                                }
                                                disabled={disabled}
                                                onChange={(e) => onUpdate(key, e.target.value)}
                                                style={{ width: '100%' }}
                                            />
                                            )
                                        ) : meta.min !== undefined && meta.max !== undefined ? (
                                            <Row gutter={10} align="middle">
                                                <Col span={15}>
                                                    <Slider 
                                                        min={meta.min} 
                                                        max={meta.max} 
                                                        step={meta.step} 
                                                        value={val as number}
                                                        disabled={disabled}
                                                        onChange={(v) => onUpdate(key, v)}
                                                        tooltip={{ open: false }}
                                                        style={{ margin: '8px 0' }}
                                                    />
                                                </Col>
                                                <Col span={9}>
                                                    <InputNumber
                                                        size="middle"
                                                        min={meta.min}
                                                        max={meta.max}
                                                        step={meta.step}
                                                        value={val as number}
                                                        disabled={disabled}
                                                        onChange={(v) => v !== null && onUpdate(key, v)}
                                                        style={{ width: '100%' }}
                                                    />
                                                </Col>
                                            </Row>
                                        ) : (
                                            <InputNumber
                                                size="middle"
                                                step={meta.step}
                                                value={val as number}
                                                disabled={disabled}
                                                onChange={(v) => v !== null && onUpdate(key, v)}
                                                style={{ width: '100%' }}
                                            />
                                        )}
                                    </Col>

                                </Row>
                                <div style={{ marginTop: '3px' }}>
                                    <Text type="secondary" style={{ fontSize: '10px', display: 'block', lineHeight: 1.35 }}>
                                        {meta.desc}
                                    </Text>
                                </div>
                            </div>
                        );
                    })}
                </Card>
            ))}
        </Space>
    );
};

export default ConfigPanel;
