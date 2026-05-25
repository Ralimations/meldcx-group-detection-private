import type { ConfigValue } from '../services/api';

export interface PerformanceProfile {
    label: string;
    description: string;
    updates: Record<string, ConfigValue>;
}

export interface AnalyticsProfile {
    label: string;
    description: string;
    updates: Record<string, ConfigValue>;
}

export const PERFORMANCE_PRESETS: Record<'performance' | 'balanced' | 'quality', PerformanceProfile> = {
    performance: {
        label: 'Performance Mode',
        description: 'Prioritizes FPS by disabling heavier age overlays, pose drawing, and run artifacts.',
        updates: {
            detect_face: false,
            enable_face_analysis: false,
            face_age_override: false,
            use_pose_visibility_gate: false,
            save_session_artifacts: false,
            show_pose_keypoints: false,
            show_age_height_stats: false,
            show_age_height_unlocked: false,
            show_height_measurement_line: false,
        },
    },
    balanced: {
        label: 'Balanced',
        description: 'Keeps detection, tracking, grouping, and body demographics while avoiding the heaviest live processing.',
        updates: {
            imgsz: 416,
            skip_frames: 2,
            preview_width: 960,
            preview_height: 540,
            stable: true,
            group_detect: true,
            detect_gender: true,
            detect_age: true,
            detect_face: false,
            enable_face_analysis: false,
            face_age_override: false,
            use_pose_visibility_gate: true,
            show_pose_keypoints: false,
            show_age_height_stats: false,
            show_age_height_unlocked: false,
            show_height_measurement_line: false,
            save_session_artifacts: false,
            show_ui_overlay: true,
        },
    },
    quality: {
        label: 'Quality',
        description: 'Turns on the full live analytics path, visual overlays, and session artifacts.',
        updates: {
            imgsz: 640,
            skip_frames: 1,
            preview_width: 1280,
            preview_height: 720,
            stable: true,
            group_detect: true,
            detect_gender: true,
            detect_age: true,
            detect_face: true,
            enable_face_analysis: true,
            face_age_override: true,
            use_pose_visibility_gate: true,
            show_pose_keypoints: true,
            show_age_height_stats: false,
            show_age_height_unlocked: false,
            show_height_measurement_line: true,
            save_session_artifacts: true,
            show_ui_overlay: true,
        },
    },
};

export const ANALYTICS_PRESETS: Record<'bodyParOnly' | 'heightAgeOnly' | 'faceOverrideValidation' | 'fullAnalytics', AnalyticsProfile> = {
    bodyParOnly: {
        label: 'Body PAR Only',
        description: 'Validate body attributes and gender labels without height-age or face override paths.',
        updates: {
            detect_gender: true,
            show_gender: true,
            show_gender_confidence: true,
            show_attributes: true,
            detect_age: false,
            show_age: false,
            detect_face: false,
            enable_face_analysis: false,
            face_age_override: false,
            gender_use_face_override: false,
            show_pose_keypoints: false,
            show_height_measurement_line: false,
            show_age_height_stats: false,
            show_age_height_unlocked: false,
        },
    },
    heightAgeOnly: {
        label: 'Height Age Only',
        description: 'Validate ROI locking, pose gating, and height-derived age labels without face overrides.',
        updates: {
            detect_gender: false,
            show_gender: false,
            show_attributes: false,
            detect_age: true,
            show_age: true,
            show_age_confidence: true,
            detect_face: false,
            enable_face_analysis: false,
            face_age_override: false,
            use_pose_visibility_gate: true,
            show_pose_keypoints: true,
            show_height_measurement_line: true,
            show_age_height_stats: true,
            show_age_height_unlocked: false,
        },
    },
    faceOverrideValidation: {
        label: 'Face Override Check',
        description: 'Validate face age/gender override behavior against the body PAR and height baseline paths.',
        updates: {
            detect_gender: true,
            show_gender: true,
            show_gender_confidence: true,
            show_attributes: true,
            detect_age: true,
            show_age: true,
            show_age_confidence: true,
            detect_face: true,
            enable_face_analysis: true,
            face_age_override: true,
            use_pose_visibility_gate: true,
            gender_use_face_override: true,
            show_pose_keypoints: false,
            show_height_measurement_line: false,
            show_age_height_stats: false,
            show_age_height_unlocked: false,
        },
    },
    fullAnalytics: {
        label: 'Full Analytics',
        description: 'Enable the full PAR, pose, height-age, face override, and artifact-saving path.',
        updates: {
            detect_gender: true,
            show_gender: true,
            show_gender_confidence: true,
            show_attributes: true,
            detect_age: true,
            show_age: true,
            show_age_confidence: true,
            detect_face: true,
            enable_face_analysis: true,
            face_age_override: true,
            use_pose_visibility_gate: true,
            gender_use_face_override: true,
            show_pose_keypoints: true,
            show_height_measurement_line: true,
            show_age_height_stats: true,
            show_age_height_unlocked: false,
            save_session_artifacts: true,
        },
    },
};
