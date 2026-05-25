import type { ConfigValue } from '../services/api';

export interface CategoryBundle {
    id: string;
    label: string;
    description: string;
    categories: string[];
    updates: Record<string, ConfigValue>;
}

export const RAW_VIDEO_BASELINE: Record<string, ConfigValue> = {
    detect_body: false,
    detect_head: false,
    show_body: false,
    show_head: false,
    show_body_confidence: false,
    show_head_confidence: false,
    detect_face: false,
    show_face: false,
    show_face_confidence: false,
    show_ui_overlay: false,
    use_roi: false,
    show_roi_mask: false,
    show_height_roi_mask: false,
    use_doorway_monitor: false,
    show_doorway_status: false,
    show_carry_status: false,
    nvr_record_enable: false,
    group_log_enable: false,
    stable: false,
    show_unconfirmed: false,
    group_detect: false,
    detect_gender: false,
    show_gender: false,
    show_attributes: false,
    detect_age: false,
    show_age: false,
    enable_face_analysis: false,
    face_age_override: false,
    use_pose_visibility_gate: false,
    show_pose_keypoints: false,
    show_age_height_stats: false,
    show_age_height_unlocked: false,
    show_height_measurement_line: false,
    save_session_artifacts: false,
    undistort_enable: false,
};

export const CATEGORY_BUNDLES: CategoryBundle[] = [
    {
        id: 'detection',
        label: 'Detection',
        description: 'Run person detection and draw body boxes.',
        categories: ['Detection'],
        updates: {
            detect_body: true,
            detect_head: false,
            show_body: true,
            show_head: true,
        },
    },
    {
        id: 'detection-roi',
        label: 'Detection ROI',
        description: 'Enable the detection ROI gate and its overlay.',
        categories: ['Detection ROI'],
        updates: {
            use_roi: true,
            show_roi_mask: true,
        },
    },
    {
        id: 'height-roi',
        label: 'Height ROI',
        description: 'Show the separate height ROI overlay.',
        categories: ['Height ROI'],
        updates: {
            show_height_roi_mask: true,
        },
    },
    {
        id: 'doorway-roi',
        label: 'Doorway ROI',
        description: 'Configure the doorway region used by room-presence alerts.',
        categories: ['Doorway ROI'],
        updates: {
            use_doorway_monitor: true,
            show_doorway_status: true,
        },
    },
    {
        id: 'tracking',
        label: 'Tracking',
        description: 'Enable the stable person tracker.',
        categories: ['Tracking'],
        updates: {
            stable: true,
            show_unconfirmed: false,
        },
    },
    {
        id: 'groups',
        label: 'Groups',
        description: 'Enable group detection and group overlays.',
        categories: ['Groups'],
        updates: {
            group_detect: true,
        },
    },
    {
        id: 'demographics',
        label: 'Demographics',
        description: 'Enable gender, age, body attributes, and height-driven age overlays.',
        categories: ['Demographics'],
        updates: {
            detect_gender: true,
            show_gender: true,
            show_attributes: true,
            detect_age: true,
            show_age: true,
            detect_face: true,
            enable_face_analysis: true,
            face_age_override: true,
            use_pose_visibility_gate: true,
            show_pose_keypoints: true,
            show_age_height_stats: false,
            show_age_height_unlocked: false,
            show_height_measurement_line: false,
        },
    },
    {
        id: 'alerts',
        label: 'Alerts',
        description: 'Enable doorway room timers and carry-alert overlays for the final program flow.',
        categories: ['Alerts'],
        updates: {
            use_doorway_monitor: true,
            show_doorway_status: true,
            show_carry_status: true,
        },
    },
    {
        id: 'archive',
        label: 'Archive',
        description: 'Enable rolling archive recording and group-event logging outputs.',
        categories: ['Archive'],
        updates: {
            nvr_record_enable: true,
            group_log_enable: true,
        },
    },
    {
        id: 'display',
        label: 'Display',
        description: 'Enable the main HUD and confidence-free visual overlays.',
        categories: ['Display'],
        updates: {
            show_ui_overlay: true,
            show_body_confidence: false,
            show_head_confidence: false,
            show_face: true,
            show_face_confidence: false,
        },
    },
    {
        id: 'performance',
        label: 'Performance',
        description: 'Enable session artifact saving and live-file processing behavior.',
        categories: ['Performance'],
        updates: {
            mimic_live: true,
            turbo_mode: false,
            save_session_artifacts: true,
        },
    },
    {
        id: 'lens',
        label: 'Lens',
        description: 'Enable lens undistortion.',
        categories: ['Lens'],
        updates: {
            undistort_enable: true,
        },
    },
];

export const ALL_CATEGORY_BUNDLE_IDS = CATEGORY_BUNDLES.map((bundle) => bundle.id);

export function buildCategoryBundleUpdates(selectedIds: string[]): Record<string, ConfigValue> {
    const selected = new Set(selectedIds);
    return CATEGORY_BUNDLES.reduce<Record<string, ConfigValue>>(
        (acc, bundle) => {
            if (!selected.has(bundle.id)) {
                return acc;
            }
            return { ...acc, ...bundle.updates };
        },
        { ...RAW_VIDEO_BASELINE },
    );
}

export function bundleIdsToCategories(selectedIds: string[]): string[] {
    const selected = new Set(selectedIds);
    const categories = new Set<string>();
    for (const bundle of CATEGORY_BUNDLES) {
        if (!selected.has(bundle.id)) {
            continue;
        }
        for (const category of bundle.categories) {
            categories.add(category);
        }
    }
    return Array.from(categories);
}
