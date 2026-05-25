const BASE_URL = 'http://localhost:8765/api';

export type ConfigValue = string | number | boolean | null | ConfigValue[] | { [key: string]: ConfigValue };
export type ConfigValues = Record<string, ConfigValue>;

export interface ParamMeta {
    cat: string;
    min?: number;
    max?: number;
    step?: number;
    type: 'float' | 'int' | 'bool' | 'select' | 'str';
    desc: string;
    options?: string;
    needs_restart?: boolean;
    advanced?: boolean;
}

export interface ConfigResponse {
    values: ConfigValues;
    meta: Record<string, ParamMeta>;
}

export interface EngineStatus {
    running: boolean;
    paused: boolean;
    people?: number;
    groups?: number;
    total_groups?: number;
    fps?: number;
}

export interface LogEntry {
    time: string;
    msg: string;
}

export interface TestSummary {
    id: number;
    timestamp: string;
    source: string;
    max_people: number;
    max_groups: number;
    avg_fps: number;
    duration: number;
    annotated_video_path?: string | null;
    detection_json_path?: string | null;
}

export interface Sensor {
    id: number;
    name: string;
    config: ConfigValues;
}

export interface PlaybackStatus {
    is_file: boolean;
    playing: boolean;
    speed: number;
    loop: boolean;
    current_frame: number;
    total_frames: number;
    fps: number;
    position_seconds: number;
    duration_seconds: number;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE_URL}${path}`, init);
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed: ${res.status}`);
    }
    return res.json() as Promise<T>;
}

async function requestVoid(path: string, init?: RequestInit): Promise<void> {
    const res = await fetch(`${BASE_URL}${path}`, init);
    if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Request failed: ${res.status}`);
    }
}

export const api = {
    async getConfig(): Promise<ConfigResponse> {
        return request<ConfigResponse>('/config');
    },

    async setConfig(values: ConfigValues): Promise<void> {
        await requestVoid('/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(values),
        });
    },

    async getStatus(): Promise<EngineStatus> {
        return request<EngineStatus>('/engine/status');
    },

    async startEngine(): Promise<void> {
        await requestVoid('/engine/start', { method: 'POST' });
    },

    async stopEngine(): Promise<void> {
        await requestVoid('/engine/stop', { method: 'POST' });
    },

    async pauseEngine(): Promise<boolean> {
        const data = await request<{ paused: boolean }>('/engine/pause', { method: 'POST' });
        return data.paused;
    },

    async captureWorkspaceSnapshot(): Promise<void> {
        await requestVoid('/workspace/snapshot', { method: 'POST' });
    },

    async getLogs(): Promise<{ logs: LogEntry[] }> {
        return request<{ logs: LogEntry[] }>('/logs');
    },

    async getVideos(): Promise<{ videos: string[], current: string }> {
        return request<{ videos: string[], current: string }>('/videos');
    },

    async resetEngine(): Promise<void> {
        await requestVoid('/reset', { method: 'POST' });
    },

    async resetDefaults(): Promise<ConfigValues> {
        const data = await request<{ values: ConfigValues }>('/defaults', { method: 'POST' });
        return data.values;
    },

    async getPresets(): Promise<{ presets: string[] }> {
        return request<{ presets: string[] }>('/presets');
    },

    async savePreset(name: string): Promise<void> {
        await requestVoid('/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
    },

    async loadPreset(name: string): Promise<ConfigValues> {
        const data = await request<{ values: ConfigValues }>('/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
        return data.values;
    },
    async fetchSummaries(): Promise<TestSummary[]> {
        return request<TestSummary[]>('/summaries');
    },

    async getPlayback(): Promise<PlaybackStatus> {
        return request<PlaybackStatus>('/playback');
    },

    async updatePlayback(payload: Partial<Pick<PlaybackStatus, 'playing' | 'speed' | 'loop'>> & { seek_frame?: number; restart?: boolean }): Promise<PlaybackStatus> {
        return request<PlaybackStatus>('/playback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
    },

    // --- Sensors ---
    async getSensors(): Promise<{ sensors: Sensor[], active_id: number | null }> {
        return request<{ sensors: Sensor[], active_id: number | null }>('/sensors');
    },

    async createSensor(name: string, config: ConfigValues): Promise<{ ok: boolean, id: number }> {
        return request<{ ok: boolean, id: number }>('/sensors', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, config }),
        });
    },

    async deleteSensor(id: number): Promise<{ ok: boolean }> {
        return request<{ ok: boolean }>(`/sensors/${id}`, { method: 'DELETE' });
    },

    async updateSensorName(id: number, name: string): Promise<{ ok: boolean }> {
        return request<{ ok: boolean }>(`/sensors/${id}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name }),
        });
    },

    async loadSensor(id: number): Promise<{ ok: boolean, values: ConfigValues }> {
        return request<{ ok: boolean, values: ConfigValues }>('/sensors/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ id }),
        });
    },

    async deactivateSensor(): Promise<{ ok: boolean }> {
        return request<{ ok: boolean }>('/sensors/deactivate', { method: 'POST' });
    },

    async clearLogs(): Promise<void> {
        await requestVoid('/logs/clear', { method: 'POST' });
    },
};
