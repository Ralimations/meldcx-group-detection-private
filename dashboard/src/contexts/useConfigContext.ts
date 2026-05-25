import { useContext } from 'react';
import { ConfigContext } from './config-context';

export function useConfigContext() {
    const context = useContext(ConfigContext);
    if (context === undefined) {
        throw new Error('useConfigContext must be used within a ConfigProvider');
    }
    return context;
}
