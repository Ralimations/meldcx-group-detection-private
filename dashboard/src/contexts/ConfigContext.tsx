import React from 'react';
import type { ReactNode } from 'react';
import { ConfigContext } from './config-context';
import { useConfig } from '../hooks/useConfig';

export const ConfigProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
    const config = useConfig();
    return (
        <ConfigContext.Provider value={config}>
            {children}
        </ConfigContext.Provider>
    );
};
