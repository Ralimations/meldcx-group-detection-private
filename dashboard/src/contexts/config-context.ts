import { createContext } from 'react';
import { useConfig } from '../hooks/useConfig';

export type ConfigContextType = ReturnType<typeof useConfig>;

export const ConfigContext = createContext<ConfigContextType | undefined>(undefined);
