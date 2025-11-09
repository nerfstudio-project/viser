import React from "react";
// @ts-ignore - troika-three-text doesn't have type definitions
import { Text as TroikaText } from "troika-three-text";

/**
 * Context for accessing the global batched text manager.
 */
export interface BatchedTextManagerContextValue {
  registerText: (text: TroikaText, nodeName: string, depthTest: boolean) => void;
  unregisterText: (text: TroikaText) => void;
  syncBatchedText: () => void;
  syncText: (text: TroikaText) => void;
}

export const BatchedTextManagerContext =
  React.createContext<BatchedTextManagerContextValue | null>(null);
