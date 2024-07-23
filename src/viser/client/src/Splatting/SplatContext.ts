import { create } from "zustand";
import React from "react";

interface SplatState {
  groupBufferFromName: { [name: string]: Uint32Array };
  setBuffer: (name: string, buffer: Uint32Array) => void;
  removeBuffer: (name: string) => void;
}

export function useGaussianSplatStore() {
  return React.useState(() =>
    create<SplatState>((set) => ({
      groupBufferFromName: {},
      setBuffer: (name, buffer) => {
        return set((state) => ({
          groupBufferFromName: { [name]: buffer, ...state.groupBufferFromName },
        }));
      },
      removeBuffer: (name) => {
        return set((state) => {
          // eslint-disable-next-line no-unused-vars
          const { [name]: _, ...buffers } = state.groupBufferFromName;
          return { groupBufferFromName: buffers };
        });
      },
    })),
  )[0];
}
export const GaussianSplatsContext = React.createContext<ReturnType<
  typeof useGaussianSplatStore
> | null>(null);
