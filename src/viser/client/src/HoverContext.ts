import React from "react";

// Extended hover context to include instanceId for instanced meshes
export interface HoverState {
  isHovered: boolean;
  instanceId: number | null;
}

export const HoverableContext =
  React.createContext<React.MutableRefObject<HoverState> | null>(null);
