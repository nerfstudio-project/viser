import React from "react";

// This context needs to be created outside of FloatingPanel.tsx, otherwise
// fast refresh + hot module reloads on the panel component break the whole
// application.
//
// The behavior seems to match this "fixed" GitHub issue:
// https://github.com/vitejs/vite/issues/3301
//
// (confirmed on vitejs/plugin-react 4.0.1)

const FloatingPanelRefContext =
  React.createContext<React.RefObject<HTMLDivElement> | null>(null);

export default FloatingPanelRefContext;
