import React from "react";

export const HoverableContext =
  React.createContext<React.MutableRefObject<boolean> | null>(null);
