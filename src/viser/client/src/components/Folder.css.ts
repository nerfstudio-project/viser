import { style } from "@vanilla-extract/css";
import { vars } from "../AppTheme";

export const folderWrapper = style({
  borderWidth: "1px",
  position: "relative",
  marginLeft: vars.spacing.xs,
  marginRight: vars.spacing.xs,
  // If there's a GUI element above, we need more margin.
  marginTop: vars.spacing.xs,
  // If there's a GUI element below, we need more margin.
  // Note: 0.5em is the vertical margin below general GUI elements.
  marginBottom: "1.2em",
  ":last-child": {
    marginBottom: "0.5em",
  },
  paddingBottom: `calc(${vars.spacing.xs} - 0.5em)`,
});

export const folderLabel = style({
  fontSize: "0.875em",
  position: "absolute",
  padding: "0 0.375em 0 0.375em",
  top: 0,
  left: "0.375em",
  transform: "translateY(-50%)",
  userSelect: "none",
  fontWeight: 500,
});

export const folderToggleIcon = style({
  width: "0.9em",
  height: "0.9em",
  strokeWidth: 3,
  top: "0.1em",
  position: "relative",
  marginLeft: "0.25em",
  marginRight: "-0.1em",
  opacity: 0.5,
});
