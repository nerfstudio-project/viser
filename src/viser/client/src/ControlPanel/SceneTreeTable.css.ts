import { globalStyle, style } from "@vanilla-extract/css";
import { vars } from "../AppTheme";

export const tableWrapper = style({
  borderRadius: vars.radius.xs,
  padding: "0.1em 0",
  overflowX: "auto",
  display: "flex",
  flexDirection: "column",
  gap: "0",
});

export const propsWrapper = style({
  position: "relative",
  borderRadius: vars.radius.xs,
  border: "1px solid",
  borderColor: vars.colors.defaultBorder,
  padding: vars.spacing.xs,
  boxSizing: "border-box",
  margin: vars.spacing.xs,
  marginTop: "0.2em",
  marginBottom: "0.2em",
  overflowX: "auto",
  display: "flex",
  flexDirection: "column",
  gap: vars.spacing.xs,
});

export const editIconWrapper = style({
  opacity: "0",
});

export const tableRow = style({
  display: "flex",
  alignItems: "center",
  gap: "0.2em",
  padding: "0 0.25em",
  lineHeight: "2em",
  fontSize: "0.875em",
});

globalStyle(`${tableRow}:hover ${editIconWrapper}`, {
  opacity: "1.0",
});
