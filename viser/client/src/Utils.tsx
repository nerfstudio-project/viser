import { GuiConfig } from "./ControlPanel/GuiState";
import * as Messages from "./WebsocketMessages";

// GUIState Type Guard
export function isGuiConfig(message: Messages.Message): message is GuiConfig {
  return message.type.startsWith("GuiAdd");
}
// Drag Utils
export interface DragEvents {
  move: "touchmove" | "mousemove";
  end: "touchend" | "mouseup";
}
export const touchEvents: DragEvents = { move: "touchmove", end: "touchend" };
export const mouseEvents: DragEvents = { move: "mousemove", end: "mouseup" };

export function isTouchEvent(
  event: TouchEvent | MouseEvent
): event is TouchEvent {
  return event.type === "touchmove";
}
export function isMouseEvent(
  event: TouchEvent | MouseEvent
): event is MouseEvent {
  return event.type === "mousemove";
}
