// Drag Utils
export interface DragEvents {
  move: "touchmove" | "mousemove";
  end: "touchend" | "mouseup";
}
export const touchEvents: DragEvents = { move: "touchmove", end: "touchend" };
export const mouseEvents: DragEvents = { move: "mousemove", end: "mouseup" };

export function isTouchEvent(
  event: TouchEvent | MouseEvent,
): event is TouchEvent {
  return event.type === "touchmove";
}
export function isMouseEvent(
  event: TouchEvent | MouseEvent,
): event is MouseEvent {
  return event.type === "mousemove";
}
