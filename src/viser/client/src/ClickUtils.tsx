import React from "react";
import * as THREE from "three";
import { ViewerContextContents } from "./App";

/** Normalize click coordinates to be between -1 and 1, with (0, 0) being the center of the screen.
 * Uses offsetX/Y, and clientWidth/Height to get the coordinates.
 */
export function normalizeClick(
    viewer: ViewerContextContents,
    event: React.PointerEvent<HTMLDivElement>
): THREE.Vector2 {
    const mouseVector = new THREE.Vector2();
    mouseVector.x = 2 * (event.nativeEvent.offsetX / viewer.canvasRef.current!.clientWidth) - 1;
    mouseVector.y = 1 - 2 * (event.nativeEvent.offsetY / viewer.canvasRef.current!.clientHeight);
    return mouseVector;
}

/** Given a normalized click (using `normalizeClick`), check if it is within the bounds of the canvas.
 */
export function isClickValid(
    mouseVector: THREE.Vector2
): boolean {
    return (
        mouseVector.x < 1 && mouseVector.x > -1 &&
        mouseVector.y < 1 && mouseVector.y > -1
    );
}