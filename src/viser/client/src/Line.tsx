/** Adapted from: https://github.com/pmndrs/drei/blob/d5ee73265a49d59ab87aab0fad89e997e5495daa/src/core/Line.tsx
 *
 * Modifications:
 * - Takes typed arrays (Float32Array/Uint8Array) as input instead of vanilla arrays.
 * - Updates geometry in place using useLayoutEffect to avoid flickering on prop changes.
 */

import * as React from "react";
import { ColorRepresentation } from "three";
import { ThreeElement, useThree } from "@react-three/fiber";
import {
  LineGeometry,
  LineSegmentsGeometry,
  LineMaterial,
  LineMaterialParameters,
  Line2,
  LineSegments2,
} from "three-stdlib";
import { ForwardRefComponent } from "@react-three/drei/helpers/ts-utils";

export type LineProps = {
  points: Float32Array; // length must be n * 3
  vertexColors?: Uint8Array; // length must be n * 3, values 0-255 for RGB
  lineWidth?: number;
  segments?: boolean;
} & Omit<LineMaterialParameters, "vertexColors" | "color"> &
  Omit<ThreeElement<typeof Line2>, "args"> &
  Omit<ThreeElement<typeof LineMaterial>, "color" | "vertexColors" | "args"> & {
    color?: ColorRepresentation;
  };

export const Line: ForwardRefComponent<LineProps, Line2 | LineSegments2> =
  /* @__PURE__ */ React.forwardRef<Line2 | LineSegments2, LineProps>(
    function Line(
      {
        points,
        color = 0xffffff,
        vertexColors,
        linewidth,
        lineWidth,
        segments,
        dashed,
        ...rest
      },
      ref,
    ) {
      const size = useThree((state) => state.size);
      const line2 = React.useMemo(
        () => (segments ? new LineSegments2() : new Line2()),
        [segments],
      );
      const [lineMaterial] = React.useState(() => new LineMaterial());
      const itemSize = 3; // We're now always using RGB colors (3 components)

      // Create geometry once, then update it in place to avoid flicker.
      const [lineGeom] = React.useState(() =>
        segments ? new LineSegmentsGeometry() : new LineGeometry(),
      );

      // Update geometry positions when points change.
      React.useLayoutEffect(() => {
        lineGeom.setPositions(points);
        line2.computeLineDistances();
      }, [points, line2, lineGeom]);

      // Update geometry colors when vertexColors change.
      React.useLayoutEffect(() => {
        if (vertexColors) {
          // Convert Uint8Array (0-255) to Float32Array (0-1)
          const normalizedColors = new Float32Array(vertexColors).map(
            (c) => c / 255,
          );
          color = 0xffffff;
          lineGeom.setColors(normalizedColors, itemSize);
        }
      }, [vertexColors, lineGeom, itemSize]);

      React.useLayoutEffect(() => {
        if (dashed) {
          lineMaterial.defines.USE_DASH = "";
        } else {
          // Setting lineMaterial.defines.USE_DASH to undefined is apparently not sufficient.
          delete lineMaterial.defines.USE_DASH;
        }
        lineMaterial.needsUpdate = true;
      }, [dashed, lineMaterial]);

      // Clean up resources only when component unmounts.
      // lineGeom and lineMaterial are stable references from useState.
      React.useEffect(() => {
        return () => {
          lineGeom.dispose();
          lineMaterial.dispose();
        };
      }, [lineGeom, lineMaterial]);

      return (
        <primitive object={line2} ref={ref} {...rest}>
          <primitive object={lineGeom} attach="geometry" />
          <primitive
            object={lineMaterial}
            attach="material"
            color={color}
            vertexColors={Boolean(vertexColors)}
            resolution={[size.width, size.height]}
            linewidth={linewidth ?? lineWidth ?? 1}
            dashed={dashed}
            transparent={false} /*need to set to true if itemSize === 4*/
            {...rest}
          />
        </primitive>
      );
    },
  );
