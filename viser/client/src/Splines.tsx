import { CatmullRomLine } from "@react-three/drei";
import React from "react";
import { Vector3 } from "three";

export class SplineConfig {
    positions: Vector3[];
    tension: number;
    closed: boolean;
    lineWidth: number;

    constructor(positions: [number, number, number][], closed: boolean, tension: number, lineWidth: number) {
        this.tension = tension;
        this.closed = closed;
        this.lineWidth = lineWidth;

        this.positions = [];
        for (const position of positions) {
            const vec = new Vector3(position[0], position[1], position[2]);
            this.positions.push(vec);
        }
    }

    get length(): number {
        return this.positions.length;
    }
}

export const Spline = React.forwardRef<THREE.Mesh, { path: SplineConfig; }>(function Spline(props, ref) {
    if (props.path.length < 2) {
        return null;
    }
    return (
        <mesh ref={ref}>
            <CatmullRomLine
                points={props.path.positions}
                closed={props.path.closed}
                curveType="catmullrom"
                tension={props.path.tension}
                lineWidth={props.path.lineWidth}
            ></CatmullRomLine>
        </mesh>
    );
});
