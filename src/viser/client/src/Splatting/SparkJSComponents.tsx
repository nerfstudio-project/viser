/**
 * SparkJS components extended for React Three Fiber compatibility.
 *
 * This file extends SparkJS's SplatMesh and SparkRenderer to work with
 * React Three Fiber's declarative JSX syntax, following the pattern from
 * the spark-react-r3f example.
 */

import { extend } from "@react-three/fiber";
import { SplatMesh as SparkSplatMesh, SparkRenderer as SparkSparkRenderer } from "@sparkjsdev/spark";

// Extend SparkJS components for use in React Three Fiber
extend({ SplatMesh: SparkSplatMesh, SparkRenderer: SparkSparkRenderer });
