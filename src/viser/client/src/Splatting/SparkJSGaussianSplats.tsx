/**
 * SparkJS Gaussian Splat renderer component for Viser.
 *
 * This component uses SparkJS to render Gaussian splats with spherical
 * harmonics support. It converts binary SPZ data to a Blob URL for loading.
 */

import React, { useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import "./SparkJSComponents";

// TypeScript declarations for extended elements
declare module "@react-three/fiber" {
  interface ThreeElements {
    splatMesh: any;
  }
}

interface SparkJSGaussianSplatsProps {
  spzData: Uint8Array;
  children?: React.ReactNode;
}

export const SparkJSGaussianSplats = React.forwardRef<
  THREE.Group,
  SparkJSGaussianSplatsProps
>(function SparkJSGaussianSplats({ spzData, children }, ref) {
  const splatMeshRef = useRef<any>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Create a Blob URL from the SPZ binary data
  const splatUrl = useMemo(() => {
    // Create a new Uint8Array to ensure ArrayBuffer type compatibility
    const data = new Uint8Array(spzData);
    const blob = new Blob([data], { type: "application/octet-stream" });
    return URL.createObjectURL(blob);
  }, [spzData]);

  // Clean up the Blob URL when component unmounts
  React.useEffect(() => {
    return () => {
      URL.revokeObjectURL(splatUrl);
    };
  }, [splatUrl]);

  // Memoize SplatMesh args
  const splatMeshArgs = useMemo(
    () =>
      ({
        url: splatUrl,
      }) as const,
    [splatUrl],
  );

  // Track loading state using the initialized Promise
  React.useEffect(() => {
    if (splatMeshRef.current && splatMeshRef.current.initialized) {
      setIsLoading(true);
      splatMeshRef.current.initialized
        .then(() => {
          setIsLoading(false);
          console.log("SparkJS Gaussian splat loaded successfully");
        })
        .catch((error: Error) => {
          setIsLoading(false);
          console.error("Failed to load SparkJS Gaussian splat:", error);
        });
    }
  }, [splatUrl]);

  // Check isInitialized property each frame as fallback
  useFrame(() => {
    if (
      splatMeshRef.current &&
      splatMeshRef.current.isInitialized &&
      isLoading
    ) {
      setIsLoading(false);
      console.log("SparkJS Gaussian splat initialized (detected via frame)");
    }
  });

  return (
    <group ref={ref}>
      <splatMesh ref={splatMeshRef} args={[splatMeshArgs]} />
      {children}
      {/* Optional loading indicator - can be styled or replaced */}
      {isLoading && (
        <mesh position={[0, 0, 0]}>
          <sphereGeometry args={[0.05, 16, 16]} />
          <meshBasicMaterial color="#3b82f6" transparent opacity={0.5} />
        </mesh>
      )}
    </group>
  );
});
