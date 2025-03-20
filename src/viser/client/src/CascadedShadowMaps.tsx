import { useFrame, useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo } from "react";
import { Color, Vector3, Vector3Tuple } from "three";
import { CSM, CSMParameters } from "three/examples/jsm/csm/CSM";

interface CascadedShadowMapProps
  extends Omit<CSMParameters, "lightDirection" | "camera" | "parent"> {
  fade?: boolean;
  lightDirection?: Vector3Tuple;
  color?: Color;
}

class CSMProxy {
  instance: CSM | undefined;
  args: CSMParameters;

  constructor(args: CSMParameters) {
    this.args = args;
  }

  attach() {
    this.instance = new CSM(this.args);
  }

  dispose() {
    if (this.instance) {
      this.instance.dispose();
    }
  }
}

export function CascadedShadowMap({
  maxFar = 50,
  shadowMapSize = 1024,
  lightIntensity = 0.25,
  cascades = 2,
  fade,
  lightDirection = [1, -1, 1],
  shadowBias = 0.000001,
  lightFar,
  lightMargin,
  lightNear,
  mode,
  color,
}: CascadedShadowMapProps) {
  const camera = useThree((three) => three.camera);
  const parent = useThree((three) => three.scene);
  const proxyInstance = useMemo(
    () => {
      const proxy = new CSMProxy({
        camera,
        cascades,
        lightDirection: new Vector3().fromArray(lightDirection).normalize(),
        lightFar,
        lightIntensity,
        lightMargin,
        lightNear,
        maxFar,
        mode,
        parent,
        shadowBias,
        shadowMapSize,
      });
      return proxy;
    },
    // These values will cause CSM to re-instantiate itself.
    // This is an expensive operation and should be avoided.
    [
      // Values that can be updated during runtime are omitted from this deps check.
      cascades,
      fade,
      lightFar,
      lightIntensity,
      lightMargin,
      lightNear,
      maxFar,
      mode,
      shadowBias,
      shadowMapSize,
    ],
  );

  useFrame(() => {
    if (proxyInstance && proxyInstance.instance) {
      proxyInstance.instance.update();
    }
  });

  useEffect(() => {
    if (color === undefined) return;
    proxyInstance.instance!.lights.forEach((light) => {
      light.color = color;
    });
  }, []);

  useLayoutEffect(() => {
    proxyInstance.attach();

    return () => {
      proxyInstance.dispose();
    };
  }, [proxyInstance]);

  return <primitive object={proxyInstance} />;
}
