import React from "react";
import { CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer";
import { useFrame } from "@react-three/fiber";
import { ViewerContext } from ".";

/** Component for rendering text labels on scene nodes. */
export default function LabelRenderer() {
  const { wrapperRef } = React.useContext(ViewerContext)!;
  const labelRenderer = new CSS2DRenderer();

  React.useEffect(() => {
    const wrapper = wrapperRef.current;
    labelRenderer.domElement.style.overflow = "hidden";
    labelRenderer.domElement.style.position = "absolute";
    labelRenderer.domElement.style.pointerEvents = "none";
    labelRenderer.domElement.style.top = "0px";
    wrapper && wrapper.appendChild(labelRenderer.domElement);

    function updateDimensions() {
      wrapper &&
        labelRenderer.setSize(wrapper.offsetWidth, wrapper.offsetHeight);
    }
    updateDimensions();

    window.addEventListener("resize", updateDimensions);
    return () => {
      window.removeEventListener("resize", updateDimensions);
    };
  });

  useFrame(({ scene, camera }) => {
    labelRenderer.render(scene, camera);
  });
  return <></>;
}
