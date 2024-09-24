import * as React from "react";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { GuiVector3Message } from "../WebsocketMessages";
import { VectorInput, ViserInputComponent } from "./common";

export default function Vector3Component({
  id,
  value,
  props: { hint, label, visible, disabled, min, max, step, precision },
}: GuiVector3Message) {
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return (
    <ViserInputComponent {...{ id, hint, label }}>
      <VectorInput
        id={id}
        n={3}
        value={value}
        onChange={(value: any) => setValue(id, value)}
        min={min}
        max={max}
        step={step}
        precision={precision}
        disabled={disabled}
      />
    </ViserInputComponent>
  );
}
