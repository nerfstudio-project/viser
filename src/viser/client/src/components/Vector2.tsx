import * as React from "react";
import { GuiComponentContext } from "../ControlPanel/GuiComponentContext";
import { GuiAddVector2Message } from "../WebsocketMessages";
import { VectorInput, ViserInputComponent } from "./common";

export default function Vector2Component({ id, hint, label, visible, disabled, value, ...otherProps }: GuiAddVector2Message) {
  const { min, max, step, precision } = otherProps;
  const { setValue } = React.useContext(GuiComponentContext)!;
  if (!visible) return <></>;
  return <ViserInputComponent {...{ id, hint, label }}>
    <VectorInput
      id={id}
      n={2}
      value={value}
      onChange={(value: any) => setValue(id, value)}
      min={min}
      max={max}
      step={step}
      precision={precision}
      disabled={disabled}
    />
  </ViserInputComponent>;
}