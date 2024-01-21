import { WrapInputDefault, VectorInput } from "./utils";
import { GuiProps } from "../ControlPanel/GuiState";
import { Vector3InputProps } from "../WebsocketMessages";


export default function Vector3InputComponent({ disabled, value, update, ...conf }: GuiProps<Vector3InputProps>) {
  return (<WrapInputDefault>
    <VectorInput
      id={conf.id}
      n={3}
      value={value}
      onChange={(value) => update({ value: value as [number, number, number] })}
      min={conf.min}
      max={conf.max}
      step={conf.step}
      precision={conf.precision ?? undefined}
      disabled={disabled}
    />
  </WrapInputDefault>);
}