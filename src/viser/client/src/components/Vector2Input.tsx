import { WrapInputDefault, VectorInput } from "./utils";
import { GuiProps } from "../ControlPanel/GuiState";
import { Vector2InputProps } from "../WebsocketMessages";


export default function Vector2InputComponent({ disabled, value, update, ...conf }: GuiProps<Vector2InputProps>) {
  return (<WrapInputDefault>
        <VectorInput
          id={conf.id}
          n={2}
          value={value}
          onChange={(value) => update({ value: value as [number, number] })}
          min={conf.min}
          max={conf.max}
          step={conf.step}
          precision={conf.precision ?? undefined}
          disabled={disabled}
        />
  </WrapInputDefault>);
}