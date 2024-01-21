// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.
import {
  AllComponentProps,
  ButtonProps,
  TextInputProps,
  NumberInputProps,
  SliderProps,
  CheckboxProps,
  RgbInputProps,
  RgbaInputProps,
  FolderProps,
  MarkdownProps,
  TabGroupProps,
  ModalProps,
  Vector2InputProps,
  Vector3InputProps,
  DropdownProps,
} from "./WebsocketMessages";
import Button from "./components/Button";
import TextInput from "./components/TextInput";
import NumberInput from "./components/NumberInput";
import Slider from "./components/Slider";
import Checkbox from "./components/Checkbox";
import RgbInput from "./components/RgbInput";
import RgbaInput from "./components/RgbaInput";
import Folder from "./components/Folder";
import Markdown from "./components/Markdown";
import TabGroup from "./components/TabGroup";
import Modal from "./components/Modal";
import Vector2Input from "./components/Vector2Input";
import Vector3Input from "./components/Vector3Input";
import Dropdown from "./components/Dropdown";
import { GuiProps, GuiGenerateContextProps } from "./ControlPanel/GuiState";


export type GeneratedComponentProps<T extends AllComponentProps> = T & GuiGenerateContextProps<Omit<T, "type">> & { id: string };


export default function GeneratedComponent({ type, ...props }: AllComponentProps) {
  switch (type) {
    case "Button":
      return <Button {...props} id="" />;
    case "TextInput":
      return <TextInput {...props} />;
    case "NumberInput":
      return <NumberInput {...props} />;
    case "Slider":
      return <Slider {...props} />;
    case "Checkbox":
      return <Checkbox {...props} />;
    case "RgbInput":
      return <RgbInput {...props} />;
    case "RgbaInput":
      return <RgbaInput {...props} />;
    case "Folder":
      return <Folder {...props} />;
    case "Markdown":
      return <Markdown {...props} />;
    case "TabGroup":
      return <TabGroup {...props} />;
    case "Modal":
      return <Modal {...props} />;
    case "Vector2Input":
      return <Vector2Input {...props} />;
    case "Vector3Input":
      return <Vector3Input {...props} />;
    case "Dropdown":
      return <Dropdown {...props} />;
    default:
      throw new Error(`Unknown component type ${type}`);
  }
}
