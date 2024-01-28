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
import React, { useContext } from "react";
import { GuiProps, GuiGenerateContextProps, GuiGenerateContext } from "./ControlPanel/GuiState";


type GenericGeneratedComponentProps<T> = T extends AllComponentProps ? T & GuiGenerateContextProps<Omit<T, "type">> & { id: string }: never;
export type GeneratedComponentProps = GenericGeneratedComponentProps<AllComponentProps>;


function assertUnknownComponent(x: string): never {
    throw new Error(`Component type ${x} is not known.`);
}


export default function GeneratedComponent(props: AllComponentProps) {
  const { type } = props;
  const id = "test";
  const contextProps = useContext(GuiGenerateContext)! as GuiGenerateContextProps<Omit<typeof props, "type">>;
  let component = null;
  switch (type) {
    case "Button":
      component = <Button {...props} {...contextProps} />;
      break;
    case "TextInput":
      component = <TextInput {...props} {...contextProps} />;
      break;
    case "NumberInput":
      component = <NumberInput {...props} {...contextProps} />;
      break;
    case "Slider":
      component = <Slider {...props} {...contextProps} />;
      break;
    case "Checkbox":
      component = <Checkbox {...props} {...contextProps} />;
      break;
    case "RgbInput":
      component = <RgbInput {...props} {...contextProps} />;
      break;
    case "RgbaInput":
      component = <RgbaInput {...props} {...contextProps} />;
      break;
    case "Folder":
      component = <Folder {...props} {...contextProps} />;
      break;
    case "Markdown":
      component = <Markdown {...props} {...contextProps} />;
      break;
    case "TabGroup":
      component = <TabGroup {...props} {...contextProps} />;
      break;
    case "Modal":
      component = <Modal {...props} {...contextProps} />;
      break;
    case "Vector2Input":
      component = <Vector2Input {...props} />;
      break;
    case "Vector3Input":
      component = <Vector3Input {...props} />;
      break;
    case "Dropdown":
      component = <Dropdown {...props} />;
      break;
    default:
      assertUnknownComponent(type);
  }
  const currentContextProps = { ...contextProps, id };
  return <GuiGenerateContext.Provider value={currentContextProps}>{component}</GuiGenerateContext.Provider>
}
