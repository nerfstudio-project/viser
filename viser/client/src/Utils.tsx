import { GuiConfig } from "./ControlPanel/GuiState";
import * as Messages from "./WebsocketMessages";

// GUIState Type Guard
export function isGuiConfig(message: Messages.Message): message is GuiConfig {
  return message.type.startsWith("GuiAdd");
}
