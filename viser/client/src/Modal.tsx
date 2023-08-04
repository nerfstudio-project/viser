import { ViewerContext } from ".";
import { ThemeConfigurationMessage } from "./WebsocketMessages";
import { GuiConfig } from "./ControlPanel/GuiState";
import GeneratedControls from "./ControlPanel/Generated";
import { Modal } from "@mantine/core";
import React, { useContext } from "react";

export function ViserModal() {
  const viewer = useContext(ViewerContext)!;

  const guiConfigFromId = viewer.useGui((state) => state.guiConfigFromId);
  const modalGuiConfigs = [...Object.keys(guiConfigFromId)]
    .sort((a, b) => guiConfigFromId[a].order - guiConfigFromId[b].order)
    .filter((conf) => guiConfigFromId[conf].type === "GuiAddModal");

  const showGenerated = modalGuiConfigs.length > 0;
  return (showGenerated && <GeneratedInput conf={guiConfigFromId[modalGuiConfigs[0]]}></GeneratedInput>)
}

function GeneratedInput({ conf }: { conf: GuiConfig }) {
  const viewer = React.useContext(ViewerContext)!;

  const value = viewer.useGui((state) => state.guiValueFromId[conf.id]) ?? conf.initial_value;
  const setVisible = viewer.useGui((state) => state.setGuiVisible);

  let { visible, disabled } =
    viewer.useGui((state) => state.guiAttributeFromId[conf.id]) || {};

  visible = visible ?? true;
  disabled = disabled ?? false;

    return (
      <Modal opened={visible} onClose={() => setVisible(conf.id, false)} title={value} centered>
        <GeneratedControls destination="MODAL" />
      </Modal>
    );
}
