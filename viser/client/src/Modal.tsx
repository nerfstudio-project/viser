import { ViewerContext } from "./App";
import { GuiAddModalMessage } from "./WebsocketMessages";
import GeneratedGuiContainer from "./ControlPanel/Generated";
import { Modal } from "@mantine/core";
import React, { useContext } from "react";

export function ViserModal() {
  const viewer = useContext(ViewerContext)!;

  const guiConfigFromId = viewer.useGui((state) => state.guiConfigFromId);
  const modalGuiConfigs = [...Object.keys(guiConfigFromId)]
    .map((id) => guiConfigFromId[id])
    .sort((a, b) => a.order - b.order)
    .map((conf) => {
      if (conf.type == "GuiAddModalMessage")
        return (
          <GeneratedModal conf={conf} />
        );
      else
        return;
    });

  return modalGuiConfigs;
}

function GeneratedModal({ conf }: { conf: GuiAddModalMessage }) {
  const viewer = useContext(ViewerContext)!;
  const remove = viewer.useGui((state) => state.removeGuiContainer);

  const [modalVisible, setModalVisible] = React.useState<boolean>(true);

  return (
    <Modal
      opened={modalVisible}
      title={conf.label}
      onClose={() => {
        setModalVisible(false);
        remove(conf.id);
      }}
      centered>
        <GeneratedGuiContainer containerId={conf.id} />
    </Modal>
  )
}
