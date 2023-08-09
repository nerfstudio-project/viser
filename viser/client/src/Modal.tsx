import { ViewerContext } from "./App";
import { GuiModalMessage } from "./WebsocketMessages";
import GeneratedGuiContainer from "./ControlPanel/Generated";
import { Modal } from "@mantine/core";
import React, { useContext } from "react";

export function ViserModal() {
  const viewer = useContext(ViewerContext)!;

  const modalList = viewer.useGui((state) => state.modals);
  const modals = modalList
    .map((conf) => {
      return (
        <GeneratedModal key={conf.id} conf={conf} />
      );
    });

  return modals;
}

function GeneratedModal({ conf }: { conf: GuiModalMessage }) {
  const viewer = useContext(ViewerContext)!;
  const popModal = viewer.useGui((state) => state.popModal);
  const removeContainer = viewer.useGui((state) => state.removeGuiContainer);

  const [modalVisible, setModalVisible] = React.useState<boolean>(true);

  return (
    <Modal
      opened={modalVisible}
      title={conf.label}
      onClose={() => {
        setModalVisible(false);
        removeContainer(conf.id);
        popModal();
      }}
      centered>
        <GeneratedGuiContainer containerId={conf.id} />
    </Modal>
  )
}
