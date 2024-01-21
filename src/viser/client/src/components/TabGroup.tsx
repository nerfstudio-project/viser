import React from 'react';
import { Tabs, Image } from '@mantine/core';
import { TabGroupProps } from '../WebsocketMessages';
import { GuiGenerateContext } from '../ControlPanel/GuiState';


export default function GeneratedTabGroup({ conf }: { conf: TabGroupProps }) {
    const [tabState, setTabState] = React.useState<TabsValue>("0");
    const { renderContainer } = React.useContext(GuiGenerateContext)!;
    const icons = conf.tab_icons_base64;
  
    return (
      <Tabs
        radius="xs"
        value={tabState}
        onTabChange={setTabState}
        sx={{ marginTop: "-0.75em" }}
      >
        <Tabs.List>
          {conf.tab_labels.map((label, index) => (
            <Tabs.Tab
              value={index.toString()}
              key={index}
              icon={
                icons[index] === null ? undefined : (
                  <Image
                    /*^In Safari, both the icon's height and width need to be set, otherwise the icon is clipped.*/
                    height={"1.125em"}
                    width={"1.125em"}
                    sx={(theme) => ({
                      filter:
                        theme.colorScheme == "dark" ? "invert(1)" : undefined,
                    })}
                    src={"data:image/svg+xml;base64," + icons[index]}
                  />
                )
              }
            >
              {label}
            </Tabs.Tab>
          ))}
        </Tabs.List>
        {conf.tab_container_ids.map((containerId, index) => (
          <Tabs.Panel value={index.toString()} key={containerId}>
            {renderContainer(containerId)}
          </Tabs.Panel>
        ))}
      </Tabs>
    );
  }