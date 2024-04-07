import Plot from "react-plotly.js";
import { GuiAddPlotlyMessage } from "../WebsocketMessages";
import Plotly from "plotly.js";
import { Flex } from "@mantine/core";
import { useDisclosure } from '@mantine/hooks';
import { Modal, Button, Box, Paper, Text } from '@mantine/core';

import { useEffect, useState } from "react";
import { useElementSize } from '@mantine/hooks';
import { IconWindowMinimize, IconWindowMaximize, IconHandOff, IconHandFinger } from '@tabler/icons-react';
import { ViserInputComponent } from "./common";

function generatePlotWithAspect(json_str: string, aspect_ratio: number) {
    // Parse json string, to construct plotly object.
    // Note that only the JSON string is kept as state, not the json object.
    const plot_json = JSON.parse(json_str);

    // This keeps the zoom-in state, etc, see https://plotly.com/javascript/uirevision/. 
    plot_json.layout.uirevision = "true";

    // Box size change -> width value change -> plot rerender trigger.
    // This doesn't actually work for the *first* time a modal is opened...
    const { ref, width } = useElementSize();
    plot_json.layout.width = width;
    plot_json.layout.height = width * aspect_ratio;

    return (
        <Box ref={ref}>
            <Plot
                data={plot_json.data}
                layout={plot_json.layout}
            />
        </Box>
    );
}

export default function PlotlyComponent({
    id,
    label,
    visible,
    plotly_json_str,
    aspect_ratio,
}: GuiAddPlotlyMessage) {
    if (!visible) return <></>;

    // Create two plots; one for the control panel, and one for the modal.
    // They should have different sizes, so we need to generate them separately.
    const plot_controlpanel = generatePlotWithAspect(plotly_json_str, aspect_ratio);
    const plot_modal = generatePlotWithAspect(plotly_json_str, aspect_ratio);

    // Create a modal with the plot, and a button to open it.
    const [opened, { open, close }] = useDisclosure(false);
    const gui_modal = (
        <Box>
            <Modal opened={opened} onClose={close} title={`Expanded Plot Visualization`} fullScreen
                closeButtonProps={{
                    icon: <IconWindowMinimize stroke={1.625} />
                }}
            >
                <Box p="0.5em">
                    <Text fw={500} c="dimmed"> {label} </Text>
                    <Paper p="xs" withBorder radius="md">
                        {plot_modal}
                    </Paper>
                </Box>
            </Modal>
            <Button onClick={open} variant="transparent" size="xs" c="dimmed">
                <IconWindowMaximize stroke={1.625} size={"1.5em"}/>
            </Button>
        </Box>
    );

    const gui_controlpanel = (
        <Box pr="0.5em" pl="0.5em" pb="0.5em">
            <Paper p="xs" withBorder radius="md">
                {plot_controlpanel}
            </Paper>
        </Box>
    )

    return (
        <Box>
            <ViserInputComponent {...{ id, label }}>
                {gui_modal}
            </ViserInputComponent>
            {gui_controlpanel}
        </Box>
    )
}