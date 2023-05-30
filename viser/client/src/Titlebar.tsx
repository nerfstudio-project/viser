import Box from "@mui/material/Box";
import { Grid } from "@mui/material";

export function Titlebar() {
    const useTitlebar = true;

    if (!useTitlebar) {
        return null;
    }

    return (
        <Grid
            container
            sx={{
                width: "100%",
                height: "2.5em",
                zIndex: "1000",
                backgroundColor: "rgba(255, 255, 255, 0.85)",
                borderBottom: "1px solid",
                borderBottomColor: "divider",
                direction: "row",
                justifyContent: "space-between",
                alignItems: "center",
                paddingX: "1em"
            }}
        >
                <Box component="div">
                    title
                </Box>
                <Box component="div">
                    title
                </Box>
                <Box component="div">
                    title
                </Box>
        </Grid>
    );
}