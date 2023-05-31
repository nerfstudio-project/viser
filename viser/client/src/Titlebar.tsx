import Box from "@mui/material/Box";
import { Grid } from "@mui/material";

export function Titlebar() {
    const viewer = useContext(ViewerContext)!;
    const showTitlebar = viewer.useGui(
        (state) => state.theme.show_titlebar
    );
    const content: Array<Array<TitlebarButtonProps | TitlebarImageProps | TitlebarPaddingProps>> = viewer.useGui(
        (state) => JSON.parse(state.theme.titlebar_content)
    )
    
    if (showTitlebar == null || content == null || !showTitlebar) {
        return null;
    }

    const left = content[0];
    const center = content[1];
    const right = content[2];

    return (
        <Grid
            container
            sx={{
                width: "100%",
                zIndex: "1000",
                backgroundColor: "rgba(255, 255, 255, 0.85)",
                borderBottom: "1px solid",
                borderBottomColor: "divider",
                direction: "row",
                justifyContent: "space-between",
                alignItems: "center",
                paddingX: "0.875em",
                height: "2.5em"
            }}
        >
            <Grid item xs="auto" component="div"
                sx={{
                    display: "flex",
                    direction: "row",
                    alignItems: "center",
                    justifyContent: "left",
                    overflow: 'visible'
                }}
            >
                {left.map((object, index) => renderTitlebarObject(object, index))}
            </Grid>
            <Grid item xs component="div"
                sx={{
                    display: "flex",
                    direction: "row",
                    alignItems: "center",
                    justifyContent: "center"
                }} 
            >
                {center.map((object, index) => renderTitlebarObject(object, index))}
            </Grid>
            <Grid item xs={4} component="div"
                sx={{
                    display: "flex",
                    direction: "row",
                    alignItems: "center",
                    justifyContent: "right"
                }}
            >
                {right.map((object, index) => renderTitlebarObject(object, index))}
            </Grid>
        </Grid>
    );
}