import Box from "@mui/material/Box";
import { Button, Grid, Icon, IconButton, SvgIcon } from "@mui/material";
import { useContext } from "react";
import { ViewerContext } from ".";

import * as Icons from '@mui/icons-material';
type IconName = keyof typeof Icons;

interface TitlebarButtonProps {
    text: string | null;
    href: string | null;
    icon: "GitHub" | "Description" | "Keyboard" | null;
    variant: "text" | "contained" | "outlined" | null;
}

export function TitlebarButton(props: TitlebarButtonProps) {
    if (props.icon !== null && props.text === null) {
        return (
            <IconButton
            href={props.href ?? ''}>
                <SvgIcon component={Icons[props.icon as IconName] ?? null} />
            </IconButton>
        )
    }
    return (
        <Button
            variant={props.variant ?? 'contained'}
            href={props.href ?? ''}
            sx={{
                marginY: '0.4em',
                marginX: '0.125em',
                alignItems: 'center'
            }}
            size="small"
            target="_blank"
            startIcon={props.icon ? (<SvgIcon component={Icons[props.icon as IconName] ?? null} />) : null}
        >
            {props.text ?? ''}
        </Button>
    )
}

interface TitlebarImageProps {
    image_url: string;
    image_alt: string;
    href: string | null;
}

export function TitlebarImage(props: TitlebarImageProps | null) {
    if (props == null) {
        return null;
    }
    else {
        if (props.href == null) {
            return (<img src={props.image_url} alt={props.image_alt} style={{ height: "2em", marginLeft: '0.125em', marginRight: '0.125em' }} />)
        }
        return (<a href={props.href} style={{ height: "2em", marginLeft: '0.125em', marginRight: '0.125em' }}>
        <img src={props.image_url} alt={props.image_alt} style={{ height: "2em", marginLeft: '0.125em', marginRight: '0.125em' }} />
        </a>)
    }
}

export function Titlebar() {
    const viewer = useContext(ViewerContext)!;
    const content = viewer.useGui(
        (state) => state.theme.titlebar_content
    )

    if (content == null) {
        return null;
    }

    const buttons = content.buttons;
    const imageData = content.image;

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
                {buttons?.map((btn) => TitlebarButton(btn))}
            </Grid>
            <Grid item xs={3} component="div"
                sx={{
                    display: "flex",
                    direction: "row",
                    alignItems: "center",
                    justifyContent: "right"
                }}
            >
                {imageData != null ? TitlebarImage(imageData) : null}
            </Grid>
        </Grid>
    );
}
