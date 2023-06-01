import Box from "@mui/material/Box";
import { Button, Grid, Icon, IconButton, SvgIcon } from "@mui/material";
import { useContext } from "react";
import { ViewerContext } from ".";

import * as Icons from '@mui/icons-material';
type IconName = keyof typeof Icons;

const buttonVariants = ['text', 'contained', 'outlined'];
function isVariant(variant: unknown): variant is ('text' | 'contained' | 'outlined') {
    return typeof variant === 'string' && buttonVariants.includes(variant);
}

interface TitlebarButtonProps {
    text?: string;
    href?: string;
    icon?: string;
    variant?: string;
}
function isButtonData(obj: any): obj is TitlebarButtonProps {
    return (
        typeof obj === "object" &&
        (obj.text === null || typeof obj.text === "string") &&
        (obj.href === null || typeof obj.href === "string") &&
        (obj.icon === null || typeof obj.icon === "string") &&
        (obj.variant === null || typeof obj.variant === "string")
    );
}

interface TitlebarImageProps {
    imageSource: string;
    alt: string;
}
function isImageData(obj: any): obj is TitlebarImageProps {
    return (
        typeof obj === "object" &&
        typeof obj.imageSource === "string" &&
        typeof obj.alt === "string"
    );
}

interface TitlebarPaddingProps {
    width: string;
}
function isPaddingData(obj: any): obj is TitlebarPaddingProps {
    return (
        typeof obj === "object" &&
        typeof obj.width == "string"
    );
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
            variant={isVariant(props.variant) ? props.variant : 'contained'}
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

/**
 * 
 * @param props src is a data url or a link to an image
 * @returns image
 */
export function TitlebarImage(props: TitlebarImageProps) {
    return (
        <img src={props.imageSource} alt={props.alt} style={{ height: "2em", marginLeft: '0.125em', marginRight: '0.125em' }} />
    )
}


export function TitlebarPadding(props: TitlebarPaddingProps) {
    return (
        <div style={{ width: props.width, height: '100%' }}></div>
    )
}

function renderTitlebarObject(object: TitlebarButtonProps | TitlebarImageProps | TitlebarPaddingProps, index: number) {
    if (isImageData(object)) {
        return (<TitlebarImage key={index} imageSource={object.imageSource} alt={object.alt} />);
    }
    if (isPaddingData(object)) {
        return (<TitlebarPadding key={index} width={object.width} />);
    }
    if (isButtonData(object)) {
        return (<TitlebarButton key={index} text={object.text} icon={object.icon} href={object.href} variant={object.variant} />);
    }
    return null;
}

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