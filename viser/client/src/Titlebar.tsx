import Box from "@mui/material/Box";
import { Button, Grid, Icon, SvgIcon } from "@mui/material";
import { Server } from "ws";

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
        (obj.text === undefined || typeof obj.text === "string") &&
        (obj.href === undefined || typeof obj.href === "string") &&
        (obj.icon === undefined || typeof obj.icon === "string") &&
        (obj.variant === undefined || typeof obj.variant === "string")
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
        >
            {props.icon !== undefined ? (<Icon style={{marginRight: '0.2em', fontSize: '1em'}}>{props.icon}</Icon>) : null}
            <span style={{lineHeight: '1', marginTop: '0.25em'}}>{props.text ?? ''}</span>
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

export function Titlebar(props: { useTitlebar: boolean; }) {

    const left: Array<TitlebarButtonProps | TitlebarImageProps | TitlebarPaddingProps> = [{ text: "Getting Started", href: "https://docs.nerf.studio/en/latest/quickstart/viewer_quickstart.html", variant: "outlined" }, { text: "Github", icon: "github", href: "https://github.com/nerfstudio-project/nerfstudio", variant: "outlined" }, { text: "Documentation", icon: "description", href: "https://docs.nerf.studio/", variant: "outlined" }, { text: "Viewport Controls", icon: "keyboard", href: "https://viewer.nerf.studio/", variant: "outlined" }]
    const center: Array<TitlebarButtonProps | TitlebarImageProps | TitlebarPaddingProps> = []
    const right: Array<TitlebarButtonProps | TitlebarImageProps | TitlebarPaddingProps> = [{ imageSource: "https://docs.nerf.studio/en/latest/_images/logo.png", alt: "Nerfstudio Logo" }, {width: '5em'}]

    if (!props.useTitlebar) {
        return null;
    }

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
                paddingX: "0.875em"
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