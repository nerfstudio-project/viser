// AUTOMATICALLY GENERATED message interfaces, from Python dataclass definitions.
// This file should not be manually modified.
/** Variant of CameraMessage used for visualizing camera frustums.
 *
 * OpenCV convention, +Z forward.
 *
 * (automatically generated)
 */
export interface CameraFrustumMessage {
  type: "CameraFrustumMessage";
  name: string;
  props: {
    fov: number;
    aspect: number;
    scale: number;
    line_width: number;
    color: [number, number, number];
    image_media_type: "image/jpeg" | "image/png" | null;
    _image_data: Uint8Array | null;
    cast_shadow: boolean;
    receive_shadow: boolean;
  };
}
/** GlTF message.
 *
 * (automatically generated)
 */
export interface GlbMessage {
  type: "GlbMessage";
  name: string;
  props: {
    glb_data: Uint8Array;
    scale: number;
    cast_shadow: boolean;
    receive_shadow: boolean;
  };
}
/** Coordinate frame message.
 *
 * (automatically generated)
 */
export interface FrameMessage {
  type: "FrameMessage";
  name: string;
  props: {
    show_axes: boolean;
    axes_length: number;
    axes_radius: number;
    origin_radius: number;
    origin_color: [number, number, number];
  };
}
/** Batched axes message.
 *
 * Positions and orientations should follow a `T_parent_local` convention, which
 * corresponds to the R matrix and t vector in `p_parent = [R | t] p_local`.
 *
 * (automatically generated)
 */
export interface BatchedAxesMessage {
  type: "BatchedAxesMessage";
  name: string;
  props: {
    batched_wxyzs: Uint8Array;
    batched_positions: Uint8Array;
    axes_length: number;
    axes_radius: number;
  };
}
/** Grid message. Helpful for visualizing things like ground planes.
 *
 * (automatically generated)
 */
export interface GridMessage {
  type: "GridMessage";
  name: string;
  props: {
    width: number;
    height: number;
    width_segments: number;
    height_segments: number;
    plane: "xz" | "xy" | "yx" | "yz" | "zx" | "zy";
    cell_color: [number, number, number];
    cell_thickness: number;
    cell_size: number;
    section_color: [number, number, number];
    section_thickness: number;
    section_size: number;
    shadow_opacity: number;
  };
}
/** Add a 2D label to the scene.
 *
 * (automatically generated)
 */
export interface LabelMessage {
  type: "LabelMessage";
  name: string;
  props: { text: string };
}
/** Add a 3D gui element to the scene.
 *
 * (automatically generated)
 */
export interface Gui3DMessage {
  type: "Gui3DMessage";
  name: string;
  props: { order: number; container_uuid: string };
}
/** Point cloud message.
 *
 * Positions are internally canonicalized to float32, colors to uint8.
 *
 * Float color inputs should be in the range [0,1], int color inputs should be in the
 * range [0,255].
 *
 * (automatically generated)
 */
export interface PointCloudMessage {
  type: "PointCloudMessage";
  name: string;
  props: {
    points: Uint8Array;
    colors: Uint8Array;
    point_size: number;
    point_shape: "square" | "diamond" | "circle" | "rounded" | "sparkle";
    precision: "float16" | "float32";
  };
}
/** Directional light message.
 *
 * (automatically generated)
 */
export interface DirectionalLightMessage {
  type: "DirectionalLightMessage";
  name: string;
  props: {
    color: [number, number, number];
    intensity: number;
    cast_shadow: boolean;
  };
}
/** Ambient light message.
 *
 * (automatically generated)
 */
export interface AmbientLightMessage {
  type: "AmbientLightMessage";
  name: string;
  props: { color: [number, number, number]; intensity: number };
}
/** Hemisphere light message.
 *
 * (automatically generated)
 */
export interface HemisphereLightMessage {
  type: "HemisphereLightMessage";
  name: string;
  props: {
    sky_color: [number, number, number];
    ground_color: [number, number, number];
    intensity: number;
  };
}
/** Point light message.
 *
 * (automatically generated)
 */
export interface PointLightMessage {
  type: "PointLightMessage";
  name: string;
  props: {
    color: [number, number, number];
    intensity: number;
    distance: number;
    decay: number;
    cast_shadow: boolean;
  };
}
/** Rectangular Area light message.
 *
 * (automatically generated)
 */
export interface RectAreaLightMessage {
  type: "RectAreaLightMessage";
  name: string;
  props: {
    color: [number, number, number];
    intensity: number;
    width: number;
    height: number;
  };
}
/** Spot light message.
 *
 * (automatically generated)
 */
export interface SpotLightMessage {
  type: "SpotLightMessage";
  name: string;
  props: {
    color: [number, number, number];
    intensity: number;
    distance: number;
    angle: number;
    penumbra: number;
    decay: number;
    cast_shadow: boolean;
  };
}
/** Mesh message.
 *
 * Vertices are internally canonicalized to float32, faces to uint32.
 *
 * (automatically generated)
 */
export interface MeshMessage {
  type: "MeshMessage";
  name: string;
  props: {
    vertices: Uint8Array;
    faces: Uint8Array;
    color: [number, number, number];
    wireframe: boolean;
    opacity: number | null;
    flat_shading: boolean;
    side: "front" | "back" | "double";
    material: "standard" | "toon3" | "toon5";
    cast_shadow: boolean;
    receive_shadow: boolean;
  };
}
/** Skinned mesh message.
 *
 * (automatically generated)
 */
export interface SkinnedMeshMessage {
  type: "SkinnedMeshMessage";
  name: string;
  props: {
    vertices: Uint8Array;
    faces: Uint8Array;
    color: [number, number, number];
    wireframe: boolean;
    opacity: number | null;
    flat_shading: boolean;
    side: "front" | "back" | "double";
    material: "standard" | "toon3" | "toon5";
    cast_shadow: boolean;
    receive_shadow: boolean;
    bone_wxyzs: Uint8Array;
    bone_positions: Uint8Array;
    skin_indices: Uint8Array;
    skin_weights: Uint8Array;
  };
}
/** Message from server->client carrying batched meshes information.
 *
 * (automatically generated)
 */
export interface BatchedMeshesMessage {
  type: "BatchedMeshesMessage";
  name: string;
  props: {
    batched_wxyzs: Uint8Array;
    batched_positions: Uint8Array;
    lod: "auto" | "off" | [number, number][];
    vertices: Uint8Array;
    faces: Uint8Array;
    color: [number, number, number];
    wireframe: boolean;
    opacity: number | null;
    flat_shading: boolean;
    side: "front" | "back" | "double";
    material: "standard" | "toon3" | "toon5";
    cast_shadow: boolean;
    receive_shadow: boolean;
  };
}
/** Message from server->client carrying batched GLB information.
 *
 * (automatically generated)
 */
export interface BatchedGlbMessage {
  type: "BatchedGlbMessage";
  name: string;
  props: {
    batched_wxyzs: Uint8Array;
    batched_positions: Uint8Array;
    lod: "auto" | "off" | [number, number][];
    glb_data: Uint8Array;
    scale: number;
    cast_shadow: boolean;
    receive_shadow: boolean;
  };
}
/** Message for transform gizmos.
 *
 * (automatically generated)
 */
export interface TransformControlsMessage {
  type: "TransformControlsMessage";
  name: string;
  props: {
    scale: number;
    line_width: number;
    fixed: boolean;
    auto_transform: boolean;
    active_axes: [boolean, boolean, boolean];
    disable_axes: boolean;
    disable_sliders: boolean;
    disable_rotations: boolean;
    translation_limits: [[number, number], [number, number], [number, number]];
    rotation_limits: [[number, number], [number, number], [number, number]];
    depth_test: boolean;
    opacity: number;
  };
}
/** Message for rendering 2D images.
 *
 * (automatically generated)
 */
export interface ImageMessage {
  type: "ImageMessage";
  name: string;
  props: {
    media_type: "image/jpeg" | "image/png";
    _data: Uint8Array;
    render_width: number;
    render_height: number;
    cast_shadow: boolean;
    receive_shadow: boolean;
  };
}
/** Message from server->client carrying line segments information.
 *
 * (automatically generated)
 */
export interface LineSegmentsMessage {
  type: "LineSegmentsMessage";
  name: string;
  props: { points: Uint8Array; line_width: number; colors: Uint8Array };
}
/** Message from server->client carrying Catmull-Rom spline information.
 *
 * (automatically generated)
 */
export interface CatmullRomSplineMessage {
  type: "CatmullRomSplineMessage";
  name: string;
  props: {
    positions: [number, number, number][];
    curve_type: "centripetal" | "chordal" | "catmullrom";
    tension: number;
    closed: boolean;
    line_width: number;
    color: [number, number, number];
    segments: number | null;
  };
}
/** Message from server->client carrying Cubic Bezier spline information.
 *
 * (automatically generated)
 */
export interface CubicBezierSplineMessage {
  type: "CubicBezierSplineMessage";
  name: string;
  props: {
    positions: [number, number, number][];
    control_points: [number, number, number][];
    line_width: number;
    color: [number, number, number];
    segments: number | null;
  };
}
/** Message from server->client carrying splattable Gaussians.
 *
 * (automatically generated)
 */
export interface GaussianSplatsMessage {
  type: "GaussianSplatsMessage";
  name: string;
  props: { buffer: Uint8Array };
}
/** Remove a particular node from the scene.
 *
 * (automatically generated)
 */
export interface RemoveSceneNodeMessage {
  type: "RemoveSceneNodeMessage";
  name: string;
}
/** GuiFolderMessage(uuid: 'str', container_uuid: 'str', props: 'GuiFolderProps')
 *
 * (automatically generated)
 */
export interface GuiFolderMessage {
  type: "GuiFolderMessage";
  uuid: string;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    visible: boolean;
    expand_by_default: boolean;
  };
}
/** GuiMarkdownMessage(uuid: 'str', container_uuid: 'str', props: 'GuiMarkdownProps')
 *
 * (automatically generated)
 */
export interface GuiMarkdownMessage {
  type: "GuiMarkdownMessage";
  uuid: string;
  container_uuid: string;
  props: { order: number; _markdown: string; visible: boolean };
}
/** GuiHtmlMessage(uuid: 'str', container_uuid: 'str', props: 'GuiHtmlProps')
 *
 * (automatically generated)
 */
export interface GuiHtmlMessage {
  type: "GuiHtmlMessage";
  uuid: string;
  container_uuid: string;
  props: { order: number; content: string; visible: boolean };
}
/** GuiProgressBarMessage(uuid: 'str', value: 'float', container_uuid: 'str', props: 'GuiProgressBarProps')
 *
 * (automatically generated)
 */
export interface GuiProgressBarMessage {
  type: "GuiProgressBarMessage";
  uuid: string;
  value: number;
  container_uuid: string;
  props: {
    order: number;
    animated: boolean;
    color:
      | "dark"
      | "gray"
      | "red"
      | "pink"
      | "grape"
      | "violet"
      | "indigo"
      | "blue"
      | "cyan"
      | "green"
      | "lime"
      | "yellow"
      | "orange"
      | "teal"
      | null;
    visible: boolean;
  };
}
/** GuiPlotlyMessage(uuid: 'str', container_uuid: 'str', props: 'GuiPlotlyProps')
 *
 * (automatically generated)
 */
export interface GuiPlotlyMessage {
  type: "GuiPlotlyMessage";
  uuid: string;
  container_uuid: string;
  props: {
    order: number;
    _plotly_json_str: string;
    aspect: number;
    visible: boolean;
  };
}
/** GuiImageMessage(uuid: 'str', container_uuid: 'str', props: 'GuiImageProps')
 *
 * (automatically generated)
 */
export interface GuiImageMessage {
  type: "GuiImageMessage";
  uuid: string;
  container_uuid: string;
  props: {
    order: number;
    label: string | null;
    _data: Uint8Array | null;
    media_type: "image/jpeg" | "image/png";
    visible: boolean;
  };
}
/** GuiTabGroupMessage(uuid: 'str', container_uuid: 'str', props: 'GuiTabGroupProps')
 *
 * (automatically generated)
 */
export interface GuiTabGroupMessage {
  type: "GuiTabGroupMessage";
  uuid: string;
  container_uuid: string;
  props: {
    _tab_labels: string[];
    _tab_icons_html: (string | null)[];
    _tab_container_ids: string[];
    order: number;
    visible: boolean;
  };
}
/** GuiButtonMessage(uuid: 'str', value: 'bool', container_uuid: 'str', props: 'GuiButtonProps')
 *
 * (automatically generated)
 */
export interface GuiButtonMessage {
  type: "GuiButtonMessage";
  uuid: string;
  value: boolean;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    color:
      | "dark"
      | "gray"
      | "red"
      | "pink"
      | "grape"
      | "violet"
      | "indigo"
      | "blue"
      | "cyan"
      | "green"
      | "lime"
      | "yellow"
      | "orange"
      | "teal"
      | null;
    _icon_html: string | null;
  };
}
/** GuiUploadButtonMessage(uuid: 'str', container_uuid: 'str', props: 'GuiUploadButtonProps')
 *
 * (automatically generated)
 */
export interface GuiUploadButtonMessage {
  type: "GuiUploadButtonMessage";
  uuid: string;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    color:
      | "dark"
      | "gray"
      | "red"
      | "pink"
      | "grape"
      | "violet"
      | "indigo"
      | "blue"
      | "cyan"
      | "green"
      | "lime"
      | "yellow"
      | "orange"
      | "teal"
      | null;
    _icon_html: string | null;
    mime_type: string;
  };
}
/** GuiSliderMessage(uuid: 'str', value: 'float', container_uuid: 'str', props: 'GuiSliderProps')
 *
 * (automatically generated)
 */
export interface GuiSliderMessage {
  type: "GuiSliderMessage";
  uuid: string;
  value: number;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    min: number;
    max: number;
    step: number;
    precision: number;
    _marks: { value: number; label: string | null }[] | null;
  };
}
/** GuiMultiSliderMessage(uuid: 'str', value: 'Tuple[float, ...]', container_uuid: 'str', props: 'GuiMultiSliderProps')
 *
 * (automatically generated)
 */
export interface GuiMultiSliderMessage {
  type: "GuiMultiSliderMessage";
  uuid: string;
  value: number[];
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    min: number;
    max: number;
    step: number;
    min_range: number | null;
    precision: number;
    fixed_endpoints: boolean;
    _marks: { value: number; label: string | null }[] | null;
  };
}
/** GuiNumberMessage(uuid: 'str', value: 'float', container_uuid: 'str', props: 'GuiNumberProps')
 *
 * (automatically generated)
 */
export interface GuiNumberMessage {
  type: "GuiNumberMessage";
  uuid: string;
  value: number;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    precision: number;
    step: number;
    min: number | null;
    max: number | null;
  };
}
/** GuiRgbMessage(uuid: 'str', value: 'Tuple[int, int, int]', container_uuid: 'str', props: 'GuiRgbProps')
 *
 * (automatically generated)
 */
export interface GuiRgbMessage {
  type: "GuiRgbMessage";
  uuid: string;
  value: [number, number, number];
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
  };
}
/** GuiRgbaMessage(uuid: 'str', value: 'Tuple[int, int, int, int]', container_uuid: 'str', props: 'GuiRgbaProps')
 *
 * (automatically generated)
 */
export interface GuiRgbaMessage {
  type: "GuiRgbaMessage";
  uuid: string;
  value: [number, number, number, number];
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
  };
}
/** GuiCheckboxMessage(uuid: 'str', value: 'bool', container_uuid: 'str', props: 'GuiCheckboxProps')
 *
 * (automatically generated)
 */
export interface GuiCheckboxMessage {
  type: "GuiCheckboxMessage";
  uuid: string;
  value: boolean;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
  };
}
/** GuiVector2Message(uuid: 'str', value: 'Tuple[float, float]', container_uuid: 'str', props: 'GuiVector2Props')
 *
 * (automatically generated)
 */
export interface GuiVector2Message {
  type: "GuiVector2Message";
  uuid: string;
  value: [number, number];
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    min: [number, number] | null;
    max: [number, number] | null;
    step: number;
    precision: number;
  };
}
/** GuiVector3Message(uuid: 'str', value: 'Tuple[float, float, float]', container_uuid: 'str', props: 'GuiVector3Props')
 *
 * (automatically generated)
 */
export interface GuiVector3Message {
  type: "GuiVector3Message";
  uuid: string;
  value: [number, number, number];
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    min: [number, number, number] | null;
    max: [number, number, number] | null;
    step: number;
    precision: number;
  };
}
/** GuiTextMessage(uuid: 'str', value: 'str', container_uuid: 'str', props: 'GuiTextProps')
 *
 * (automatically generated)
 */
export interface GuiTextMessage {
  type: "GuiTextMessage";
  uuid: string;
  value: string;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    multiline: boolean;
  };
}
/** GuiDropdownMessage(uuid: 'str', value: 'str', container_uuid: 'str', props: 'GuiDropdownProps')
 *
 * (automatically generated)
 */
export interface GuiDropdownMessage {
  type: "GuiDropdownMessage";
  uuid: string;
  value: string;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    options: string[];
  };
}
/** GuiButtonGroupMessage(uuid: 'str', value: 'str', container_uuid: 'str', props: 'GuiButtonGroupProps')
 *
 * (automatically generated)
 */
export interface GuiButtonGroupMessage {
  type: "GuiButtonGroupMessage";
  uuid: string;
  value: string;
  container_uuid: string;
  props: {
    order: number;
    label: string;
    hint: string | null;
    visible: boolean;
    disabled: boolean;
    options: string[];
  };
}
/** Sent server->client to remove a GUI element.
 *
 * (automatically generated)
 */
export interface GuiRemoveMessage {
  type: "GuiRemoveMessage";
  uuid: string;
}
/** Message for running some arbitrary Javascript on the client.
 * We use this to set up the Plotly.js package, via the plotly.min.js source
 * code.
 *
 * (automatically generated)
 */
export interface RunJavascriptMessage {
  type: "RunJavascriptMessage";
  source: string;
}
/** Notification message.
 *
 * (automatically generated)
 */
export interface NotificationMessage {
  type: "NotificationMessage";
  mode: "show" | "update";
  uuid: string;
  props: {
    title: string;
    body: string;
    loading: boolean;
    with_close_button: boolean;
    auto_close: number | false;
    color:
      | "dark"
      | "gray"
      | "red"
      | "pink"
      | "grape"
      | "violet"
      | "indigo"
      | "blue"
      | "cyan"
      | "green"
      | "lime"
      | "yellow"
      | "orange"
      | "teal"
      | null;
  };
}
/** Remove a specific notification.
 *
 * (automatically generated)
 */
export interface RemoveNotificationMessage {
  type: "RemoveNotificationMessage";
  uuid: string;
}
/** Message for a posed viewer camera.
 * Pose is in the form T_world_camera, OpenCV convention, +Z forward.
 *
 * (automatically generated)
 */
export interface ViewerCameraMessage {
  type: "ViewerCameraMessage";
  wxyz: [number, number, number, number];
  position: [number, number, number];
  fov: number;
  near: number;
  far: number;
  image_height: number;
  image_width: number;
  look_at: [number, number, number];
  up_direction: [number, number, number];
}
/** Message for a raycast-like pointer in the scene.
 * origin is the viewing camera position, in world coordinates.
 * direction is the vector if a ray is projected from the camera through the clicked pixel,
 *
 *
 * (automatically generated)
 */
export interface ScenePointerMessage {
  type: "ScenePointerMessage";
  event_type: "click" | "rect-select";
  ray_origin: [number, number, number] | null;
  ray_direction: [number, number, number] | null;
  screen_pos: [number, number][];
}
/** Message to enable/disable scene click events.
 *
 * (automatically generated)
 */
export interface ScenePointerEnableMessage {
  type: "ScenePointerEnableMessage";
  enable: boolean;
  event_type: "click" | "rect-select";
}
/** Environment Map message.
 *
 * (automatically generated)
 */
export interface EnvironmentMapMessage {
  type: "EnvironmentMapMessage";
  hdri:
    | "apartment"
    | "city"
    | "dawn"
    | "forest"
    | "lobby"
    | "night"
    | "park"
    | "studio"
    | "sunset"
    | "warehouse"
    | null;
  background: boolean;
  background_blurriness: number;
  background_intensity: number;
  background_wxyz: [number, number, number, number];
  environment_intensity: number;
  environment_wxyz: [number, number, number, number];
}
/** Default light message.
 *
 * (automatically generated)
 */
export interface EnableLightsMessage {
  type: "EnableLightsMessage";
  enabled: boolean;
  cast_shadow: boolean;
}
/** Server -> client message to set a skinned mesh bone's orientation.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetBoneOrientationMessage {
  type: "SetBoneOrientationMessage";
  name: string;
  bone_index: number;
  wxyz: [number, number, number, number];
}
/** Server -> client message to set a skinned mesh bone's position.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetBonePositionMessage {
  type: "SetBonePositionMessage";
  name: string;
  bone_index: number;
  position: [number, number, number];
}
/** Server -> client message to set the camera's position.
 *
 * (automatically generated)
 */
export interface SetCameraPositionMessage {
  type: "SetCameraPositionMessage";
  position: [number, number, number];
}
/** Server -> client message to set the camera's up direction.
 *
 * (automatically generated)
 */
export interface SetCameraUpDirectionMessage {
  type: "SetCameraUpDirectionMessage";
  position: [number, number, number];
}
/** Server -> client message to set the camera's look-at point.
 *
 * (automatically generated)
 */
export interface SetCameraLookAtMessage {
  type: "SetCameraLookAtMessage";
  look_at: [number, number, number];
}
/** Server -> client message to set the camera's near clipping plane.
 *
 * (automatically generated)
 */
export interface SetCameraNearMessage {
  type: "SetCameraNearMessage";
  near: number;
}
/** Server -> client message to set the camera's far clipping plane.
 *
 * (automatically generated)
 */
export interface SetCameraFarMessage {
  type: "SetCameraFarMessage";
  far: number;
}
/** Server -> client message to set the camera's field of view.
 *
 * (automatically generated)
 */
export interface SetCameraFovMessage {
  type: "SetCameraFovMessage";
  fov: number;
}
/** Server -> client message to set a scene node's orientation.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetOrientationMessage {
  type: "SetOrientationMessage";
  name: string;
  wxyz: [number, number, number, number];
}
/** Server -> client message to set a scene node's position.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface SetPositionMessage {
  type: "SetPositionMessage";
  name: string;
  position: [number, number, number];
}
/** Client -> server message when a transform control is updated.
 *
 * As with all other messages, transforms take the `T_parent_local` convention.
 *
 * (automatically generated)
 */
export interface TransformControlsUpdateMessage {
  type: "TransformControlsUpdateMessage";
  name: string;
  wxyz: [number, number, number, number];
  position: [number, number, number];
}
/** Message for rendering a background image.
 *
 * (automatically generated)
 */
export interface BackgroundImageMessage {
  type: "BackgroundImageMessage";
  media_type: "image/jpeg" | "image/png";
  rgb_data: Uint8Array | null;
  depth_data: Uint8Array | null;
}
/** Set the visibility of a particular node in the scene.
 *
 * (automatically generated)
 */
export interface SetSceneNodeVisibilityMessage {
  type: "SetSceneNodeVisibilityMessage";
  name: string;
  visible: boolean;
}
/** Set the clickability of a particular node in the scene.
 *
 * (automatically generated)
 */
export interface SetSceneNodeClickableMessage {
  type: "SetSceneNodeClickableMessage";
  name: string;
  clickable: boolean;
}
/** Message for clicked objects.
 *
 * (automatically generated)
 */
export interface SceneNodeClickMessage {
  type: "SceneNodeClickMessage";
  name: string;
  instance_index: number | null;
  ray_origin: [number, number, number];
  ray_direction: [number, number, number];
  screen_pos: [number, number];
}
/** Reset GUI.
 *
 * (automatically generated)
 */
export interface ResetGuiMessage {
  type: "ResetGuiMessage";
}
/** GuiModalMessage(order: 'float', uuid: 'str', title: 'str')
 *
 * (automatically generated)
 */
export interface GuiModalMessage {
  type: "GuiModalMessage";
  order: number;
  uuid: string;
  title: string;
}
/** GuiCloseModalMessage(uuid: 'str')
 *
 * (automatically generated)
 */
export interface GuiCloseModalMessage {
  type: "GuiCloseModalMessage";
  uuid: string;
}
/** Sent client<->server when any property of a GUI component is changed.
 *
 * (automatically generated)
 */
export interface GuiUpdateMessage {
  type: "GuiUpdateMessage";
  uuid: string;
  updates: { [key: string]: any };
}
/** Sent client<->server when any property of a scene node is changed.
 *
 * (automatically generated)
 */
export interface SceneNodeUpdateMessage {
  type: "SceneNodeUpdateMessage";
  name: string;
  updates: { [key: string]: any };
}
/** Message from server->client to configure parts of the GUI.
 *
 * (automatically generated)
 */
export interface ThemeConfigurationMessage {
  type: "ThemeConfigurationMessage";
  titlebar_content: {
    buttons:
      | {
          text: string | null;
          icon: "GitHub" | "Description" | "Keyboard" | null;
          href: string | null;
        }[]
      | null;
    image: {
      image_url_light: string;
      image_url_dark: string | null;
      image_alt: string;
      href: string | null;
    } | null;
  } | null;
  control_layout: "floating" | "collapsible" | "fixed";
  control_width: "small" | "medium" | "large";
  show_logo: boolean;
  show_share_button: boolean;
  dark_mode: boolean;
  colors:
    | [
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
        string,
      ]
    | null;
}
/** Message from server->client requesting a render from a specified camera
 * pose.
 *
 * (automatically generated)
 */
export interface GetRenderRequestMessage {
  type: "GetRenderRequestMessage";
  format: "image/jpeg" | "image/png";
  height: number;
  width: number;
  quality: number;
  wxyz: [number, number, number, number];
  position: [number, number, number];
  fov: number;
}
/** Message from client->server carrying a render.
 *
 * (automatically generated)
 */
export interface GetRenderResponseMessage {
  type: "GetRenderResponseMessage";
  payload: Uint8Array;
}
/** Signal that a file is about to be sent.
 *
 * This message is used to upload files from clients to the server.
 *
 *
 * (automatically generated)
 */
export interface FileTransferStartUpload {
  type: "FileTransferStartUpload";
  source_component_uuid: string;
  transfer_uuid: string;
  filename: string;
  mime_type: string;
  part_count: number;
  size_bytes: number;
}
/** Signal that a file is about to be sent.
 *
 * This message is used to send files to clients from the server.
 *
 *
 * (automatically generated)
 */
export interface FileTransferStartDownload {
  type: "FileTransferStartDownload";
  save_immediately: boolean;
  transfer_uuid: string;
  filename: string;
  mime_type: string;
  part_count: number;
  size_bytes: number;
}
/** Send a file for clients to download or upload files from client.
 *
 * (automatically generated)
 */
export interface FileTransferPart {
  type: "FileTransferPart";
  source_component_uuid: string | null;
  transfer_uuid: string;
  part: number;
  content: Uint8Array;
}
/** Send a file for clients to download or upload files from client.
 *
 * (automatically generated)
 */
export interface FileTransferPartAck {
  type: "FileTransferPartAck";
  source_component_uuid: string | null;
  transfer_uuid: string;
  transferred_bytes: number;
  total_bytes: number;
}
/** Message from client->server to connect to the share URL server.
 *
 * (automatically generated)
 */
export interface ShareUrlRequest {
  type: "ShareUrlRequest";
}
/** Message from server->client to indicate that the share URL has been updated.
 *
 * (automatically generated)
 */
export interface ShareUrlUpdated {
  type: "ShareUrlUpdated";
  share_url: string | null;
}
/** Message from client->server to disconnect from the share URL server.
 *
 * (automatically generated)
 */
export interface ShareUrlDisconnect {
  type: "ShareUrlDisconnect";
}
/** Message from server->client to set the label of the GUI panel.
 *
 * (automatically generated)
 */
export interface SetGuiPanelLabelMessage {
  type: "SetGuiPanelLabelMessage";
  label: string | null;
}

export type Message =
  | CameraFrustumMessage
  | GlbMessage
  | FrameMessage
  | BatchedAxesMessage
  | GridMessage
  | LabelMessage
  | Gui3DMessage
  | PointCloudMessage
  | DirectionalLightMessage
  | AmbientLightMessage
  | HemisphereLightMessage
  | PointLightMessage
  | RectAreaLightMessage
  | SpotLightMessage
  | MeshMessage
  | SkinnedMeshMessage
  | BatchedMeshesMessage
  | BatchedGlbMessage
  | TransformControlsMessage
  | ImageMessage
  | LineSegmentsMessage
  | CatmullRomSplineMessage
  | CubicBezierSplineMessage
  | GaussianSplatsMessage
  | RemoveSceneNodeMessage
  | GuiFolderMessage
  | GuiMarkdownMessage
  | GuiHtmlMessage
  | GuiProgressBarMessage
  | GuiPlotlyMessage
  | GuiImageMessage
  | GuiTabGroupMessage
  | GuiButtonMessage
  | GuiUploadButtonMessage
  | GuiSliderMessage
  | GuiMultiSliderMessage
  | GuiNumberMessage
  | GuiRgbMessage
  | GuiRgbaMessage
  | GuiCheckboxMessage
  | GuiVector2Message
  | GuiVector3Message
  | GuiTextMessage
  | GuiDropdownMessage
  | GuiButtonGroupMessage
  | GuiRemoveMessage
  | RunJavascriptMessage
  | NotificationMessage
  | RemoveNotificationMessage
  | ViewerCameraMessage
  | ScenePointerMessage
  | ScenePointerEnableMessage
  | EnvironmentMapMessage
  | EnableLightsMessage
  | SetBoneOrientationMessage
  | SetBonePositionMessage
  | SetCameraPositionMessage
  | SetCameraUpDirectionMessage
  | SetCameraLookAtMessage
  | SetCameraNearMessage
  | SetCameraFarMessage
  | SetCameraFovMessage
  | SetOrientationMessage
  | SetPositionMessage
  | TransformControlsUpdateMessage
  | BackgroundImageMessage
  | SetSceneNodeVisibilityMessage
  | SetSceneNodeClickableMessage
  | SceneNodeClickMessage
  | ResetGuiMessage
  | GuiModalMessage
  | GuiCloseModalMessage
  | GuiUpdateMessage
  | SceneNodeUpdateMessage
  | ThemeConfigurationMessage
  | GetRenderRequestMessage
  | GetRenderResponseMessage
  | FileTransferStartUpload
  | FileTransferStartDownload
  | FileTransferPart
  | FileTransferPartAck
  | ShareUrlRequest
  | ShareUrlUpdated
  | ShareUrlDisconnect
  | SetGuiPanelLabelMessage;
export type SceneNodeMessage =
  | CameraFrustumMessage
  | GlbMessage
  | FrameMessage
  | BatchedAxesMessage
  | GridMessage
  | LabelMessage
  | Gui3DMessage
  | PointCloudMessage
  | DirectionalLightMessage
  | AmbientLightMessage
  | HemisphereLightMessage
  | PointLightMessage
  | RectAreaLightMessage
  | SpotLightMessage
  | MeshMessage
  | SkinnedMeshMessage
  | BatchedMeshesMessage
  | BatchedGlbMessage
  | TransformControlsMessage
  | ImageMessage
  | LineSegmentsMessage
  | CatmullRomSplineMessage
  | CubicBezierSplineMessage
  | GaussianSplatsMessage;
export type GuiComponentMessage =
  | GuiFolderMessage
  | GuiMarkdownMessage
  | GuiHtmlMessage
  | GuiProgressBarMessage
  | GuiPlotlyMessage
  | GuiImageMessage
  | GuiTabGroupMessage
  | GuiButtonMessage
  | GuiUploadButtonMessage
  | GuiSliderMessage
  | GuiMultiSliderMessage
  | GuiNumberMessage
  | GuiRgbMessage
  | GuiRgbaMessage
  | GuiCheckboxMessage
  | GuiVector2Message
  | GuiVector3Message
  | GuiTextMessage
  | GuiDropdownMessage
  | GuiButtonGroupMessage;
const typeSetSceneNodeMessage = new Set([
  "CameraFrustumMessage",
  "GlbMessage",
  "FrameMessage",
  "BatchedAxesMessage",
  "GridMessage",
  "LabelMessage",
  "Gui3DMessage",
  "PointCloudMessage",
  "DirectionalLightMessage",
  "AmbientLightMessage",
  "HemisphereLightMessage",
  "PointLightMessage",
  "RectAreaLightMessage",
  "SpotLightMessage",
  "MeshMessage",
  "SkinnedMeshMessage",
  "BatchedMeshesMessage",
  "BatchedGlbMessage",
  "TransformControlsMessage",
  "ImageMessage",
  "LineSegmentsMessage",
  "CatmullRomSplineMessage",
  "CubicBezierSplineMessage",
  "GaussianSplatsMessage",
]);
export function isSceneNodeMessage(
  message: Message,
): message is SceneNodeMessage {
  return typeSetSceneNodeMessage.has(message.type);
}
const typeSetGuiComponentMessage = new Set([
  "GuiFolderMessage",
  "GuiMarkdownMessage",
  "GuiHtmlMessage",
  "GuiProgressBarMessage",
  "GuiPlotlyMessage",
  "GuiImageMessage",
  "GuiTabGroupMessage",
  "GuiButtonMessage",
  "GuiUploadButtonMessage",
  "GuiSliderMessage",
  "GuiMultiSliderMessage",
  "GuiNumberMessage",
  "GuiRgbMessage",
  "GuiRgbaMessage",
  "GuiCheckboxMessage",
  "GuiVector2Message",
  "GuiVector3Message",
  "GuiTextMessage",
  "GuiDropdownMessage",
  "GuiButtonGroupMessage",
]);
export function isGuiComponentMessage(
  message: Message,
): message is GuiComponentMessage {
  return typeSetGuiComponentMessage.has(message.type);
}
