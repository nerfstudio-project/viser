# GUI Handles

A handle is created for each GUI element that is added. Handles can be used to
read and write state.

When a GUI element is added to a server (for example, via
:func:`viser.ViserServer.add_gui_text()`), state is synchronized between all
connected clients. When a GUI element is added to a client (for example, via
:func:`viser.ClientHandle.add_gui_text()`), state is local to a specific client.

<!-- prettier-ignore-start -->

.. autoclass:: viser.GuiInputHandle()

.. autoclass:: viser.GuiButtonHandle()

.. autoclass:: viser.GuiButtonGroupHandle()

.. autoclass:: viser.GuiDropdownHandle()

.. autoclass:: viser.GuiFolderHandle()

.. autoclass:: viser.GuiMarkdownHandle()

.. autoclass:: viser.GuiTabGroupHandle()

.. autoclass:: viser.GuiTabHandle()

<!-- prettier-ignore-end -->
