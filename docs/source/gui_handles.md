# GUI Handles

A handle is created for each GUI element that is added. Handles can be used to
read and write state.

When a GUI element is added to a server (for example, via
:func:`ViserServer.add_gui_text()`), state is synchronized between all connected
clients. When a GUI element is added to a client (for example, via
:func:`ClientHandle.add_gui_text()`), state is local to a specific client.

<!-- prettier-ignore-start -->

.. autoapiclass:: viser.GuiHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.GuiButtonHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.GuiButtonGroupHandle
   :members:
   :undoc-members:
   :inherited-members:

.. autoapiclass:: viser.GuiDropdownHandle
   :members:
   :undoc-members:
   :inherited-members:

<!-- prettier-ignore-end -->
