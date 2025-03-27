Development
==========

In this note, we outline current practices, tools, and workflows for ``viser``
development. We assume that the repository is cloned to ``~/viser``.

Python install
-------------

We recommend an editable install for Python development, ideally in a virtual
environment (eg via conda).

.. code-block:: bash

   # Install package.
   cd ~/viser
   pip install -e .

   # Install example dependencies.
   pip install -e .[examples]

After installation, any of the example scripts (``~/viser/examples``) should be
runnable. A few of them require downloading assets, which can be done via the
scripts in ``~/viser/examples/assets``.

**Linting, formatting, type-checking.**

First, install developer tools:

.. code-block:: bash

   # Using pip.
   pip install -e .[dev]
   pre-commit install

For code quality, rely primarily on ``pyright`` and ``ruff``:

.. code-block:: bash

   # Check static types.
   pyright

   # Lint and format.
   ruff check --fix .
   ruff format .

Message updates
--------------

The ``viser`` frontend and backend communicate via a shared set of message
definitions:

- On the server, these are defined as Python dataclasses in
  ``~/viser/src/viser/_messages.py``.
- On the client, these are defined as TypeScript interfaces in
  ``~/viser/src/viser/client/src/WebsocketMessages.tsx``.

Note that there is a 1:1 correspondence between the dataclasses message types
and the TypeScript ones.

The TypeScript definitions should not be manually modified. Instead, changes
should be made in Python and synchronized via the ``sync_message_defs.py`` script:

.. code-block:: bash

   cd ~/viser
   python sync_message_defs.py

Client development
----------------

For client development, we can start by launching a relevant Python script. The
examples are a good place to start:

.. code-block:: bash

   cd ~/viser/examples
   python 05_camera_commands.py

When a ``viser`` script is launched, two URLs will be printed:

- An HTTP URL, like ``http://localhost:8080``, which can be used to open a
  *pre-built* version of the React frontend.
- A websocket URL, like ``ws://localhost:8080``, which client applications can
  connect to.

If changes to the client source files are detected on startup, ``viser`` will
re-build the client automatically. This is okay for quick changes, but for
faster iteration we can also launch a development version of the frontend, which
will reflect changes we make to the client source files
(``~/viser/src/viser/client/src``) without a full build. This requires a few more
steps.

**Installing dependencies.**

1. `Install nodejs. <https://nodejs.dev/en/download/package-manager>`_
2. `Install yarn. <https://yarnpkg.com/getting-started/install>`_
3. Install dependencies.
   
   .. code-block:: bash
   
      cd ~/viser/src/viser/client
      yarn install

**Launching client.**

To launch the client, we can run:

.. code-block:: bash

   cd ~/viser/src/viser/client
   yarn start

from the ``viser/src/viser/client`` directory. After opening the client in a web
browser, the websocket server address typically needs to be updated in the
"Server" tab.

**Formatting.**

We use `prettier <https://prettier.io/docs/en/install.html>`_. This can be run via
one of:

- ``prettier -w .``
- ``npx prettier -w .``

from ``~/viser/src/viser/client``.