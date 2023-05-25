# Development

In this note, we outline current practices, tools, and workflows for `viser`
development. We assume that the repository is cloned to `~/viser`.

There's a lot of improvement that can be made here. PRs for tooling are welcome.

## Python install

We recommend an editable install for Python development, ideally in a virtual
environment (eg via conda).

```bash
# Install package.
cd ~/viser
pip install -e .

# Install example dependencies.
pip install -r examples/requirements.txt
```

_Alternatively_, `poetry` can also be used:

```Bash
# Install poetry.
curl -sSL https://install.python-poetry.org | python3 -

# Install package.
cd ~/viser
poetry install

# Install example dependencies.
poetry run pip install -r examples/requirements.txt
```

After installation, any of the example scripts (`~/viser/examples`) should be
runnable. A few of them require downloading assets, which can be done via the
scripts in `~/viser/examples/assets`.

**Linting, formatting, type-checking.**

First install developer tools:

```bash
# Using pip.
pip install -e .[dev]

# Using poetry.
poetry install -E dev
```

It would be hard to write unit tests for `viser`. We rely on static typing for
robustness. From the `~/viser` root directory, you can run:

- `pyright . && mypy .`

For formatting and linting:

- `black . && ruff --fix .`

## Message updates

The `viser` frontend and backend communicate via a shared set of message
definitions:

- On the server, these are defined as Python dataclasses in
  `~/viser/viser/_messages.py`.
- On the client, these are defined as TypeScript interfaces in
  `~/viser/viser/client/src/WebsocketMessages.tsx`.

Note that there is a 1:1 correspondence between the dataclasses message types
and the TypeScript ones.

The TypeScript definitions should not be manually modified. Instead, changes
should be made in Python and synchronized via the `sync_message_defs.py` script:

```
cd ~/viser
python sync_message_defs.py
```

## Client development

For client development, we can start by launching a relevant Python script. The
examples are a good place to start:

```
cd ~/viser/examples
python 05_camera_commands.py
```

When a `viser` script is launched, two URLs will be printed:

- An HTTP URL, like `http://localhost:8080`, which can be used to open a
  _pre-built_ version of the React frontend.
- A websocket URL, like `ws://localhost:8080`, which client applications can
  connect to.

For client-side development, the HTTP URL to the pre-built frontend should be
ignored. Instead, we want to launch a development version of the frontend, which
will reflect changes we make to the client source files
(`~/viser/viser/client/src`). This requires a few more steps.

**Installing dependencies.**

1. [Install nodejs.](https://nodejs.dev/en/download/package-manager)
2. [Install yarn.](https://yarnpkg.com/getting-started/install)
3. Install dependencies.
   ```
   cd ~/viser/viser/client
   yarn install
   ```

**Launching client.**

To launch the client, we can then simply run

```
cd ~/viser/viser/client
yarn start
```

from the `viser/viser/client` directory. After opening the client in a web
browser, the websocket server address typically needs to be updated in the
"Server" tab.

**Building the client.**

When changes to the client are finished, we can wrap up by (a) building the
client and (b) committing the built viewer. This is what the printed HTTP server
points to, and allows use of the viewer without installing all of the web
dependencies.

```
cd ~/viser/viser/client
yarn build
git add build
```

This manual build step and version control of the `build/` directory are
unideal. Open to suggestions on how to improve this workflow.

**Formatting.**

We use [prettier](https://prettier.io/docs/en/install.html). This can be run via
one of:

- `prettier -w .`
- `npx prettier -w .`

from `~/viser/client`.
