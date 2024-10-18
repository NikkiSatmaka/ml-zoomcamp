# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY .python-version pyproject.toml README.md uv.lock /app/
COPY src/ml_zoomcamp/__init__.py /app/src/ml_zoomcamp/__init__.py
COPY src/ml_zoomcamp/predict.py /app/src/ml_zoomcamp/predict.py

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY model/ /app/model/

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind", "0.0.0.0:9696", "ml_zoomcamp.predict:app" ]
