FROM svizor/zoomcamp-model:3.11.5-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=apps/svizor-bank-marketing/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY uv.lock /app
COPY apps/svizor-bank-marketing /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 9696

ENTRYPOINT []

CMD [ "fastapi", "run", "--host", "0.0.0.0", "--port", "9696", "app.py" ]
