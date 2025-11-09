"""
Simple optimization entrypoint to match docker-compose 'optimize' service.
Delegates to CLI 'tune'.
"""
from src import cli as _cli  # ensure package path

if __name__ == "__main__":
    # Defer to CLI 'tune' subcommand, letting env vars drive symbols/TF etc.
    # Equivalent to: python -m src.cli tune
    raise SystemExit(_cli.main(["tune"]))

