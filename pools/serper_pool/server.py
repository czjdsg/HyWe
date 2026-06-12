from serper_pool.config import PoolSettings
from serper_pool.proxy import SerperPoolApplication


def main() -> int:
    app = SerperPoolApplication(PoolSettings.from_env())
    app.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
