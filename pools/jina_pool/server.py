from jina_pool.config import PoolSettings
from jina_pool.proxy import JinaPoolApplication


def main() -> int:
    app = JinaPoolApplication(PoolSettings.from_env())
    app.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
