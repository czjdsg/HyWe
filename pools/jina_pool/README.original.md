# Jina API Pool

这是一个本地 Jina Reader key 池代理。你把多个有额度的 Jina key 放进池子里，业务代码统一访问本地 Reader 代理，代理会自动挑选可用 key；遇到额度耗尽、失效或临时错误时，会自动切到下一个 key。

## 目录

- `server.py`: 启动本地代理服务
- `manage_keys.py`: 管理 key 池
- `jina_pool.db`: SQLite 持久化数据库，首次启动后自动创建
- `jina_pool_ctl.sh`: 后台常驻管理脚本

## 启动

```bash
cd /data/czj/jina_pool
python3 server.py
```

默认监听 `127.0.0.1:8766`。

可选环境变量：

```bash
export JINA_POOL_HOST=127.0.0.1
export JINA_POOL_PORT=8766
export JINA_POOL_DB=/data/czj/jina_pool/jina_pool.db
export JINA_POOL_REQUEST_TIMEOUT=60
export JINA_POOL_COOLDOWN_SECONDS=60
export JINA_POOL_ALLOW_ANONYMOUS=false
```

## 后台常驻

```bash
cd /data/czj/jina_pool
bash jina_pool_ctl.sh start
bash jina_pool_ctl.sh status
```

停止和重启：

```bash
bash jina_pool_ctl.sh stop
bash jina_pool_ctl.sh restart
```

后台日志默认写到 `logs/jina_pool.log`。

## 管理 Key

新增单个 key：

```bash
cd /data/czj/jina_pool
python3 manage_keys.py add YOUR_JINA_KEY --label main-1
```

一次新增多个 key：

```bash
cd /data/czj/jina_pool
python3 manage_keys.py add KEY_A KEY_B KEY_C --label-prefix batch
```

从文件导入：

```bash
cd /data/czj/jina_pool
python3 manage_keys.py import ./keys.txt
```

`keys.txt` 每行支持：

```text
YOUR_KEY
label-1,YOUR_KEY
label-2<TAB>YOUR_KEY
```

查看状态：

```bash
python3 manage_keys.py list
python3 manage_keys.py stats
```

## 业务侧接入

现有 `visit` 工具已经补了代理兼容。只要把 Reader 基础地址改到本地池子即可：

```bash
export JINA_READER_BASE_URL=http://127.0.0.1:8766
```

然后按原来的方式运行项目即可。业务代码里原本的 `JINA_API_KEY` / `JINA_API_KEYS` 可以继续留着，也可以不配；如果走本地池子，最终用的是池子里轮换出的 key。

## 调试接口

- `GET /healthz`: 查看池子状态
- `GET /admin/keys`: 查看已加载 key 的脱敏状态
- 其它 `GET /<target-url>` 或 `POST /<target-url>`：原样转发到 `https://r.jina.ai/<target-url>`

例如：

```bash
curl "http://127.0.0.1:8766/https://example.com"
```

## 自动切换策略

- `200` 成功：记录成功并继续保留该 key
- `401/403` 且像无效 key：标记 `disabled`
- `402` 或响应里包含额度不足语义：标记 `exhausted`
- `429`、`5xx`、网络错误：短暂 `cooldown` 后换下一个 key
- `400/404/422` 这类请求本身有问题：直接透传，不触发轮换
