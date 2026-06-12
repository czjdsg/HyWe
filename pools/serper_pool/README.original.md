# Serper API Pool

这是一个本地 Serper key 池代理。你把多个有额度的 key 放进池子里，业务代码统一请求本地代理，代理会自动挑选可用 key；遇到配额耗尽、失效或临时错误时，会自动切到下一个 key。

## 目录

- `server.py`: 启动本地代理服务
- `manage_keys.py`: 管理 key 池
- `serper_pool.db`: SQLite 持久化数据库，首次启动后自动创建

## 启动

```bash
cd /data/czj/serper_pool
python3 server.py
```

默认监听 `127.0.0.1:8765`。

如果你要后台常驻，直接用：

```bash
cd /data/czj/serper_pool
bash serper_pool_ctl.sh start
bash serper_pool_ctl.sh status
```

停止和重启：

```bash
bash serper_pool_ctl.sh stop
bash serper_pool_ctl.sh restart
```

后台日志默认写到 `logs/serper_pool.log`，pid 写到 `serper_pool.pid`。

可选环境变量：

```bash
export SERPER_POOL_HOST=127.0.0.1
export SERPER_POOL_PORT=8765
export SERPER_POOL_DB=/data/czj/serper_pool/serper_pool.db
export SERPER_POOL_REQUEST_TIMEOUT=20
export SERPER_POOL_COOLDOWN_SECONDS=60
```

## 管理 Key

新增单个 key：

```bash
cd /data/czj/serper_pool
python3 manage_keys.py add YOUR_SERPER_KEY --label main-1
```

一次新增多个 key：

```bash
cd /data/czj/serper_pool
python3 manage_keys.py add KEY_A KEY_B KEY_C --label-prefix batch
```

从文件导入：

```bash
cd /data/czj/serper_pool
python3 manage_keys.py import ./keys.txt
```

`keys.txt` 每行支持以下任一格式：

```text
YOUR_KEY
label-1,YOUR_KEY
label-2<TAB>YOUR_KEY
```

查看状态：

```bash
cd /data/czj/serper_pool
python3 manage_keys.py list
python3 manage_keys.py stats
```

禁用 / 启用 / 删除：

```bash
python3 manage_keys.py disable 1
python3 manage_keys.py enable 1
python3 manage_keys.py delete 1
```

## 业务侧接入

现有代码已经补了代理兼容。只要把 Serper 基础地址改到本地池子即可：

```bash
export SERPER_BASE_URL=http://127.0.0.1:8765
```

然后按原来的方式运行项目即可，不需要再给业务代码配置单个 Serper key。

## 调试接口

- `GET /healthz`: 查看池子状态
- `GET /admin/keys`: 查看已加载 key 的脱敏状态
- `POST /search`
- `POST /scholar`
- 其它 Serper POST 路径也会原样转发到 `https://google.serper.dev`

## 自动切换策略

- `200` 成功：记录成功并继续保留该 key
- `401/403` 且像无效 key：标记 `disabled`
- `429` 或响应里包含额度不足语义：标记 `exhausted`
- `5xx`、网络错误、临时限流：短暂 `cooldown` 后换下一个 key
- `400/404/422` 这类请求本身有问题：直接透传，不消耗轮换逻辑
