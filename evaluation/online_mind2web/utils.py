import base64
import io
from PIL import Image
from openai import (
    APIConnectionError,
    APIError,
    RateLimitError,
    AzureOpenAI,
    OpenAI
)
import os
import backoff

def encode_image(image):
    """Convert a PIL image to base64 string."""
    if image.mode == "RGBA":
        image = image.convert("RGB")
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_predication(response, mode):
    """Extract the prediction from the response."""
    if mode == "Autonomous_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "AgentTrek_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0
    elif mode == "WebVoyager_eval":
        if "FAILURE" in response:
            return 0
        else:
            return 1
    elif mode == "WebJudge_Online_Mind2Web_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0  
    elif mode == "WebJudge_general_eval":
        try:
            if "success" in response.lower().split('status:')[1]:
                return 1
            else:
                return 0
        except:
            return 0      
    else:
        raise ValueError(f"Unknown mode: {mode}")


class OpenaiEngine():
    def __init__(
        self,
        api_key=None,
        stop=[],
        rate_limit=-1,
        model=None,
        tokenizer=None,
        temperature=0,
        port=-1,
        endpoint_target_uri = "",
        **kwargs,
    ) -> None:
        """Init an OpenAI GPT/Codex engine

        Args:
            api_key (_type_, optional): Auth key from OpenAI. Defaults to None.
            stop (list, optional): Tokens indicate stop of sequence. Defaults to ["\n"].
            rate_limit (int, optional): Max number of requests per minute. Defaults to -1.
            model (_type_, optional): Model family. Defaults to None.
        """
        assert (
                os.getenv("OPENAI_API_KEY", api_key) is not None
        ), "must pass on the api_key or set OPENAI_API_KEY in the environment"
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", api_key)
        if isinstance(api_key, str):
            self.api_keys = [api_key]
        elif isinstance(api_key, list):
            self.api_keys = api_key
        else:
            raise ValueError("api_key must be a string or list")
        self.stop = stop
        self.temperature = temperature
        self.model = model
        # convert rate limit to minmum request interval
        self.request_interval = 0 if rate_limit == -1 else 60.0 / rate_limit
        self.next_avil_time = [0] * len(self.api_keys)
        self.client = OpenAI(
                        api_key=api_key,
                    )

    def log_error(details):
        print(f"Retrying in {details['wait']:0.1f} seconds due to {details['exception']}")

    @backoff.on_exception(
        backoff.expo,
        (APIError, RateLimitError, APIConnectionError),
        max_tries=3,
        on_backoff=log_error
    )
    def generate(self, messages, max_new_tokens=512, temperature=0, model=None, **kwargs):
        model = model if model else self.model
        # 推理模型(o4-mini 等)开关:JUDGE_MIN_COMPLETION_TOKENS>0 时改传 max_completion_tokens
        # 且不传 temperature。o 系列的输出预算**含推理 token**,关键点抽取和逐图打分默认只给 512,
        # 实测中转站会把 max_tokens 静默当成 max_completion_tokens,推理把额度吃光后返回空串、
        # finish_reason=length、**不报错** —— 判分就静默变成 0 或丢关键点。未设置时行为与原来完全一致。
        min_comp = int(os.environ.get("JUDGE_MIN_COMPLETION_TOKENS", "0") or 0)
        if min_comp > 0:
            response = self.client.chat.completions.create(
                model=model if model else self.model,
                messages=messages,
                max_completion_tokens=max(max_new_tokens, min_comp),
                **kwargs,
            )
        else:
            response = self.client.chat.completions.create(
                model=model if model else self.model,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs,
            )
        outs = [choice.message.content for choice in response.choices]
        for choice in response.choices:
            if not (choice.message.content or "").strip():
                print(f"[webjudge] 空回复 finish_reason={choice.finish_reason} model={model}", flush=True)
        return outs
    

# ---------------------------------------------------------------------------
# OSS 图片传输(适配:9000+ 张截图逐张走 base64 会把请求撑爆、也慢)
#
# 原版把每张截图 base64 内联进请求体。判 300 题需要给判官送约 9.3k 张图,
# 内联会让每个请求都带上百 KB~MB 的 payload,网关容易超时。
# 改成:PNG -> JPEG -> 上传 OSS -> 只传签名 URL。
#
# publish_bytes 内部按 sha256 内容寻址且带缓存,所以同一张图重复上传是免费的
# (断点续跑时尤其重要)。未配置 OSS 环境变量时自动退回 base64,不影响原有用法。
# ---------------------------------------------------------------------------
import threading as _threading

_INFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "inference")
_OSS_URL_CACHE: dict[str, str] = {}
_OSS_URL_LOCK = _threading.Lock()
_JPEG_QUALITY = int(os.environ.get("WEBJUDGE_JPEG_QUALITY", "85"))


def _url_transport_ready() -> bool:
    """HYBRID_IMAGE_TRANSPORT=url: images are published to HYBRID_IMAGE_PUBLISH_DIR,
    which you serve over HTTPS at HYBRID_IMAGE_URL_BASE (see inference/hybrid_runtime.py)."""
    return (os.environ.get("HYBRID_IMAGE_TRANSPORT", "").strip().lower() == "url"
            and os.environ.get("HYBRID_IMAGE_URL_BASE", "").strip()
            and os.environ.get("HYBRID_IMAGE_PUBLISH_DIR", "").strip())


def image_part(image_path, detail="high"):
    """返回 OpenAI 兼容的 image_url 片段。

    传输方式:
      1. HYBRID_IMAGE_TRANSPORT=url —— 发布到自建 HTTPS 图床, 发 URL
      2. 否则 base64 内联
    图片一律先转 JPEG:全页截图 PNG 单张约 196KB,JPEG q85 降到 84KB。
    """
    with _OSS_URL_LOCK:
        hit = _OSS_URL_CACHE.get(image_path)
    if hit:
        return {"type": "image_url", "image_url": {"url": hit, "detail": detail}}

    jpeg_bytes = None
    try:
        img = Image.open(image_path)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        jpeg_bytes = buf.getvalue()
    except Exception as exc:
        print(f"[img] 读图失败 {image_path}: {type(exc).__name__}: {str(exc)[:80]}")

    if jpeg_bytes and _url_transport_ready():
        try:
            import sys as _sys
            _sys.path.insert(0, _INFERENCE_DIR)
            from hybrid_runtime import _image_publish_config, _publish_image_bytes
            public_base, publish_dir = _image_publish_config()
            url = _publish_image_bytes(jpeg_bytes, "image/jpeg", public_base, publish_dir)
            with _OSS_URL_LOCK:
                _OSS_URL_CACHE[image_path] = url
            return {"type": "image_url", "image_url": {"url": url, "detail": detail}}
        except Exception as exc:
            print(f"[img] URL 发布失败,退回 base64: {type(exc).__name__}: {str(exc)[:100]}")

    # base64 用上面**同一份** JPEG(quality=_JPEG_QUALITY)。原来这里另走 encode_image(PIL 默认 q75),
    # 换传输方式就等于换了判官看到的画质(URL 路 q85),跨判官/跨传输的对比会混进画质差异。
    if jpeg_bytes:
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    else:
        b64 = encode_image(Image.open(image_path))
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}}
