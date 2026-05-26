"""LIM Slack operational event router.

The router is intentionally small and event-driven. It inspects raw Slack
events before the normal Hermes mention gate so high-volume operations
channels can get lightweight automation without turning every message into a
full agent conversation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


TERMINAL_REACTIONS = {
    "white_check_mark",
    "money_with_wings",
    "receipt",
    "new",
    "warning",
    "x",
}

PROCESSING_REACTION = "eyes"

PRODUCT_IMAGE_PROMPT = (
    "You are reading product package photos for Toast catalog intake. "
    "Inspect every visible package face carefully, especially barcode/UPC/EAN labels. "
    "Transcribe exact barcode digits when visible, even if they appear on a second image. "
    "Also extract product name, brand, size/weight/count, flavor/variant, and any useful "
    "front/back package text. If a barcode is partially blocked or unreadable, say so. "
    "Return concise plain text with fields: Product, Brand, Size, Barcode, Notes."
)

PRODUCT_AGENT_PROMPT = """You are LIM's Toast product intake worker.

You are processing one Slack thread from a configured product intake channel. This message was routed automatically because the thread looked like product intake; the staff member did not explicitly @mention Hermes.

Use the attached product package images and the Slack thread text below. Operate like the normal Hermes Slack agent with full image reasoning.

Rules:
- Auto-write to Toast only if complete and confident.
- Do not publish Toast config changes. Never call toast_config_publish_all.
- Price must be human-written in the Slack text/thread. Do not invent a price and do not use a price seen only in the image.
- Barcode is product identity. If the photographed barcode differs from an existing Toast row's barcode/SKU, treat it as a different item.
- If barcode lookup misses but exact name/size exists, only update that existing item if it has no barcode/SKU.
- Infer category from the product/package. Use Toast sibling/category lookup only to obtain internal Toast category IDs.
- Do not set inventory tracking unless explicit stock/count/receiving context is provided.
- Do not assign vendor unless explicitly provided.
- If ambiguous, do not write; reply with a short needs-review reason.
- Always reply in this Slack thread after a create/update/enrich/needs-review outcome with what happened.
- Successful wording must say Toast admin was changed and not published.

Follow the catalog-photo-quick-adds, catalog-manage, and toast-mcp SOPs.

Slack thread text:
{thread_text}
"""


@dataclass
class ProductIntakeConfig:
    enabled: bool = True
    channel_ids: List[str] = field(default_factory=list)
    dry_run: bool = True
    auto_write: bool = False
    debounce_seconds: float = 30.0
    react_only_success: bool = False
    reply_on_success: bool = True
    use_agent_worker: bool = True


@dataclass
class LimSlackRouterConfig:
    enabled: bool = False
    product_intake: ProductIntakeConfig = field(default_factory=ProductIntakeConfig)


@dataclass
class ThreadMessage:
    ts: str
    user: str = ""
    text: str = ""
    files: List[Dict[str, Any]] = field(default_factory=list)
    reactions: List[str] = field(default_factory=list)


@dataclass
class ThreadContext:
    channel_id: str
    thread_ts: str
    trigger_ts: str
    messages: List[ThreadMessage]
    image_paths: List[str] = field(default_factory=list)
    image_file_ids: List[str] = field(default_factory=list)

    @property
    def thread_text(self) -> str:
        return "\n".join(m.text for m in self.messages if m.text).strip()

    @property
    def all_reactions(self) -> set[str]:
        return {reaction for msg in self.messages for reaction in msg.reactions}


@dataclass
class Intent:
    name: str
    reason: str = ""
    price: Optional[float] = None


@dataclass
class ProductDecision:
    status: str
    action: str
    reason: str
    product: Dict[str, Any] = field(default_factory=dict)
    toast: Dict[str, Any] = field(default_factory=lambda: {"published": False, "verified": False})


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def load_router_config(config: Dict[str, Any]) -> LimSlackRouterConfig:
    raw = config.get("lim_slack_router") if isinstance(config, dict) else None
    if not isinstance(raw, dict):
        return LimSlackRouterConfig()
    intake_raw = raw.get("product_intake") if isinstance(raw.get("product_intake"), dict) else {}
    return LimSlackRouterConfig(
        enabled=_as_bool(raw.get("enabled"), False),
        product_intake=ProductIntakeConfig(
            enabled=_as_bool(intake_raw.get("enabled"), True),
            channel_ids=[str(c) for c in (intake_raw.get("channel_ids") or []) if str(c).strip()],
            dry_run=_as_bool(intake_raw.get("dry_run"), True),
            auto_write=_as_bool(intake_raw.get("auto_write"), False),
            debounce_seconds=float(intake_raw.get("debounce_seconds", 30) or 0),
            react_only_success=_as_bool(intake_raw.get("react_only_success"), False),
            reply_on_success=_as_bool(intake_raw.get("reply_on_success"), True),
            use_agent_worker=_as_bool(intake_raw.get("use_agent_worker"), True),
        ),
    )


_PRICE_RE = re.compile(r"(?<!\d)\$?\s*(\d{1,4})[.:](\d{2})(?!\d)")


def parse_human_price(text: str) -> Optional[float]:
    """Return the last human-written decimal price in text, or None."""
    if not text:
        return None
    matches = list(_PRICE_RE.finditer(text))
    if not matches:
        return None
    dollars, cents = matches[-1].groups()
    return float(f"{int(dollars)}.{cents}")


def _file_is_image(file_obj: Dict[str, Any]) -> bool:
    mimetype = str(file_obj.get("mimetype") or "")
    filetype = str(file_obj.get("filetype") or "").lower()
    name = str(file_obj.get("name") or "").lower()
    if mimetype.startswith("image/"):
        return True
    if filetype in {"jpg", "jpeg", "png", "webp", "gif"}:
        return True
    return name.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif"))


def _has_image_event(event: Dict[str, Any]) -> bool:
    return any(_file_is_image(f) for f in event.get("files") or [])


def _reactions_from_message(message: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for reaction in message.get("reactions") or []:
        name = reaction.get("name")
        if name:
            names.append(str(name))
    return names


def has_terminal_reaction(message_or_context: Dict[str, Any] | ThreadContext) -> bool:
    if isinstance(message_or_context, ThreadContext):
        return bool(message_or_context.all_reactions & TERMINAL_REACTIONS)
    return bool(set(_reactions_from_message(message_or_context)) & TERMINAL_REACTIONS)


_ADD_WORDS = re.compile(r"\b(add|new|create|put this|ring this|price this)\b", re.I)
_UPDATE_WORDS = re.compile(r"\b(update|change|set|actually|back to|edit)\b", re.I)
_CHATTER_WORDS = re.compile(r"\b(thanks|thank you|lol|ok|okay|yes|no|sounds good)\b", re.I)
_USER_MENTION_RE = re.compile(r"<@[A-Z0-9]+(?:\|[^>]+)?>")


def classify_product_intake_event(event: Dict[str, Any], *, target_channel_ids: Iterable[str]) -> Intent:
    channel_id = str(event.get("channel") or event.get("channel_id") or "")
    if channel_id not in set(target_channel_ids):
        return Intent("ignore", "outside configured channel")
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return Intent("ignore", "bot message")
    if has_terminal_reaction(event):
        return Intent("ignore", "terminal reaction present")

    text = str(event.get("text") or "")
    has_image = _has_image_event(event)
    price = parse_human_price(text)
    has_action = bool(_ADD_WORDS.search(text) or _UPDATE_WORDS.search(text))

    if has_image and price is not None:
        if _UPDATE_WORDS.search(text):
            return Intent("product_intake_update_existing", "image plus update language and price", price)
        return Intent("product_intake_create", "image plus price", price)
    if has_image and has_action and price is None:
        return Intent("needs_review", "price missing")
    if has_image:
        return Intent("needs_review", "product image present but price missing")
    if price is not None and has_action:
        return Intent("price_update", "price update language without image", price)
    if _CHATTER_WORDS.search(text) or text.strip():
        return Intent("ignore", "ordinary chatter")
    return Intent("ignore", "empty/non-actionable")


def decide_barcode_identity(
    *,
    barcode: Optional[str],
    barcode_lookup_hit: Optional[Dict[str, Any]],
    name_size_hit: Optional[Dict[str, Any]],
) -> str:
    """Apply LIM's barcode identity rule."""
    if barcode and barcode_lookup_hit:
        return "update_barcode_match"
    if barcode and name_size_hit:
        skus = [str(s) for s in (name_size_hit.get("skus") or []) if str(s).strip()]
        if not skus:
            return "enrich_barcode_less_match"
        if barcode not in skus:
            return "create_separate_different_barcode"
        return "update_barcode_match"
    if barcode:
        return "create_new_barcode"
    if name_size_hit:
        return "needs_review_no_barcode_existing_match"
    return "needs_review_no_barcode"


class ProcessedCache:
    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or get_hermes_home() / "lim" / "slack-product-router" / "processed.jsonl"
        self._seen: set[str] = set()
        self._loaded = False

    def _key(self, channel_id: str, thread_ts: str) -> str:
        return f"{channel_id}:{thread_ts}"

    def load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                channel_id = str(row.get("channel_id") or "")
                thread_ts = str(row.get("thread_ts") or "")
                if channel_id and thread_ts:
                    self._seen.add(self._key(channel_id, thread_ts))
        except Exception:
            logger.warning("LIM Slack router: failed to read processed cache", exc_info=True)

    def contains(self, channel_id: str, thread_ts: str) -> bool:
        self.load()
        return self._key(channel_id, thread_ts) in self._seen

    def append(self, row: Dict[str, Any]) -> None:
        self.load()
        channel_id = str(row.get("channel_id") or "")
        thread_ts = str(row.get("thread_ts") or "")
        if not channel_id or not thread_ts:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"ts": datetime.now(timezone.utc).isoformat(), **row}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        self._seen.add(self._key(channel_id, thread_ts))


class ProductIntakeWorker:
    def __init__(
        self,
        *,
        dry_run: bool,
        auto_write: bool,
        vision_analyzer: Optional[Callable[[str, List[str]], Awaitable[str]]] = None,
        toast_client: Optional[Any] = None,
    ) -> None:
        self.dry_run = dry_run
        self.auto_write = auto_write
        self.vision_analyzer = vision_analyzer
        self.toast_client = toast_client

    async def process(self, context: ThreadContext) -> ProductDecision:
        price = parse_human_price(context.thread_text)
        if price is None:
            return ProductDecision("needs_review", "none", "price missing")
        if not context.image_paths:
            return ProductDecision("needs_review", "none", "product image missing")

        vision_text = await self._analyze_images(context.image_paths)

        barcode = extract_barcode(context.thread_text + "\n" + vision_text)
        product_name = extract_product_name(vision_text) or "product from package image"
        product = {
            "name": product_name,
            "barcode": barcode,
            "price": price,
        }

        if self.dry_run or not self.auto_write:
            return ProductDecision(
                status="needs_review",
                action="dry_run",
                reason="dry run: would create or update Toast item, not publish",
                product=product,
                toast={"published": False, "verified": False},
            )

        if not self.toast_client:
            return ProductDecision(
                status="needs_review",
                action="none",
                reason="auto_write enabled but Toast tool client is unavailable",
                product=product,
                toast={"published": False, "verified": False},
            )
        if not barcode:
            return ProductDecision("needs_review", "none", "barcode missing or unreadable", product, {"published": False, "verified": False})
        if product_name == "product from package image":
            return ProductDecision("needs_review", "none", "product name could not be read confidently", product, {"published": False, "verified": False})

        barcode_hit = await self.toast_client.search_barcode(barcode)
        identity = decide_barcode_identity(barcode=barcode, barcode_lookup_hit=barcode_hit, name_size_hit=None)
        if identity == "update_barcode_match":
            product_id = str(barcode_hit.get("product_id") or barcode_hit.get("id") or "")
            if not product_id:
                return ProductDecision("needs_review", "none", "Toast barcode match did not include a product id", product, {"published": False, "verified": False})
            await self.toast_client.update_price(product_id=product_id, price=price)
            verified = await self.toast_client.search_barcode(barcode)
            product["product_id"] = product_id
            return ProductDecision(
                status="updated",
                action="price_update",
                reason="barcode matched existing Toast item; price updated, not published",
                product=product,
                toast={"published": False, "verified": bool(verified)},
            )

        return ProductDecision(
            status="needs_review",
            action="none",
            reason="barcode did not resolve; create/enrich requires category and duplicate confirmation",
            product=product,
            toast={"published": False, "verified": False},
        )

    async def _analyze_images(self, image_paths: List[str]) -> str:
        targeted_text = await analyze_product_images(image_paths)
        if targeted_text:
            return targeted_text
        if self.vision_analyzer:
            try:
                return await self.vision_analyzer(PRODUCT_IMAGE_PROMPT, image_paths)
            except Exception:
                logger.warning("LIM Slack router: vision analysis failed", exc_info=True)
        return ""


async def analyze_product_images(image_paths: List[str]) -> str:
    if not image_paths:
        return ""
    try:
        from tools.vision_tools import vision_analyze_tool
    except Exception:
        logger.warning("LIM Slack router: vision tool unavailable", exc_info=True)
        return ""

    parts: List[str] = []
    for index, path in enumerate(image_paths, start=1):
        try:
            result_json = await vision_analyze_tool(
                image_url=path,
                user_prompt=PRODUCT_IMAGE_PROMPT,
            )
            result = json.loads(result_json)
            if result.get("success") and result.get("analysis"):
                parts.append(f"Image {index}:\n{result['analysis']}")
            else:
                logger.warning("LIM Slack router: product vision analysis failed for image %s: %s", index, result)
        except Exception:
            logger.warning("LIM Slack router: product vision analysis error for image %s", index, exc_info=True)
    return "\n\n".join(parts)


class ToastCatalogClient:
    """Narrow Toast MCP wrapper for safe v1 auto-write operations.

    The client deliberately exposes no publish method.
    """

    def _call_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        from tools.registry import registry

        entry = registry.get_entry(name)
        if not entry:
            raise RuntimeError(f"Toast MCP tool is unavailable: {name}")
        raw = entry.handler(args)
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            return {"result": raw}
        if isinstance(parsed, dict) and parsed.get("error"):
            raise RuntimeError(str(parsed["error"]))
        return parsed if isinstance(parsed, dict) else {"result": parsed}

    async def search_barcode(self, barcode: str) -> Optional[Dict[str, Any]]:
        result = await asyncio.to_thread(
            self._call_tool,
            "mcp_toast_web_toast_search_suggest_barcode",
            {"barcode": barcode},
        )
        payload = result.get("result", result)
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                return None
        if isinstance(payload, list):
            return payload[0] if payload else None
        if isinstance(payload, dict):
            for key in ("item", "product", "match", "result"):
                value = payload.get(key)
                if isinstance(value, dict):
                    return value
                if isinstance(value, list) and value:
                    return value[0] if isinstance(value[0], dict) else None
            return payload if payload else None
        return None

    async def update_price(self, *, product_id: str, price: float) -> Dict[str, Any]:
        return await asyncio.to_thread(
            self._call_tool,
            "mcp_toast_web_toast_items_update_price",
            {"product_id": product_id, "price": price},
        )


_BARCODE_RE = re.compile(r"\b(\d{8}|\d{12,14})\b")


def extract_barcode(text: str) -> Optional[str]:
    match = _BARCODE_RE.search(text or "")
    return match.group(1) if match else None


def extract_product_name(vision_text: str) -> Optional[str]:
    for line in (vision_text or "").splitlines():
        cleaned = line.strip(" -:\t")
        if 4 <= len(cleaned) <= 90 and not _BARCODE_RE.search(cleaned) and "image" not in cleaned.lower():
            return cleaned
    return None


def _merge_skills(existing: Optional[str | List[str]], required: List[str]) -> List[str]:
    merged: List[str] = []
    if isinstance(existing, str):
        merged.extend([existing])
    elif isinstance(existing, list):
        merged.extend([str(item) for item in existing if str(item).strip()])
    for skill in required:
        if skill not in merged:
            merged.append(skill)
    return merged


class LimSlackRouter:
    def __init__(
        self,
        config: LimSlackRouterConfig,
        *,
        processed_cache: Optional[ProcessedCache] = None,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self.config = config
        self.processed_cache = processed_cache or ProcessedCache()
        self.sleep = sleep
        self._tasks: Dict[str, asyncio.Task] = {}

    def _thread_key(self, channel_id: str, thread_ts: str) -> str:
        return f"{channel_id}:{thread_ts}"

    async def inspect_slack_message(
        self,
        event: Dict[str, Any],
        *,
        slack_client: Any,
        adapter: Any,
        bot_user_id: Optional[str] = None,
    ) -> bool:
        intake = self.config.product_intake
        if not self.config.enabled or not intake.enabled:
            return False
        channel_id = str(event.get("channel") or "")
        if channel_id not in set(intake.channel_ids):
            return False

        text = str(event.get("text") or "")
        explicitly_mentioned = bool(bot_user_id and f"<@{bot_user_id}>" in text)
        if explicitly_mentioned:
            return False
        if _USER_MENTION_RE.search(text):
            return True

        intent = classify_product_intake_event(event, target_channel_ids=intake.channel_ids)
        if intent.name == "ignore":
            return True

        thread_ts = str(event.get("thread_ts") or event.get("ts") or "")
        if not thread_ts:
            return True
        if self.processed_cache.contains(channel_id, thread_ts):
            return True

        key = self._thread_key(channel_id, thread_ts)
        previous = self._tasks.get(key)
        if previous and not previous.done():
            previous.cancel()
        self._tasks[key] = asyncio.create_task(
            self._debounced_process(
                channel_id=channel_id,
                thread_ts=thread_ts,
                trigger_ts=str(event.get("ts") or thread_ts),
                slack_client=slack_client,
                adapter=adapter,
            )
        )
        return True

    async def _debounced_process(self, *, channel_id: str, thread_ts: str, trigger_ts: str, slack_client: Any, adapter: Any) -> None:
        delay = max(0.0, self.config.product_intake.debounce_seconds)
        if delay:
            await self.sleep(delay)
        try:
            await self.process_thread(channel_id=channel_id, thread_ts=thread_ts, trigger_ts=trigger_ts, slack_client=slack_client, adapter=adapter)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("LIM Slack router: failed processing %s/%s", channel_id, thread_ts)
            await self._safe_react(adapter, channel_id, trigger_ts, "x")
            await self._safe_reply(slack_client, channel_id, thread_ts, ":x: I tried to process this product intake but hit an internal error. No item was changed.")

    async def process_thread(self, *, channel_id: str, thread_ts: str, trigger_ts: str, slack_client: Any, adapter: Any) -> ProductDecision:
        if self.processed_cache.contains(channel_id, thread_ts):
            return ProductDecision("ignored", "none", "already processed")
        context = await self.build_thread_context(channel_id=channel_id, thread_ts=thread_ts, trigger_ts=trigger_ts, slack_client=slack_client, adapter=adapter)
        if has_terminal_reaction(context):
            return ProductDecision("ignored", "none", "terminal reaction present")

        await self._safe_react(adapter, channel_id, trigger_ts, PROCESSING_REACTION)
        if self.config.product_intake.use_agent_worker:
            result = await self._dispatch_agent_worker(context=context, adapter=adapter)
            await self._safe_remove_reaction(adapter, channel_id, trigger_ts, PROCESSING_REACTION)
            if result.status not in {"ignored"}:
                self.processed_cache.append({
                    "channel_id": channel_id,
                    "thread_ts": thread_ts,
                    "message_ts": trigger_ts,
                    "status": result.status,
                    "product_id": result.product.get("product_id"),
                    "barcode": result.product.get("barcode"),
                    "price": result.product.get("price"),
                })
            return result

        worker = ProductIntakeWorker(
            dry_run=self.config.product_intake.dry_run,
            auto_write=self.config.product_intake.auto_write,
            vision_analyzer=getattr(adapter, "_enrich_message_with_vision", None),
            toast_client=ToastCatalogClient() if self.config.product_intake.auto_write and not self.config.product_intake.dry_run else None,
        )
        result = await worker.process(context)
        terminal = reaction_for_status(result)
        if terminal:
            await self._safe_react(adapter, channel_id, trigger_ts, terminal)
        await self._safe_remove_reaction(adapter, channel_id, trigger_ts, PROCESSING_REACTION)
        reply = reply_for_result(result, dry_run=self.config.product_intake.dry_run)
        if reply and (self.config.product_intake.reply_on_success or result.status in {"needs_review", "failed"}):
            await self._safe_reply(slack_client, channel_id, thread_ts, reply)
        if result.status not in {"ignored"}:
            self.processed_cache.append({
                "channel_id": channel_id,
                "thread_ts": thread_ts,
                "message_ts": trigger_ts,
                "status": result.status,
                "product_id": result.product.get("product_id"),
                "barcode": result.product.get("barcode"),
                "price": result.product.get("price"),
            })
        return result

    async def _dispatch_agent_worker(self, *, context: ThreadContext, adapter: Any) -> ProductDecision:
        from gateway.platforms.base import MessageEvent, MessageType, resolve_channel_prompt, resolve_channel_skills

        primary = context.messages[0] if context.messages else ThreadMessage(ts=context.trigger_ts)
        user_id = primary.user or ""
        try:
            user_name = await adapter._resolve_user_name(user_id, chat_id=context.channel_id) if user_id else ""
        except Exception:
            user_name = user_id

        source = adapter.build_source(
            chat_id=context.channel_id,
            chat_name=context.channel_id,
            chat_type="group",
            user_id=user_id,
            user_name=user_name,
            thread_id=context.thread_ts,
        )

        base_prompt = resolve_channel_prompt(adapter.config.extra, context.channel_id, None)
        router_prompt = PRODUCT_AGENT_PROMPT.format(thread_text=context.thread_text or "(no text)")
        channel_prompt = "\n\n".join(part for part in (base_prompt, router_prompt) if part)
        configured_skills = resolve_channel_skills(adapter.config.extra, context.channel_id, None)
        auto_skill = _merge_skills(
            configured_skills,
            ["catalog-photo-quick-adds", "catalog-manage", "toast-mcp"],
        )

        event = MessageEvent(
            text=router_prompt,
            message_type=MessageType.PHOTO if context.image_paths else MessageType.TEXT,
            source=source,
            raw_message={"lim_slack_router": True, "thread_ts": context.thread_ts},
            message_id=context.trigger_ts,
            media_urls=context.image_paths,
            media_types=["image/jpeg"] * len(context.image_paths),
            reply_to_message_id=context.thread_ts,
            channel_prompt=channel_prompt,
            auto_skill=auto_skill,
        )
        await adapter.handle_message(event)
        return ProductDecision(
            status="agent_dispatched",
            action="agent_worker",
            reason="dispatched to normal Hermes Slack agent pipeline",
            product={"price": parse_human_price(context.thread_text)},
            toast={"published": False, "verified": False},
        )

    async def build_thread_context(self, *, channel_id: str, thread_ts: str, trigger_ts: str, slack_client: Any, adapter: Any) -> ThreadContext:
        result = await slack_client.conversations_replies(channel=channel_id, ts=thread_ts, inclusive=True, limit=50)
        raw_messages = result.get("messages") or []
        messages: List[ThreadMessage] = []
        image_paths: List[str] = []
        image_file_ids: List[str] = []
        for raw in raw_messages:
            files = list(raw.get("files") or [])
            media_urls, media_types, _notices = await adapter._cache_image_files_from_slack_message(
                files=files,
                channel_id=channel_id,
                team_id=str(raw.get("team") or raw.get("team_id") or ""),
            )
            image_paths.extend(media_urls)
            for f in files:
                if _file_is_image(f) and f.get("id"):
                    image_file_ids.append(str(f["id"]))
            messages.append(ThreadMessage(
                ts=str(raw.get("ts") or ""),
                user=str(raw.get("user") or raw.get("username") or ""),
                text=str(raw.get("text") or "").strip(),
                files=files,
                reactions=_reactions_from_message(raw),
            ))
        return ThreadContext(channel_id=channel_id, thread_ts=thread_ts, trigger_ts=trigger_ts, messages=messages, image_paths=image_paths, image_file_ids=image_file_ids)

    async def _safe_react(self, adapter: Any, channel_id: str, ts: str, reaction: str) -> None:
        try:
            await adapter._add_reaction(channel_id, ts, reaction)
        except Exception:
            logger.debug("LIM Slack router: add reaction failed", exc_info=True)

    async def _safe_remove_reaction(self, adapter: Any, channel_id: str, ts: str, reaction: str) -> None:
        try:
            await adapter._remove_reaction(channel_id, ts, reaction)
        except Exception:
            logger.debug("LIM Slack router: remove reaction failed", exc_info=True)

    async def _safe_reply(self, slack_client: Any, channel_id: str, thread_ts: str, text: str) -> None:
        try:
            await slack_client.chat_postMessage(channel=channel_id, thread_ts=thread_ts, text=text)
        except Exception:
            logger.warning("LIM Slack router: Slack reply failed", exc_info=True)


def reaction_for_status(result: ProductDecision) -> Optional[str]:
    if result.action == "dry_run":
        return "warning"
    if result.status == "added":
        return "white_check_mark"
    if result.status == "updated":
        return "money_with_wings"
    if result.status == "enriched":
        return "receipt"
    if result.status == "needs_review":
        return "warning"
    if result.status == "failed":
        return "x"
    return None


def reply_for_result(result: ProductDecision, *, dry_run: bool) -> str:
    if result.status == "ignored":
        return ""
    product = result.product or {}
    name = product.get("name") or "item"
    price = product.get("price")
    barcode = product.get("barcode")
    detail = f"{name}"
    if price is not None:
        detail += f" - ${float(price):.2f}"
    if barcode:
        detail += f" - UPC {barcode}"
    if result.action == "dry_run" or dry_run:
        return f":test_tube: Dry run: would create or update Toast item, not publish:\n{detail}"
    if result.status == "needs_review":
        return f":warning: Needs review: {result.reason}."
    if result.status in {"added", "updated", "enriched"}:
        return f":white_check_mark: Updated Toast, not published:\n{detail}"
    if result.status == "failed":
        return f":x: {result.reason}"
    return ""
