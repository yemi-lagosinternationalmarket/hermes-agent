from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from gateway.lim_slack_router import (
    LimSlackRouter,
    LimSlackRouterConfig,
    ProcessedCache,
    ProductIntakeConfig,
    classify_product_intake_event,
    decide_barcode_identity,
    parse_human_price,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("$5.99", 5.99),
        ("5.99", 5.99),
        ("$5:99", 5.99),
        ("Update this to 6.99", 6.99),
        ("actually change it back to 6.99", 6.99),
        ("barcode 822514227021", None),
    ],
)
def test_parse_human_price(text, expected):
    assert parse_human_price(text) == expected


def test_classifier_image_plus_price_is_product_candidate():
    event = {
        "channel": "C_PRODUCTS",
        "text": "$5:99",
        "files": [{"mimetype": "image/jpeg", "id": "F1"}],
    }

    intent = classify_product_intake_event(event, target_channel_ids=["C_PRODUCTS"])

    assert intent.name == "product_intake_create"
    assert intent.price == 5.99


def test_classifier_text_only_chatter_ignored():
    event = {"channel": "C_PRODUCTS", "text": "okay thanks"}

    intent = classify_product_intake_event(event, target_channel_ids=["C_PRODUCTS"])

    assert intent.name == "ignore"


def test_classifier_update_image_price_is_update_candidate():
    event = {
        "channel": "C_PRODUCTS",
        "text": "Update this to 6.99",
        "files": [{"mimetype": "image/png", "id": "F1"}],
    }

    intent = classify_product_intake_event(event, target_channel_ids=["C_PRODUCTS"])

    assert intent.name == "product_intake_update_existing"
    assert intent.price == 6.99


def test_classifier_image_only_needs_review():
    event = {
        "channel": "C_PRODUCTS",
        "text": "",
        "files": [{"mimetype": "image/jpeg", "id": "F1"}],
    }

    intent = classify_product_intake_event(event, target_channel_ids=["C_PRODUCTS"])

    assert intent.name == "needs_review"
    assert "price missing" in intent.reason


def test_classifier_bot_message_ignored():
    event = {
        "channel": "C_PRODUCTS",
        "text": "$5.99",
        "bot_id": "B123",
        "files": [{"mimetype": "image/jpeg", "id": "F1"}],
    }

    intent = classify_product_intake_event(event, target_channel_ids=["C_PRODUCTS"])

    assert intent.name == "ignore"


def test_classifier_outside_configured_channel_ignored():
    event = {
        "channel": "C_OTHER",
        "text": "$5.99",
        "files": [{"mimetype": "image/jpeg", "id": "F1"}],
    }

    intent = classify_product_intake_event(event, target_channel_ids=["C_PRODUCTS"])

    assert intent.name == "ignore"


def test_barcode_identity_updates_existing_barcode_match():
    decision = decide_barcode_identity(
        barcode="822514227021",
        barcode_lookup_hit={"id": "P1"},
        name_size_hit=None,
    )

    assert decision == "update_barcode_match"


def test_barcode_identity_enriches_exact_match_without_sku():
    decision = decide_barcode_identity(
        barcode="822514227021",
        barcode_lookup_hit=None,
        name_size_hit={"id": "P1", "skus": []},
    )

    assert decision == "enrich_barcode_less_match"


def test_barcode_identity_creates_separate_item_for_different_sku():
    decision = decide_barcode_identity(
        barcode="822514227021",
        barcode_lookup_hit=None,
        name_size_hit={"id": "P1", "skus": ["000000000000"]},
    )

    assert decision == "create_separate_different_barcode"


def test_processed_cache_prevents_duplicate_thread_processing(tmp_path: Path):
    cache = ProcessedCache(tmp_path / "processed.jsonl")

    assert not cache.contains("C_PRODUCTS", "111.1")

    cache.append({"channel_id": "C_PRODUCTS", "thread_ts": "111.1", "status": "needs_review"})

    assert cache.contains("C_PRODUCTS", "111.1")


@pytest.mark.asyncio
async def test_router_does_not_consume_explicit_mentions(tmp_path: Path):
    config = LimSlackRouterConfig(
        enabled=True,
        product_intake=ProductIntakeConfig(
            enabled=True,
            channel_ids=["C_PRODUCTS"],
            debounce_seconds=0,
        ),
    )
    router = LimSlackRouter(config, processed_cache=ProcessedCache(tmp_path / "processed.jsonl"))

    consumed = await router.inspect_slack_message(
        {
            "channel": "C_PRODUCTS",
            "ts": "111.1",
            "text": "<@U_BOT> what is this?",
            "files": [{"id": "F1", "mimetype": "image/jpeg"}],
        },
        slack_client=AsyncMock(),
        adapter=AsyncMock(),
        bot_user_id="U_BOT",
    )

    assert consumed is False


@pytest.mark.asyncio
async def test_router_consumes_non_hermes_mentions_without_processing(tmp_path: Path):
    config = LimSlackRouterConfig(
        enabled=True,
        product_intake=ProductIntakeConfig(
            enabled=True,
            channel_ids=["C_PRODUCTS"],
            debounce_seconds=0,
        ),
    )
    router = LimSlackRouter(config, processed_cache=ProcessedCache(tmp_path / "processed.jsonl"))

    consumed = await router.inspect_slack_message(
        {
            "channel": "C_PRODUCTS",
            "ts": "111.1",
            "text": "<@UOTHER> add this $5.99",
            "files": [{"id": "F1", "mimetype": "image/jpeg"}],
        },
        slack_client=AsyncMock(),
        adapter=AsyncMock(),
        bot_user_id="U_BOT",
    )

    assert consumed is True
    assert router._tasks == {}


@pytest.mark.asyncio
async def test_router_dry_run_reacts_replies_and_records_processed(tmp_path: Path):
    config = LimSlackRouterConfig(
        enabled=True,
        product_intake=ProductIntakeConfig(
            enabled=True,
            channel_ids=["C_PRODUCTS"],
            dry_run=True,
            auto_write=False,
            debounce_seconds=0,
            use_agent_worker=False,
        ),
    )
    cache = ProcessedCache(tmp_path / "processed.jsonl")
    router = LimSlackRouter(config, processed_cache=cache, sleep=AsyncMock())
    slack_client = AsyncMock()
    slack_client.conversations_replies = AsyncMock(
        return_value={
            "messages": [
                {
                    "ts": "111.1",
                    "user": "U1",
                    "text": "$3.99",
                    "files": [{"id": "F1", "mimetype": "image/jpeg"}],
                }
            ]
        }
    )
    adapter = AsyncMock()
    adapter._cache_image_files_from_slack_message = AsyncMock(return_value=(["/tmp/product.jpg"], ["image/jpeg"], []))
    adapter._enrich_message_with_vision = AsyncMock(return_value="Baraka Whole Mini Okra 14 oz\nUPC 822514227021")

    result = await router.process_thread(
        channel_id="C_PRODUCTS",
        thread_ts="111.1",
        trigger_ts="111.1",
        slack_client=slack_client,
        adapter=adapter,
    )

    assert result.action == "dry_run"
    assert result.product["price"] == 3.99
    assert cache.contains("C_PRODUCTS", "111.1")
    adapter._add_reaction.assert_any_await("C_PRODUCTS", "111.1", "eyes")
    adapter._add_reaction.assert_any_await("C_PRODUCTS", "111.1", "warning")
    adapter._remove_reaction.assert_awaited_once_with("C_PRODUCTS", "111.1", "eyes")
    slack_client.chat_postMessage.assert_awaited_once()
    assert "Dry run" in slack_client.chat_postMessage.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_router_agent_worker_dispatches_normal_message_pipeline(tmp_path: Path):
    config = LimSlackRouterConfig(
        enabled=True,
        product_intake=ProductIntakeConfig(
            enabled=True,
            channel_ids=["C_PRODUCTS"],
            dry_run=False,
            auto_write=True,
            debounce_seconds=0,
            use_agent_worker=True,
        ),
    )
    cache = ProcessedCache(tmp_path / "processed.jsonl")
    router = LimSlackRouter(config, processed_cache=cache, sleep=AsyncMock())
    slack_client = AsyncMock()
    slack_client.conversations_replies = AsyncMock(
        return_value={
            "messages": [
                {
                    "ts": "111.1",
                    "user": "U1",
                    "text": "$7:99",
                    "files": [{"id": "F1", "mimetype": "image/jpeg"}],
                },
                {
                    "ts": "111.2",
                    "user": "U1",
                    "text": "",
                    "files": [{"id": "F2", "mimetype": "image/jpeg"}],
                },
            ]
        }
    )

    class FakeAdapter:
        config = type("Config", (), {"extra": {}})()

        def __init__(self):
            self._add_reaction = AsyncMock()
            self._remove_reaction = AsyncMock()
            self._cache_image_files_from_slack_message = AsyncMock(
                side_effect=[
                    (["/tmp/front.jpg"], ["image/jpeg"], []),
                    (["/tmp/barcode.jpg"], ["image/jpeg"], []),
                ]
            )
            self._resolve_user_name = AsyncMock(return_value="yemi")
            self.handle_message = AsyncMock()

        def build_source(self, **kwargs):
            return kwargs

    adapter = FakeAdapter()

    result = await router.process_thread(
        channel_id="C_PRODUCTS",
        thread_ts="111.1",
        trigger_ts="111.1",
        slack_client=slack_client,
        adapter=adapter,
    )

    assert result.action == "agent_worker"
    adapter.handle_message.assert_awaited_once()
    event = adapter.handle_message.await_args.args[0]
    assert event.media_urls == ["/tmp/front.jpg", "/tmp/barcode.jpg"]
    assert "Toast product intake worker" in event.text
    assert "catalog-photo-quick-adds" in event.auto_skill
    assert cache.contains("C_PRODUCTS", "111.1")
    adapter._add_reaction.assert_awaited_once_with("C_PRODUCTS", "111.1", "eyes")
    adapter._remove_reaction.assert_awaited_once_with("C_PRODUCTS", "111.1", "eyes")
