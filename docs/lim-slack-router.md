# LIM Slack Product Intake Router

Hermes can watch a configured Slack channel for LIM product intake posts without requiring an `@hermes` mention. The router is event-driven from the Slack gateway and runs before normal mention routing, so ordinary chatter in the target channel is ignored without starting a full Hermes agent turn.

## Configuration

Defaults are safe: the router is disabled globally, product intake defaults to dry-run, and Toast publish is not exposed.

```yaml
lim_slack_router:
  enabled: false
  product_intake:
    enabled: true
    channel_ids:
      - C1234567890
    debounce_seconds: 30
    dry_run: true
    auto_write: false
    react_only_success: false
    reply_on_success: true
    use_agent_worker: true
```

To test in Slack, set `enabled: true`, keep `dry_run: true`, and add the product intake channel ID. To allow write mode later, set `dry_run: false` and `auto_write: true`.

## What It Processes

The first supported route is Toast product intake:

- A product/package image plus a human-written price, such as `$5.99`, `5.99`, `$5:99`, or `Update this to 6.99`.
- Threaded posts where the image and price arrive in separate messages.
- Price-update language with product context.

The router silently ignores messages outside configured channels, bot messages, normal chatter, and messages already marked with a terminal reaction.

## Slack Reactions

- `eyes`: processing started.
- `white_check_mark`: item added to Toast admin, not published.
- `money_with_wings`: price updated in Toast admin, not published.
- `receipt`: existing Toast item enriched, not duplicated.
- `new`: separate item created because barcode differed.
- `warning`: needs review or dry-run result.
- `x`: failed.

The router removes `eyes` after it adds a terminal reaction when possible.

## Dry Run

Dry run does not call Toast write tools. It fetches thread context and images, extracts the human-written price, runs available image analysis, and replies with what it would do:

```text
:test_tube: Dry run: would create or update Toast item, not publish:
Baraka Whole Mini Okra 14 oz - $3.99 - UPC 822514227021
```

## Write Mode

Write mode is guarded by both flags:

```yaml
dry_run: false
auto_write: true
```

The default worker path dispatches qualifying product-intake threads into the normal Hermes Slack agent pipeline with all thread images attached and product-intake rules injected. This keeps the router cheap for channel noise while giving real product work the same image reasoning and Toast skills as an explicit `@Hermes` request.

The direct fallback write path is intentionally narrow and never publishes. It can update the price of an existing Toast item when the package barcode resolves to a Toast item. If the barcode does not resolve, or creation/enrichment would require category or duplicate confirmation, the fallback marks the thread as needs review instead of guessing.

Toast config publishing is not available through this router.

## Barcode Rule

Barcode is identity. If a readable package barcode differs from an existing Toast row's barcode/SKU, the existing barcode is not overwritten. That product is treated as a different item.

## Idempotency And Reprocessing

Slack reactions are the visible state. A tiny local JSONL cache prevents duplicate writes after Slack redelivery or process restarts:

```text
~/.hermes/lim/slack-product-router/processed.jsonl
```

To reprocess a thread, remove the terminal Slack reaction and remove the matching `channel_id` plus `thread_ts` line from the cache.

## Notes

- Do not put the intake channel in `free_response_channels`; that routes casual channel chatter through the full Hermes agent pipeline.
- Inventory tracking remains off unless explicit receiving/count context is added in the thread.
- Vendor is not required and should not block product intake.
- Toast admin changes remain pending manual publish.
