use crate::common::ResponseEvent;
use crate::common::ResponseStream;
use crate::error::ApiError;
use crate::telemetry::SseTelemetry;
use codex_client::ByteStream;
use codex_client::StreamResponse;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::TokenUsage;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use serde::Deserialize;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio::time::Instant;
use tokio::time::timeout;
use tracing::debug;
use tracing::trace;

const REQUEST_ID_HEADER: &str = "request-id";

pub fn spawn_anthropic_response_stream(
    stream_response: StreamResponse,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) -> ResponseStream {
    let upstream_request_id = stream_response
        .headers
        .get(REQUEST_ID_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::to_string);
    let (tx_event, rx_event) = mpsc::channel::<Result<ResponseEvent, ApiError>>(1600);
    tokio::spawn(async move {
        process_anthropic_sse(stream_response.bytes, tx_event, idle_timeout, telemetry).await;
    });

    ResponseStream {
        rx_event,
        upstream_request_id,
    }
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    message: Option<AnthropicMessageStart>,
    #[serde(default)]
    content_block: Option<AnthropicContentBlock>,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
    #[serde(default)]
    index: Option<usize>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
    #[serde(default)]
    error: Option<AnthropicError>,
}

#[derive(Debug, Deserialize)]
struct AnthropicMessageStart {
    id: String,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: i64,
    #[serde(default)]
    output_tokens: i64,
    #[serde(default)]
    cache_read_input_tokens: i64,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    #[serde(default)]
    message: String,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(default, rename = "type")]
    kind: Option<String>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    partial_json: Option<String>,
    #[serde(default)]
    stop_reason: Option<String>,
}

#[derive(Debug)]
enum AnthropicBlock {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
    Other,
}

#[derive(Debug, Default)]
struct AnthropicStreamState {
    response_id: Option<String>,
    blocks: HashMap<usize, AnthropicBlock>,
    input_tokens: i64,
    output_tokens: i64,
    cached_input_tokens: i64,
}

pub async fn process_anthropic_sse(
    stream: ByteStream,
    tx_event: mpsc::Sender<Result<ResponseEvent, ApiError>>,
    idle_timeout: Duration,
    telemetry: Option<Arc<dyn SseTelemetry>>,
) {
    let mut stream = stream.eventsource();
    let mut state = AnthropicStreamState::default();

    loop {
        let start = Instant::now();
        let response = timeout(idle_timeout, stream.next()).await;
        if let Some(t) = telemetry.as_ref() {
            t.on_sse_poll(&response, start.elapsed());
        }
        let sse = match response {
            Ok(Some(Ok(sse))) => sse,
            Ok(Some(Err(e))) => {
                debug!("Anthropic SSE error: {e:#}");
                let _ = tx_event.send(Err(ApiError::Stream(e.to_string()))).await;
                return;
            }
            Ok(None) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream(
                        "stream closed before message_stop".into(),
                    )))
                    .await;
                return;
            }
            Err(_) => {
                let _ = tx_event
                    .send(Err(ApiError::Stream("idle timeout waiting for SSE".into())))
                    .await;
                return;
            }
        };

        trace!("Anthropic SSE event: {}", &sse.data);
        let event: AnthropicStreamEvent = match serde_json::from_str(&sse.data) {
            Ok(event) => event,
            Err(e) => {
                debug!(
                    "failed to parse Anthropic SSE event: {e}, data: {}",
                    &sse.data
                );
                continue;
            }
        };

        match event.kind.as_str() {
            "message_start" => {
                if let Some(message) = event.message {
                    state.response_id = Some(message.id);
                    if let Some(usage) = message.usage {
                        state.input_tokens = usage.input_tokens;
                        state.cached_input_tokens = usage.cache_read_input_tokens;
                        state.output_tokens = usage.output_tokens;
                    }
                }
                if tx_event.send(Ok(ResponseEvent::Created)).await.is_err() {
                    return;
                }
            }
            "content_block_start" => {
                let index = event.index.unwrap_or(0);
                let block = match event.content_block.unwrap_or(AnthropicContentBlock::Other) {
                    AnthropicContentBlock::Text { text } => AnthropicBlock::Text(text),
                    AnthropicContentBlock::ToolUse { id, name, input } => AnthropicBlock::ToolUse {
                        id,
                        name,
                        input_json: if input.is_null()
                            || input.as_object().is_some_and(serde_json::Map::is_empty)
                        {
                            String::new()
                        } else {
                            input.to_string()
                        },
                    },
                    AnthropicContentBlock::Other => AnthropicBlock::Other,
                };
                state.blocks.insert(index, block);
            }
            "content_block_delta" => {
                let index = event.index.unwrap_or(0);
                match (state.blocks.get_mut(&index), event.delta) {
                    (Some(AnthropicBlock::Text(text)), Some(delta))
                        if delta.kind.as_deref() == Some("text_delta") =>
                    {
                        let delta = delta.text.unwrap_or_default();
                        text.push_str(&delta);
                        if tx_event
                            .send(Ok(ResponseEvent::OutputTextDelta(delta)))
                            .await
                            .is_err()
                        {
                            return;
                        }
                    }
                    (Some(AnthropicBlock::ToolUse { input_json, .. }), Some(delta))
                        if delta.kind.as_deref() == Some("input_json_delta") =>
                    {
                        let partial_json = delta.partial_json.unwrap_or_default();
                        input_json.push_str(&partial_json);
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = event.index.unwrap_or(0);
                if let Some(item) = state.blocks.remove(&index).and_then(block_to_response_item)
                    && tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(item)))
                        .await
                        .is_err()
                {
                    return;
                }
            }
            "message_delta" => {
                if let Some(usage) = event.usage {
                    state.output_tokens = usage.output_tokens;
                }
                if let Some(delta) = event.delta
                    && delta.stop_reason.as_deref() == Some("tool_use")
                {
                    trace!("Anthropic stream stopped for tool use");
                }
            }
            "message_stop" => {
                let response_id = state
                    .response_id
                    .take()
                    .unwrap_or_else(|| "anthropic-message".to_string());
                let token_usage = Some(TokenUsage {
                    input_tokens: state.input_tokens,
                    cached_input_tokens: state.cached_input_tokens,
                    output_tokens: state.output_tokens,
                    reasoning_output_tokens: 0,
                    total_tokens: state.input_tokens + state.output_tokens,
                });
                let _ = tx_event
                    .send(Ok(ResponseEvent::Completed {
                        response_id,
                        token_usage,
                        end_turn: Some(true),
                    }))
                    .await;
                return;
            }
            "error" => {
                let message = event
                    .error
                    .map(|error| error.message)
                    .filter(|message| !message.is_empty())
                    .unwrap_or_else(|| "Anthropic stream error".to_string());
                let _ = tx_event.send(Err(ApiError::Stream(message))).await;
                return;
            }
            other => trace!("unhandled Anthropic event: {other}"),
        }
    }
}

fn block_to_response_item(block: AnthropicBlock) -> Option<ResponseItem> {
    match block {
        AnthropicBlock::Text(text) => Some(ResponseItem::Message {
            id: None,
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText { text }],
            phase: None,
        }),
        AnthropicBlock::ToolUse {
            id,
            name,
            input_json,
        } => Some(ResponseItem::FunctionCall {
            id: None,
            name,
            namespace: None,
            arguments: if input_json.trim().is_empty() {
                "{}".to_string()
            } else {
                input_json
            },
            call_id: id,
        }),
        AnthropicBlock::Other => None,
    }
}
