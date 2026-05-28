use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::common::ResponsesApiRequest;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::requests::Compression;
use crate::sse::anthropic::spawn_anthropic_response_stream;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde::Serialize;
use serde_json::Value;
use std::sync::Arc;
use tracing::instrument;

pub struct AnthropicMessagesClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
}

impl<T: HttpTransport> AnthropicMessagesClient<T> {
    pub fn new(transport: T, provider: Provider, auth: SharedAuthProvider) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
        }
    }

    #[instrument(
        name = "anthropic.messages.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "anthropic_http",
            http.method = "POST",
            api.path = "messages"
        )
    )]
    pub async fn stream_request(
        &self,
        request: ResponsesApiRequest,
        extra_headers: HeaderMap,
        compression: Compression,
    ) -> Result<ResponseStream, ApiError> {
        let body = AnthropicMessagesRequest::from_responses_request(request)?;
        let body = serde_json::to_value(body).map_err(|e| {
            ApiError::Stream(format!("failed to encode anthropic messages request: {e}"))
        })?;

        let request_compression = match compression {
            Compression::None => RequestCompression::None,
            Compression::Zstd => RequestCompression::Zstd,
        };

        let stream_response = self
            .session
            .stream_with(Method::POST, "messages", extra_headers, Some(body), |req| {
                req.headers.insert(
                    http::header::ACCEPT,
                    HeaderValue::from_static("text/event-stream"),
                );
                req.compression = request_compression;
            })
            .await?;

        Ok(spawn_anthropic_response_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
        ))
    }
}

#[derive(Debug, Serialize)]
struct AnthropicMessagesRequest {
    model: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    system: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<AnthropicToolChoice>,
    stream: bool,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<AnthropicOutputConfig>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicToolChoice {
    Auto,
}

#[derive(Debug, Serialize)]
struct AnthropicOutputConfig {
    effort: AnthropicEffort,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
enum AnthropicEffort {
    Minimal,
    Low,
    Medium,
    High,
}

impl AnthropicMessagesRequest {
    fn from_responses_request(request: ResponsesApiRequest) -> Result<Self, ApiError> {
        let mut messages = Vec::new();
        let mut pending_tool_results = Vec::new();

        for item in request.input {
            match item {
                codex_protocol::models::ResponseItem::Message { role, content, .. } => {
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    let role = if role == "assistant" {
                        "assistant"
                    } else {
                        "user"
                    };
                    let text = content
                        .into_iter()
                        .filter_map(|item| match item {
                            codex_protocol::models::ContentItem::InputText { text }
                            | codex_protocol::models::ContentItem::OutputText { text } => {
                                Some(text)
                            }
                            codex_protocol::models::ContentItem::InputImage { .. } => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");
                    if !text.is_empty() {
                        messages.push(AnthropicMessage {
                            role: role.to_string(),
                            content: vec![AnthropicContent::Text { text }],
                        });
                    }
                }
                codex_protocol::models::ResponseItem::FunctionCall {
                    name,
                    arguments,
                    call_id,
                    ..
                } => {
                    flush_tool_results(&mut messages, &mut pending_tool_results);
                    let input =
                        serde_json::from_str(&arguments).unwrap_or(Value::String(arguments));
                    messages.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: vec![AnthropicContent::ToolUse {
                            id: call_id,
                            name,
                            input,
                        }],
                    });
                }
                codex_protocol::models::ResponseItem::FunctionCallOutput { call_id, output } => {
                    pending_tool_results.push(AnthropicContent::ToolResult {
                        tool_use_id: call_id,
                        content: output.to_string(),
                    });
                }
                _ => {}
            }
        }
        flush_tool_results(&mut messages, &mut pending_tool_results);

        if messages.is_empty() {
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: vec![AnthropicContent::Text {
                    text: String::new(),
                }],
            });
        }

        let tools = anthropic_tools_from_responses_tools(request.tools);
        let tool_choice = (!tools.is_empty()).then_some(AnthropicToolChoice::Auto);
        Ok(Self {
            model: request.model,
            system: request.instructions,
            messages,
            tools,
            tool_choice,
            stream: true,
            max_tokens: 4096,
            output_config: request.reasoning.and_then(anthropic_output_config),
        })
    }
}

fn anthropic_output_config(reasoning: crate::common::Reasoning) -> Option<AnthropicOutputConfig> {
    let effort = reasoning.effort?;
    let effort = match effort {
        ReasoningEffortConfig::None => return None,
        ReasoningEffortConfig::Minimal => AnthropicEffort::Minimal,
        ReasoningEffortConfig::Low => AnthropicEffort::Low,
        ReasoningEffortConfig::Medium => AnthropicEffort::Medium,
        ReasoningEffortConfig::High | ReasoningEffortConfig::XHigh => AnthropicEffort::High,
    };

    Some(AnthropicOutputConfig { effort })
}

fn flush_tool_results(
    messages: &mut Vec<AnthropicMessage>,
    pending_tool_results: &mut Vec<AnthropicContent>,
) {
    if pending_tool_results.is_empty() {
        return;
    }

    messages.push(AnthropicMessage {
        role: "user".to_string(),
        content: std::mem::take(pending_tool_results),
    });
}

fn anthropic_tools_from_responses_tools(tools: Vec<Value>) -> Vec<AnthropicTool> {
    tools
        .into_iter()
        .filter_map(|tool| {
            let object = tool.as_object()?;
            let type_name = object.get("type").and_then(Value::as_str);
            if type_name != Some("function") {
                return None;
            }
            let name = object.get("name")?.as_str()?.to_string();
            let description = object
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let input_schema = object.get("parameters").cloned().unwrap_or(Value::Object(
                serde_json::Map::from_iter([("type".to_string(), Value::String("object".into()))]),
            ));
            Some(AnthropicTool {
                name,
                description,
                input_schema,
            })
        })
        .collect()
}
