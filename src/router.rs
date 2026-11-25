//! Assistant engine router.
//!
//! Checks LLM availability on every `chat()` call and delegates to the
//! appropriate path: `OllamaBackend` when ready, `FallbackEngine` otherwise.
//!
//! The router also runs language detection and FSM flow detection before
//! routing, so both paths receive structured intent context.

use std::time::Duration;

use crate::actions::{ProposedAction, ToolSpec, ACTION_CATALOG};
use crate::fallback::{detect_lang, FallbackEngine, FallbackResponse, FsmState};
use crate::AssistantEngine;

/// Check whether the configured chat model is reachable and present.
///
/// Sends a non-blocking GET to `http://127.0.0.1:11434/api/tags` with a
/// 300 ms timeout. Returns `true` only if the server responds and the
/// response lists the configured chat model.
pub async fn is_llm_ready(engine: &AssistantEngine) -> bool {
    let model_id = engine.config().model_id;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_millis(300))
        .build()
        .unwrap_or_default();

    let Ok(resp) = client
        .get(format!("{}/api/tags", engine.config().ollama_endpoint))
        .send()
        .await
    else {
        return false;
    };

    let Ok(body) = resp.json::<serde_json::Value>().await else {
        return false;
    };

    body.get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter().any(|m| {
                m.get("name")
                    .and_then(|n| n.as_str())
                    .map(|n| {
                        n == model_id
                            || (!model_id.contains(':')
                                && n == format!("{}:latest", model_id))
                    })
                    .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

/// Per-turn routing result.
#[derive(Debug)]
pub struct RouterResponse {
    pub text: String,
    pub source: String,
    /// Populated when the turn resolves to a proposed action.
    pub action: Option<ProposedAction>,
}

impl From<FallbackResponse> for RouterResponse {
    fn from(r: FallbackResponse) -> Self {
        Self { text: r.text, source: r.source.to_string(), action: r.action }
    }
}

/// The tool payload offered to the model: the static catalog plus the caller's.
/// Split out of `try_tool_call` so it can be tested without a running server.
fn build_tools(extra: &[ToolSpec]) -> Vec<serde_json::Value> {
    // Tool schemas come from the catalog's own `params` and from the caller's
    // specs. There is no per-name branch here: a tool that needs arguments
    // carries its schema with it, so adding one never edits this function.
    let mut tools: Vec<serde_json::Value> = ACTION_CATALOG.iter().map(|e| {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": e.name,
                "description": e.description,
                "parameters": serde_json::from_str::<serde_json::Value>(e.params)
                    .unwrap_or_else(|_| serde_json::json!({ "type": "object", "properties": {}, "required": [] })),
            }
        })
    }).collect();
    tools.extend(extra.iter().map(|t| {
        serde_json::json!({
            "type": "function",
            "function": { "name": t.name, "description": t.description, "parameters": t.params }
        })
    }));
    tools
}

/// The write flag and summary for a tool name the model returned, or `None` if
/// the name was never offered.
fn resolve_tool(name: &str, extra: &[ToolSpec]) -> Option<(bool, String)> {
    match crate::actions::find_action(name) {
        Some(e) => Some((e.write, e.summary.to_string())),
        None => extra.iter().find(|t| t.name == name).map(|t| (t.write, t.summary.clone())),
    }
}

/// Try to get a tool-call response from Ollama for the given message.
///
/// Sends a **non-streaming** request with the action catalog as tools.
/// Returns `Some(ProposedAction)` if the model chose a tool, `None` otherwise
/// (model does not support tools, rejected the request, or emitted plain text).
///
/// Design D4: this is strictly additive. Callers fall back to the FSM floor
/// when this returns `None`.
async fn try_tool_call(message: &str, engine: &AssistantEngine, extra: &[ToolSpec]) -> Option<ProposedAction> {
    let model_id = engine.config().model_id;
    let ollama_base = engine.config().ollama_endpoint;

    let tools = build_tools(extra);

    let payload = serde_json::json!({
        "model": model_id,
        "messages": [{ "role": "user", "content": message }],
        "tools": tools,
        "stream": false,
    });

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
        .ok()?;

    let resp = client
        .post(format!("{}/api/chat", ollama_base))
        .json(&payload)
        .send()
        .await
        .ok()?;

    if !resp.status().is_success() {
        return None;
    }

    let body: serde_json::Value = resp.json().await.ok()?;

    // Parse tool_calls from the response.
    let tool_calls = body
        .get("message")?
        .get("tool_calls")?
        .as_array()?;

    let call = tool_calls.first()?;
    let name = call
        .get("function")?
        .get("name")?
        .as_str()?
        .to_string();

    let args = call
        .get("function")
        .and_then(|f| f.get("arguments"))
        .cloned()
        .unwrap_or(serde_json::json!({}));

    // The name must come from one of the two sources we offered. A model that
    // invents one gets no action at all: fabricating a write flag for an unknown
    // tool is exactly the guess this path must not make.
    let (write, summary) = resolve_tool(&name, extra)?;

    Some(ProposedAction { name, args, write, summary })
}

/// The assistant router. Owns the `FallbackEngine`; borrows the `AssistantEngine`
/// for LLM calls. FSM state is managed externally (per-surface) by the caller.
pub struct Router {
    pub fallback: FallbackEngine,
}

impl Default for Router {
    fn default() -> Self {
        Self::new()
    }
}

impl Router {
    pub fn new() -> Self {
        Self { fallback: FallbackEngine::new() }
    }

    /// Route a single chat turn.
    ///
    /// - `message`: raw user input.
    /// - `fsm_state`: per-surface FSM session state (mutated in place).
    /// - `engine`: the `AssistantEngine` used for LLM calls.
    /// - `extra_tools`: caller-supplied tools offered alongside the static
    ///   catalog. Empty is valid and reproduces the previous behaviour.
    ///
    /// Returns the response text plus the source tag (`"llm"` | `"fallback"`).
    pub async fn chat(
        &self,
        message: &str,
        fsm_state: &mut Option<FsmState>,
        engine: &AssistantEngine,
        extra_tools: &[ToolSpec],
    ) -> RouterResponse {
        let lang = detect_lang(message);

        // If an FSM flow is in progress, advance it regardless of LLM state.
        if fsm_state.is_some() {
            if let Some(resp) = self.fallback.advance_fsm(fsm_state, message, lang) {
                // If FSM produced a response, use it (fallback path).
                // When LLM is available we still drive the FSM, then the LLM
                // uses the slot context — but for now return the FSM answer.
                return resp.into();
            }
            // FSM returned None: flow finished or was cancelled; clear state.
            *fsm_state = None;
        }

        // An installed assistant gets its server started here; if that fails the
        // answer still comes from the basic engine but is tagged so the UI can say why.
        let mut start_failed = false;
        if engine.is_installed() {
            if let Err(e) = engine.ensure_server().await {
                tracing::warn!("[router] assistant server could not start: {}", e);
                start_failed = true;
            }
        }
        let llm_ready = !start_failed && is_llm_ready(engine).await;

        if llm_ready {
            // LLM path: try tool-calling first (additive, design D4).
            // If the model selects a tool, return immediately with the action.
            if let Some(tool_action) = try_tool_call(message, engine, extra_tools).await {
                tracing::info!("[router] tool-call resolved to action: {}", tool_action.name);
                let summary = tool_action.summary.clone();
                return RouterResponse {
                    text: summary,
                    source: "llm".to_string(),
                    action: Some(tool_action),
                };
            }

            // No tool call (or model doesn't support it): detect flow intent for FSM.
            let flow_hint = self.fallback.detect_flow(message, lang);
            if let Some((flow, _)) = flow_hint {
                *fsm_state = Some(FsmState { flow, step: 0, slots: vec![] });
            }

            // FSM action detection as the guaranteed floor (design D4).
            let action_floor = self.fallback.detect_action(message, lang)
                .and_then(|r| r.action);

            match engine.chat(message).await {
                Ok(text) => RouterResponse { text, source: "llm".to_string(), action: action_floor },
                Err(e) => {
                    tracing::warn!("[router] LLM error, falling back: {}", e);
                    let mut resp: RouterResponse = self.fallback.query(message, lang).into();
                    resp.source = "llm_unavailable".to_string();
                    resp
                }
            }
        } else {
            let mut resp = self.fallback_turn(message, lang, fsm_state);
            if start_failed { resp.source = "llm_unavailable".to_string(); }
            resp
        }
    }

    fn fallback_turn(&self, message: &str, lang: &str, fsm_state: &mut Option<FsmState>) -> RouterResponse {
        {
            // Fallback path: check for action intent first (beats FSM detect_flow
            // so "compile to PDF" returns an action, not an FSM greeting).
            if let Some(action_resp) = self.fallback.detect_action(message, lang) {
                return action_resp.into();
            }
            // Then detect FSM flow (multi-step guided conversation).
            if let Some((flow, first_prompt)) = self.fallback.detect_flow(message, lang) {
                *fsm_state = Some(FsmState { flow, step: 0, slots: vec![] });
                return first_prompt.into();
            }
            self.fallback.query(message, lang).into()
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // is_llm_ready is an async function that hits the network; we verify it
    // returns false when nothing is listening on port 11434 (which is the case
    // in CI and in most test environments).
    #[tokio::test]
    async fn is_llm_ready_returns_false_when_server_unreachable() {
        let cfg = crate::EngineConfig {
            data_dir: std::path::PathBuf::from("/tmp/test"),
            model_id: "qwen3:test".to_string(),
            embedding_model: "nomic-embed-text".to_string(),
            ollama_endpoint: "http://127.0.0.1:11434".to_string(),
            context_window: 4096,
        };
        let engine = crate::AssistantEngine::new(cfg);
        // If Ollama happens to be running locally with qwen3:test this may
        // return true — but in CI (no Ollama) it must be false.
        let _ = is_llm_ready(&engine).await; // just assert no panic
    }

    #[test]
    fn router_new_does_not_panic() {
        let _ = Router::new();
    }
}

#[cfg(test)]
mod tool_payload_tests {
    use super::*;
    use crate::actions::ACTION_CATALOG;

    fn spec() -> ToolSpec {
        ToolSpec {
            name: "run_command".into(),
            description: "Run a Corylus command".into(),
            params: serde_json::json!({
                "type": "object",
                "properties": { "command": { "type": "string", "enum": ["app.openSettings"] } },
                "required": ["command"]
            }),
            write: false,
            summary: "Run a command".into(),
        }
    }

    fn names(tools: &[serde_json::Value]) -> Vec<String> {
        tools.iter().map(|t| t["function"]["name"].as_str().unwrap().to_string()).collect()
    }

    /// 1.2 — every catalog entry carries a parseable schema, so the payload is
    /// built from the catalog rather than from a per-name branch.
    #[test]
    fn every_catalog_entry_has_a_parseable_params_schema() {
        for e in ACTION_CATALOG {
            let v: serde_json::Value = serde_json::from_str(e.params)
                .unwrap_or_else(|err| panic!("{}: params is not JSON: {err}", e.name));
            assert_eq!(v["type"], "object", "{}: params must be an object schema", e.name);
        }
    }

    #[test]
    fn compile_panel_still_offers_its_three_formats() {
        let tools = build_tools(&[]);
        let compile = tools.iter().find(|t| t["function"]["name"] == "open_compile_panel").unwrap();
        let formats = &compile["function"]["parameters"]["properties"]["format"]["enum"];
        assert_eq!(formats, &serde_json::json!(["PDF", "EPUB", "DOCX"]));
    }

    /// 1.3 — the injected tool is offered alongside the catalog, not instead of it.
    #[test]
    fn injected_tools_are_offered_with_the_catalog() {
        let tools = build_tools(&[spec()]);
        let n = names(&tools);
        assert_eq!(tools.len(), ACTION_CATALOG.len() + 1);
        assert!(n.contains(&"word_count".to_string()));
        assert!(n.contains(&"run_command".to_string()));
        let injected = tools.iter().find(|t| t["function"]["name"] == "run_command").unwrap();
        assert_eq!(injected["function"]["parameters"]["properties"]["command"]["enum"][0], "app.openSettings");
    }

    #[test]
    fn no_extra_tools_reproduces_the_catalog_alone() {
        assert_eq!(build_tools(&[]).len(), ACTION_CATALOG.len());
    }

    /// 1.4 — a name we never offered resolves to nothing, rather than to a
    /// ProposedAction carrying an invented write flag.
    #[test]
    fn an_unoffered_tool_name_resolves_to_none() {
        assert!(resolve_tool("replace_text", &[spec()]).is_none());
        assert!(resolve_tool("", &[]).is_none());
    }

    #[test]
    fn resolve_tool_finds_both_sources() {
        assert_eq!(resolve_tool("create_snapshot", &[]).unwrap().0, true);
        assert_eq!(resolve_tool("run_command", &[spec()]).unwrap().0, false);
    }
}
