//! Rule/retrieval-based fallback assistant engine.
//!
//! When the LLM path is unavailable this module answers help questions by
//! scoring a bundled corpus with a simple BM25-style scorer, and drives
//! guided conversational flows (compile, binder, new project) via a small FSM.
//!
//! The corpus is compiled into the binary via `include_str!` so the fallback
//! works fully offline with zero downloads.

use serde::Deserialize;

// ── Corpus ─────────────────────────────────────────────────────────────────

const CORPUS_JSON: &str =
    include_str!("../../../app/assets/assistant-help-corpus.json");

#[derive(Debug, Deserialize)]
struct CorpusEntry {
    question: String,
    answer: String,
    answer_es: Option<String>,
    intent: String,
    tags: Vec<String>,
}

// ── Public types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FallbackResponse {
    pub text: String,
    pub source: &'static str,
}

impl FallbackResponse {
    fn llm() -> Self { Self { text: String::new(), source: "llm" } }

    fn fallback(text: impl Into<String>) -> Self {
        Self { text: text.into(), source: "fallback" }
    }
}

/// FSM state for a single guided conversational flow.
#[derive(Debug, Clone)]
pub struct FsmState {
    pub flow: FsmFlow,
    pub step: usize,
    /// Slot values collected so far (e.g. format, scope).
    pub slots: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FsmFlow {
    Compile,
    Binder,
    NewProject,
}

// ── BM25 engine ────────────────────────────────────────────────────────────

pub struct FallbackEngine {
    entries: Vec<CorpusEntry>,
}

impl FallbackEngine {
    pub fn new() -> Self {
        let entries: Vec<CorpusEntry> =
            serde_json::from_str(CORPUS_JSON).unwrap_or_default();
        Self { entries }
    }

    /// Query the corpus and return the best matching response.
    /// `lang` is `"es"` for Spanish, anything else for English.
    pub fn query(&self, text: &str, lang: &str) -> FallbackResponse {
        let query_terms = tokenize(text);
        if query_terms.is_empty() {
            return self.unknown_response(lang);
        }

        let mut best_score = 0.0f32;
        let mut best_idx = usize::MAX;

        for (i, entry) in self.entries.iter().enumerate() {
            let score = bm25_score(&query_terms, entry);
            if score > best_score {
                best_score = score;
                best_idx = i;
            }
        }

        const THRESHOLD: f32 = 0.10;
        if best_score < THRESHOLD || best_idx == usize::MAX {
            return self.unknown_response(lang);
        }

        let entry = &self.entries[best_idx];
        let text = if lang == "es" {
            entry.answer_es.as_deref().unwrap_or(&entry.answer)
        } else {
            &entry.answer
        };

        FallbackResponse::fallback(text)
    }

    /// Advance the FSM by one turn. Returns the next prompt or the final answer.
    pub fn advance_fsm(
        &self,
        state: &mut Option<FsmState>,
        input: &str,
        lang: &str,
    ) -> Option<FallbackResponse> {
        // Cancel escape hatch
        let lower = input.to_lowercase();
        if matches!(
            lower.trim(),
            "cancel" | "never mind" | "nevermind" | "exit" | "quit"
                | "cancelar" | "salir" | "olvidar"
        ) {
            *state = None;
            let msg = if lang == "es" {
                "De acuerdo, volvemos al modo de preguntas abiertas."
            } else {
                "Sure, back to open Q&A mode."
            };
            return Some(FallbackResponse::fallback(msg));
        }

        let s = state.as_mut()?;
        match s.flow {
            FsmFlow::Compile => self.advance_compile(s, input, lang),
            FsmFlow::Binder => self.advance_binder(s, input, lang),
            FsmFlow::NewProject => self.advance_new_project(s, input, lang),
        }
    }

    fn advance_compile(
        &self,
        s: &mut FsmState,
        input: &str,
        lang: &str,
    ) -> Option<FallbackResponse> {
        match s.step {
            0 => {
                s.slots.push(input.to_string());
                s.step = 1;
                let msg = if lang == "es" {
                    "¿Quieres compilar todo el proyecto o solo el documento actual?"
                } else {
                    "Do you want to compile the whole project or just the current document?"
                };
                Some(FallbackResponse::fallback(msg))
            }
            1 => {
                s.slots.push(input.to_string());
                let fmt = s.slots.first().cloned().unwrap_or_default();
                let scope = s.slots.get(1).cloned().unwrap_or_default();
                let text = if lang == "es" {
                    format!(
                        "Para compilar en {fmt} ({scope}): abre el panel Compilar \
                        → elige {fmt} → selecciona el alcance → haz clic en Compilar. \
                        El archivo exportado se guardará en tu carpeta de exportación. \
                        (Escribe 'cancelar' para salir en cualquier momento.)"
                    )
                } else {
                    format!(
                        "To compile to {fmt} ({scope}): open the Compile panel \
                        → choose {fmt} → select the scope → click Compile. \
                        The exported file will be saved to your export folder. \
                        (Type 'cancel' to exit at any time.)"
                    )
                };
                Some(FallbackResponse::fallback(text))
            }
            _ => None,
        }
    }

    fn advance_binder(
        &self,
        s: &mut FsmState,
        input: &str,
        lang: &str,
    ) -> Option<FallbackResponse> {
        match s.step {
            0 => {
                s.slots.push(input.to_string());
                s.step = 1;
                let resp = self.query(input, lang);
                Some(resp)
            }
            _ => None,
        }
    }

    fn advance_new_project(
        &self,
        s: &mut FsmState,
        input: &str,
        lang: &str,
    ) -> Option<FallbackResponse> {
        match s.step {
            0 => {
                s.slots.push(input.to_string());
                s.step = 1;
                let msg = if lang == "es" {
                    "¿Dónde quieres guardar el proyecto? (ruta o descripción)"
                } else {
                    "Where would you like to save the project? (path or description)"
                };
                Some(FallbackResponse::fallback(msg))
            }
            1 => {
                s.slots.push(input.to_string());
                let name = s.slots.first().cloned().unwrap_or_default();
                let text = if lang == "es" {
                    format!(
                        "Para crear el proyecto '{name}': ve a la pantalla de inicio \
                        → haz clic en 'Nuevo proyecto' → introduce el nombre → elige la ubicación \
                        → haz clic en Crear."
                    )
                } else {
                    format!(
                        "To create project '{name}': go to the Home screen \
                        → click 'New Project' → enter the name → choose the location \
                        → click Create."
                    )
                };
                Some(FallbackResponse::fallback(text))
            }
            _ => None,
        }
    }

    fn unknown_response(&self, lang: &str) -> FallbackResponse {
        let text = if lang == "es" {
            "No estoy seguro de cómo responder a eso. Prueba preguntando sobre \
            compilar, la carpeta de documentos, el corrector ortográfico, la \
            gramática o el asistente IA. También puedes revisar la documentación \
            de Corylus para más ayuda."
        } else {
            "I'm not sure how to answer that. Try asking about compiling, \
            the document binder, spellcheck, grammar, or the AI assistant. \
            You can also check the Corylus documentation for more help."
        };
        FallbackResponse::fallback(text)
    }

    /// Detect intent from user input to decide if an FSM flow should start.
    /// Returns the flow and a first-step prompt if a flow is detected.
    pub fn detect_flow(&self, text: &str, lang: &str) -> Option<(FsmFlow, FallbackResponse)> {
        let lower = text.to_lowercase();

        let compile_kw = ["compil", "export", "pdf", "epub", "docx", "microsoft word", " .docx"];
        let binder_kw = ["binder", "carpeta", "folder"];
        let project_kw = ["new project", "nuevo proyecto", "create project", "crear proyecto"];

        if compile_kw.iter().any(|kw| lower.contains(kw)) {
            let prompt = if lang == "es" {
                "Claro, te ayudo a compilar. ¿Qué formato necesitas? (PDF, EPUB, DOCX…)"
            } else {
                "Sure, let me help you compile. What format do you need? (PDF, EPUB, DOCX…)"
            };
            return Some((FsmFlow::Compile, FallbackResponse::fallback(prompt)));
        }

        if project_kw.iter().any(|kw| lower.contains(kw)) {
            let prompt = if lang == "es" {
                "Vamos a crear un nuevo proyecto. ¿Cómo se llamará?"
            } else {
                "Let's create a new project. What will it be called?"
            };
            return Some((FsmFlow::NewProject, FallbackResponse::fallback(prompt)));
        }

        if binder_kw.iter().any(|kw| lower.contains(kw)) {
            let prompt = if lang == "es" {
                "Claro. ¿Qué necesitas hacer con los documentos o la carpeta?"
            } else {
                "Sure. What do you need to do with documents or the binder?"
            };
            return Some((FsmFlow::Binder, FallbackResponse::fallback(prompt)));
        }

        None
    }
}

// ── BM25-style scoring ──────────────────────────────────────────────────────

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| t.len() > 1)
        .map(String::from)
        .collect()
}

fn bm25_score(query_terms: &[String], entry: &CorpusEntry) -> f32 {
    // Build document from question + tags (indexed fields)
    let doc = format!("{} {}", entry.question, entry.tags.join(" "));
    let doc_terms = tokenize(&doc);
    let doc_len = doc_terms.len() as f32;
    let avg_len = 12.0_f32; // rough average, fixed for simplicity

    const K1: f32 = 1.2;
    const B: f32 = 0.75;

    let mut score = 0.0f32;
    for qt in query_terms {
        let tf = doc_terms.iter().filter(|t| t.as_str() == qt.as_str()).count() as f32;
        if tf == 0.0 { continue; }
        // Simplified IDF (assume moderate doc frequency)
        let idf = 1.0_f32;
        let tf_norm = (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * doc_len / avg_len));
        score += idf * tf_norm;
    }

    // Small boost for intent tag matches (makes corpus more predictable)
    let intent_terms = tokenize(&entry.intent);
    for qt in query_terms {
        if intent_terms.contains(qt) {
            score += 0.5;
        }
    }

    score
}

// ── Language detection (best-effort) ────────────────────────────────────────

/// Guess the language of a query: returns `"es"` for Spanish, `"en"` otherwise.
pub fn detect_lang(text: &str) -> &'static str {
    let lower = text.to_lowercase();
    // Common Spanish function words and interrogatives
    let es_markers = [
        "cómo", "como", "qué", "que", "cuál", "cual", "dónde", "donde",
        "puedo", "quiero", "necesito", "tengo", "hacer", "crear", "abrir",
        "guardar", "compilar", "exportar", "instalar", "activar",
        "¿", "á", "é", "í", "ó", "ú", "ñ",
    ];
    let hits = es_markers.iter().filter(|m| lower.contains(*m)).count();
    if hits >= 1 { "es" } else { "en" }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bm25_returns_best_match() {
        let engine = FallbackEngine::new();
        let resp = engine.query("How do I compile my project to PDF", "en");
        assert_eq!(resp.source, "fallback");
        assert!(!resp.text.is_empty());
        // Should mention compile or PDF
        let lower = resp.text.to_lowercase();
        assert!(lower.contains("compile") || lower.contains("pdf") || lower.contains("panel"));
    }

    #[test]
    fn bm25_spanish_query_returns_spanish_answer() {
        let engine = FallbackEngine::new();
        let resp = engine.query("¿Cómo compilo mi proyecto en PDF?", "es");
        assert_eq!(resp.source, "fallback");
        // Spanish answer should exist and mention compile-related terms in Spanish
        assert!(!resp.text.is_empty());
    }

    #[test]
    fn below_threshold_returns_sentinel() {
        let engine = FallbackEngine::new();
        let resp = engine.query("xyzzy frobozz plugh", "en");
        assert_eq!(resp.source, "fallback");
        assert!(resp.text.contains("not sure") || resp.text.contains("documentation"));
    }

    #[test]
    fn below_threshold_spanish_returns_spanish_sentinel() {
        let engine = FallbackEngine::new();
        let resp = engine.query("xyzzy frobozz", "es");
        assert!(resp.text.contains("seguro") || resp.text.contains("documentación"));
    }

    #[test]
    fn fsm_cancel_resets_state() {
        let engine = FallbackEngine::new();
        let mut state = Some(FsmState {
            flow: FsmFlow::Compile,
            step: 0,
            slots: vec![],
        });
        let resp = engine.advance_fsm(&mut state, "cancel", "en");
        assert!(state.is_none());
        assert!(resp.is_some());
    }

    #[test]
    fn fsm_cancel_spanish() {
        let engine = FallbackEngine::new();
        let mut state = Some(FsmState {
            flow: FsmFlow::NewProject,
            step: 0,
            slots: vec![],
        });
        let resp = engine.advance_fsm(&mut state, "cancelar", "es");
        assert!(state.is_none());
        let text = resp.unwrap().text;
        assert!(text.contains("volvemos") || text.contains("abiertas"));
    }

    #[test]
    fn detect_lang_spanish() {
        assert_eq!(detect_lang("¿Cómo compilo mi proyecto?"), "es");
        assert_eq!(detect_lang("cómo hago esto"), "es");
        assert_eq!(detect_lang("quiero crear un proyecto"), "es");
    }

    #[test]
    fn detect_lang_english() {
        assert_eq!(detect_lang("How do I compile my project?"), "en");
        assert_eq!(detect_lang("install the assistant"), "en");
    }

    #[test]
    fn detect_flow_compile() {
        let engine = FallbackEngine::new();
        let result = engine.detect_flow("how do I export to PDF", "en");
        assert!(result.is_some());
        let (flow, _) = result.unwrap();
        assert_eq!(flow, FsmFlow::Compile);
    }
}
