//! Proposed-action shape and static action catalog.
//!
//! The assistant crate knows action *names and schemas* so it can offer them to
//! the LLM as tools and match them in the FSM, but it does NOT know how to
//! execute them — that lives in the app layer (design D1).

use serde::{Deserialize, Serialize};

// ── Proposed action ──────────────────────────────────────────────────────────

/// A structured action the mascot proposes to execute on the user's behalf.
///
/// Produced by the FSM (fallback path) and by LLM tool-calling; consumed by
/// the app-layer executor.  `write` is the crate's *advisory* classification;
/// the executor re-reads it from its own registry (design D3).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProposedAction {
    /// Stable action name from the catalog (e.g. `"word_count"`).
    pub name: String,
    /// Arguments collected from the message or guided-flow slots.
    pub args: serde_json::Value,
    /// Advisory read/write flag.  `true` = write/outward-effecting.
    pub write: bool,
    /// Short human-readable summary for display in the bubble.
    pub summary: String,
}

// ── Static catalog ───────────────────────────────────────────────────────────

/// One entry in the crate-level action catalog.
/// The app layer mirrors this with its own executor registry; the crate's copy
/// is for FSM matching and LLM tool-schema generation only.
#[derive(Debug, Clone)]
pub struct ActionEntry {
    pub name: &'static str,
    pub description: &'static str,
    pub description_es: &'static str,
    pub write: bool,
    /// Simple keyword signals that indicate this action (all lowercase).
    pub intent_signals: &'static [&'static str],
    /// Summary template (English).
    pub summary: &'static str,
    /// Summary template (Spanish).
    pub summary_es: &'static str,
}

/// v1 action catalog — four action families.
pub static ACTION_CATALOG: &[ActionEntry] = &[
    ActionEntry {
        name: "word_count",
        description: "Count the total words and documents in the open project",
        description_es: "Contar el total de palabras y documentos del proyecto abierto",
        write: false,
        // Specific signals that unambiguously mean word-count, not compile.
        // Avoid single-word "word" to prevent false positives.
        intent_signals: &[
            "count words", "word count", "how many words", "cuántas palabras",
            "cuenta palabras", "total words", "palabras totales",
            "count my words", "words in my project", "words so far",
        ],
        summary: "Count words in the project",
        summary_es: "Contar palabras del proyecto",
    },
    ActionEntry {
        name: "create_snapshot",
        description: "Save a manual snapshot of the current document",
        description_es: "Guardar una instantánea manual del documento actual",
        write: true,
        intent_signals: &[
            "snapshot", "instantánea", "guardar versión", "save version",
            "create snapshot", "take snapshot", "make snapshot",
            "crear instantánea", "tomar instantánea",
        ],
        summary: "Save a snapshot of the current document",
        summary_es: "Guardar una instantánea del documento actual",
    },
    ActionEntry {
        name: "open_compile_panel",
        description: "Open the compile/export panel, optionally pre-filled with a format",
        description_es: "Abrir el panel de compilación/exportación, opcionalmente pre-rellenado con un formato",
        write: false, // navigation only — does not export headlessly
        intent_signals: &[
            "compil", "export", "generate pdf", "generate epub", "create pdf",
            "exportar", "compilar", "compile to", "export to", "open compile",
            "pdf export", "epub export", "docx export", "print",
        ],
        summary: "Open the compile panel",
        summary_es: "Abrir el panel de compilación",
    },
    ActionEntry {
        name: "open_statistics_panel",
        description: "Open the project statistics panel",
        description_es: "Abrir el panel de estadísticas del proyecto",
        write: false,
        intent_signals: &[
            "statistics", "stats", "estadísticas", "open stats",
            "show statistics", "reading time", "tiempo de lectura",
            "pages", "open statistics", "project stats",
        ],
        summary: "Open the statistics panel",
        summary_es: "Abrir el panel de estadísticas",
    },
    ActionEntry {
        name: "ambient_sounds_play",
        description: "Resume the current ambient-sounds mix",
        description_es: "Reanudar la mezcla de sonidos ambientales actual",
        write: true,
        // Require a qualifier alongside "play"/"reproduce" so this doesn't steal
        // turns from unrelated "play"-adjacent chat.
        intent_signals: &[
            "play sounds", "play ambient", "play the ambient sounds", "play my sounds",
            "play my mix", "resume sounds", "resume ambient sounds", "play the rain",
            "reproduce sonido", "reproducir sonido", "pon la lluvia", "reanudar sonidos",
            "reproducir mezcla", "reproduce mi mezcla",
        ],
        summary: "Play the current ambient-sounds mix",
        summary_es: "Reproducir la mezcla de sonidos ambientales actual",
    },
    ActionEntry {
        name: "ambient_sounds_stop",
        description: "Pause the current ambient-sounds mix",
        description_es: "Pausar la mezcla de sonidos ambientales actual",
        write: true,
        intent_signals: &[
            "stop sounds", "stop ambient", "stop the ambient sounds", "stop my sounds",
            "pause sounds", "pause ambient sounds", "stop the rain", "pause my mix",
            "para el sonido", "parar sonido", "detener sonido", "pausar sonido",
            "pausar mezcla", "detener mezcla",
        ],
        summary: "Stop the current ambient-sounds mix",
        summary_es: "Detener la mezcla de sonidos ambientales actual",
    },
    ActionEntry {
        name: "open_ambient_mixer",
        description: "Open the ambient-sounds mixer dialog",
        description_es: "Abrir el mezclador de sonidos ambientales",
        write: false,
        intent_signals: &[
            "open the mixer", "open sound mixer", "open ambient sounds", "sound mixer",
            "abrir mezclador", "abrir sonidos ambientales", "abrir el mezclador de sonido",
        ],
        summary: "Open the ambient-sounds mixer",
        summary_es: "Abrir el mezclador de sonidos ambientales",
    },
    ActionEntry {
        name: "toggle_distraction_free",
        description: "Enter or exit distraction-free writing mode",
        description_es: "Entrar o salir del modo de escritura sin distracciones",
        write: true,
        intent_signals: &[
            "distraction free", "distraction-free", "focus mode", "go distraction free",
            "enter distraction free", "exit distraction free", "modo sin distracciones",
            "pantalla completa de escritura", "modo enfoque", "escritura sin distracciones",
        ],
        summary: "Toggle distraction-free mode",
        summary_es: "Alternar el modo sin distracciones",
    },
    ActionEntry {
        name: "open_settings",
        description: "Open the Settings screen, optionally at a named category",
        description_es: "Abrir la pantalla de Ajustes, opcionalmente en una categoría concreta",
        write: false,
        intent_signals: &[
            "open settings", "show settings", "open preferences", "abrir ajustes",
            "abrir configuración", "abrir preferencias", "mostrar ajustes",
            "ambient sound settings", "ajustes de sonido",
        ],
        summary: "Open Settings",
        summary_es: "Abrir Ajustes",
    },
];

/// Look up an entry by name.
pub fn find_action(name: &str) -> Option<&'static ActionEntry> {
    ACTION_CATALOG.iter().find(|e| e.name == name)
}
