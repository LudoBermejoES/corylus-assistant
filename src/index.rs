//! SQLite vector index for RAG retrieval.
//!
//! Schema
//! ──────
//!   meta        — key/value store: embedding_model, embedding_dim, schema_version
//!   chunks      — id, source, content, content_hash (blake3 hex)
//!   embeddings  — chunk_id FK → chunks.id, embedding BLOB (raw f32 LE bytes)
//!
//! Similarity search
//! ─────────────────
//! Embeddings are stored as raw f32 little-endian blobs. Retrieval loads all
//! vectors into memory, computes cosine similarity in Rust, and returns top-K.
//! For the manuscript sizes Corylus targets (thousands of chunks at most) this
//! is fast enough without a native vector extension.
//!
//! Incremental update (task 3.4)
//! ──────────────────────────────
//! Each source file is hashed (blake3). On re-index we skip files whose hash
//! hasn't changed; we delete rows for files that have been removed.
//!
//! Model pinning (task 3.6)
//! ──────────────────────────
//! The `meta` table records which embedding model + dimension built the index.
//! If the configured model changes, `check_model_compatibility` returns
//! `Err(EmbeddingModelMismatch)`. The caller must set `full_reindex = true`.

use rusqlite::{Connection, params, OptionalExtension};
use tracing::{info, warn};

use crate::{
    Result,
    error::AssistantError,
    rag::{AsyncEmbedFn, embed_chunks},
    state::index_dir,
    EngineConfig,
};

const INDEX_SCHEMA_VERSION: u32 = 1;

pub struct VectorIndex {
    conn: Connection,
}

impl VectorIndex {
    /// Open (or create) the vector index for this engine config.
    pub fn open(config: &EngineConfig) -> Result<Self> {
        let dir = index_dir(config);
        std::fs::create_dir_all(&dir)?;
        let db_path = dir.join("rag.db");
        let conn = Connection::open(&db_path)?;
        let idx = Self { conn };
        idx.migrate()?;
        Ok(idx)
    }

    fn migrate(&self) -> Result<()> {
        self.conn.execute_batch("
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS chunks (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source       TEXT    NOT NULL,
                content      TEXT    NOT NULL,
                content_hash TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source);
            CREATE INDEX IF NOT EXISTS chunks_hash   ON chunks(content_hash);
            CREATE TABLE IF NOT EXISTS embeddings (
                chunk_id  INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                embedding BLOB    NOT NULL
            );
        ")?;
        Ok(())
    }

    // ── Model pinning ─────────────────────────────────────────────────────────

    fn meta_get(&self, key: &str) -> Result<Option<String>> {
        Ok(self.conn
            .query_row(
                "SELECT value FROM meta WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?)
    }

    fn meta_set(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?1,?2)
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            params![key, value],
        )?;
        Ok(())
    }

    /// Check whether the current index was built with the same model as `config`.
    /// Returns `Err(EmbeddingModelMismatch)` if the model changed; `Ok(())` if
    /// compatible or if the index is empty (first run).
    pub fn check_model_compatibility(&self, config: &EngineConfig) -> Result<()> {
        let stored_model = self.meta_get("embedding_model")?;
        let stored_dim   = self.meta_get("embedding_dim")?.and_then(|s| s.parse::<u32>().ok());
        match (stored_model, stored_dim) {
            (Some(m), Some(d)) if m != config.embedding_model => {
                Err(AssistantError::EmbeddingModelMismatch {
                    index_model:  m,
                    index_dim:    d,
                    config_model: config.embedding_model.clone(),
                })
            }
            _ => Ok(()),
        }
    }

    fn pin_model(&self, model: &str, dim: usize) -> Result<()> {
        self.meta_set("embedding_model", model)?;
        self.meta_set("embedding_dim", &dim.to_string())?;
        self.meta_set("schema_version", &INDEX_SCHEMA_VERSION.to_string())?;
        Ok(())
    }

    // ── Indexing ──────────────────────────────────────────────────────────────

    /// Full re-index: wipe existing data then index from scratch.
    pub async fn full_reindex(
        &mut self,
        documents: Vec<(String, String)>,
        config: &EngineConfig,
        embed_fn: &(impl AsyncEmbedFn + Sync),
        on_progress: impl Fn(usize, usize),
    ) -> Result<()> {
        self.reset()?;
        self.incremental_update(documents, config, embed_fn, on_progress).await
    }

    /// Incremental update: skip unchanged files, remove deleted files, add new/changed ones.
    pub async fn incremental_update(
        &mut self,
        documents: Vec<(String, String)>,
        config: &EngineConfig,
        embed_fn: &(impl AsyncEmbedFn + Sync),
        on_progress: impl Fn(usize, usize),
    ) -> Result<()> {
        let total = documents.len();
        let live_sources: std::collections::HashSet<String> =
            documents.iter().map(|(s, _)| s.clone()).collect();

        // Delete rows for sources no longer present
        let existing_sources: Vec<String> = {
            let mut stmt = self.conn.prepare("SELECT DISTINCT source FROM chunks")?;
            let sources: Vec<String> = stmt.query_map([], |r| r.get(0))?.filter_map(|r| r.ok()).collect();
            sources
        };
        for src in existing_sources {
            if !live_sources.contains(&src) {
                self.delete_source(&src)?;
            }
        }

        let mut done = 0usize;
        for (source, content) in documents {
            let file_hash = blake3_hex(&content);

            let existing_hash: Option<String> = self.conn
                .query_row(
                    "SELECT content_hash FROM chunks WHERE source=?1 LIMIT 1",
                    params![source],
                    |r| r.get(0),
                )
                .optional()?;

            if existing_hash.as_deref() == Some(&file_hash) {
                done += 1;
                on_progress(done, total);
                continue;
            }

            self.delete_source(&source)?;

            let chunks = crate::rag::chunk_text(&source, &content, 800, 200, 50);
            if chunks.is_empty() {
                done += 1;
                on_progress(done, total);
                continue;
            }

            let embedded = embed_chunks(chunks, embed_fn).await;
            if embedded.is_empty() {
                warn!("[rag] no embeddings produced for {}", source);
                done += 1;
                on_progress(done, total);
                continue;
            }

            let dim = embedded[0].1.len();
            self.pin_model(&config.embedding_model, dim)?;

            for (chunk, vec) in &embedded {
                let chunk_id: i64 = self.conn.query_row(
                    "INSERT INTO chunks(source,content,content_hash) VALUES(?1,?2,?3) RETURNING id",
                    params![chunk.source, chunk.text, file_hash],
                    |r| r.get(0),
                )?;
                let blob = f32_slice_to_blob(vec);
                self.conn.execute(
                    "INSERT INTO embeddings(chunk_id, embedding) VALUES(?1, ?2)",
                    params![chunk_id, blob],
                )?;
            }
            info!("[rag] indexed {} chunks from {}", embedded.len(), source);

            done += 1;
            on_progress(done, total);
        }
        Ok(())
    }

    fn delete_source(&self, source: &str) -> Result<()> {
        self.conn.execute("DELETE FROM chunks WHERE source=?1", params![source])?;
        Ok(())
    }

    /// Drop all indexed data and model pin.
    pub fn reset(&self) -> Result<()> {
        self.conn.execute_batch("
            DELETE FROM embeddings;
            DELETE FROM chunks;
            DELETE FROM meta WHERE key IN ('embedding_model','embedding_dim');
        ")?;
        Ok(())
    }

    // ── Retrieval ─────────────────────────────────────────────────────────────

    /// Find the top-K chunks most similar to `query_vec` by cosine similarity.
    /// Returns closest-first (highest cosine similarity first).
    pub fn top_k(&self, query_vec: &[f32], k: usize) -> Result<Vec<RetrievedRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.source, c.content, e.embedding
             FROM embeddings e
             JOIN chunks c ON c.id = e.chunk_id"
        )?;

        let query_norm = l2_norm(query_vec);
        if query_norm == 0.0 {
            return Ok(vec![]);
        }

        let mut scored: Vec<(f32, String, String)> = stmt
            .query_map([], |row| {
                let source: String = row.get(0)?;
                let content: String = row.get(1)?;
                let blob: Vec<u8> = row.get(2)?;
                Ok((source, content, blob))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(source, content, blob)| {
                let vec = blob_to_f32_slice(&blob);
                if vec.is_empty() { return None; }
                let sim = cosine_similarity(query_vec, &vec, query_norm);
                Some((sim, source, content))
            })
            .collect();

        // Sort descending by similarity
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored.into_iter().map(|(sim, source, text)| RetrievedRow {
            source,
            text,
            distance: (1.0 - sim) as f64, // distance = 1 - cosine_sim; 0 = perfect match
        }).collect())
    }

    /// Number of indexed chunks.
    pub fn chunk_count(&self) -> Result<usize> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks", [], |r| r.get(0)
        )?;
        Ok(n as usize)
    }
}

#[derive(Debug, Clone)]
pub struct RetrievedRow {
    pub source:   String,
    pub text:     String,
    /// Cosine distance (0 = identical, 2 = opposite). Lower is more relevant.
    pub distance: f64,
}

// ── Math helpers ──────────────────────────────────────────────────────────────

fn cosine_similarity(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    if a.len() != b.len() { return 0.0; }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let b_norm = l2_norm(b);
    if b_norm == 0.0 { return 0.0; }
    dot / (a_norm * b_norm)
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ── Blob encoding ─────────────────────────────────────────────────────────────

fn blake3_hex(s: &str) -> String {
    blake3::hash(s.as_bytes()).to_hex().to_string()
}

fn f32_slice_to_blob(v: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(v.len() * 4);
    for f in v {
        buf.extend_from_slice(&f.to_le_bytes());
    }
    buf
}

fn blob_to_f32_slice(blob: &[u8]) -> Vec<f32> {
    if blob.len() % 4 != 0 { return vec![]; }
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
