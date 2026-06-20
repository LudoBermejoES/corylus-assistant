//! RAG chunking and embedding helpers.
//!
//! Chunking strategy: overlapping fixed-size windows over plain text, skipping
//! very short fragments. Each chunk carries its source path for citation.
//!
//! Embedding: delegates to `OllamaBackend::embed()` which calls `/api/embeddings`.

use crate::Result;

/// A single retrievable chunk of project content.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Source file path, relative to the project root.
    pub source: String,
    /// The plain-text content of this chunk.
    pub text: String,
}

/// Split `text` from `source` into overlapping chunks.
///
/// - `window`: target size in characters (default 800)
/// - `overlap`: overlap between consecutive windows (default 200)
/// - Chunks shorter than `min_len` are dropped.
pub fn chunk_text(source: &str, text: &str, window: usize, overlap: usize, min_len: usize) -> Vec<Chunk> {
    if text.is_empty() {
        return vec![];
    }
    let window = window.max(64);
    let overlap = overlap.min(window / 2);
    let _step = window.saturating_sub(overlap).max(1);

    // Work on character boundary-safe slices
    let chars: Vec<char> = text.chars().collect();
    let total = chars.len();
    let mut chunks = Vec::new();
    let mut start = 0usize;

    while start < total {
        let end = (start + window).min(total);
        // Extend to the next sentence boundary ('. ', '? ', '! ', '\n\n') within +120 chars
        let search_end = (end + 120).min(total);
        let slice: String = chars[start..search_end].iter().collect();
        let cut = find_sentence_end(&slice, end - start)
            .map(|i| start + i)
            .unwrap_or(end);

        let chunk_text: String = chars[start..cut].iter().collect();
        let trimmed = chunk_text.trim();
        if trimmed.len() >= min_len {
            chunks.push(Chunk {
                source: source.to_string(),
                text: trimmed.to_string(),
            });
        }

        if cut >= total {
            break;
        }
        start = cut.saturating_sub(overlap).max(start + 1);
    }
    chunks
}

/// Find a sentence-end boundary (`. `, `? `, `! `, `\n\n`) near `target_idx`
/// within `text`, searching forward up to 120 chars. Returns the index after the boundary.
fn find_sentence_end(text: &str, target_idx: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let search_start = target_idx.min(bytes.len().saturating_sub(1));
    for i in search_start..bytes.len().saturating_sub(1) {
        match (bytes[i], bytes[i + 1]) {
            (b'.', b' ') | (b'?', b' ') | (b'!', b' ') => return Some(i + 2),
            (b'\n', b'\n') => return Some(i + 2),
            _ => {}
        }
    }
    None
}

/// Embed a batch of chunks using the provided async embed function.
/// Returns `(Chunk, Vec<f32>)` pairs; chunks whose embedding fails are dropped with a warning.
pub async fn embed_chunks(
    chunks: Vec<Chunk>,
    embed_fn: impl AsyncEmbedFn,
) -> Vec<(Chunk, Vec<f32>)> {
    let mut results = Vec::with_capacity(chunks.len());
    for chunk in chunks {
        match embed_fn.embed(&chunk.text).await {
            Ok(vec) if !vec.is_empty() => results.push((chunk, vec)),
            Ok(_) => tracing::warn!("[rag] empty embedding for chunk from {}", chunk.source),
            Err(e) => tracing::warn!("[rag] embed error for chunk from {}: {}", chunk.source, e),
        }
    }
    results
}

/// Trait alias for an async embed function — lets us pass `&OllamaBackend` or a mock.
pub trait AsyncEmbedFn: Sync {
    fn embed(&self, text: &str) -> impl std::future::Future<Output = Result<Vec<f32>>> + Send;
}

// Blanket impl so &T works wherever T: AsyncEmbedFn
impl<T: AsyncEmbedFn + Sync> AsyncEmbedFn for &T {
    fn embed(&self, text: &str) -> impl std::future::Future<Output = Result<Vec<f32>>> + Send {
        (*self).embed(text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_empty_text() {
        let chunks = chunk_text("doc.md", "", 800, 200, 20);
        assert!(chunks.is_empty());
    }

    #[test]
    fn chunk_short_text_single_chunk() {
        let text = "The quick brown fox jumps over the lazy dog.";
        let chunks = chunk_text("doc.md", text, 800, 200, 5);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].source, "doc.md");
        assert!(chunks[0].text.contains("fox"));
    }

    #[test]
    fn chunk_long_text_multiple_chunks() {
        // Build a text that is definitely longer than one window
        let para = "The quick brown fox jumps over the lazy dog. ";
        let text = para.repeat(40); // ~1800 chars
        let chunks = chunk_text("doc.md", &text, 500, 100, 20);
        assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
        // Overlap means adjacent chunks share content
        // Just verify sources are correct
        for c in &chunks {
            assert_eq!(c.source, "doc.md");
        }
    }

    #[test]
    fn chunk_drops_short_fragments() {
        let text = "Hi. Yes.";
        let chunks = chunk_text("doc.md", text, 800, 200, 20);
        assert!(chunks.is_empty(), "short text should be filtered: {:?}", chunks);
    }

    #[test]
    fn sentence_boundary_detection() {
        let text = "Hello world. Next sentence here.";
        let idx = find_sentence_end(text, 5);
        // Should land after "Hello world. " at index 13
        assert_eq!(idx, Some(13));
    }
}
