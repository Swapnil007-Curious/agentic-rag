// Shapes of public/data/manifest.json (written by scripts/build_static_indexes.py) and of the
// objects the app passes around.

export interface SourceInfo {
  doc_id: string;
  title: string;
  authors: string;
  edition: string;
  licence: string;
  licence_url: string;
  source_url: string;
}

export interface FileInfo {
  path: string;
  bytes: number;
  gzip_bytes: number;
  sha256: string;
}

export interface GroundedSettings {
  budget_tokens: number;
  revision_budget_tokens: number;
  min_passages: number;
  max_synth_calls: number;
  max_critic_calls: number;
  critic_on_guard_fail: boolean;
  on_final_revise: "abstain" | "answer";
}

export interface CollectionMeta {
  id: string;
  label: string;
  description: string;
  notice: string;
  examples: string[];
  evaluated: boolean;
  profile: string;
  abstention_message: string;
  threshold: number;
  grounded: GroundedSettings;
  count: number;
  dim: number;
  sources: SourceInfo[];
  files: { passages: FileInfo; vectors: FileInfo };
}

export interface Manifest {
  format: number;
  version: string;
  model: {
    id: string;
    source: string;
    pooling: string;
    normalize: boolean;
    max_tokens: number;
    query_instruction: string;
  };
  gate: {
    default_threshold: number;
    upload_threshold: number;
    follow_up_max_words: number;
    search_k: number;
    max_passages: number;
  };
  context_tokens: number;
  uploads_grounded: GroundedSettings;
  history: {
    max_turns: number;
    budget_tokens: number;
    summary_target_share: number;
    summary_input_chars: number;
  };
  limits: { max_message_chars: number; max_output_tokens: number };
  uploads: { max_files: number; max_mb: number; max_pages: number; max_chunks_per_file: number };
  general_examples: string[];
  collections: CollectionMeta[];
}

/** A searchable chunk: a collection passage or a chunk of an uploaded PDF. */
export interface Chunk {
  chunk_id: string;
  doc_id: string;
  heading_path: string;
  section_ids: string[];
  pages: string;
  n_tokens: number;
  text: string;
}

export interface Hit {
  chunk: Chunk;
  score: number;
  source: string; // collection id or "upload"
}

/** A passage given to the model, labelled C1, C2, ... */
export interface Passage extends Chunk {
  label: string;
  source: string;
  score: number;
  title: string; // upload file name, else ""
}
