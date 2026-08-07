-- PRISM v3 source-tree delivery: persist the validated miner tree so the
-- orchestrator can stage kernels / helpers / tokenizer/ onto the Lium pod
-- (not only the architecture.py + training.py seam projections).
--
-- Blob is the deterministic USTAR produced by prism_tree::StagedTree::pack
-- (self-describing: includes the entry marker). Cap matches
-- prism_tree::MAX_TREE_TOTAL_BYTES plus USTAR framing headroom.

ALTER TABLE prism_submission
    ADD COLUMN tree_blob BYTEA;

ALTER TABLE prism_submission
    ADD CONSTRAINT prism_submission_tree_blob_len
        CHECK (tree_blob IS NULL OR octet_length(tree_blob) <= 17825792);
