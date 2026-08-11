-- Tip leaf supersede: allow replacing a raw_weight_snapshot row when the
-- payload_digest changes for the same (challenge_id, epoch, miner_hotkey).
--
-- `base_app` still has no direct UPDATE privilege on append-only tables
-- (schema tests keep that invariant). Tip supersede runs through this
-- SECURITY DEFINER helper owned by the migration role.

CREATE OR REPLACE FUNCTION upsert_raw_weight_tip(
    p_id uuid,
    p_challenge_id text,
    p_epoch bigint,
    p_miner_hotkey text,
    p_kind text,
    p_score bigint,
    p_absence_reason text,
    p_payload bytea,
    p_payload_digest bytea,
    p_signature bytea,
    p_nonce bytea
) RETURNS uuid
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    result_id uuid;
BEGIN
    INSERT INTO raw_weight_snapshot (
        id, challenge_id, epoch, miner_hotkey, kind, score, absence_reason,
        payload, payload_digest, signature, nonce
    ) VALUES (
        p_id, p_challenge_id, p_epoch, p_miner_hotkey, p_kind, p_score,
        p_absence_reason, p_payload, p_payload_digest, p_signature, p_nonce
    )
    ON CONFLICT (challenge_id, epoch, miner_hotkey) DO UPDATE SET
        id = EXCLUDED.id,
        kind = EXCLUDED.kind,
        score = EXCLUDED.score,
        absence_reason = EXCLUDED.absence_reason,
        payload = EXCLUDED.payload,
        payload_digest = EXCLUDED.payload_digest,
        signature = EXCLUDED.signature,
        nonce = EXCLUDED.nonce
    WHERE raw_weight_snapshot.payload_digest IS DISTINCT FROM EXCLUDED.payload_digest
    RETURNING id INTO result_id;

    RETURN result_id;
END;
$$;

REVOKE ALL ON FUNCTION upsert_raw_weight_tip(
    uuid, text, bigint, text, text, bigint, text, bytea, bytea, bytea, bytea
) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION upsert_raw_weight_tip(
    uuid, text, bigint, text, text, bigint, text, bytea, bytea, bytea, bytea
) TO base_app;
