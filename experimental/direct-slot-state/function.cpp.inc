// Caller has synchronized all users of both sessions. KV remains on the existing
// paged/ring snapshot path; this only moves the live checkpoint state.
bool conversation_state_transfer(const SessionState& from, SessionState& to,
                                 const ModelGeometry& g, const std::vector<int32_t>& ids,
                                 std::string& error) {
    static const bool direct = [] {
        const char* value = std::getenv("STRATA_DIRECT_SLOT_STATE");
        return value && value[0] == '1';
    }();
    if (!direct) {
        ConversationCheckpoint c; c.ids = ids;
        return conversation_checkpoint_save(c, from, g, error) &&
               conversation_checkpoint_restore(c, to, g, error);
    }
    ConversationStateSizes a, b;
    if (!checkpoint_targets(from, g, ids.size(), a, error) ||
        !checkpoint_targets(to, g, ids.size(), b, error)) return false;
    if (from.layer_lo != to.layer_lo || from.layer_hi != to.layer_hi ||
        from.gdn_ord0 != to.gdn_ord0 || from.gdn_alloc != to.gdn_alloc ||
        from.qsa_ord0 != to.qsa_ord0 || from.qsa_alloc != to.qsa_alloc ||
        a.gdn != b.gdn || bool(from.ple_hist) != bool(to.ple_hist))
        return fail(error, "incompatible live-state transfer");
    if (&from == &to) return fail(error, "live-state transfer requires distinct sessions");
    if (!copy(to.gdn_state, from.gdn_state, a.gdn, error) ||
        !copy(to.ple_hist, from.ple_hist, from.ple_hist ? a.ple : 0, error)) return false;
    for (size_t j = 0; j < owned_qsa(from); ++j) {
        const auto& src = owned(from, j);
        const auto& dst = owned(to, j);
        if (!copy(dst.idx_tail, src.idx_tail, a.tail, error) ||
            !copy(dst.idx_dead, src.idx_dead, a.dead, error) ||
            !copy(dst.idx_block_pos, src.idx_block_pos, a.block_pos, error)) return false;
        if (!ids.empty()) {
            const size_t row = ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(dst.idx_pooled + row * g.idx_key_dim, src.idx_dead, a.dead, error)) return false;
        }
    }
    to.ple_prev[0] = ids.size() >= 2 ? ids[ids.size() - 2] : -1;
    to.ple_prev[1] = ids.empty() ? -1 : ids.back();
    return sync(error);
}
