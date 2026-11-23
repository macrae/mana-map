"""THE CORE: the contracts this bench may never break, and the tests that hold them.

`make smoke` runs exactly these (`-m core`, ~20 s). Every commit's gate runs them,
whatever else it selects; a harness edit that cannot know what it touched runs them
first. They are the rules a deck lives or dies by — a broken one gives the pilot a
confident wrong answer, which is worse than no answer:

  * the INSTRUMENT is honest: the same seed gives the same game, two lists are read
    on the same draws, a list against itself reads exactly zero, a family of rows
    carries its correction, an interval sits on the difference, absent is not zero;
  * the PILOT'S WORD binds: the keep list refuses every way out, a sleeved deck
    changes only through a branch, undo restores the bytes, the paper lock moves
    only on his act, a decided study is frozen;
  * the RULES hold: the bracket floor (Game Changers, two-card infinites), the
    deck's legality, and an engine edge exists only when one card's event is
    accepted by another card's filter;
  * the GATES gate: every validator fails through its exit code.

One home, reviewed as a list: an entry is `"<file>::<test function>"` (a
parametrized test is matched by its function name, every case included) and the
rule it guards. `tests/conftest.py` marks them at collection; `test_core_contracts.py`
holds every entry to a collected UNIT-tier test, so the smoke run can never read
tracked data, need Forge, or silently shrink.
"""

CORE = {
    # ── the instrument is honest ─────────────────────────────────────────────
    "test_pilot_goldfish.py::test_determinism_same_seed":
        "the same seed plays the same game — every paired read rests on it",
    "test_pilot_goldfish.py::test_different_seeds_differ":
        "a seed is a real input, not a constant",
    "test_pilot_goldfish.py::test_every_model_channel_is_off_unless_a_deck_opts_in":
        "a channel a deck did not declare changes nothing about its games",
    "test_pilot_goldfish.py::test_a_death_rate_without_a_source_is_refused":
        "an authored rate must name where it came from",
    "test_pilot_try_and_protected.py::test_align_keeps_shared_cards_in_their_slots":
        "two lists are read on the same draws — shared cards keep their slots",
    "test_pilot_net_change.py::test_objective_not_met_is_a_refusal_even_when_other_things_improved":
        "the pre-registered objective rules its own row; side gains don't rescue it",
    "test_pilot_net_change.py::test_a_clock_is_never_described_as_a_win_rate":
        "the goldfish's clock is never sold as a win rate",
    "test_pilot_net_change.py::test_a_champion_reading_of_zero_does_not_divide_by_it":
        "a zero baseline never produces an infinite effect",
    "test_sim_stats.py::test_the_difference_interval_is_oriented_b_minus_a":
        "a difference reads B minus A, always",
    "test_sim_stats.py::test_two_overlapping_wilson_intervals_can_still_differ":
        "the interval is on the difference, not the overlap of two",
    "test_sim_stats.py::test_two_constant_arms_do_not_claim_an_estimate":
        "no variance, no estimate",
    "test_sim_stats.py::test_permutation_is_seeded_and_replays":
        "a p-value replays from its seed",
    "test_sim_stats.py::test_permutation_finds_a_real_difference_and_not_a_fake_one":
        "the test has power and keeps its false-positive rate",
    "test_pilot_diagnostic.py::test_a_frequency_never_erases_a_difference_the_interval_shows":
        "a rounded frequency never hides what the interval shows",

    # ── the pilot's word binds ───────────────────────────────────────────────
    "test_pilot_deck_edit.py::test_the_keep_list_refuses_every_way_out":
        "the keep list: no edit path removes a protected card",
    "test_pilot_deck_edit.py::test_a_sleeved_deck_is_refused_with_the_branch_path":
        "a sleeved list changes only through a branch",
    "test_pilot_deck_edit.py::test_every_lifecycle_status_is_refused_with_revive":
        "an archived deck is read-only until revived",
    "test_pilot_deck_edit.py::test_undo_and_redo_round_trip_the_bytes":
        "undo restores the exact list",
    "test_pilot_deck_edit.py::test_a_new_edit_clears_redo":
        "redo never replays over a newer edit",
    "test_pilot_deck_edit.py::test_a_stale_page_is_refused":
        "an edit from a stale page never overwrites a newer list",
    "test_pilot_deck_edit.py::test_a_lock_held_by_another_process_blocks":
        "two writers never interleave on one deck",
    "test_pilot_deck_state.py::test_archiving_withdraws_the_paper_lock":
        "the paper lock leaves with the deck",
    "test_pilot_deck_state.py::test_superseding_a_sleeved_deck_leaves_the_lock_alone":
        "superseding is not unsleeving",
    "test_pilot_deck_state.py::test_locking_an_archived_deck_is_refused":
        "an archived deck cannot be sleeved",
    "test_pilot_deck_versions.py::test_restore_refuses_to_cut_a_protected_card":
        "the keep list holds through a version restore",
    "test_pilot_deck_versions.py::test_a_sleeved_deck_restores_only_to_its_paper_version":
        "a sleeved deck restores only to what is in the sleeves",

    "test_pilot_study.py::test_goldfish_may_never_be_the_sole_decider":
        "the goldfish reports; it never decides alone",
    "test_pilot_study.py::test_goldfish_in_decided_by_is_refused_when_a_named_card_is_dark":
        "the goldfish never decides about a card it cannot read",
    "test_pilot_study.py::test_validate_study_refuses_a_decision_with_no_decided_by":
        "a decision names who decided, in the pilot's words",
    "test_pilot_study.py::test_an_uncited_figure_in_the_reading_is_refused_and_ids_are_not_numbers":
        "every figure in a call cites a fact",
    "test_pilot_study.py::test_a_decided_study_is_frozen_unless_forced":
        "a decided study is frozen",

    # ── the rules hold ───────────────────────────────────────────────────────
    "test_pilot_bracket.py::test_no_game_changers_leaves_floor_at_one":
        "bracket floor: no Game Changers, no raise",
    "test_pilot_bracket.py::test_up_to_three_game_changers_forces_bracket_three":
        "bracket floor: one to three Game Changers is bracket 3",
    "test_pilot_bracket.py::test_more_than_three_game_changers_forces_bracket_four":
        "bracket floor: four or more is bracket 4",
    "test_pilot_bracket.py::test_early_two_card_infinite_forces_four":
        "bracket floor: an early two-card infinite is bracket 4",
    "test_pilot_bracket.py::test_three_card_infinite_is_not_a_two_card_infinite":
        "bracket floor: three cards is not two",
    "test_pilot_bracket.py::test_excluded_combo_does_not_raise_the_floor":
        "bracket floor: an excluded line never raises it",
    "test_pilot_deck_edit.py::test_a_duplicate_is_refused_in_a_singleton_format":
        "singleton is singleton",
    "test_pilot_deck_edit.py::test_an_off_identity_add_is_refused":
        "colour identity is enforced",
    "test_pilot_deck_edit.py::test_a_lone_add_to_a_commander_hundred_is_refused":
        "a Commander deck stays at one hundred",
    "test_pilot_ability_graph.py::test_blood_artist_consumes_dies_of_self_or_another_creature":
        "a trigger reads exactly the event its script names",
    "test_pilot_ability_graph.py::test_teysa_refuses_a_white_token_and_takes_a_black_one":
        "a filter's every predicate is checked — colour included",
    "test_pilot_ability_graph.py::test_rhystic_study_consumes_an_opponents_cast_not_yours":
        "whose event it is matters",
    "test_pilot_ability_graph.py::test_an_unmapped_mode_is_reported_never_guessed":
        "an unread ability is named, never invented",
    "test_pilot_engine_graph.py::test_an_edge_needs_the_event_and_every_predicate":
        "an engine edge exists only when one card's event is accepted by another's filter",
    "test_pilot_engine_graph.py::test_a_doubler_is_a_modifier_not_an_edge":
        "a doubler scales an event; it does not feed a card",
    "test_pilot_engine_graph.py::test_a_stack_stamp_needs_a_passing_scenario_naming_every_card":
        "only a checker-passed stack stamps a line",
    "test_pilot_engine_graph.py::test_the_record_is_byte_identical_across_runs":
        "the computed engine is deterministic",

    # ── the gates gate ───────────────────────────────────────────────────────
    "test_validators_actually_fail.py::test_a_validator_signals_failure_through_the_exit_code":
        "every validator fails through its exit code",
    "test_validators_actually_fail.py::test_the_dispatcher_still_discards_return_values":
        "the dispatcher cannot swallow a failure",
    "test_mechanical_tags.py::test_tags_are_sorted":
        "the model-facing tag vector is stable",
    "test_mechanical_tags.py::test_encode_tags_multihot_basic":
        "the frozen tag encoding the ability model was trained on",
}


#: SACRED RULES WITH NO FAST GUARD YET. Each is held today only by a test that reads the
#: tracked fleet (regression tier), so it cannot sit in the smoke run. Phase C of the
#: 2026-11-22 test plan writes an inline-data version of each and moves it into CORE;
#: until then `test_core_contracts.py` holds every name here to a real collected test,
#: so the debt cannot be forgotten or renamed away.
PENDING = {
    "test_pilot_diagnostic.py::test_an_absent_declaration_gives_an_absent_figure_never_a_zero":
        "absent means absent, never zero",
    "test_pilot_net_change.py::test_a_staged_swap_pairs_BOTH_ways_and_the_reason_is_recoverable":
        "a branch and its deck are paired game by game, both directions",
    "test_pilot_try_and_protected.py::test_a_list_against_itself_pairs_to_exactly_zero_and_two_seeds_make_no_call":
        "A/A reads exactly zero; noise is never a call",
    "test_pilot_try_and_protected.py::test_a_protected_cut_is_refused_by_the_swap_arithmetic":
        "the keep list: a protected card is never cut by a swap",
    "test_pilot_try_and_protected.py::test_propose_and_merge_refuse_a_branch_that_cuts_it":
        "the keep list: a branch cutting it can't be proposed or merged",
    "test_pilot_try_and_protected.py::test_the_validator_fails_the_broken_shapes":
        "the keep list's own shape is gated",
    "test_pilot_try_and_protected.py::test_try_refuses_a_protected_cut_before_measuring_anything":
        "the keep list: try refuses before spending a game",
}
