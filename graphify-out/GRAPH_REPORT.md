# Graph Report - /home/alvin/Code/_labs/total-recall  (2026-08-06)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 195 nodes · 442 edges · 9 communities (8 shown, 1 thin omitted)
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Community 0
- Community 1
- Community 2
- Community 3
- Community 4
- Community 5
- Community 6
- Community 7
- Community 8

## God Nodes (most connected - your core abstractions)
1. `scan()` - 14 edges
2. `stage()` - 13 edges
3. `acquire()` - 13 edges
4. `test_paths()` - 12 edges
5. `semantic_search()` - 12 edges
6. `BankPaths` - 11 edges
7. `init_repo()` - 11 edges
8. `main()` - 10 edges
9. `write()` - 10 edges
10. `resolve_bank_id()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `scan()` --references--> `BankPaths`  [EXTRACTED]
  src/curator.rs → src/bank.rs
- `stage_candidate()` --references--> `BankPaths`  [EXTRACTED]
  src/curator.rs → src/bank.rs
- `test_paths()` --references--> `BankPaths`  [EXTRACTED]
  src/curator.rs → src/bank.rs
- `semantic_search()` --references--> `Embedder`  [EXTRACTED]
  src/wiki.rs → src/embeddings.rs
- `acquire_with_retry()` --references--> `LockGuard`  [EXTRACTED]
  src/embeddings.rs → src/lock.rs

## Import Cycles
- None detected.

## Communities (9 total, 1 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.13
Nodes (31): append_index_entry(), append_index_entry_adds_a_line_with_slug_and_summary(), data_root(), empty_mf_bank_file_falls_through_to_hash(), ensure_index(), ensure_index_creates_template_once_and_is_idempotent(), explicit_flag_still_beats_mf_bank_file(), find_repo_root() (+23 more)

### Community 1 - "Community 1"
Cohesion: 0.15
Nodes (29): Into, BankPaths, as_prompt_includes_kind_description_and_sources(), complete(), complete_writes_wiki_entry_updates_index_and_clears_pending(), direct_source_gets_no_untrusted_warning(), external_source_gets_untrusted_warning(), get_prompt() (+21 more)

### Community 2 - "Community 2"
Cohesion: 0.18
Nodes (28): ExitCode, Cli, Commands, complete_handover(), curator_scan(), explicit_bank_flag_overrides_auto_resolution(), find_md_files(), import() (+20 more)

### Community 3 - "Community 3"
Cohesion: 0.20
Nodes (23): embedder(), list(), list_returns_all_written_slugs_sorted(), RankedMatch, read(), Path, Result, String (+15 more)

### Community 4 - "Community 4"
Cohesion: 0.17
Nodes (19): Display, Drop, Formatter, acquire(), acquire_on_fresh_bank_succeeds_and_writes_our_pid(), acquiring_a_live_held_lock_fails_fast(), acquiring_a_stale_dead_pid_lock_reclaims_it(), is_pid_alive() (+11 more)

### Community 5 - "Community 5"
Cohesion: 0.15
Nodes (13): Duration, acquire_with_retry(), build_model(), concurrent_first_time_downloads_do_not_corrupt_each_other(), Embedder, once_warm_concurrent_loads_all_succeed_quickly_via_retry(), Path, PathBuf (+5 more)

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (17): empty_bank_produces_no_candidates(), exact_dup_pair_is_not_also_staged_by_the_fuzzy_pass(), exact_duplicate_content_is_staged_as_exact_dup_not_fuzzy_curation(), fully_contained_content_is_staged_as_exact_dup(), is_exact_or_contained(), non_duplicate_entries_still_reach_the_similarity_pass(), Result, String (+9 more)

### Community 7 - "Community 7"
Cohesion: 0.53
Nodes (5): Path, Result, write(), write_creates_final_file_with_no_leftover_temp(), write_overwrites_existing_file()

## Knowledge Gaps
- **1 isolated node(s):** `total-recall`
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BankPaths` connect `Community 1` to `Community 0`, `Community 6`?**
  _High betweenness centrality (0.421) - this node is a cross-community bridge._
- **Why does `semantic_search()` connect `Community 3` to `Community 5`?**
  _High betweenness centrality (0.359) - this node is a cross-community bridge._
- **Why does `Embedder` connect `Community 5` to `Community 3`?**
  _High betweenness centrality (0.341) - this node is a cross-community bridge._
- **What connects `total-recall` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Community 0` be split into smaller, more focused modules?**
  _Cohesion score 0.12941176470588237 - nodes in this community are weakly interconnected._
- **Should `Community 1` be split into smaller, more focused modules?**
  _Cohesion score 0.1497326203208556 - nodes in this community are weakly interconnected._