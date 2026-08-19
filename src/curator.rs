use crate::{atomic, bank, embeddings, handover, wiki};

/// Local duplicate-candidate finder: cosine similarity between every pair of wiki entries. Not a judgment -- stages Curation handovers for the harness.
/// Known limitation: all-MiniLM-L6-v2's fixed 256-token context truncates long entries before embedding.
pub fn scan(paths: &bank::BankPaths, threshold: f64) -> Result<Vec<String>, String> {
    let slugs = wiki::list(&paths.wiki).map_err(|e| e.to_string())?;
    if slugs.len() < 2 {
        return Ok(Vec::new());
    }

    let mut contents = Vec::with_capacity(slugs.len());
    for slug in &slugs {
        contents.push(wiki::read(&paths.wiki, slug).map_err(|e| e.to_string())?);
    }

    let mut embedder = embeddings::Embedder::new(bank::data_root().join("models"))?;
    let vectors = embedder.embed(&contents)?;

    let mut staged = Vec::new();
    for i in 0..slugs.len() {
        for j in (i + 1)..slugs.len() {
            // Zero-tuning case: an exact/contained match skips the embedding comparison entirely.
            if is_exact_or_contained(&contents[i], &contents[j]) {
                let job_id = format!("exact-dup-{}-{}", slugs[i], slugs[j]);
                let reason = format!(
                    "{} and {} are identical or fully-contained content — high-confidence duplicate, no threshold involved",
                    slugs[i], slugs[j]
                );
                stage_candidate(paths, &job_id, reason, &slugs[i], &slugs[j])
                    .map_err(|e| e.to_string())?;
                staged.push(job_id);
                continue;
            }

            let sim = embeddings::cosine_similarity(&vectors[i], &vectors[j]) as f64;
            if sim < threshold {
                continue;
            }
            let job_id = format!("curation-{}-{}", slugs[i], slugs[j]);
            let reason = format!(
                "{} and {} are {:.0}% semantically similar (local embedding cosine similarity) — judge whether to merge/retire",
                slugs[i],
                slugs[j],
                sim * 100.0
            );
            stage_candidate(paths, &job_id, reason, &slugs[i], &slugs[j])
                .map_err(|e| e.to_string())?;
            staged.push(job_id);
        }
    }
    Ok(staged)
}

fn stage_candidate(
    paths: &bank::BankPaths,
    job_id: &str,
    reason: String,
    slug_a: &str,
    slug_b: &str,
) -> std::io::Result<()> {
    let sources = vec![
        paths.wiki.join(format!("{slug_a}.md")),
        paths.wiki.join(format!("{slug_b}.md")),
    ];
    let task = handover::HandoverTask::new(
        handover::HandoverKind::Curation,
        reason,
        sources,
        "internal",
    );
    atomic::write(
        &paths.pending.join(format!("{job_id}.md")),
        &task.as_prompt(),
    )
}

/// True if the two contents are identical after trimming, or one is fully contained in the other.
fn is_exact_or_contained(a: &str, b: &str) -> bool {
    let (a, b) = (a.trim(), b.trim());
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a == b || a.contains(b) || b.contains(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_paths() -> (tempfile::TempDir, bank::BankPaths) {
        let data_root = tempfile::tempdir().unwrap();
        let paths = bank::paths_for(data_root.path(), "test-bank");
        (data_root, paths)
    }

    #[test]
    fn semantically_similar_entries_get_staged_as_curation_candidates() {
        let (_data_root, paths) = test_paths();
        wiki::write_named(
            &paths.wiki,
            "a",
            "the user prefers short, terse commit messages",
        )
        .unwrap();
        wiki::write_named(
            &paths.wiki,
            "b",
            "commits from this user should be brief and to the point",
        )
        .unwrap();

        let staged = scan(&paths, 0.6).unwrap();
        assert_eq!(staged.len(), 1);
        assert!(paths.pending.join(format!("{}.md", staged[0])).exists());
    }

    #[test]
    fn unrelated_entries_are_not_staged() {
        let (_data_root, paths) = test_paths();
        wiki::write_named(&paths.wiki, "a", "user prefers terse commit messages").unwrap();
        wiki::write_named(
            &paths.wiki,
            "b",
            "podman containers require explicit network setup",
        )
        .unwrap();

        let staged = scan(&paths, 0.6).unwrap();
        assert!(staged.is_empty());
    }

    #[test]
    fn staged_curation_marker_names_both_entries_and_similarity() {
        let (_data_root, paths) = test_paths();
        wiki::write_named(
            &paths.wiki,
            "boris-cherny",
            "sub-agents get a fresh, isolated context window from the parent",
        )
        .unwrap();
        wiki::write_named(
            &paths.wiki,
            "jack-roberts",
            "a sub-agent's context is isolated and separate from its parent's",
        )
        .unwrap();

        let staged = scan(&paths, 0.6).unwrap();
        assert_eq!(staged.len(), 1);
        let marker =
            std::fs::read_to_string(paths.pending.join(format!("{}.md", staged[0]))).unwrap();
        assert!(marker.contains("Curation"));
        assert!(marker.contains("boris-cherny"));
        assert!(marker.contains("jack-roberts"));
        assert!(marker.contains('%'));
    }

    #[test]
    fn empty_bank_produces_no_candidates() {
        let (_data_root, paths) = test_paths();
        std::fs::create_dir_all(&paths.wiki).unwrap();
        assert!(scan(&paths, 0.6).unwrap().is_empty());
    }

    #[test]
    fn single_entry_bank_produces_no_candidates_without_loading_a_model() {
        let (_data_root, paths) = test_paths();
        wiki::write_named(&paths.wiki, "only-one", "a lone entry").unwrap();
        // No pairs to compare -- scan should return early, never construct an Embedder.
        assert!(scan(&paths, 0.6).unwrap().is_empty());
    }

    #[test]
    fn exact_duplicate_content_is_staged_as_exact_dup_not_fuzzy_curation() {
        let (_data_root, paths) = test_paths();
        wiki::write_named(&paths.wiki, "a", "identical content here").unwrap();
        wiki::write_named(&paths.wiki, "b", "identical content here").unwrap();

        // Threshold above cosine's max of 1.0 rejects every embedding match, proving the exact check caught this pair.
        let staged = scan(&paths, 1.01).unwrap();
        assert_eq!(staged.len(), 1);
        assert!(staged[0].starts_with("exact-dup-"));
    }

    #[test]
    fn fully_contained_content_is_staged_as_exact_dup() {
        let (_data_root, paths) = test_paths();
        wiki::write_named(&paths.wiki, "short", "the core fact").unwrap();
        wiki::write_named(&paths.wiki, "long", "prefix: the core fact :suffix").unwrap();

        let staged = scan(&paths, 1.01).unwrap();
        assert_eq!(staged.len(), 1);
        assert!(staged[0].starts_with("exact-dup-"));
    }

    #[test]
    fn exact_dup_pair_is_not_also_staged_by_the_fuzzy_pass() {
        let (_data_root, paths) = test_paths();
        wiki::write_named(&paths.wiki, "a", "identical content here").unwrap();
        wiki::write_named(&paths.wiki, "b", "identical content here").unwrap();

        // Low threshold too -- if the exact check didn't `continue`, the fuzzy pass would also match and double-stage.
        let staged = scan(&paths, 0.1).unwrap();
        assert_eq!(
            staged.len(),
            1,
            "pair should be staged exactly once, not twice"
        );
    }

    #[test]
    fn scan_completes_within_a_bounded_time_at_realistic_bank_scale() {
        // Tripwire, not a performance guarantee. 90s budget (not 30s) because CI's debug-mode candle matrix ops measured 50s on GitHub's runner.
        const TOPICS: [&str; 50] = [
            "the user prefers tabs over spaces in Python",
            "podman containers require explicit network setup on this host",
            "the coffee machine in the office needs descaling monthly",
            "quarterly revenue grew twelve percent year over year",
            "the cat is allergic to a specific brand of cat litter",
            "rust's borrow checker rejects this pattern at compile time",
            "the hiking trail closes seasonally due to snowfall",
            "the invoice template needs a new tax line item",
            "the guitar has a buzzing fret on the low E string",
            "database migrations must run before the app boots",
            "the recipe calls for browned butter, not melted",
            "the flight was delayed due to a mechanical issue",
            "the garden needs more shade for the ferns",
            "the meeting was rescheduled to Thursday afternoon",
            "the printer is out of magenta toner again",
            "the bridge is closed for structural repairs",
            "the API rate limit resets at midnight UTC",
            "the dog needs a rabies booster this year",
            "the novel's plot twist happens in chapter twelve",
            "the thermostat schedule needs updating for winter",
            "the client wants the logo in a warmer blue",
            "the marathon route changed due to construction",
            "the spreadsheet formula has a circular reference",
            "the plane's boarding group was announced early",
            "the orchestra is rehearsing a new symphony",
            "the server's disk usage crossed eighty percent",
            "the bakery sells out of croissants by nine",
            "the lecture covered thermodynamics in depth",
            "the router firmware update fixed the dropouts",
            "the museum exhibit closes at the end of the month",
            "the compiler warning points to an unused import",
            "the vineyard's harvest starts earlier this year",
            "the subway line is running a reduced schedule",
            "the thesis defense is scheduled for next week",
            "the greenhouse humidity sensor needs calibration",
            "the podcast episode ran twenty minutes over",
            "the firmware bug only reproduces on cold boot",
            "the choir is learning a piece in a minor key",
            "the tide pools are best visited at low tide",
            "the spreadsheet macro fails on empty rows",
            "the violin needs new strings before the recital",
            "the drone's battery lasts about twenty minutes",
            "the bakery's sourdough starter is ten years old",
            "the satellite passes overhead twice a day",
            "the lecture hall projector needs a new bulb",
            "the marathon training plan peaks in week ten",
            "the greenhouse tomatoes are ready to harvest",
            "the orchestra's brass section needs more rehearsal",
            "the router's firmware has a known security patch",
            "the museum's new wing opens next spring",
        ];
        let (_data_root, paths) = test_paths();
        for (i, topic) in TOPICS.iter().enumerate() {
            wiki::write_named(&paths.wiki, &format!("entry-{i}"), topic).unwrap();
        }

        let start = std::time::Instant::now();
        let staged = scan(&paths, 0.9).unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_secs() < 90,
            "scan of 50 entries took {elapsed:?}, expected well under 90s"
        );
        // Genuinely distinct topics at a high (0.9) threshold shouldn't false-positive into staged candidates.
        assert!(staged.is_empty(), "unexpected candidates: {staged:?}");
    }

    #[test]
    fn non_duplicate_entries_still_reach_the_similarity_pass() {
        let (_data_root, paths) = test_paths();
        wiki::write_named(
            &paths.wiki,
            "a",
            "the user prefers short, terse commit messages",
        )
        .unwrap();
        wiki::write_named(
            &paths.wiki,
            "b",
            "commits from this user should be brief and to the point",
        )
        .unwrap();

        let staged = scan(&paths, 0.6).unwrap();
        assert_eq!(staged.len(), 1);
        assert!(
            staged[0].starts_with("curation-"),
            "expected a similarity match, got {}",
            staged[0]
        );
    }
}
