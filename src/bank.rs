use crate::atomic;
use std::io;
use std::path::{Path, PathBuf};

const INDEX_TEMPLATE: &str = "# Bank Index\n\n## Pending\n\n(none)\n\n## Entries\n\n";

/// The tiers every bank operation touches — one small, stable surface
/// per the board's Hashimoto/Cherny guidance, rather than each caller
/// building its own paths. No `clean` tier: CONTEXT.md's design allowed
/// for one, but nothing has ever written to it — removed rather than
/// carried as dead structure (retro action item), to be reintroduced
/// only once a real ingestion pathway actually needs a clean-tier stage.
pub struct BankPaths {
    pub root: PathBuf,
    pub wiki: PathBuf,
    pub raw: PathBuf,
    pub index: PathBuf,
    pub pending: PathBuf,
}

/// The data root for all banks. `MF_DATA_ROOT` overrides for tests /
/// alternate installs; defaults to `~/.trm` per ADR-0001.
pub fn data_root() -> PathBuf {
    if let Ok(over) = std::env::var("MF_DATA_ROOT") {
        return PathBuf::from(over);
    }
    let home = std::env::var("HOME").expect("HOME must be set to locate ~/.trm");
    PathBuf::from(home).join(".trm")
}

pub fn paths_for(data_root: &Path, bank_id: &str) -> BankPaths {
    let root = data_root.join("banks").join(bank_id);
    BankPaths {
        wiki: root.join("wiki"),
        raw: root.join("raw"),
        index: root.join("index.md"),
        pending: root.join("pending"),
        root,
    }
}

/// Create index.md with its fixed sections if it doesn't exist yet.
pub fn ensure_index(index_path: &Path) -> io::Result<()> {
    if !index_path.exists() {
        atomic::write(index_path, INDEX_TEMPLATE)?;
    }
    Ok(())
}

/// Append a one-line entry for `slug` to index.md's Entries section.
/// Milestone 1 keeps this intentionally simple (append at end of file,
/// no structural markdown parsing) — revisit if that gets confusing once
/// Milestone 2 needs to remove/reorder entries during curation.
pub fn append_index_entry(index_path: &Path, slug: &str, summary: &str) -> io::Result<()> {
    ensure_index(index_path)?;
    let mut contents = std::fs::read_to_string(index_path)?;
    contents.push_str(&format!("- `{slug}` — {summary}\n"));
    atomic::write(index_path, &contents)
}

/// First ~60 chars of content, single line, for the index summary.
pub fn summarize(content: &str) -> String {
    let one_line: String = content.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.chars().count() <= 60 {
        one_line
    } else {
        format!("{}…", one_line.chars().take(60).collect::<String>())
    }
}

/// Resolve which bank a given invocation should read/write.
/// Precedence: explicit flag > `.trm-bank` file at the repo root > git
/// remote (of the enclosing repo) > hash of the enclosing repo's root
/// path > "global". The `.trm-bank` file exists for the case a remote-
/// derived or hashed name is wrong — a repo without a remote (or one
/// whose remote slug doesn't match the bank you actually want) can pin
/// an explicit, checked-in bank name instead.
pub fn resolve_bank_id(explicit: Option<&str>, cwd: &Path) -> String {
    if let Some(explicit) = explicit {
        return explicit.to_string();
    }

    match find_repo_root(cwd) {
        Some(root) => mf_bank_file(&root)
            .or_else(|| git_remote_slug(&root))
            .unwrap_or_else(|| hash_path_id(&root)),
        None => "global".to_string(),
    }
}

/// Read a `.trm-bank` file at `repo_root`, if present — its trimmed
/// contents are the bank id, taking priority over remote/hash derivation.
fn mf_bank_file(repo_root: &Path) -> Option<String> {
    let contents = std::fs::read_to_string(repo_root.join(".trm-bank")).ok()?;
    let trimmed = contents.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Walk upward from `start` looking for a `.git` directory.
fn find_repo_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start.to_path_buf();
    loop {
        if dir.join(".git").is_dir() {
            return Some(dir);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Read `.git/config` under `repo_root` and pull the `origin` remote's URL,
/// slugified to `<owner>-<repo>` (or just `<repo>` if no owner segment).
fn git_remote_slug(repo_root: &Path) -> Option<String> {
    let config = std::fs::read_to_string(repo_root.join(".git").join("config")).ok()?;
    let mut in_origin = false;
    for line in config.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_origin = trimmed == "[remote \"origin\"]";
            continue;
        }
        if in_origin
            && let Some(url) = trimmed
                .strip_prefix("url = ")
                .or_else(|| trimmed.strip_prefix("url="))
        {
            return Some(slugify_remote_url(url.trim()));
        }
    }
    None
}

fn slugify_remote_url(url: &str) -> String {
    let stripped = url.trim_end_matches(".git").trim_end_matches('/');
    let tail = stripped.rsplit(['/', ':']).take(2).collect::<Vec<_>>();
    tail.into_iter().rev().collect::<Vec<_>>().join("-")
}

/// Fallback bank id for a repo with no origin remote: a short stable hash
/// of its absolute path, so the same repo always resolves to the same bank.
fn hash_path_id(repo_root: &Path) -> String {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    repo_root.hash(&mut hasher);
    format!("repo-{:x}", hasher.finish() & 0xffff_ffff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn init_repo(root: &Path) {
        fs::create_dir_all(root.join(".git")).unwrap();
    }

    fn set_remote(root: &Path, url: &str) {
        fs::write(
            root.join(".git").join("config"),
            format!("[core]\n\tbare = false\n[remote \"origin\"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n"),
        )
        .unwrap();
    }

    #[test]
    fn explicit_flag_wins_over_everything() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(resolve_bank_id(Some("chosen"), tmp.path()), "chosen");
    }

    #[test]
    fn outside_any_repo_resolves_to_global() {
        let tmp = tempfile::tempdir().unwrap();
        assert_eq!(resolve_bank_id(None, tmp.path()), "global");
    }

    #[test]
    fn repo_with_origin_remote_uses_remote_slug() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path());
        set_remote(tmp.path(), "git@github.com:vectorize-io/hindsight.git");
        assert_eq!(resolve_bank_id(None, tmp.path()), "vectorize-io-hindsight");
    }

    #[test]
    fn repo_without_remote_falls_back_to_path_hash() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path());
        let id = resolve_bank_id(None, tmp.path());
        assert!(id.starts_with("repo-"), "expected repo-<hash>, got {id}");
    }

    #[test]
    fn two_different_repos_resolve_to_different_banks() {
        let tmp_a = tempfile::tempdir().unwrap();
        let tmp_b = tempfile::tempdir().unwrap();
        init_repo(tmp_a.path());
        init_repo(tmp_b.path());
        assert_ne!(
            resolve_bank_id(None, tmp_a.path()),
            resolve_bank_id(None, tmp_b.path())
        );
    }

    #[test]
    fn resolves_from_a_subdirectory_of_the_repo() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path());
        set_remote(tmp.path(), "https://github.com/artisenalcode/mindforge");
        let sub = tmp.path().join("src").join("deep");
        fs::create_dir_all(&sub).unwrap();
        assert_eq!(resolve_bank_id(None, &sub), "artisenalcode-mindforge");
    }

    #[test]
    fn mf_bank_file_overrides_remote_derived_name() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path());
        set_remote(tmp.path(), "git@github.com:someorg/somerepo.git");
        fs::write(tmp.path().join(".trm-bank"), "mindforge\n").unwrap();
        assert_eq!(resolve_bank_id(None, tmp.path()), "mindforge");
    }

    #[test]
    fn mf_bank_file_overrides_hash_fallback_when_no_remote() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path());
        fs::write(tmp.path().join(".trm-bank"), "mindforge").unwrap();
        assert_eq!(resolve_bank_id(None, tmp.path()), "mindforge");
    }

    #[test]
    fn mf_bank_file_applies_from_a_subdirectory_too() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path());
        fs::write(tmp.path().join(".trm-bank"), "mindforge").unwrap();
        let sub = tmp.path().join("trm").join("src");
        fs::create_dir_all(&sub).unwrap();
        assert_eq!(resolve_bank_id(None, &sub), "mindforge");
    }

    #[test]
    fn explicit_flag_still_beats_mf_bank_file() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path());
        fs::write(tmp.path().join(".trm-bank"), "mindforge").unwrap();
        assert_eq!(resolve_bank_id(Some("override"), tmp.path()), "override");
    }

    #[test]
    fn empty_mf_bank_file_falls_through_to_hash() {
        let tmp = tempfile::tempdir().unwrap();
        init_repo(tmp.path());
        fs::write(tmp.path().join(".trm-bank"), "   \n").unwrap();
        assert!(resolve_bank_id(None, tmp.path()).starts_with("repo-"));
    }

    #[test]
    fn paths_for_derives_the_four_bank_subpaths() {
        let data_root = Path::new("/tmp/example-root");
        let paths = paths_for(data_root, "global");
        assert_eq!(paths.root, data_root.join("banks/global"));
        assert_eq!(paths.wiki, data_root.join("banks/global/wiki"));
        assert_eq!(paths.raw, data_root.join("banks/global/raw"));
        assert_eq!(paths.index, data_root.join("banks/global/index.md"));
        assert_eq!(paths.pending, data_root.join("banks/global/pending"));
    }

    #[test]
    fn ensure_index_creates_template_once_and_is_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().join("index.md");
        ensure_index(&index_path).unwrap();
        let first = fs::read_to_string(&index_path).unwrap();
        assert!(first.contains("## Pending"));
        assert!(first.contains("## Entries"));
        ensure_index(&index_path).unwrap();
        assert_eq!(fs::read_to_string(&index_path).unwrap(), first);
    }

    #[test]
    fn append_index_entry_adds_a_line_with_slug_and_summary() {
        let tmp = tempfile::tempdir().unwrap();
        let index_path = tmp.path().join("index.md");
        append_index_entry(&index_path, "some-slug-abc123", "a one-line summary").unwrap();
        let contents = fs::read_to_string(&index_path).unwrap();
        assert!(contents.contains("`some-slug-abc123` — a one-line summary"));
    }

    #[test]
    fn summarize_truncates_long_content_with_ellipsis() {
        let long = "word ".repeat(30);
        let summary = summarize(&long);
        assert!(summary.ends_with('…'));
        assert!(summary.chars().count() <= 61);
    }

    #[test]
    fn summarize_leaves_short_content_untouched() {
        assert_eq!(summarize("short fact"), "short fact");
    }
}
