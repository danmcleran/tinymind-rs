---
name: crate-release
description: "Release a new version of the tinymind-rs crate (crate `tinymind`) end to end: docs audit of README, crate docs and the GitHub Pages site against the tree (a hard gate before the release PR), the release PR, cargo publish, annotated tag, GitHub release with the Pages link last, verification on crates.io, docs.rs and Pages, and the LinkedIn post. Use when the user asks to release, publish or tag a new tinymind-rs / `tinymind` crate version (e.g. 'release 0.1.5', 'cut a crate release'). For the C++ tinymind library use the user-level `release` skill instead."
---

# crate-release

Release `tinymind` (repo `danmcleran/tinymind-rs`). Every standing rule is
enforced by a command whose output is checked, not remembered. The point of
the docs gate: crates.io and docs.rs show the README and crate docs **as
published**, frozen with the version, so whatever is stale at `cargo publish`
stays stale on crates.io until the next release.

## Constants

- Repo: `danmcleran/tinymind-rs`, default branch `master`
- Pages site: `https://danmcleran.github.io/tinymind-rs/` (the link that ends
  the release notes and leads the LinkedIn post)
- crates.io: `https://crates.io/crates/tinymind`; docs.rs: `https://docs.rs/tinymind`
- LinkedIn post: `~/Documents/tinymind/tinymind-rs-vX.Y.Z/linkedin-post.md`

## Hard rules

1. **Go-ahead.** `cargo publish` only after Dan's explicit go-ahead for *this*
   version; a version can be yanked, never deleted. Agents run the rest of the
   release in-session.
2. **Token.** Dan runs `cargo login` himself. Never read or print
   `~/.cargo/credentials.toml` (checking it exists is fine). If publish fails on
   authentication, stop and ask Dan to log in again.
3. **Docs before the tag.** Step 3 (audit) must find nothing open, and
   `scripts/check-docs.sh --release X.Y.Z` must print `ok` on the release PR,
   before anything is published or tagged.
4. **Semver.** Before 1.0: additions only is a patch bump, anything that can
   break a caller is a minor bump. A new default type parameter, a new trait
   method on an unsealed trait or a changed signature is not "additions only";
   check `git diff vLAST..master -- src` for each.
5. **Annotated tag with `--cleanup=verbatim`** (git's default cleanup deletes
   every line starting with `#`, i.e. every Markdown heading in the annotation).
6. **The release notes end with the Pages line, then the changelog link.**
   Checked in Step 7.
7. **Public prose in full sentences** (release notes, tag annotation, LinkedIn
   post), even when a compressed chat style is active. Project names lowercase;
   the C++ library is "tinymind (C++)", this project "tinymind-rs", the crate
   `tinymind` in backticks (`scripts/cpp-name.py --check` on the notes file). No invented numbers: every figure comes from the
   tree, a run, or the goldens.
8. **PR flow** for every change: branch, PR, CI green, squash-merge, delete the
   branch. Never push to `master`.

## Steps

### 1. Version and state

```bash
git fetch --tags -q && git checkout master && git pull -q
git describe --tags --abbrev=0                       # last release, vLAST
git log --oneline vLAST..master                      # what ships
scripts/upstream-changes.sh                          # tinymind commits not yet ported
```

Propose the version from rule 4 and the commits. Anything `upstream-changes.sh`
lists is ported first (its own PR) or consciously deferred and said in the report.

### 2. CHANGELOG completeness

Every user-visible change in `git log vLAST..master` has an `[Unreleased]`
entry (public API, examples, tools, fixes, MSRV). Repository-only changes
(CI, docs-only PRs) need none. `scripts/check-docs.sh` already fails if a new
top-level public item is not named in the CHANGELOG.

### 3. Docs audit (hard gate)

a. **Mechanical:** `scripts/check-docs.sh` prints `check-docs: ok`. It checks
   examples (both quick starts, the README table, a site page), public modules
   (README table, crate docs), test suites and tools (README), site links,
   nav order, data attribution, new public items in the CHANGELOG, and
   lowercase project names. The two counts the docs quote are checked where
   they are computed: run `PYTHON=<venv python> scripts/compare-with-cpp.sh`
   (it ends with `check-counts: comparisons = N ...`) and let CI's kani job
   check the harness count.

b. **Judgment checklist.** For each change in Step 1, confirm README.md,
   src/lib.rs and the site say what the code does:
   - **README.md:** Status paragraph (what is ported, hardware), Use it, Quick
     start and the example paragraph, "What is ported" table, "How equivalence
     is proven".
   - **Crate docs:** `src/lib.rs` scope paragraph, the module docs of every
     changed module, rustdoc examples still the best entry point.
   - **Site:** `docs/index.md` (buttons, "What it is", Status table, quick
     start), the example's own page (re-run the example; re-run
     `scripts/plot-examples.py` if its output changed; every number on the
     page from the goldens or a fresh run), `docs/equivalence.md` ("What is
     compared", "How it is checked"), `docs/int8.md`, `docs/trees.md`,
     `docs/benchmarks.md` and `bench/README.md` (re-run the boards if arithmetic
     changed), `docs/bit-for-bit.md` ("What is in the crate" list).
   - **"Not ported" / "Not yet" lists** everywhere: grep README.md and docs/
     for anything this release ports (`grep -rn "not ported\|Not yet" README.md docs`).
   - **CLAUDE.md:** layout rows, current state, next steps.

c. **Fresh eyes.** Spawn one read-only subagent with the facts of this release
   (what was added, what changed) and ask it to verify every claim in
   README.md, src/lib.rs, the module docs, docs/*.md, bench/README.md and the
   CHANGELOG against the code, reporting file:line, the wrong text, evidence
   and a replacement, errors separate from omissions. Fix everything real.
   (The first such audit, before 0.1.4, found 6 errors and 12 omissions that
   the authors had missed.)

d. **Land the fixes** in a docs PR before the release PR, or in the release PR
   if they are small. Do not continue with anything open.

### 4. Release PR

On a branch `release-X.Y.Z`:

- `Cargo.toml` `version = "X.Y.Z"`
- CHANGELOG: move `[Unreleased]` entries under `## [X.Y.Z] - YYYY-MM-DD`,
  leave `[Unreleased]` empty, update the comparison links at the bottom
- `cargo update -p tinymind` in `.`, `ci/panic-never` and `bench`
- `docs/bit-for-bit.md`: `As of X.Y.Z:` and its "What is in the crate" list
- CLAUDE.md: add the version to the release list under "Publishing"

Then, all of which must pass:

```bash
scripts/check-docs.sh --release X.Y.Z       # must print: check-docs: ok (release X.Y.Z)
cargo fmt --check && (cd ci/panic-never && cargo fmt --check)
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
PYTHON=<venv python> scripts/compare-with-cpp.sh
```

Open the PR, wait for every check (`rust`, `kani`, `cpp-equivalence`, `build
site`), squash-merge, delete the branch, and wait for CI and Pages to pass on
`master`.

### 5. Publish

From a clean `master` at the merge commit:

```bash
git status --short                            # must be empty
cargo publish --dry-run
cargo publish                                 # rule 1 and rule 2
```

### 6. Tag

Write the notes file (`$NOTES`, in the scratchpad):

```
## tinymind X.Y.Z: <themed title>

<the CHANGELOG entry for X.Y.Z: ### Added / ### Changed / ### Fixed>

Documentation, examples and the API reference: https://danmcleran.github.io/tinymind-rs/

Full changelog: https://github.com/danmcleran/tinymind-rs/compare/vLAST...vX.Y.Z
```

```bash
git tag -a vX.Y.Z --cleanup=verbatim -F "$NOTES" <merge commit>
ANN=$(git for-each-ref --format='%(contents)' refs/tags/vX.Y.Z)
[ "$(printf '%s\n' "$ANN" | grep -c '^#')" = "$(grep -c '^#' "$NOTES")" ] \
  && [ "$(printf '%s\n' "$ANN" | head -1)" = "$(head -1 "$NOTES")" ] \
  && echo "OK: annotation keeps every heading" \
  || echo "FAIL: git tag -d vX.Y.Z and retag with --cleanup=verbatim"
git push origin vX.Y.Z                        # only after OK
```

### 7. GitHub release

```bash
gh release create vX.Y.Z --repo danmcleran/tinymind-rs --verify-tag \
  --title "tinymind X.Y.Z: <themed title>" --notes-file "$NOTES"
B=$(gh release view vX.Y.Z --repo danmcleran/tinymind-rs --json body -q .body)
printf '%s\n' "$B" | tail -3 | grep -q 'danmcleran.github.io/tinymind-rs' \
  && echo "OK: Pages link at the end" || echo "FAIL: Pages link missing at the end"
printf '%s\n' "$B" | tail -1 | grep -q "compare/vLAST...vX.Y.Z" \
  && echo "OK: changelog link last" || echo "FAIL: changelog link"
gh release view vX.Y.Z --repo danmcleran/tinymind-rs --json name,body -q '.name + .body' \
  | grep -qE 'TinyMind|Tinymind' && echo "FAIL: camel-case name" || echo "OK: lowercase"
```

Fix with `gh release edit vX.Y.Z --notes-file` and re-run until all print OK.

### 8. Verify what users see

Python `urllib` with a generic User-Agent (no curl on this machine; never an
email in headers):

- crates.io API `https://crates.io/api/v1/crates/tinymind`: `max_version` is X.Y.Z.
- crates.io readme `https://crates.io/api/v1/crates/tinymind/X.Y.Z/readme`:
  contains a phrase from this release's README changes.
- docs.rs: the page of an item new in this release
  (`https://docs.rs/tinymind/X.Y.Z/tinymind/<module>/...`) returns 200 and
  contains its doc text (docs.rs can take a few minutes; retry, do not guess).
  Module items live under their module path, e.g. `tinymind/tree/struct.QGbdt.html`.
- Pages: the pages this release changed return 200 and show the new content.

### 9. LinkedIn post (when Dan wants one)

Use the `linkedin-post` skill, saving to
`~/Documents/tinymind/tinymind-rs-vX.Y.Z/linkedin-post.md`. For this crate the
**first URL in the body must be the tinymind-rs Pages site** (or one of its
pages), so LinkedIn's card is the site's; end with the release notes and the
crates.io links. Check:

```bash
F=~/Documents/tinymind/tinymind-rs-vX.Y.Z/linkedin-post.md
tail -n +4 "$F" | grep -oiE 'https?://[^ ]+|\b[A-Za-z0-9-]+(\.[A-Za-z0-9-]+)*\.(com|org|net|io|dev|ai|co|me|app|edu|gov|info|ly|gg|tv|us|uk)\b' | head -1
# must print a danmcleran.github.io/tinymind-rs URL
grep -cE 'TinyMind|Tinymind' "$F"           # must print 0
```

### 10. Report

The audit result (what Step 3 found and the PR that fixed it, or nothing), the
release PR, crates.io / docs.rs / Pages verification, the release URL with
Step 7's checks, any deferred upstream changes, and the LinkedIn file with its
length. Do not report success unless Step 3, Step 4's `--release` check and
Step 7 all passed.
