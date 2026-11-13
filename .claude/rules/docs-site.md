---
paths:
  - "docs/**"
  - ".github/workflows/pages.yml"
  - "scripts/plot-examples.py"
---

# Docs site (GitHub Pages)

Porting notes moved out of CLAUDE.md (2026-11-11). They load when a file under
`paths:` is read. The Rust <- C++ map is the layout table in CLAUDE.md; add new
hard-won detail here, and new files to `paths:`.
What every PR must update is in CLAUDE.md, "Keeping the docs current".

`docs/` is a Jekyll site with the just-the-docs theme, the same setup as tinymind's
site, published at <https://danmcleran.github.io/tinymind-rs/> by
`.github/workflows/pages.yml`. That workflow builds on every PR (deploys only from
`master`) and adds the crate's rustdoc under `/api/`. Its `build site` job is not a
required check.

- Pages: Home, XOR example, Iris example, Energy example, Anomaly detection, Decision trees, Maze (Q-learning), Activity recognition (LSTM), Fine-tuning on the device, Int8 inference, Q-format in Rust, On-board
  results, Equivalence with tinymind, API reference, and "Bit for bit" (the launch write-up:
  how the port is checked and what Rust bought; its numbers must be kept current or the
  page dated). Keep them about what is ported; link to tinymind's site for concepts
  rather than copying its pages.
- Do not document tinymind bugs on the site (no "findings" sections, no links to
  tinymind fix PRs, no "before #N fixed it" asides). Dan's call, 2026-11-07. They
  belong in CLAUDE.md, commit messages and write-ups to Dan.
- Code examples live in rustdoc (the crate-level XOR doctest runs in `cargo test`),
  not in the site's Markdown, where nothing would compile them.
- Numbers on the pages (settling step, predictions, surface shares) come from the
  golden outputs. When outputs change, recheck them and rerun
  `scripts/plot-examples.py` (needs matplotlib in a venv; see its docstring).
- `docs/assets/tinymind_logo.png` is copied from tinymind; the OG banner
  `tinymind-rs_og.png` was drawn with Pillow in tinymind's OG layout.
- Not yet: a Releases page (tinymind's `render_releases_page.py` exits when a repo
  has no releases; the crate has had releases since 0.1.0, so it and `pages-on-release.yml` can be added now) and
  the knowledge graph (not worth it at this size).
- No local Ruby is needed: the PR build checks the site. To preview locally:
  `cd docs && bundle install && bundle exec jekyll serve`.
