#!/usr/bin/env python3
"""The C++ library is "tinymind (C++)" in user-facing prose.

Three names collide here: tinymind (the C++ library), tinymind-rs (this
project) and `tinymind` (the crate, the title of every docs.rs page). So prose
never says a bare "tinymind": the C++ library is "tinymind (C++)", the project
is "tinymind-rs", the crate is `tinymind` in backticks. Code spans, fenced
code, URLs, paths and identifiers are left alone.

  python3 -I scripts/cpp-name.py --check FILE...   list bare mentions, exit 1 if any
  python3 -I scripts/cpp-name.py --fix FILE...     rewrite them in place

Markdown files are checked as a whole; Rust files only in their doc comments
(`//!` and `///`, what docs.rs shows).
"""
import re, sys

BARE = re.compile(r"(?<![\w/`.\-])tinymind(?![\w`/\-]|::| \(C\+\+\))")

def prose(seg):
    seg = re.sub(r"tinymind's C\+\+ (and|bit)\b", r"tinymind (C++) \1", seg)
    seg = re.sub(r"tinymind's C\+\+ (?=\w)", "the tinymind (C++) ", seg)
    seg = re.sub(r"tinymind C\+\+ (?=\w)", "tinymind (C++) ", seg)
    return BARE.sub("tinymind (C++)", seg)

def line_fix(text):
    # Only the parts outside inline code spans.
    parts = text.split("`")
    for i in range(0, len(parts), 2):
        # Leave URLs and liquid alone: split them out.
        chunks = re.split(r"(https?://\S+|\{\{[^}]*\}\}|\]\([^)]*\))", parts[i])
        for j in range(0, len(chunks), 2):
            chunks[j] = prose(chunks[j])
        parts[i] = "".join(chunks)
    return "`".join(parts)

def fix_markdown(src):
    out, fenced = [], False
    for line in src.split("\n"):
        if line.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line); continue
        if fenced or line.lstrip().startswith("<img"):
            out.append(line); continue
        out.append(line_fix(line))
    return "\n".join(out)

def fix_rust(src):
    out, fenced = [], False
    for line in src.split("\n"):
        m = re.match(r"(\s*//[!/] ?)(.*)$", line)
        if not m:
            out.append(line); continue
        body = m.group(2)
        if body.lstrip().startswith("```"):
            fenced = not fenced
            out.append(line); continue
        out.append(line if fenced else m.group(1) + line_fix(body))
    return "\n".join(out)

def main():
    mode, paths = sys.argv[1], sys.argv[2:]
    if mode not in ("--check", "--fix"):
        sys.exit(__doc__)
    found = 0
    for path in paths:
        src = open(path).read()
        new = fix_rust(src) if path.endswith(".rs") else fix_markdown(src)
        if new == src:
            continue
        if mode == "--fix":
            open(path, "w").write(new)
            print("fixed", path)
            continue
        for n, (a, b) in enumerate(zip(src.split("\n"), new.split("\n")), 1):
            if a != b:
                found += 1
                print("%s:%d: bare 'tinymind' (write 'tinymind (C++)'): %s" % (path, n, a.strip()))
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
