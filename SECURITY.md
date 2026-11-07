# Security Policy

## Reporting a vulnerability

Please report security issues privately through GitHub's private vulnerability
reporting: open the repository's **Security** tab and choose **Report a
vulnerability** (or go to
<https://github.com/danmcleran/tinymind-rs/security/advisories/new>). Do not open
a public issue for a suspected vulnerability.

Include what you can of:

- the affected module or function, and the tinymind-rs commit or crate version;
- the Rust toolchain version and target (for example `thumbv7em-none-eabihf`);
- a minimal reproducer and what goes wrong.

tinymind-rs is maintained by one person, so responses are best effort. You can
expect an acknowledgement within a week and an assessment once the report has been
reproduced. Fixes land on `master` and ship in the next release, and the advisory
credits the reporter unless they prefer otherwise.

## Supported versions

The crate has not been released yet; until it is, only `master` is supported.
After the first release, security fixes go to `master` and the latest release.
Older releases are not patched.

## Scope

The library contains no `unsafe` code (`#![forbid(unsafe_code)]` enforces this), so
memory-safety defects should not be possible in it. In scope:

- a panic reachable through the public API, which would halt firmware built on the
  crate;
- wrong results for inputs the API accepts, where they break a documented guarantee.

tinymind-rs reproduces tinymind's arithmetic bit for bit, so a defect may come from
the C++ library itself. If it does, please also report it to
[tinymind](https://github.com/danmcleran/tinymind/security/advisories/new).

Out of scope: numerical accuracy of a trained model, and the `xor` example binary,
which is a demonstration rather than hardened code. Reports about them are still
welcome as ordinary issues.
