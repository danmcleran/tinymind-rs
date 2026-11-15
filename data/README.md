# data

- `iris.data`: the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)
  (R. A. Fisher, 1936; UCI Machine Learning Repository,
  [doi:10.24432/C56C76](https://doi.org/10.24432/C56C76)), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is a byte-for-byte
  copy of tinymind (C++)'s `examples/iris/iris.data`, which the `iris` example reads;
  `scripts/compare-with-cpp.sh` checks that the two copies still match.
- `optdigits.tra`, `optdigits.tes`: the [Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)
  (E. Alpaydin and C. Kaynak, 1998; UCI Machine Learning Repository,
  [doi:10.24432/C50P49](https://doi.org/10.24432/C50P49)), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Byte-for-byte
  copies of tinymind (C++)'s `examples/optical_digits/optdigits.tra` and `.tes`,
  which the `optical_digits` example reads; `scripts/compare-with-cpp.sh`
  checks that the copies still match. They are left out of the published
  crate (`exclude` in `Cargo.toml`): 828 KB, too large to ship to every user.
- `ENB2012_data.csv`: the [Energy Efficiency dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency)
  (A. Tsanas and A. Xifara, 2012; UCI Machine Learning Repository,
  [doi:10.24432/C51307](https://doi.org/10.24432/C51307)), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is a byte-for-byte
  copy of tinymind (C++)'s `examples/energy_efficiency/ENB2012_data.csv` (CRLF line
  endings, kept by `.gitattributes`), which the `energy_efficiency` example
  reads; `scripts/compare-with-cpp.sh` checks that the two copies still match.
