# CCCS Iceberg

This is a fork of [apache-iceberg](https://github.com/apache/iceberg)

You can review the original [README.md](https://github.com/apache/iceberg/blob/main/README.md).

## What's in this fork

This fork includes pull-requests made `upstream` that have not yet been merged and have been deemed to be either useful or necessary for CCCS' usecases.

It also includes some pull-requests made by CCCS developers that are specific to CCCS usecases and are therefore only issued against `origin`.

#### Upstream Pull-Request

- [Spark: Support rewrite file with z-order for nested Struct type](https://github.com/apache/iceberg/pull/9818)
- [Parquet: Make row-group filters cooperate to filter #10090](https://github.com/apache/iceberg/pull/10090)
- [Spark: Custom snapshot property from session configuration #12999](https://github.com/apache/iceberg/pull/12999)

#### Origin Pull-Request

- [Added azure specific oauth things](https://github.com/CybercentreCanada/iceberg/pull/19)

#### Cherry-picked ahead of official release
- [Spark 3.5: Backport: Refactor Spark procedures to consistently use ProcedureInput for parameter handling.](https://github.com/apache/iceberg/pull/14179)
- [Spark: enable stream-results option for remove orphan files](https://github.com/apache/iceberg/pull/14278)
