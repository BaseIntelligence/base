//! Live data adapters behind `site-api`: TAO/USD quote cache, sealed weight
//! vector projection, and metrics frames. Pure functions / explicit handles so
//! the HTTP crate stays a thin router.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::doc_markdown)]

pub mod metrics;
pub mod price;
pub mod timefmt;
pub mod weights;
