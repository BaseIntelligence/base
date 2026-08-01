//! SSRF guard for miner-announced base URLs.
//!
//! Whatever a miner announces here becomes the target of an outbound HTTP
//! request issued by the challenge service, from inside the operator's network.
//! That makes this module a request-forgery sink, not a formatting check: the
//! rejection list below is the security boundary and every class in it has a
//! test.
//!
//! # Hostnames are deliberately not resolved
//!
//! [`validate_base_url`] never calls DNS. Resolving here would only prove what
//! the name pointed at during the announcement; the challenge service connects
//! minutes to hours later, and nothing stops a miner from serving a public A
//! record now and `169.254.169.254` at dispatch time (DNS rebinding, or simply
//! a short TTL). Resolving would therefore buy no real protection while making
//! validation depend on the resolver — a DNS blip would 400 an honest miner,
//! and a slow resolver would block the handler.
//!
//! The residual risk is explicit: a hostname that resolves to a private or
//! link-local address passes this check. The authoritative control is at the
//! egress point, which is why [`is_forbidden_ip`] is public — the dispatcher
//! must re-check the address it actually connected to before it sends the
//! request, and that check is the one that closes rebinding.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

use thiserror::Error;
use url::{Host, Url};

/// Longest accepted announcement URL, in bytes.
pub const MAX_BASE_URL_LEN: usize = 2048;

/// Lowest port accepted outside the two well-known HTTP ports.
///
/// Everything below this is a privileged service port (ssh, smtp, dns, ...)
/// that a miner has no reason to serve `agent-v1` on, and that an SSRF probe
/// has every reason to aim at.
pub const MIN_UNPRIVILEGED_PORT: u16 = 1024;

/// Name suffixes that only ever resolve inside somebody's network.
///
/// A public miner endpoint cannot legitimately live under any of these, and a
/// resolver that is asked for one answers from a local zone, bypassing every
/// literal-address check.
const FORBIDDEN_SUFFIXES: [&str; 11] = [
    ".local",
    ".localhost",
    ".internal",
    ".intranet",
    ".lan",
    ".corp",
    ".home",
    ".home.arpa",
    ".onion",
    ".test",
    ".invalid",
];

/// Why a base URL was refused. One variant per rejection class.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum UrlRejection {
    /// Empty or whitespace-padded input.
    #[error("base_url must be a non-empty, untrimmed-whitespace-free URL")]
    Empty,
    /// Longer than [`MAX_BASE_URL_LEN`].
    #[error("base_url exceeds {MAX_BASE_URL_LEN} bytes")]
    TooLong,
    /// Not parseable as an absolute URL.
    #[error("base_url is not a well-formed absolute URL")]
    Malformed,
    /// Scheme other than `http` / `https`.
    #[error("base_url scheme must be http or https")]
    Scheme,
    /// Embedded `user:pass@` credentials.
    #[error("base_url must not carry embedded credentials")]
    Credentials,
    /// No host component at all.
    #[error("base_url must have a host")]
    MissingHost,
    /// Carries a path, query, or fragment.
    #[error("base_url must be an origin only: no path, query, or fragment")]
    PathQueryFragment,
    /// Port zero or a privileged port other than 80 / 443.
    #[error("base_url port must be 80, 443, or {MIN_UNPRIVILEGED_PORT}-65535")]
    Port,
    /// Literal address in a loopback / private / link-local / reserved range.
    #[error("base_url host is a non-public IP address")]
    ForbiddenAddress,
    /// Single-label or internal-only hostname.
    #[error("base_url host is not a public, fully qualified hostname")]
    ForbiddenHostname,
}

/// Accept a miner-announced base URL, or say exactly why not.
///
/// On success the caller stores the input **verbatim**: that is the byte string
/// the miner signed, so normalising it here would leave a stored URL nobody
/// signed.
///
/// # Errors
///
/// Any [`UrlRejection`] class listed on this module.
pub fn validate_base_url(raw: &str) -> Result<(), UrlRejection> {
    if raw.is_empty() || raw.trim() != raw {
        return Err(UrlRejection::Empty);
    }
    if raw.len() > MAX_BASE_URL_LEN {
        return Err(UrlRejection::TooLong);
    }
    let url = Url::parse(raw).map_err(|_| UrlRejection::Malformed)?;

    // Anything but http/https reaches a different client stack entirely
    // (file:, gopher:, redis: via a permissive proxy), so gate it first.
    if !matches!(url.scheme(), "http" | "https") {
        return Err(UrlRejection::Scheme);
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(UrlRejection::Credentials);
    }
    // An origin-only URL keeps the dispatcher in control of the request line:
    // a path or query in the announcement would be prepended to every route the
    // dispatcher builds, and `@`-style host confusion hides in the same field.
    if url.query().is_some() || url.fragment().is_some() {
        return Err(UrlRejection::PathQueryFragment);
    }
    if !matches!(url.path(), "" | "/") {
        return Err(UrlRejection::PathQueryFragment);
    }

    let port = url.port_or_known_default().ok_or(UrlRejection::Port)?;
    if port != 80 && port != 443 && port < MIN_UNPRIVILEGED_PORT {
        return Err(UrlRejection::Port);
    }

    match url.host() {
        Some(Host::Ipv4(v4)) if is_forbidden_ip(&IpAddr::V4(v4)) => {
            Err(UrlRejection::ForbiddenAddress)
        }
        Some(Host::Ipv6(v6)) if is_forbidden_ip(&IpAddr::V6(v6)) => {
            Err(UrlRejection::ForbiddenAddress)
        }
        Some(Host::Domain(d)) => check_domain(d),
        Some(_) => Ok(()),
        None => Err(UrlRejection::MissingHost),
    }
}

/// Reject hostnames that cannot name a public endpoint.
///
/// `url` has already lowercased the host and punycoded any IDN, so this only
/// has to reason about ASCII.
fn check_domain(host: &str) -> Result<(), UrlRejection> {
    let host = host.strip_suffix('.').unwrap_or(host);
    if host.is_empty() {
        return Err(UrlRejection::MissingHost);
    }
    // A single label (`localhost`, `gateway`, `metadata`) is resolved through
    // the search domain of whoever is dialling, i.e. the operator's network.
    if !host.contains('.') {
        return Err(UrlRejection::ForbiddenHostname);
    }
    if FORBIDDEN_SUFFIXES.iter().any(|s| host.ends_with(s)) {
        return Err(UrlRejection::ForbiddenHostname);
    }
    Ok(())
}

/// Whether `addr` is outside the publicly routable unicast space.
///
/// Public on purpose: the component that opens the socket must run this against
/// the address it actually resolved and connected to. This module's name-based
/// checks cannot see through DNS.
#[must_use]
pub fn is_forbidden_ip(addr: &IpAddr) -> bool {
    match addr {
        IpAddr::V4(v4) => is_forbidden_v4(*v4),
        IpAddr::V6(v6) => is_forbidden_v6(v6),
    }
}

/// IANA IPv4 special-purpose ranges (RFC 6890 + RFC 1918 + RFC 3927).
fn is_forbidden_v4(addr: Ipv4Addr) -> bool {
    let o = addr.octets();
    o[0] == 0                                        // 0.0.0.0/8 "this network"
        || o[0] == 10                                // RFC1918
        || (o[0] == 100 && (64..128).contains(&o[1]))// 100.64/10 CGNAT
        || o[0] == 127                               // loopback
        || (o[0] == 169 && o[1] == 254)              // link-local, incl. cloud metadata
        || (o[0] == 172 && (16..32).contains(&o[1])) // RFC1918
        || (o[0] == 192 && o[1] == 0 && o[2] == 0)   // IETF protocol assignments
        || (o[0] == 192 && o[1] == 0 && o[2] == 2)   // TEST-NET-1
        || (o[0] == 192 && o[1] == 168)              // RFC1918
        || (o[0] == 198 && (o[1] == 18 || o[1] == 19))    // benchmarking
        || (o[0] == 198 && o[1] == 51 && o[2] == 100)     // TEST-NET-2
        || (o[0] == 203 && o[1] == 0 && o[2] == 113)      // TEST-NET-3
        || o[0] >= 224 // multicast, reserved 240/4, and 255.255.255.255
}

/// IPv6 special-purpose ranges, plus every transition format that smuggles an
/// IPv4 address through an IPv6 literal.
fn is_forbidden_v6(addr: &Ipv6Addr) -> bool {
    if let Some(v4) = embedded_v4(addr) {
        return is_forbidden_v4(v4);
    }
    let s = addr.segments();
    addr.is_unspecified()
        || addr.is_loopback()
        || addr.is_multicast()
        || (s[0] & 0xfe00) == 0xfc00 // fc00::/7 unique local
        || (s[0] & 0xffc0) == 0xfe80 // fe80::/10 link-local
        || (s[0] & 0xffc0) == 0xfec0 // fec0::/10 deprecated site-local
        || (s[0] == 0x2001 && s[1] == 0x0db8) // 2001:db8::/32 documentation
        || (s[0] == 0x0100 && s[1] == 0 && s[2] == 0 && s[3] == 0) // 100::/64 discard
}

/// The IPv4 address carried by a mapped / compatible / 6to4 / NAT64 literal.
///
/// `::ffff:169.254.169.254` and `2002:a9fe:a9fe::` both reach cloud metadata
/// through a stack that speaks IPv6, so they have to be unwrapped and judged as
/// the IPv4 addresses they are.
fn embedded_v4(addr: &Ipv6Addr) -> Option<Ipv4Addr> {
    let s = addr.segments();
    let from_pair = |hi: u16, lo: u16| {
        Ipv4Addr::new(
            u8::try_from(hi >> 8).unwrap_or(0),
            u8::try_from(hi & 0xff).unwrap_or(0),
            u8::try_from(lo >> 8).unwrap_or(0),
            u8::try_from(lo & 0xff).unwrap_or(0),
        )
    };
    // ::ffff:a.b.c.d (mapped) and ::a.b.c.d (deprecated compatible).
    if s[0] == 0
        && s[1] == 0
        && s[2] == 0
        && s[3] == 0
        && s[4] == 0
        && (s[5] == 0xffff || s[5] == 0)
    {
        return Some(from_pair(s[6], s[7]));
    }
    // 2002:a.b.c.d::/16 6to4.
    if s[0] == 0x2002 {
        return Some(from_pair(s[1], s[2]));
    }
    // 64:ff9b::/96 NAT64 well-known prefix.
    if s[0] == 0x0064 && s[1] == 0xff9b && s[2] == 0 && s[3] == 0 && s[4] == 0 && s[5] == 0 {
        return Some(from_pair(s[6], s[7]));
    }
    None
}
