//! One test per SSRF rejection class for miner-announced base URLs.
//!
//! Every case here is a URL a miner could sign and a challenge service could be
//! made to dial. A regression that re-admits any of them turns the announce
//! endpoint back into a request-forgery primitive, so each class is asserted on
//! its own rejection reason, not merely on `is_err`.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use miner_endpoint::{validate_base_url, UrlRejection, MAX_BASE_URL_LEN};

fn reject(url: &str) -> UrlRejection {
    validate_base_url(url).expect_err(&format!("{url} must be rejected"))
}

fn assert_all(urls: &[&str], want: UrlRejection) {
    for url in urls {
        assert_eq!(reject(url), want, "wrong rejection class for {url}");
    }
}

#[test]
fn public_https_and_http_origins_are_accepted() {
    for url in [
        "https://cvm.example.com",
        "https://cvm.example.com/",
        "http://cvm.example.com",
        "https://cvm.example.com:8443",
        "http://cvm.example.com:8080/",
        "https://8.8.8.8:9000",
        "https://[2606:4700:4700::1111]:8443",
        "https://a.b.c.example.co.uk",
    ] {
        validate_base_url(url).unwrap_or_else(|e| panic!("{url} must be accepted, got {e}"));
    }
}

#[test]
fn non_http_schemes_are_rejected() {
    assert_all(
        &[
            "ftp://example.com",
            "file:///etc/passwd",
            "gopher://example.com:70",
            "redis://example.com:6379",
            "data:text/plain,hi",
        ],
        UrlRejection::Scheme,
    );
}

#[test]
fn embedded_credentials_are_rejected() {
    // `user@host` is also the classic parser-confusion trick: a naive consumer
    // reads the text before the `@` as the host.
    assert_all(
        &[
            "https://user:pass@example.com",
            "https://user@example.com",
            "https://:pass@example.com",
            "https://evil.example.com@169.254.169.254",
        ],
        UrlRejection::Credentials,
    );
}

#[test]
fn loopback_literals_are_rejected() {
    // The last two are the same address written in the alternative integer /
    // shorthand forms the WHATWG parser accepts.
    assert_all(
        &[
            "http://127.0.0.1:8080",
            "http://127.9.9.9",
            "http://[::1]:8080",
            "http://2130706433",
            "http://127.1",
        ],
        UrlRejection::ForbiddenAddress,
    );
}

#[test]
fn rfc1918_private_literals_are_rejected() {
    assert_all(
        &[
            "http://10.0.0.5:8080",
            "http://10.255.255.255",
            "http://172.16.0.1:8080",
            "http://172.31.255.254",
            "http://192.168.1.1:8080",
        ],
        UrlRejection::ForbiddenAddress,
    );
}

#[test]
fn link_local_and_cloud_metadata_are_rejected() {
    assert_all(
        &[
            "http://169.254.169.254",
            "http://169.254.169.254/",
            "http://169.254.0.1:8080",
            "http://[fe80::1]",
            "http://[fe80::a9fe:a9fe]:8080",
        ],
        UrlRejection::ForbiddenAddress,
    );
}

#[test]
fn unique_local_ipv6_is_rejected() {
    assert_all(
        &[
            "http://[fd00::1]",
            "http://[fc00::1]:8443",
            "http://[fec0::1]",
        ],
        UrlRejection::ForbiddenAddress,
    );
}

#[test]
fn multicast_literals_are_rejected() {
    assert_all(
        &[
            "http://224.0.0.1",
            "http://239.255.255.250",
            "http://[ff02::1]",
        ],
        UrlRejection::ForbiddenAddress,
    );
}

#[test]
fn unspecified_literals_are_rejected() {
    assert_all(
        &["http://0.0.0.0", "http://0.0.0.0:8080", "http://[::]"],
        UrlRejection::ForbiddenAddress,
    );
}

#[test]
fn reserved_and_special_purpose_literals_are_rejected() {
    assert_all(
        &[
            "http://100.64.0.1",   // CGNAT
            "http://192.0.0.1",    // IETF protocol assignments
            "http://192.0.2.1",    // TEST-NET-1
            "http://198.18.0.1",   // benchmarking
            "http://198.51.100.1", // TEST-NET-2
            "http://203.0.113.1",  // TEST-NET-3
            "http://240.0.0.1",    // reserved 240/4
            "http://255.255.255.255",
            "http://[2001:db8::1]", // documentation
        ],
        UrlRejection::ForbiddenAddress,
    );
}

#[test]
fn ipv6_transition_forms_do_not_smuggle_private_v4() {
    assert_all(
        &[
            "http://[::ffff:169.254.169.254]", // IPv4-mapped
            "http://[::ffff:127.0.0.1]",
            "http://[::127.0.0.1]",      // deprecated IPv4-compatible
            "http://[2002:a9fe:a9fe::]", // 6to4 wrapping 169.254.169.254
            "http://[64:ff9b::7f00:1]",  // NAT64 wrapping 127.0.0.1
        ],
        UrlRejection::ForbiddenAddress,
    );
}

#[test]
fn internal_only_hostnames_are_rejected() {
    assert_all(
        &[
            "http://localhost:8080",
            "http://metadata", // single label, resolved via search domain
            "http://gateway.local",
            "http://svc.internal",
            "http://db.intranet",
            "http://host.lan",
            "http://app.corp",
            "http://x.home.arpa",
            "http://abc.onion",
            "http://foo.test",
            "http://foo.invalid",
            "http://api.localhost",
        ],
        UrlRejection::ForbiddenHostname,
    );
}

#[test]
fn path_query_and_fragment_are_rejected() {
    assert_all(
        &[
            "https://example.com/v1",
            "https://example.com/v1/agent",
            "https://example.com/?a=b",
            "https://example.com?a=b",
            "https://example.com/#frag",
            "https://example.com#frag",
        ],
        UrlRejection::PathQueryFragment,
    );
}

#[test]
fn out_of_range_ports_are_rejected() {
    assert_all(
        &[
            "https://example.com:0",
            "https://example.com:1",
            "https://example.com:22",
            "https://example.com:25",
            "https://example.com:1023",
        ],
        UrlRejection::Port,
    );
    // A port above 65535 is not even a URL.
    assert_eq!(reject("https://example.com:70000"), UrlRejection::Malformed);
}

#[test]
fn empty_padded_and_relative_inputs_are_rejected() {
    assert_eq!(reject(""), UrlRejection::Empty);
    assert_eq!(reject(" https://example.com"), UrlRejection::Empty);
    assert_eq!(reject("https://example.com\n"), UrlRejection::Empty);
    assert_eq!(reject("example.com"), UrlRejection::Malformed);
    assert_eq!(reject("//example.com"), UrlRejection::Malformed);
    assert_eq!(reject("https://"), UrlRejection::Malformed);
}

#[test]
fn over_long_urls_are_rejected() {
    let long = format!("https://{}.example.com", "a".repeat(MAX_BASE_URL_LEN));
    assert_eq!(reject(&long), UrlRejection::TooLong);
}
