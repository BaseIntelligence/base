use std::path::PathBuf;
fn main() {
    let src = PathBuf::from(std::env::args().nth(1).expect("src"));
    let cache = PathBuf::from(std::env::args().nth(2).expect("cache"));
    let man = agent_pack::materialize_catalog(&src, &cache).expect("materialize");
    println!("entries={} pin={} catalog_digest={}", man.entries.len(), man.pin, man.catalog_digest);
    for e in &man.entries {
        println!("{} {} {}", e.pack_id, e.pack_digest, e.environment_image_digest);
    }
}
