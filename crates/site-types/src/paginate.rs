//! Pagination helpers.

use crate::types::Paginated;

/// Slice `items` into a [`Paginated`] page (1-based `page`).
#[must_use]
pub fn page_slice<T: Clone>(items: &[T], page: u32, page_size: u32) -> Paginated<T> {
    let page = page.max(1);
    let page_size = page_size.clamp(1, 100);
    let total = u32::try_from(items.len()).unwrap_or(u32::MAX);
    let page_count = total.div_ceil(page_size).max(1);
    let start = (page.saturating_sub(1)).saturating_mul(page_size) as usize;
    let end = start.saturating_add(page_size as usize).min(items.len());
    let slice = if start >= items.len() {
        Vec::new()
    } else {
        items[start..end].to_vec()
    };
    Paginated {
        items: slice,
        page,
        page_size,
        total,
        page_count,
    }
}
