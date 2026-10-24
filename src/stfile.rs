//! minimal safetensors writer — F32 tensors only. Shared by
//! `genmask` (selector masks) and `weights slice` (checkpoint cuts).

/// `name → (shape, f32 payload)` rendered as a `.safetensors` byte
/// buffer: u64le header len, JSON header padded to 8, raw LE f32 data.
pub fn safetensors_f32(name: &str, shape: &[usize], data: &[f32]) -> Vec<u8> {
    let nbytes = data.len() * 4;
    let shape_s = shape
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let header = format!(
        "{{\"{name}\":{{\"dtype\":\"F32\",\"shape\":[{shape_s}],\"data_offsets\":[0, {nbytes}]}}}}"
    );
    let pad = (8 - (header.len() % 8)) % 8;
    let header = format!("{header}{:pad$}", "", pad = pad);
    let mut out = Vec::with_capacity(8 + header.len() + nbytes);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}
