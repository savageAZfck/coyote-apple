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

/// Multi-tensor `.safetensors` — `(name, shape, f32 data, is_f16)` per
/// entry. F16 payloads store f16 bits, everything else F32 LE. This is
/// the checkpoint-export writer: `coyote-apple run --export` emits the
/// VM's trained `param` globals in the same layout `param @ "src"`
/// loads, so a trained brain round-trips through `milc convert`.
pub fn safetensors_multi(tensors: &[(String, Vec<usize>, Vec<f32>, bool)]) -> Vec<u8> {
    let mut header = String::from("{");
    let mut data: Vec<u8> = Vec::new();
    for (i, (name, shape, vals, f16)) in tensors.iter().enumerate() {
        if i > 0 {
            header.push(',');
        }
        let (dtype, esz) = if *f16 {
            ("F16", 2usize)
        } else {
            ("F32", 4usize)
        };
        let start = data.len();
        let end = start + vals.len() * esz;
        let shape_s = shape
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        header.push_str(&format!(
            "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":[{shape_s}],\"data_offsets\":[{start}, {end}]}}"
        ));
        if *f16 {
            for v in vals {
                data.extend_from_slice(&half::f16::from_f32(*v).to_le_bytes());
            }
        } else {
            for v in vals {
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    header.push('}');
    let pad = (8 - (header.len() % 8)) % 8;
    header.push_str(&" ".repeat(pad));
    let mut out = Vec::with_capacity(8 + header.len() + data.len());
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
}
