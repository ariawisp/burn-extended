use burn_extended::loader::mxfp4::dequant_mxfp4_bytes;

#[test]
fn mxfp4_dequant_simple() {
    // One row, 2 packed bytes (4 outputs)
    // blocks: [low, high nibbles] => indices into FP4_VALUES
    // Choose bytes so indices are easy: 0x01 => lo=1 (0.5), hi=0 (0.0)
    let blocks = [0x10u8, 0x21u8]; // [0.0,0.5, 0.5,1.0] before scale
    let scales = [127u8]; // exponent bias 127 -> scale 2^(0) = 1
    let out = dequant_mxfp4_bytes(&blocks, &[1, 2], &scales, &[1]);
    assert_eq!(out.len(), 4);
    // Expected: [0.0, 0.5, 0.5, 1.0]
    let expect = [0.0f32, 0.5, 0.5, 1.0];
    for i in 0..4 {
        assert!((out[i] - expect[i]).abs() < 1e-6, "mismatch at {}: {} vs {}", i, out[i], expect[i]);
    }

    // With a scale exponent +1 (128 -> 1 after bias), scale=2^1=2
    let scales = [128u8];
    let out2 = dequant_mxfp4_bytes(&blocks, &[1, 2], &scales, &[1]);
    let expect2 = [0.0f32, 1.0, 1.0, 2.0];
    for i in 0..4 {
        assert!((out2[i] - expect2[i]).abs() < 1e-6);
    }
}
