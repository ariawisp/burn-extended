use burn_extended::attention::AttnWindow;
use burn_extended::cache::{MhaCacheManager, MqaCacheManager, WindowPolicy};
use burn_ndarray::NdArray;
use burn_tensor::backend::Backend;

type TB = NdArray<f32>;

fn device() -> <TB as Backend>::Device {
    Default::default()
}

#[test]
fn window_policy_maps_correctly() {
    assert!(matches!(WindowPolicy::Full.window_for(0), AttnWindow::Full));
    assert!(matches!(
        WindowPolicy::Fixed(7).window_for(3),
        AttnWindow::Window(7)
    ));

    let p = WindowPolicy::EveryOther {
        window: 5,
        full_on_even: true,
    };
    match p.window_for(0) {
        AttnWindow::Full => {}
        _ => panic!("even layer should be Full"),
    }
    match p.window_for(1) {
        AttnWindow::Window(w) => assert_eq!(w, 5),
        _ => panic!("odd layer should be Window(5)"),
    }
}

#[test]
fn cache_managers_initialize_caches() {
    let device = device();
    // MQA manager
    let mut mqa = MqaCacheManager::<TB>::new(&device, 3, 2, 8, 32, 1, 1);
    assert_eq!(mqa.caches.len(), 3);
    // Mutate through cache_mut and observe state change
    mqa.cache_mut(1).local_end_index = 7;
    assert_eq!(mqa.caches[1].local_end_index, 7);

    // MHA manager
    let mut mha = MhaCacheManager::<TB>::new(&device, 4, 4, 8, 64, 2, 1);
    assert_eq!(mha.caches.len(), 4);
    mha.cache_mut(2).local_end_index = 9;
    assert_eq!(mha.caches[2].local_end_index, 9);
}
