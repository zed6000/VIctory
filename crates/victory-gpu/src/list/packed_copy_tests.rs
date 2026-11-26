use super::*;
use crate::resource::PlacedFootprint;
use std::ptr::null;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
use windows::core::{IUnknown, Interface};

fn create(call: impl FnOnce(*mut This) -> HRESULT) -> IUnknown {
    let mut object = null();
    assert_eq!(call(&mut object), S_OK);
    // SAFETY: the successful guest call returned an owned COM reference.
    unsafe { IUnknown::from_raw(object.cast_mut().cast()) }
}

fn check_packed_copy(format: u32, red: u16) {
    let Some(host) = crate::host::host() else {
        eprintln!("skipping packed footprint copy (no D3D12 device)");
        return;
    };
    let allocator = create(|out| create_allocator(null(), 0, &ALLOCATOR_IIDS[0], out));
    let title = create(|out| {
        create_list(
            null(),
            0,
            0,
            allocator.as_raw().cast(),
            null(),
            &LIST_IIDS[1],
            out,
        )
    });
    let title = title.as_raw().cast();
    let list = host_list(title).expect("host command list");
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 7,
        Height: 4,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT(format as i32),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let texture = create(|out| {
        crate::resource::create_committed(
            null(),
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            0,
            null(),
            &PAGEABLE_IIDS[0],
            out,
        )
    });
    let texture = texture.as_raw().cast();
    let native = crate::resource::host_resource(texture).expect("writable packed texture");
    // SAFETY: the live host resource and device can be queried before recording a draw.
    let host_desc = unsafe { native.GetDesc() };
    let mut support = D3D12_FEATURE_DATA_FORMAT_SUPPORT {
        Format: host_desc.Format,
        ..Default::default()
    };
    // SAFETY: the output is a format-support query of the declared size.
    unsafe {
        host.device.CheckFeatureSupport(
            D3D12_FEATURE_FORMAT_SUPPORT,
            (&raw mut support).cast(),
            size_of_val(&support) as u32,
        )
    }
    .expect("query writable texture format");
    assert!(
        host_desc.Flags.contains(desc.Flags),
        "host flags {:?}",
        host_desc.Flags
    );
    assert!(
        support.Support1.contains(
            D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW
        ),
        "guest format {format} maps to {:?}, which cannot serve both writable uses",
        host_desc.Format
    );
    const SIZE: usize = 4096;
    const UPLOAD: usize = 512;
    const READBACK: usize = 2048;
    const PITCH: usize = 256;
    // SAFETY: test_buffer returns one owned guest reference.
    let buffer =
        unsafe { IUnknown::from_raw(crate::resource::test_buffer(SIZE as u64).cast_mut().cast()) };
    let buffer = buffer.as_raw().cast();
    let (mapped, len) = crate::resource::buffer_bytes(buffer).expect("mapped guest buffer");
    assert_eq!(len, SIZE as u64);
    let mut expected = [0xa5; SIZE];
    for y in 0..4 {
        for x in 0..7 {
            let word = (0x1234u16).wrapping_mul((1 + y * 7 + x) as u16);
            let at = UPLOAD + y * PITCH + x * 2;
            expected[at..at + 2].copy_from_slice(&word.to_le_bytes());
        }
    }
    // SAFETY: the live guest buffer holds SIZE bytes, and no submission is in flight.
    unsafe { std::ptr::copy_nonoverlapping(expected.as_ptr(), mapped, SIZE) };
    let footprint = PlacedFootprint {
        offset: UPLOAD as u64,
        format,
        width: 7,
        height: 4,
        depth: 1,
        row_pitch: PITCH as u32,
    };
    let upload = CopyLocation {
        resource: buffer,
        kind: 1,
        footprint,
    };
    let image = CopyLocation {
        resource: texture,
        kind: 0,
        footprint: PlacedFootprint {
            offset: 0,
            ..footprint
        },
    };
    copy_texture_region(title, &image, 0, 0, 0, &upload, null());

    // A render-target write checks the channel order independently of the upload conversion.
    ensure_state(list, texture, D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32);
    let view = host.rtv(&native, None);
    let pixel = RECT {
        left: 3,
        top: 2,
        right: 4,
        bottom: 3,
    };
    // SAFETY: the view names the live render target and the rectangle lies within it.
    unsafe {
        list.list
            .ClearRenderTargetView(view, &[1.0, 0.0, 0.0, 0.0], Some(&[pixel]))
    };

    let output = CopyLocation {
        resource: buffer,
        kind: 1,
        footprint: PlacedFootprint {
            offset: READBACK as u64,
            width: 8,
            height: 5,
            ..footprint
        },
    };
    let region = [1, 1, 0, 5, 4, 1];
    copy_texture_region(title, &output, 2, 1, 0, &image, &region);
    // SAFETY: all referenced resources are live until the host waits for this list.
    unsafe { list.list.Close().expect("record packed copy and clear") };
    host.execute(&[list]);
    for y in 1..4 {
        for x in 1..5 {
            let src = UPLOAD + y * PITCH + x * 2;
            let dst = READBACK + y * PITCH + (x + 1) * 2;
            if (x, y) == (3, 2) {
                expected[dst..dst + 2].copy_from_slice(&red.to_le_bytes());
            } else {
                expected.copy_within(src..src + 2, dst);
            }
        }
    }
    // SAFETY: submission and deferred readback finished, with the buffer still live.
    let actual = unsafe { std::slice::from_raw_parts(mapped, SIZE) };
    for (offset, (&got, want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(got, want, "guest format {format}, byte {offset}");
    }
}

#[test]
fn b5g5r5a1_partial_copy_preserves_guest_texels_and_padding() {
    check_packed_copy(86, 0x7c00);
}

#[test]
fn b4g4r4a4_partial_copy_preserves_guest_texels_and_padding() {
    check_packed_copy(115, 0x0f00);
}
