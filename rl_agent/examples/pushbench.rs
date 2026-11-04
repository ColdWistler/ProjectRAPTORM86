//! Micro-benchmark: what `FrameSink::push` costs in the rollout's inner loop.
//!
//! Isolates the one property the streaming design claims — publishing a frame
//! must be a bounded, tiny amount of work that never depends on how fast (or
//! whether) a consumer is reading.
//!
//! ```text
//! cargo run --release --example pushbench -p rl_agent
//! ```

use rl_agent::stream::{Frame, FrameSink};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::sync_channel;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Push `n` frames and report the per-frame cost.
fn bench(label: &str, sink: &FrameSink, n: u32) {
    // Warm up so first-touch allocation is not charged to the measured loop.
    for i in 0..10_000u32 {
        sink.push(Frame {
            step: i,
            ..Default::default()
        });
    }
    let t = Instant::now();
    for i in 0..n {
        sink.push(Frame {
            step: i,
            ..Default::default()
        });
    }
    let per_frame = t.elapsed().as_nanos() as f64 / f64::from(n);
    // For scale: the env step + policy forward that surround each push is
    // roughly 600_000 ns at the ~1600 steps/s this run measures.
    println!("{label:<30} {per_frame:7.1} ns/frame");
}

fn main() {
    let n = 2_000_000u32;

    println!("{n} frames per case\n");

    // 1. Streaming off — what the loop costs today.
    bench("disabled (baseline)", &FrameSink::disabled(), n);

    // 2. Enabled, no reader at all. Every frame after the 8192-slot queue
    //    fills is dropped. This is the worst case: a GUI that crashed.
    let dead_stats = Arc::new(rl_agent::stream::StreamStats::default());
    let (dead_tx, dead_rx) = sync_channel::<Frame>(8192);
    drop(dead_rx);
    bench(
        "enabled, reader dead",
        &FrameSink::from_parts(dead_tx, Arc::clone(&dead_stats)),
        n,
    );

    // 3. Enabled with a live writer thread that keeps up.
    let live_stats = Arc::new(rl_agent::stream::StreamStats::default());
    let (live_tx, live_rx) = sync_channel::<Frame>(8192);
    let stop = Arc::new(AtomicBool::new(false));

    let dir = std::env::temp_dir().join("rl_pushbench");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let ring_path = dir.join("frames.bin");
    let stop_reader = Arc::clone(&stop);
    let reader = std::thread::spawn(move || {
        // Mirrors the real writer, minus per-episode bookkeeping.
        let Ok(mut ring) = rl_agent::stream::FrameRing::create(&ring_path) else {
            return;
        };
        while !stop_reader.load(Ordering::Relaxed) {
            match live_rx.recv_timeout(Duration::from_millis(50)) {
                Ok(frame) => {
                    let _ = ring.push(&frame);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });

    let live = FrameSink::from_parts(live_tx, Arc::clone(&live_stats));
    bench("enabled, reader keeping up", &live, n);
    stop.store(true, Ordering::Relaxed);
    drop(live);
    let _ = reader.join();

    let total = u64::from(n) + 20_000;
    println!(
        "\nlive reader: {} written, {} dropped of {total}",
        live_stats.written.load(Ordering::Relaxed),
        live_stats.dropped.load(Ordering::Relaxed)
    );
    println!(
        "dead reader: {} written, {} dropped of {total}",
        dead_stats.written.load(Ordering::Relaxed),
        dead_stats.dropped.load(Ordering::Relaxed)
    );

    let _ = std::fs::remove_dir_all(&dir);
}
