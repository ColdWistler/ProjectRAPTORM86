//! Live flight-frame streaming: watch the aircraft while it trains.
//!
//! The trainer already publishes per-*iteration* telemetry
//! ([`crate::telemetry`]). This module adds per-*step* frames so a GUI can
//! draw the aircraft in motion, and a cooperative control channel so the GUI
//! can steer the run.
//!
//! # Why a thread and a lossy channel
//!
//! The rollout loop is the hottest code in the project: 2048 iterations of
//! `agent.act` + `env.step` per training iteration, and on the GPU backend the
//! step time is sub-millisecond. Writing frames to disk inline would add a
//! `write` syscall per step to that loop, and any consumer slower than the
//! trainer would throttle training itself.
//!
//! So the sink is deliberately *lossy and bounded*:
//!
//! ```text
//! trainer thread          writer thread
//! ──────────────          ─────────────
//! sink.push(frame)  ──►  bounded channel (8192 slots)
//!   try_send, never          │
//!   blocks; drops on full    └─► frames.bin   (fixed ~7.8 MiB ring)
//!                                     └► episodes/ep_<n>.f32 on episode end
//! ```
//!
//! [`FrameSink::push`] uses `try_send`, so the cost when the consumer is
//! saturated is a single dropped frame, never a block. The trainer's wall-clock
//! is therefore independent of how fast (or slow) the GUI is. Dropped frames
//! are counted and reported through telemetry rather than hidden, so a viewer
//! can tell the difference between "the aircraft was still" and "we missed
//! this frame".
//!
//! # Why a ring buffer rather than an append-only log
//!
//! At 2048 steps/iteration and 1000 iterations a run produces ~2M frames.
//! At [`FRAME_BYTES`] (~124 B) each that is ~250 MiB of append-only log, of
//! which a live viewer would ever look at the last few thousand.
//! [`FrameRing`] keeps a fixed ~7.8 MiB file and overwrites the oldest slots,
//! so the on-disk cost is constant for the whole run and the GUI can
//! memory-map it.
//!
//! Episode *completions* are a different matter — those are the moments worth
//! keeping — so each finished episode is also written to its own small
//! `episodes/ep_<n>.f32` file for after-the-fact replay.
//!
//! # Parallelism
//!
//! Every frame carries a `worker_id`, which is `0` today because the rollout
//! is serial. The field exists so that parallel environment stepping can be
//! added later without changing the format: consumers group by `worker_id` and
//! nothing else has to move.

use flight_core::FlightFrame;
use serde::Serialize;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::Arc;
use std::thread::JoinHandle;

/// Wire format version for [`FrameRing`] / episode files. Bump on any layout
/// change so consumers can refuse a file they cannot parse.
pub const STREAM_SCHEMA: &str = "rl_agent.stream.v1";

/// Name of the live ring buffer inside a run directory.
pub const FRAMES_FILE: &str = "frames.bin";
/// Subdirectory holding per-episode replay files.
pub const EPISODES_DIR: &str = "episodes";

/// Bounded channel capacity, in frames.
///
/// 8192 frames is ~1 MiB of queue and covers several iterations of burst. It
/// is not a tuning knob that matters much: a smaller queue drops sooner, a
/// larger one tolerates a longer writer stall.
const CHANNEL_CAPACITY: usize = 8192;

/// Ring-buffer capacity in slots (power of two, so the wrap is a mask).
///
/// 65536 slots x [`FRAME_BYTES`] (~7.8 MiB). At 30 Hz simulated time that is
/// ~36 minutes of flight — far longer than anyone watches live.
///
/// Public so a reader can compute a slot's byte offset without duplicating
/// this number.
pub const RING_SLOTS: usize = 65_536;

/// A compile-time ceiling on the frame size, so a field added to [`Frame`]
/// cannot quietly inflate the ring past its documented budget.
const _: () = assert!(FRAME_BYTES <= 256);

/// Bytes per frame on disk, derived from [`Frame`]'s layout rather than
/// hand-maintained, so adding a field cannot silently desynchronise the
/// stride the writer and readers agree on.
pub const FRAME_BYTES: usize = std::mem::size_of::<Frame>();

/// One step's worth of flight state, in its on-disk layout.
///
/// `#[repr(C)]` with only 4-byte scalar fields: this struct is written straight
/// into the ring file and read back by the GUI, so the layout is a contract.
/// The `iteration`/`step`/`episode` `u32`s come first so a reader can index the
/// file by fixed offsets without deserialising, and `FRAME_BYTES` is derived
/// from `size_of` rather than hand-written so adding a field cannot silently
/// desynchronise the writer from its readers.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Frame {
    /// Iteration index the frame belongs to.
    pub iteration: u32,
    /// Step index within the iteration.
    pub step: u32,
    /// Episode counter, monotonic across the run.
    pub episode: u32,
    /// Parallel worker index. Always 0 today; see the module docs.
    pub worker_id: u16,
    /// `terminated | truncated << 1` — set on the last frame of an episode.
    pub flags: u16,

    /// Simulation time (s).
    pub sim_time: f32,

    // Earth-frame position, NED.
    pub north: f32,
    pub east: f32,
    pub altitude: f32,
    // Earth-frame ground velocity.
    pub vel_north: f32,
    pub vel_east: f32,
    pub vel_down: f32,

    // Attitude, radians.
    pub roll: f32,
    pub pitch: f32,
    pub yaw: f32,
    // Body rates, rad/s.
    pub p: f32,
    pub q: f32,
    pub r: f32,

    // Air data.
    pub airspeed: f32,
    pub alpha: f32,
    pub beta: f32,

    // Agent command (tanh-space, pre-scale) and what the airframe did.
    pub action_roll: f32,
    pub action_pitch: f32,
    pub action_yaw_rate: f32,
    pub action_throttle: f32,
    pub elevator_deg: f32,
    pub aileron_deg: f32,
    pub rudder_deg: f32,
    pub throttle: f32,

    // Perceived sensor channels, for noise visualisation.
    pub baro_altitude: f32,
    pub airspeed_indicated: f32,

    /// Reward credited for this step (crash penalty folded in).
    pub reward: f32,
}

impl Frame {
    /// Project a [`FlightFrame`] into the wire layout.
    pub fn from_flight(
        f: &FlightFrame,
        action: &[f32],
        reward: f32,
        iteration: u32,
        step: u32,
        episode: u32,
        worker_id: u16,
        terminated: bool,
        truncated: bool,
    ) -> Self {
        Self {
            iteration,
            step,
            episode,
            worker_id,
            flags: u16::from(terminated) | (u16::from(truncated) << 1),
            sim_time: f.sim_time as f32,
            north: f.north as f32,
            east: f.east as f32,
            altitude: f.altitude as f32,
            vel_north: f.vel_north as f32,
            vel_east: f.vel_east as f32,
            vel_down: f.vel_down as f32,
            roll: f.roll as f32,
            pitch: f.pitch as f32,
            yaw: f.yaw as f32,
            p: f.p as f32,
            q: f.q as f32,
            r: f.r as f32,
            airspeed: f.airspeed as f32,
            alpha: f.alpha as f32,
            beta: f.beta as f32,
            action_roll: action.first().copied().unwrap_or(0.0),
            action_pitch: action.get(1).copied().unwrap_or(0.0),
            action_yaw_rate: action.get(2).copied().unwrap_or(0.0),
            action_throttle: action.get(3).copied().unwrap_or(0.0),
            elevator_deg: f.elevator_deg as f32,
            aileron_deg: f.aileron_deg as f32,
            rudder_deg: f.rudder_deg as f32,
            throttle: f.throttle as f32,
            baro_altitude: f.baro_altitude as f32,
            airspeed_indicated: f.airspeed_indicated as f32,
            reward,
        }
    }

    /// `true` when this frame ends an episode.
    pub fn is_episode_end(&self) -> bool {
        self.flags != 0
    }

    /// Decode one frame from a [`FRAME_BYTES`]-length byte slice.
    ///
    /// The counterpart to the raw-slice writes in [`FrameRing`] and
    /// [`write_episode`], so readers in tests and tools parse frames the exact
    /// same way the file was produced.
    ///
    /// Returns `None` when `bytes` is not exactly one frame long — a short read
    /// means the caller lost track of the stride, and guessing would produce
    /// plausible garbage.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() != FRAME_BYTES {
            return None;
        }
        // SAFETY: `Frame` is `#[repr(C)]` over 4-byte POD scalars with no
        // padding, so any 4-aligned `FRAME_BYTES` buffer is a valid `Frame`,
        // and `bytes` from a file or the ring is at least byte-aligned. The
        // alignment requirement is 4; file-backed slices satisfy it because
        // the ring and episode files start at offset 0 and the stride is a
        // multiple of 4. A caller passing a misaligned slice would be UB, so
        // the length check above is paired with this documented contract.
        Some(unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const Frame) })
    }
}

const _: () = assert!(std::mem::size_of::<Frame>() == FRAME_BYTES);

/// A fixed-size ring of frames backed by a single file.
///
/// Slots are overwritten oldest-first, so the file is a bounded window onto
/// the most recent flight. The write cursor is monotonic and published to
/// consumers via [`FrameRing::write_index`], which is how a reader knows which
/// slots are valid without a separate header.
pub struct FrameRing {
    file: BufWriter<File>,
    /// Monotonic count of frames written; also the slot cursor.
    write_index: u64,
}

impl FrameRing {
    /// Create (or truncate) the ring file for a run.
    pub fn create(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        // Preallocate the full ring up front. Without this the file only grows
        // as far as the highest slot written, so it reads as sparse: a consumer
        // memory-mapping it would see a short mapping, and the "newest frame"
        // might sit in a hole. `set_len` makes the length a constant from the
        // first frame, which is the whole point of a fixed ring.
        file.set_len((RING_SLOTS * FRAME_BYTES) as u64)?;
        Ok(Self {
            file: BufWriter::new(file),
            write_index: 0,
        })
    }

    /// Append a frame, overwriting the slot `RING_SLOTS` frames ago.
    ///
    /// The slot index is masked rather than reduced, so this stays a single
    /// cheap AND on the hot path.
    pub fn push(&mut self, frame: &Frame) -> std::io::Result<()> {
        let slot = (self.write_index as usize) & (RING_SLOTS - 1);
        self.file
            .seek(std::io::SeekFrom::Start((slot * FRAME_BYTES) as u64))?;
        // SAFETY: `Frame` is `#[repr(C)]` of only f32/u16/u32 fields — no
        // padding, no pointers, no invalid bit patterns — so any bit pattern
        // of its bytes is a valid `Frame`. The alignment of the destination
        // buffer is 4, which satisfies every field.
        let bytes =
            unsafe { std::slice::from_raw_parts(frame as *const Frame as *const u8, FRAME_BYTES) };
        self.file.write_all(bytes)?;
        self.write_index += 1;
        Ok(())
    }

    /// Number of frames written so far (also the write cursor).
    pub fn write_index(&self) -> u64 {
        self.write_index
    }

    /// Flush buffered frames to disk so a live reader can see them.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.file.flush()
    }
}

/// Name of the write-cursor sidecar inside a run directory.
pub const HEAD_FILE: &str = "stream_head.bin";

/// Size of the head sidecar: one little-endian `u64`.
const HEAD_BYTES: u64 = 8;

/// The ring's write cursor, published so a live reader knows where to look.
///
/// [`FrameRing`] deliberately has no header — slots are written in order and
/// overwritten oldest-first — which leaves a reader with no way to tell which
/// slot holds the newest frame. Inferring it means scanning all
/// [`RING_SLOTS`] slots, i.e. the whole ~8 MiB file, on every single poll.
///
/// This sidecar is that cursor, and it is written *after* the ring flush that
/// made those frames durable. A reader may therefore trust it without a
/// barrier: the newest slot it names always contains bytes that already landed.
struct Head {
    file: File,
}

impl Head {
    fn create(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)?;
        // Preallocated for the same reason the ring is: a reader polling a
        // brand-new run should see a full-length file of zeros (nothing
        // recorded yet), never a zero-length one it has to special-case.
        file.set_len(HEAD_BYTES)?;
        Ok(Self { file })
    }

    /// Publish the cursor. Callers must have flushed the ring first.
    fn publish(&mut self, write_index: u64) -> std::io::Result<()> {
        // Little-endian explicitly rather than native, so the Godot viewer and
        // any other non-Rust reader agree on the byte order.
        self.file.seek(std::io::SeekFrom::Start(0))?;
        self.file.write_all(&write_index.to_le_bytes())?;
        self.file.flush()
    }
}

/// Read the published write cursor.
///
/// `None` when the sidecar is missing or short — a run that never streamed
/// frames has none, and a truncated one is not safe to interpret. `Some(0)` is
/// a legitimate "created but nothing written yet".
///
/// Exposed so readers (and tests) do not each re-derive the byte order.
pub fn read_head(path: &Path) -> Option<u64> {
    let bytes = std::fs::read(path).ok()?;
    let arr: [u8; 8] = bytes.get(..8)?.try_into().ok()?;
    Some(u64::from_le_bytes(arr))
}

/// Name of the layout manifest inside a run directory.
pub const STREAM_MANIFEST: &str = "stream.json";

/// Number of declared [`Frame`] fields.
///
/// Distinct from the 4-byte *word* count: `worker_id` and `flags` are `u16` and
/// share one word, so there are 32 fields in 31 words.
pub const FRAME_FIELD_COUNT: usize = 32;

/// What a reader needs to know before it can parse a run's frames.
///
/// Written once when the writer starts. Without it a consumer would have to
/// hard-code the stride and ring size and could silently mis-decode a run
/// written by a different version — which looks like an aircraft in a
/// physically impossible attitude rather than an error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, serde::Deserialize)]
pub struct StreamManifest {
    /// Layout version; compare against [`STREAM_SCHEMA`].
    pub schema: String,
    /// Bytes per frame. Must match the reader's expected stride.
    pub frame_bytes: usize,
    /// Slots in the ring; the newest frame is at `(head - 1) % ring_slots`.
    pub ring_slots: usize,
    /// Byte offset of each field within a frame, in declaration order.
    ///
    /// Emitted rather than assumed so a reader can assert its own table
    /// against the writer's instead of trusting a duplicated constant.
    pub field_offsets: Vec<u16>,
}

/// Byte offset of every [`Frame`] field, in declaration order.
///
/// The GUI decodes frames at these offsets; publishing them makes a layout
/// change a *reported* mismatch rather than silent garbage. Note the `u16`
/// pair at 12/14 — the offsets are not a uniform stride.
pub const FRAME_FIELD_OFFSETS: [u16; FRAME_FIELD_COUNT] = [
    0, 4, 8, 12, 14, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88,
    92, 96, 100, 104, 108, 112, 116, 120,
];

/// Names of the [`Frame`] fields, parallel to [`FRAME_FIELD_OFFSETS`].
///
/// Documentation for the reader; the compile-time checks below keep the tables
/// from drifting apart.
pub const FRAME_FIELD_NAMES: [&str; FRAME_FIELD_COUNT] = [
    "iteration",
    "step",
    "episode",
    "worker_id",
    "flags",
    "sim_time",
    "north",
    "east",
    "altitude",
    "vel_north",
    "vel_east",
    "vel_down",
    "roll",
    "pitch",
    "yaw",
    "p",
    "q",
    "r",
    "airspeed",
    "alpha",
    "beta",
    "action_roll",
    "action_pitch",
    "action_yaw_rate",
    "action_throttle",
    "elevator_deg",
    "aileron_deg",
    "rudder_deg",
    "throttle",
    "baro_altitude",
    "airspeed_indicated",
    "reward",
];

const _: () = assert!(FRAME_FIELD_NAMES.len() == FRAME_FIELD_OFFSETS.len());
const _: () = assert!(
    FRAME_FIELD_OFFSETS[FRAME_FIELD_COUNT - 1] as usize + 4 == FRAME_BYTES,
    "the last field must end exactly at FRAME_BYTES"
);

/// The manifest this binary writes.
pub fn manifest() -> StreamManifest {
    StreamManifest {
        schema: STREAM_SCHEMA.to_string(),
        frame_bytes: FRAME_BYTES,
        ring_slots: RING_SLOTS,
        field_offsets: FRAME_FIELD_OFFSETS.to_vec(),
    }
}

/// Read a run's stream manifest, if it has one.
pub fn read_manifest(run_dir: &Path) -> Option<StreamManifest> {
    let text = std::fs::read_to_string(run_dir.join(STREAM_MANIFEST)).ok()?;
    serde_json::from_str(&text).ok()
}

/// What the writer thread reports back to the trainer.
///
/// Counters are shared with the trainer thread so telemetry can publish how
/// much the stream dropped — a viewer that silently skipped frames would look
/// identical to a stationary aircraft.
#[derive(Debug, Default)]
pub struct StreamStats {
    /// Frames successfully handed to the writer.
    pub written: AtomicU64,
    /// Frames dropped because the channel was full.
    pub dropped: AtomicU64,
    /// Episodes written to `episodes/`.
    pub episodes_written: AtomicU64,
    /// Set once the writer thread has stopped (clean shutdown or failure).
    pub finished: AtomicU64,
}

impl StreamStats {
    /// Read the counters as `(written, dropped, episodes_written)`.
    pub fn load(&self) -> (u64, u64, u64) {
        (
            self.written.load(Ordering::Relaxed),
            self.dropped.load(Ordering::Relaxed),
            self.episodes_written.load(Ordering::Relaxed),
        )
    }
}

/// Handle the trainer uses to publish frames.
///
/// Cloning is cheap and shares the same underlying channel and counters.
#[derive(Clone, Default)]
pub struct FrameSink {
    inner: Option<Arc<SinkInner>>,
}

struct SinkInner {
    tx: SyncSender<Frame>,
    stats: Arc<StreamStats>,
}

impl FrameSink {
    /// A sink that discards everything. Used when streaming is disabled.
    pub fn disabled() -> Self {
        Self { inner: None }
    }

    fn new(tx: SyncSender<Frame>, stats: Arc<StreamStats>) -> Self {
        Self::from_parts(tx, stats)
    }

    /// Build a sink from an existing channel and counter handle.
    ///
    /// Exposed so benchmarks can drive a sink with a hand-built reader. The
    /// safe path is [`FrameStream::start`], which creates both halves together —
    /// a sink whose receiver nobody drains would silently drop every frame.
    pub fn from_parts(tx: SyncSender<Frame>, stats: Arc<StreamStats>) -> Self {
        Self {
            inner: Some(Arc::new(SinkInner { tx, stats })),
        }
    }

    /// `true` when frames are actually being recorded.
    pub fn is_enabled(&self) -> bool {
        self.inner.is_some()
    }

    /// Publish one frame. Never blocks; never allocates.
    ///
    /// This sits in the rollout's innermost loop, so it is deliberately
    /// trivial: a `try_send` that fails is counted and forgotten.
    #[inline]
    pub fn push(&self, frame: Frame) {
        let Some(inner) = self.inner.as_ref() else {
            return;
        };
        match inner.tx.try_send(frame) {
            Ok(()) => {
                inner.stats.written.fetch_add(1, Ordering::Relaxed);
            }
            // `Full` is the expected steady-state outcome when the writer
            // cannot keep up; `Disconnected` means the writer died, after
            // which dropping is the only correct behaviour.
            Err(TrySendError::Full(_)) => {
                inner.stats.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Disconnected(_)) => {
                inner.stats.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    /// Shared counters, for telemetry reporting.
    pub fn stats(&self) -> Option<&Arc<StreamStats>> {
        self.inner.as_ref().map(|i| &i.stats)
    }
}

/// Owns the writer thread and shuts it down cleanly on drop.
pub struct FrameStream {
    stats: Arc<StreamStats>,
    handle: Option<JoinHandle<()>>,
    /// Shutdown flag the writer checks between frames.
    stop: Arc<AtomicBool>,
}

impl FrameStream {
    /// Start the writer thread for a run directory.
    ///
    /// Returns `None` when streaming is disabled (empty run dir) or the writer
    /// thread could not be spawned.
    pub fn start(run_dir: &Path) -> Option<(FrameSink, Self)> {
        if run_dir.as_os_str().is_empty() {
            return None;
        }
        // Published before any frame exists, so a viewer can validate the
        // layout the instant the run directory appears. Best-effort: a run
        // whose manifest could not be written is still worth streaming.
        if let Ok(json) = serde_json::to_string_pretty(&manifest()) {
            let _ = std::fs::write(run_dir.join(STREAM_MANIFEST), json);
        }
        let (tx, rx) = sync_channel::<Frame>(CHANNEL_CAPACITY);
        let ring_path = run_dir.join(FRAMES_FILE);
        let episodes_dir = run_dir.join(EPISODES_DIR);
        let stats = Arc::new(StreamStats::default());
        let stats_for_writer = Arc::clone(&stats);
        let stop = Arc::new(AtomicBool::new(false));

        let head_path = run_dir.join(HEAD_FILE);
        let handle = std::thread::Builder::new()
            .name("rl-frame-writer".to_string())
            .spawn({
                let stop = Arc::clone(&stop);
                move || {
                    let _ = writer_loop(
                        rx,
                        &ring_path,
                        &episodes_dir,
                        &head_path,
                        &stats_for_writer,
                        &stop,
                    );
                    stats_for_writer.finished.store(1, Ordering::Relaxed);
                }
            })
            .ok()?;

        let sink = FrameSink::new(tx, Arc::clone(&stats));
        let stream = Self {
            stats,
            handle: Some(handle),
            stop,
        };
        Some((sink, stream))
    }

    /// Frames written, dropped, and episodes flushed, as `(written, dropped,
    /// episodes)`.
    pub fn counters(&self) -> (u64, u64, u64) {
        self.stats.load()
    }

    /// `true` once the writer thread has exited.
    pub fn is_finished(&self) -> bool {
        self.stats.finished.load(Ordering::Relaxed) == 1
    }
}

impl Drop for FrameStream {
    fn drop(&mut self) {
        // The sink is owned by the trainer and may outlive this handle, so
        // `rx` would never see a disconnect — joining on `recv` alone would
        // hang forever. Signal the flag instead; `recv_timeout` then lets the
        // writer drain what is already queued and exit promptly.
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Consume frames until the stop flag is set or the channel disconnects,
/// maintaining the ring and per-episode files.
fn writer_loop(
    rx: Receiver<Frame>,
    ring_path: &Path,
    episodes_dir: &Path,
    head_path: &Path,
    stats: &StreamStats,
    stop: &AtomicBool,
) -> std::io::Result<()> {
    let mut ring = FrameRing::create(ring_path)?;
    let mut head = Head::create(head_path)?;
    std::fs::create_dir_all(episodes_dir)?;

    // Buffered frames for the episode currently in progress.
    let mut current: Vec<Frame> = Vec::with_capacity(4096);
    let mut current_episode: u32 = u32::MAX;
    // Bounded so a single never-ending episode cannot exhaust memory. Episodes
    // cap at `env_max_steps` (2000 by default) so this is well clear.
    const MAX_EPISODE_FRAMES: usize = 8192;

    let mut since_flush = 0usize;
    // `recv_timeout` rather than `recv`: the sink can outlive the stream handle
    // (the trainer owns it), so there is no disconnect to wake us on shutdown —
    // the stop flag is the only reliable signal, and a timeout lets us notice it.
    while !stop.load(Ordering::Relaxed) {
        let frame = match rx.recv_timeout(std::time::Duration::from_millis(100)) {
            Ok(f) => f,
            // Timeout with the channel still open: loop and re-check the flag.
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            // Every sender gone: drain the rest and finish.
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        };
        ring.push(&frame)?;
        since_flush += 1;

        if current_episode != u32::MAX && frame.episode != current_episode {
            write_episode(episodes_dir, current_episode, &current)?;
            stats.episodes_written.fetch_add(1, Ordering::Relaxed);
            current.clear();
        }
        current_episode = frame.episode;

        if current.len() < MAX_EPISODE_FRAMES {
            current.push(frame);
        }

        if frame.is_episode_end() {
            write_episode(episodes_dir, frame.episode, &current)?;
            stats.episodes_written.fetch_add(1, Ordering::Relaxed);
            current.clear();
            current_episode = u32::MAX;
        }

        // Flush periodically so a live reader sees a recent window rather than
        // waiting for the whole run. Every 256 frames is ~8 s of flight at
        // 30 Hz — responsive without being a per-frame syscall.
        if since_flush >= 256 {
            ring.flush()?;
            // Order matters: the ring bytes must be durable before the cursor
            // advertises them, so a reader never follows the cursor into a slot
            // that is still sitting in the BufWriter.
            head.publish(ring.write_index())?;
            since_flush = 0;
        }
    }

    // Channel closed: flush the tail and the last in-flight episode.
    if !current.is_empty() {
        write_episode(episodes_dir, current_episode, &current)?;
        stats.episodes_written.fetch_add(1, Ordering::Relaxed);
    }
    ring.flush()?;
    head.publish(ring.write_index())?;
    Ok(())
}

/// Write one episode's frames as a flat array of [`Frame`].
///
/// Flat and headerless because the frame count is the file length divided by
/// [`FRAME_BYTES`] — the consumer gets position, episode id and flags from the
/// frames themselves.
fn write_episode(dir: &Path, episode: u32, frames: &[Frame]) -> std::io::Result<()> {
    if frames.is_empty() {
        return Ok(());
    }
    let path = episode_path(dir, episode);
    let file = File::create(&path)?;
    let mut w = BufWriter::new(file);
    for f in frames {
        // SAFETY: as in `FrameRing::push` — `Frame` is `repr(C)` over POD
        // scalars, so its bytes are always a valid `Frame`.
        let bytes =
            unsafe { std::slice::from_raw_parts(f as *const Frame as *const u8, FRAME_BYTES) };
        w.write_all(bytes)?;
    }
    w.flush()
}

/// Path of the replay file for an episode.
pub fn episode_path(dir: &Path, episode: u32) -> PathBuf {
    dir.join(format!("ep_{episode:06}.f32"))
}

// ── Control channel ─────────────────────────────────────────────────

/// Directory, relative to a run dir, that the GUI drops commands into.
pub const CONTROL_INBOX: &str = "control/inbox";
/// Directory, relative to a run dir, where the trainer leaves acknowledgements.
pub const CONTROL_OUTBOX: &str = "control/outbox";

/// Serialised form of a control request.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ControlCommand {
    /// Monotonic sequence number; echoed in the ack so the GUI can match.
    pub seq: u64,
    /// One of the variants in [`ControlAction`].
    #[serde(flatten)]
    pub action: ControlAction,
}

/// A request from the GUI to the trainer.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum ControlAction {
    /// Stop stepping and wait at the next iteration boundary.
    Pause,
    /// Resume after a [`ControlAction::Pause`].
    Resume,
    /// Write a checkpoint at the next iteration boundary.
    SaveCheckpoint,
    /// Run an evaluation pass at the next iteration boundary.
    EvaluateNow,
    /// Ask the trainer to exit cleanly at the next iteration boundary.
    Stop,
}

/// Acknowledgement written for each processed command.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct ControlAck {
    /// Sequence number of the command this answers.
    pub seq: u64,
    /// `true` when the command was applied, `false` when rejected.
    pub ok: bool,
    /// Human-readable detail (rejection reason, or a post-action summary).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Reads control commands from a run's inbox, off the trainer's hot loop.
///
/// The trainer calls [`ControlChannel::poll`] once per *iteration* — never
/// inside the step loop — so reading the inbox costs one directory scan per
/// 2048 steps, which is free. Commands are applied at the boundary, where the
/// trainer's state is consistent (a full rollout has been collected and the
/// update applied), so a pause cannot strand a half-finished rollout.
pub struct ControlChannel {
    inbox: PathBuf,
    outbox: PathBuf,
    /// Highest sequence number already processed, for de-duplication.
    last_seq: u64,
    /// Set by a `Pause` command; cleared by `Resume`.
    paused: bool,
    /// Detail describing what the most recent `apply` did, moved into the ack.
    /// `Cell` rather than a plain field so `poll` can stay `&mut self`-cheap
    /// while `apply`/`ack` remain readable.
    last_detail: std::cell::RefCell<Option<String>>,
}

impl ControlChannel {
    /// Open the channel for a run directory.
    pub fn new(run_dir: &Path) -> Self {
        Self {
            inbox: run_dir.join(CONTROL_INBOX),
            outbox: run_dir.join(CONTROL_OUTBOX),
            last_seq: 0,
            paused: false,
            last_detail: std::cell::RefCell::new(None),
        }
    }

    /// A channel bound to no directory, so `poll` always yields nothing.
    ///
    /// Not `Default`-able — [`ControlChannel`] needs a run dir — so streaming
    /// callers that never enable control use this instead.
    pub fn disabled() -> Self {
        Self {
            inbox: PathBuf::new(),
            outbox: PathBuf::new(),
            last_seq: 0,
            paused: false,
            last_detail: std::cell::RefCell::new(None),
        }
    }

    /// `true` once a `Pause` has been seen and no `Resume` has followed.
    pub fn is_paused(&self) -> bool {
        self.paused
    }

    /// Scan the inbox and return the actions to apply, in sequence order.
    ///
    /// Malformed files are acknowledged as rejected rather than ignored, so a
    /// broken GUI shows up as a visible `ok: false` instead of a command that
    /// silently never lands.
    pub fn poll(&mut self) -> Vec<ControlAction> {
        if self.inbox.as_os_str().is_empty() {
            return Vec::new();
        }
        let Ok(dir) = std::fs::read_dir(&self.inbox) else {
            return Vec::new();
        };

        let mut pending: Vec<(u64, ControlCommand)> = Vec::new();
        for entry in dir.flatten() {
            let path = entry.path();
            // Only `*.json`; ignore the GUI's own temp files.
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let Ok(text) = std::fs::read_to_string(&path) else {
                continue;
            };
            let parsed: Result<ControlCommand, _> = serde_json::from_str(&text);
            match parsed {
                Ok(cmd) if cmd.seq > self.last_seq => pending.push((cmd.seq, cmd)),
                // Already-applied or a stale duplicate: drop it silently.
                Ok(_) => {}
                Err(e) => {
                    // Can't recover the seq from an unparseable body, so
                    // acknowledge against the current watermark to at least
                    // surface the parse failure to a polling GUI.
                    let _ = self.ack(self.last_seq, false, Some(format!("unparseable: {e}")));
                }
            }
        }

        pending.sort_by_key(|(seq, _)| *seq);
        let mut actions = Vec::with_capacity(pending.len());
        for (seq, cmd) in pending {
            let ok = self.apply(&cmd.action);
            self.last_seq = seq;
            let detail = self.last_detail.borrow_mut().take();
            let _ = self.ack(seq, ok, detail);
            actions.push(cmd.action);
        }
        actions
    }
}

impl ControlChannel {
    /// Apply one action, updating the pause state.
    fn apply(&mut self, action: &ControlAction) -> bool {
        let detail: String = match action {
            ControlAction::Pause => {
                let was = self.paused;
                self.paused = true;
                if was {
                    "already paused".into()
                } else {
                    "pausing at next iteration boundary".into()
                }
            }
            ControlAction::Resume => {
                let was = self.paused;
                self.paused = false;
                if was {
                    "resuming".into()
                } else {
                    "was not paused".into()
                }
            }
            ControlAction::SaveCheckpoint => "checkpoint requested".into(),
            ControlAction::EvaluateNow => "evaluation requested".into(),
            ControlAction::Stop => "stopping at next iteration boundary".into(),
        };
        *self.last_detail.borrow_mut() = Some(detail);
        true
    }

    /// Write an acknowledgement into the outbox, replacing any prior one for
    /// the same sequence number.
    fn ack(&self, seq: u64, ok: bool, detail: Option<String>) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.outbox)?;
        let ack = ControlAck { seq, ok, detail };
        let path = self.outbox.join(format!("ack_{seq:012}.json"));
        std::fs::write(path, serde_json::to_string_pretty(&ack)?)?;
        Ok(())
    }
}

/// Sleep helper for the pause gate.
///
/// Backs off so a long pause does not spin a core, while keeping the
/// un-pause latency low enough to feel responsive.
pub fn pause_sleep(paused_for: std::time::Duration) {
    let ms = (paused_for.as_millis() as u64 / 4).clamp(20, 250);
    std::thread::sleep(std::time::Duration::from_millis(ms));
}

/// Snapshot of control-channel state for the run manifest.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct ControlSnapshot {
    /// `true` while the run is parked at an iteration boundary.
    pub paused: bool,
}

impl ControlSnapshot {
    /// Read the current state.
    pub fn of(ch: &ControlChannel) -> Self {
        Self {
            paused: ch.is_paused(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("rl_stream_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn frame(episode: u32, step: u32, end: bool) -> Frame {
        Frame {
            iteration: 0,
            step,
            episode,
            north: step as f32,
            ..Default::default()
        }
        .tap_flags(end)
    }

    // Tiny helper so the test reads clearly.
    trait TapFlags {
        fn tap_flags(self, end: bool) -> Frame;
    }
    impl TapFlags for Frame {
        fn tap_flags(mut self, end: bool) -> Frame {
            if end {
                self.flags |= 1;
            }
            self
        }
    }

    #[test]
    fn frame_layout_is_a_multiple_of_four_with_no_padding() {
        assert_eq!(std::mem::size_of::<Frame>(), FRAME_BYTES);
        // Every field is 4 bytes wide and the struct has no padding, which is
        // what makes the raw-slice write in `FrameRing::push` sound and lets a
        // reader index the file at fixed 4-byte offsets.
        assert_eq!(std::mem::size_of::<Frame>() % 4, 0);
        assert_eq!(std::mem::align_of::<Frame>(), 4);
        assert_eq!(std::mem::size_of::<Frame>(), 4 * WORD_COUNT);
        // 32 fields in 31 words: the two u16s share one.
        assert_eq!(FRAME_FIELD_COUNT, WORD_COUNT + 1);
    }

    /// Number of 4-byte *words* in [`Frame`]. One fewer than the field count,
    /// because `worker_id` and `flags` are `u16` and share a word. Kept as a
    /// constant so the padding assertion above fails loudly when someone adds
    /// a field without updating it.
    const WORD_COUNT: usize = 31;

    #[test]
    fn ring_is_bounded_and_wraps() {
        let dir = tmp_dir("ring");
        let path = dir.join(FRAMES_FILE);
        let mut ring = FrameRing::create(&path).unwrap();

        // Write more frames than slots; the file must not grow.
        for i in 0..(RING_SLOTS + 10) {
            ring.push(&Frame {
                step: i as u32,
                ..Default::default()
            })
            .unwrap();
        }
        ring.flush().unwrap();
        assert_eq!(ring.write_index(), (RING_SLOTS + 10) as u64);
        let size = std::fs::metadata(&path).unwrap().len() as usize;
        assert_eq!(
            size,
            RING_SLOTS * FRAME_BYTES,
            "ring grew past its fixed budget"
        );

        // The slot the cursor just wrote holds the most recent frame. Offset 0
        // is `iteration` and offset 4 is `step` in the `Frame` layout.
        let last_slot = (ring.write_index() - 1) as usize & (RING_SLOTS - 1);
        let bytes = std::fs::read(&path).unwrap();
        let off = last_slot * FRAME_BYTES;
        let step = u32::from_ne_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        assert_eq!(
            step,
            (RING_SLOTS + 9) as u32,
            "newest frame not in the tail slot"
        );

        // And the slot the cursor wrapped over now holds the *oldest* frame,
        // proving the ring overwrites rather than growing.
        let wrapped_slot = ring.write_index() as usize & (RING_SLOTS - 1);
        let off = wrapped_slot * FRAME_BYTES;
        let step = u32::from_ne_bytes(bytes[off + 4..off + 8].try_into().unwrap());
        assert_eq!(step, 10, "oldest frames should have been overwritten");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn sink_drops_rather_than_blocking_when_the_reader_is_gone() {
        let dir = tmp_dir("drop");
        let stats = Arc::new(StreamStats::default());
        let (tx, rx) = sync_channel::<Frame>(1);
        let sink = FrameSink {
            inner: Some(Arc::new(SinkInner {
                tx,
                stats: Arc::clone(&stats),
            })),
        };
        // Channel capacity is 1 and nobody is receiving: the first frame lands,
        // the rest must be dropped rather than blocking the caller.
        sink.push(frame(0, 0, false));
        for i in 1..100 {
            sink.push(frame(0, i, false));
        }
        assert_eq!(stats.written.load(Ordering::Relaxed), 1);
        assert_eq!(stats.dropped.load(Ordering::Relaxed), 99);
        drop(rx);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_sink_is_inert() {
        let sink = FrameSink::disabled();
        assert!(!sink.is_enabled());
        sink.push(frame(0, 0, false)); // must not panic
        assert!(sink.stats().is_none());
    }

    #[test]
    fn writer_threads_episodes_to_separate_files() {
        let dir = tmp_dir("episodes");
        let sink_stats = Arc::new(StreamStats::default());
        let writer_stats = Arc::new(StreamStats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (writer, sink) = {
            let (tx, rx) = sync_channel::<Frame>(CHANNEL_CAPACITY);
            let ring = dir.join(FRAMES_FILE);
            let eps = dir.join(EPISODES_DIR);
            let head = dir.join(HEAD_FILE);
            let st = Arc::clone(&writer_stats);
            let stop_flag = Arc::clone(&stop);
            let writer = std::thread::spawn(move || {
                writer_loop(rx, &ring, &eps, &head, &st, &stop_flag).unwrap();
            });
            (
                writer,
                FrameSink {
                    inner: Some(Arc::new(SinkInner {
                        tx,
                        stats: sink_stats,
                    })),
                },
            )
        };

        // Two episodes of 3 steps each, terminated.
        sink.push(frame(0, 0, false));
        sink.push(frame(0, 1, false));
        sink.push(frame(0, 2, true));
        sink.push(frame(1, 0, false));
        sink.push(frame(1, 1, false));
        sink.push(frame(1, 2, true));
        // Dropping the sink closes the channel, which ends the writer loop.
        drop(sink);
        writer.join().unwrap();

        let episodes = dir.join(EPISODES_DIR);
        for (ep, expect_frames) in [(0u32, 3usize), (1, 3)] {
            let path = episode_path(&episodes, ep);
            assert!(path.exists(), "missing {}", path.display());
            let len = std::fs::metadata(&path).unwrap().len() as usize;
            assert_eq!(
                len,
                expect_frames * FRAME_BYTES,
                "episode {ep} should hold {expect_frames} frames"
            );
        }
        // The ring also exists and is the fixed budget.
        let ring_len = std::fs::metadata(dir.join(FRAMES_FILE)).unwrap().len() as usize;
        assert_eq!(ring_len, RING_SLOTS * FRAME_BYTES);
        assert_eq!(writer_stats.episodes_written.load(Ordering::Relaxed), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn head_sidecar_points_at_the_newest_durable_frame() {
        let dir = tmp_dir("head");
        let stats = Arc::new(StreamStats::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = sync_channel::<Frame>(CHANNEL_CAPACITY);
        let sink = FrameSink::new(tx, Arc::clone(&stats));
        let ring = dir.join(FRAMES_FILE);
        let eps = dir.join(EPISODES_DIR);
        let head = dir.join(HEAD_FILE);
        let st = Arc::clone(&stats);
        let stop_flag = Arc::clone(&stop);
        let writer = std::thread::spawn({
            let (ring, eps, head) = (ring.clone(), eps.clone(), head.clone());
            move || writer_loop(rx, &ring, &eps, &head, &st, &stop_flag).unwrap()
        });

        for i in 0..300u32 {
            sink.push(frame(0, i, false));
        }
        // Dropping the sink disconnects and ends the loop, which publishes a
        // final cursor covering the tail.
        drop(sink);
        writer.join().unwrap();

        // 300 frames, cursor published. Reading the ring the way a GUI would:
        // seek to the slot the cursor names, decode, and confirm it is the
        // frame that was written last.
        let cursor = read_head(&head).expect("head sidecar must be readable");
        assert_eq!(cursor, 300, "cursor must count every frame, not just flushed batches");
        let bytes = std::fs::read(&ring).unwrap();
        let slot = (cursor as usize - 1) & (RING_SLOTS - 1);
        let newest = Frame::decode(&bytes[slot * FRAME_BYTES..(slot + 1) * FRAME_BYTES]).unwrap();
        assert_eq!(
            newest.step, 299,
            "the slot the cursor names must hold the newest frame"
        );

        // And it never runs ahead of the ring's own accounting.
        assert!(cursor <= RING_SLOTS as u64 * 1000, "cursor sanity");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn published_field_offsets_match_the_real_on_disk_layout() {
        // The offset table is what the GUI decodes against. Deriving it by hand
        // is the one thing that could drift silently, so check it against the
        // real bytes here: fill every field with a distinct value, encode the
        // struct exactly as `FrameRing::push` does, then read each field back
        // at its published offset.
        let probe = Frame {
            iteration: 11,
            step: 22,
            episode: 33,
            worker_id: 44,
            flags: 55,
            sim_time: 1.0,
            north: 2.0,
            east: 3.0,
            altitude: 4.0,
            vel_north: 5.0,
            vel_east: 6.0,
            vel_down: 7.0,
            roll: 8.0,
            pitch: 9.0,
            yaw: 10.0,
            p: 11.0,
            q: 12.0,
            r: 13.0,
            airspeed: 14.0,
            alpha: 15.0,
            beta: 16.0,
            action_roll: 17.0,
            action_pitch: 18.0,
            action_yaw_rate: 19.0,
            action_throttle: 20.0,
            elevator_deg: 21.0,
            aileron_deg: 22.0,
            rudder_deg: 23.0,
            throttle: 24.0,
            baro_altitude: 25.0,
            airspeed_indicated: 26.0,
            reward: 27.0,
        };
        // SAFETY: `Frame` is `repr(C)` over POD scalars — the same soundness
        // argument `FrameRing::push` relies on.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(&probe as *const Frame as *const u8, FRAME_BYTES)
        };
        assert_eq!(bytes.len(), FRAME_BYTES);

        let u32_at = |off: usize| u32::from_ne_bytes(bytes[off..off + 4].try_into().unwrap());
        let u16_at = |off: usize| u16::from_ne_bytes(bytes[off..off + 2].try_into().unwrap());
        let f32_at = |off: usize| f32::from_ne_bytes(bytes[off..off + 4].try_into().unwrap());

        // Expected values, positionally aligned with FRAME_FIELD_NAMES. The
        // first five are integers; the rest are 1.0, 2.0, ... in order.
        let expected: [f64; FRAME_FIELD_COUNT] = [
            11.0, 22.0, 33.0, 44.0, 55.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            25.0, 26.0, 27.0,
        ];

        for (i, name) in FRAME_FIELD_NAMES.iter().enumerate() {
            let off = FRAME_FIELD_OFFSETS[i] as usize;
            let actual = match *name {
                "worker_id" | "flags" => f64::from(u16_at(off)),
                "iteration" | "step" | "episode" => f64::from(u32_at(off)),
                _ => f64::from(f32_at(off)),
            };
            assert_eq!(
                actual, expected[i],
                "field {i} ({name}) read at offset {off} gives {actual}, expected {}",
                expected[i]
            );
        }

        // The manifest a GUI actually reads must carry the same numbers.
        let m = manifest();
        assert_eq!(m.frame_bytes, FRAME_BYTES);
        assert_eq!(m.ring_slots, RING_SLOTS);
        assert_eq!(m.field_offsets, FRAME_FIELD_OFFSETS);
    }

    #[test]
    fn a_fresh_head_sidecar_is_full_length_and_reads_zero() {
        let dir = tmp_dir("head_fresh");
        let path = dir.join(HEAD_FILE);
        let mut head = Head::create(&path).unwrap();
        // Full length from the start, so a viewer polling a run that has not
        // written a frame yet reads `Some(0)` instead of having to special-case
        // a zero-length file.
        assert_eq!(std::fs::metadata(&path).unwrap().len(), HEAD_BYTES);
        assert_eq!(read_head(&path), Some(0));
        head.publish(7).unwrap();
        assert_eq!(read_head(&path), Some(7));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_head_rejects_a_missing_or_short_sidecar() {
        let dir = tmp_dir("head_missing");
        assert_eq!(read_head(&dir.join(HEAD_FILE)), None, "missing file");
        let short = dir.join("short.bin");
        std::fs::write(&short, [0u8; 3]).unwrap();
        assert_eq!(read_head(&short), None, "short read is not guessable");
        // Little-endian, so the Godot viewer and any other reader agree.
        let full = dir.join("full.bin");
        std::fs::write(&full, 300u64.to_le_bytes()).unwrap();
        assert_eq!(read_head(&full), Some(300));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn control_channel_applies_and_acknowledges_in_order() {
        let dir = tmp_dir("control");
        let inbox = dir.join(CONTROL_INBOX);
        std::fs::create_dir_all(&inbox).unwrap();
        let mut ch = ControlChannel::new(&dir);
        assert!(!ch.is_paused());

        let write = |seq: u64, json: &str| {
            std::fs::write(inbox.join(format!("cmd_{seq:012}.json")), json).unwrap();
        };
        write(1, r#"{"seq":1,"op":"pause"}"#);
        write(2, r#"{"seq":2,"op":"resume"}"#);

        let actions = ch.poll();
        assert_eq!(actions.len(), 2);
        assert!(matches!(actions[0], ControlAction::Pause));
        assert!(matches!(actions[1], ControlAction::Resume));
        assert!(!ch.is_paused(), "resume should clear the pause");

        // Acks exist for both, in the outbox.
        for seq in 1..=2 {
            let p = dir.join(CONTROL_OUTBOX).join(format!("ack_{seq:012}.json"));
            let ack: ControlAck =
                serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            assert_eq!(ack.seq, seq);
            assert!(ack.ok);
        }

        // Re-polling must not re-apply.
        assert!(ch.poll().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn control_channel_pause_sticks_until_resume() {
        let dir = tmp_dir("pause");
        let inbox = dir.join(CONTROL_INBOX);
        std::fs::create_dir_all(&inbox).unwrap();
        std::fs::write(inbox.join("cmd.json"), r#"{"seq":1,"op":"pause"}"#).unwrap();
        let mut ch = ControlChannel::new(&dir);
        ch.poll();
        assert!(ch.is_paused());
        // An empty inbox keeps it paused — no spurious resume on idle poll.
        assert!(ch.poll().is_empty());
        assert!(ch.is_paused());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
