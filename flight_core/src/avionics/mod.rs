//! Simulated UAV avionics layer for reinforcement learning.
//!
//! This module provides a hardware-in-the-loop avionics stack that sits
//! between the RL agent and the 6-DOF physics engine. It simulates the
//! actual components found on a fixed-wing UAV — IMU, GPS, barometer,
//! magnetometer, airspeed sensor, servos, ESC, battery, and a PID flight
//! controller — so the agent learns to operate within real-world hardware
//! constraints.
//!
//! # Architecture
//! ```text
//! RL Agent
//!   │  commands: roll/pitch/yaw-rate/throttle
//!   ▼
//! ┌───────────────────────────────────┐
//! │  AvionicsSystem                   │
//! │  ┌──────────┐  ┌───────────────┐  │
//! │  │ Sensors  │  │ Flight        │  │
//! │  │ IMU/GPS/ │  │ Controller    │  │
//! │  │ Baro/Mag/│◄─│ PID loops     │  │
//! │  │ Airspeed │  │ Mixer         │  │
//! │  └──────────┘  └───────────────┘  │
//! │       │                ▲          │
//! │  observations      servo cmds    │
//! │       ▼                │          │
//! │  ┌──────────┐   ┌───────────┐    │
//! │  │ Battery  │   │ Actuators │    │
//! │  └──────────┘   └───────────┘    │
//! └───────────────────────────────────┘
//!   │  elevator/aileron/rudder/throttle
//!   ▼
//! flight_core 6-DOF Physics (unchanged)
//! ```
//!
//! # Design Principles
//! - **Trait-based**: every component implements `AvionicsComponent` (+ `Sensor` / `Actuator` / `Controller`)
//! - **Data bus**: components communicate only through `AvionicsBus`, never directly
//! - **Feature-gated**: each component compiles behind its own Cargo feature flag
//! - **Self-contained**: each component has its own file, config, and unit tests
//!
//! # Standards & Traceability
//! - Sensor degradation pipeline (lag → noise → drift → bias → gain → quantize)
//!   follows JSBSim's `FGSensor` model.
//! - IMU noise characteristics based on typical MEMS IMUs (ICM-42688-P / MPU-6000).
//! - PID controller architecture follows ArduPilot's `APM_Control` structure.
//! - Battery model is a simple internal-resistance Thevenin equivalent.

pub mod bus;
pub mod sensor_model;
pub mod system;
pub mod traits;

pub use bus::{AvionicsBus, FaultFlag, FaultFlags, FcMode};
pub use system::AvionicsSystem;
pub use traits::{Actuator, AvionicsComponent, Controller, Sensor};

#[cfg(feature = "imu")]
pub mod imu;

#[cfg(feature = "servo")]
pub mod actuator;

#[cfg(feature = "flight_controller")]
pub mod flight_controller;

#[cfg(feature = "gps")]
pub mod gps;

#[cfg(feature = "baro")]
pub mod baro;

#[cfg(feature = "magnetometer")]
pub mod magnetometer;

#[cfg(feature = "airspeed")]
pub mod airspeed;

#[cfg(feature = "esc")]
pub mod esc;

#[cfg(feature = "battery")]
pub mod battery;
