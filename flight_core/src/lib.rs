//! # flight_core
//!
//! A pure-Rust 6-DOF aircraft flight dynamics engine for reinforcement
//! learning training and real-time visualization. This crate implements
//! full rigid-body dynamics using a quaternion-based attitude representation,
//! 1976 US Standard Atmosphere, nonlinear post-stall aerodynamics, and an
//! RK4 numerical integrator.
//!
//! # Standards & Traceability
//! - 1976 U.S. Standard Atmosphere — NASA SP-747.
//! - Post-stall aerodynamics — Viterna & Corrigan, NASA CR 1980.
//! - Atmospheric turbulence — MIL-F-8785C / MIL-HDBK-1797 (Dryden model).
//! - Axis conventions & dynamics — Stevens & Lewis, *Aircraft Control and
//!   Simulation*, 2nd ed.
//! - Integration — Press et al., *Numerical Recipes in C*, 2nd ed., §16.1 (RK4).
//!
//! Each module header carries the specific standard/source it implements.
//! See the per-module `//!` documentation for units and conventions.

pub mod aero;
pub mod atmosphere;
pub mod config;
pub mod env;
pub mod frame;
pub mod geometry;
pub mod hitbox;
pub mod integrator;
pub mod shape;
pub mod simulator;
pub mod skyhook;
pub mod state;
pub mod terrain;
pub mod weather;
pub mod wind;

#[cfg(any(
    feature = "imu",
    feature = "gps",
    feature = "baro",
    feature = "magnetometer",
    feature = "airspeed",
    feature = "servo",
    feature = "esc",
    feature = "battery",
    feature = "flight_controller",
))]
pub mod avionics;

#[cfg(feature = "full-avionics")]
pub mod mission;

pub use aero::ControlInputs;
pub use atmosphere::Atmosphere;
pub use config::{AircraftConfig, MorphingWingConfig, Propulsion, SkyhookConfig, WingPivot};
#[cfg(feature = "full-avionics")]
pub use env::{
    AvionicsAction, AvionicsEnvStep, AvionicsEnvironment, AvionicsObservation, AVIONICS_OBS_DIM,
};
#[cfg(feature = "full-avionics")]
pub use mission::{
    sample_pickup, stall_speed, MissionConfig, MissionFeatures, MissionPhase, MissionScenario,
    MissionSignals, MissionTermination, MISSION_FEATURE_DIM,
};
pub use env::{ControlAction, EnvConfig, EnvStep, Environment, Observation};
pub use frame::FlightFrame;
pub use nalgebra;
pub use simulator::Simulator;
pub use state::AircraftState;
pub use terrain::{Terrain, TerrainGrid, TerrainHill};
pub use weather::{
    CurriculumPhase, ObservationNormalizer, PrecipitationType, TerminationReason, TurbulenceLevel,
    WeatherConfig, WeatherCurriculum, WeatherObservation, WeatherPreset, WeatherSeverity,
    WeatherSystem, WEATHER_OBS_DIM,
};
pub use wind::{TurbulenceIntensity, WindConfig, WindEnvironment};

#[cfg(test)]
mod tests;
