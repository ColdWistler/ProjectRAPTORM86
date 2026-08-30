//! Main menu: a simple Bevy UI screen that lets the user pick a simulation
//! mode. Each mode is an [`AppState`] variant, so future per-component
//! simulation tabs can be added by extending the enum (and [`AppState`]'s
//! [`States`] plumbing in `main.rs`).

use bevy::prelude::*;

use crate::AppState;

/// Marker for the root node of the main-menu screen. Tagged [`StateScoped`]
/// so Bevy removes it automatically when the app leaves [`AppState::MainMenu`].
#[derive(Component)]
struct MenuRoot;

/// Marker for the menu title text.
#[derive(Component)]
struct MenuTitle;

/// Marker for each menu option button.
#[derive(Component)]
struct MenuButton(Target);

/// The simulation/mode a menu button launches.
enum Target {
    FlightSim,
    WindTunnel,
    /// Placeholder for future individual component simulators (wing, rotor,
    /// avionics, etc.). Selecting it currently announces that it is not yet
    /// implemented rather than switching state, so the simulator menu stays
    /// navigable while these are being added.
    Placeholder(&'static str),
}

/// Systems that only run while the main menu is active.
pub struct MainMenuPlugin;

impl Plugin for MainMenuPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(OnEnter(AppState::MainMenu), spawn_main_menu)
            .add_systems(Update, menu_button_system.run_if(in_state(AppState::MainMenu)));
    }
}

/// Spawn the main menu screen: a title and a set of option buttons.
fn spawn_main_menu(mut commands: Commands) {
    commands
        .spawn((
            MenuRoot,
            StateScoped(AppState::MainMenu),
            Node {
                position_type: PositionType::Absolute,
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(18.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.08, 0.12, 1.0)),
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    MenuTitle,
                    Text::new("PROJECT RAPTORM 86"),
                    TextFont {
                        font_size: 56.0,
                        ..default()
                    },
                    TextColor(Color::srgb(0.90, 0.92, 0.98)),
                ))
                .with_children(|p| {
                    p.spawn((
                        Text::new("Select a simulation"),
                        TextFont {
                            font_size: 22.0,
                            ..default()
                        },
                        TextColor(Color::srgb(0.55, 0.62, 0.75)),
                    ));
                });

            spawn_button(parent, "6-DOF Flight Simulator", Target::FlightSim);
            spawn_button(parent, "Wind Tunnel Simulator", Target::WindTunnel);
            spawn_button(
                parent,
                "Wing / Airfoil Simulation       (coming soon)",
                Target::Placeholder("Wing"),
            );
            spawn_button(
                parent,
                "Propeller / Rotor Simulation    (coming soon)",
                Target::Placeholder("Propeller"),
            );
            spawn_button(
                parent,
                "Avionics / Systems Bus           (coming soon)",
                Target::Placeholder("Avionics"),
            );
        });
}

/// Create a single interactive menu option button.
fn spawn_button(parent: &mut ChildBuilder, label: &str, target: Target) {
    parent
        .spawn((
            MenuButton(target),
            Button,
            Node {
                width: Val::Px(420.0),
                padding: UiRect::axes(Val::Px(24.0), Val::Px(14.0)),
                justify_content: JustifyContent::Center,
                ..default()
            },
            BackgroundColor(Color::srgba(0.16, 0.22, 0.32, 1.0)),
        ))
        .with_children(|p| {
            p.spawn((
                Text::new(label),
                TextFont {
                    font_size: 20.0,
                    ..default()
                },
                TextColor(Color::srgb(0.85, 0.88, 0.95)),
            ));
        });
}

/// React to menu button clicks and hover state.
fn menu_button_system(
    mut next_state: ResMut<NextState<AppState>>,
    mut query: Query<
        (&Interaction, &MenuButton, &mut BackgroundColor),
        (Changed<Interaction>, With<Button>),
    >,
) {
    for (interaction, button, mut bg) in &mut query {
        match interaction {
            Interaction::Pressed => match button.0 {
                Target::FlightSim => {
                    next_state.set(AppState::FlightSim);
                }
                Target::WindTunnel => {
                    next_state.set(AppState::WindTunnel);
                }
                Target::Placeholder(name) => {
                    info!("{name} simulation: not implemented yet. Select the simulator to fly.");
                }
            },
            Interaction::Hovered => {
                bg.0 = Color::srgba(0.28, 0.38, 0.55, 1.0);
            }
            Interaction::None => {
                bg.0 = Color::srgba(0.16, 0.22, 0.32, 1.0);
            }
        }
    }
}
